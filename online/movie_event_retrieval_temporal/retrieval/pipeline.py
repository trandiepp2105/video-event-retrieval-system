from __future__ import annotations

from pathlib import Path
from typing import Any

from ..common import load_json, save_json
from ..config import SearchConfig
from ..embeddings import SentenceTransformerQueryEncoder, TemporalQueryEncoder
from ..indexes.faiss import FaissFullSearcher, FaissIndexLoader, FaissSubsetSearcher, SearchHitMapper
from ..indexes.ocr import OCRSearcher, OCRStore
from ..mappings.serializer import MappingSerializer
from ..metadata import MetadataRepository
from ..retrieval.event_level import EventLevelFusionService, OCRQueryExtractor
from ..retrieval.shot_level import ShotCandidateBuilder, ShotLevelFusionService
from ..schemas import EventResult, ShotResult


class TemporalMovieEventRetriever:
    def __init__(self, metadata: MetadataRepository, store_dir: Path) -> None:
        self.metadata = metadata
        self.store_dir = store_dir
        self.index_loader = FaissIndexLoader()
        self.full_searcher = FaissFullSearcher()
        self.subset_searcher = FaissSubsetSearcher()
        self.hit_mapper = SearchHitMapper()
        self.mappings = MappingSerializer().load(store_dir)

        self.event_index = self.index_loader.load(store_dir / "indexes" / "faiss" / "event.faiss")
        self.caption_index = self.index_loader.load(store_dir / "indexes" / "faiss" / "caption.faiss")
        self.shot_index = self.index_loader.load(store_dir / "indexes" / "faiss" / "shot.faiss")
        self.subtitle_index = self.index_loader.load(store_dir / "indexes" / "faiss" / "subtitle.faiss")
        self.ocr_searcher = OCRSearcher(OCRStore.load(store_dir / "indexes" / "ocr" / "documents.json"))

    def search(self, config: SearchConfig) -> dict[str, Any]:
        query = self._load_query(config)
        temporal_encoder = None
        caption_encoder = None
        subtitle_encoder = None

        if query["translated_query"]:
            if config.temporal_checkpoint_path is None:
                raise ValueError("temporal_checkpoint_path is required when translated_query is provided")
            temporal_encoder = TemporalQueryEncoder(
                checkpoint_path=config.temporal_checkpoint_path,
                clip_model_path_override=config.clip_model_path_override,
                device=config.event_device,
            )
        if query["raw_query"]:
            if config.caption_model_path is None:
                raise ValueError("caption_model_path is required when raw_query is provided")
            caption_encoder = SentenceTransformerQueryEncoder(config.caption_model_path, device=config.caption_device)
        if query["subtitle_query"]:
            if config.subtitle_model_path is None:
                raise ValueError("subtitle_model_path is required when subtitle_query is provided")
            subtitle_encoder = SentenceTransformerQueryEncoder(config.subtitle_model_path, device=config.subtitle_device)

        event_hits = []
        if temporal_encoder is not None:
            event_query = temporal_encoder.encode_event_query(query["translated_query"])
            scores, faiss_ids = self.full_searcher.search(self.event_index, event_query, config.event_top_k)
            event_hits = self.hit_mapper.map_hits(scores, faiss_ids, self.mappings.event_mapping)

        caption_hits = []
        if caption_encoder is not None:
            caption_query = caption_encoder.encode(query["raw_query"])
            scores, faiss_ids = self.full_searcher.search(self.caption_index, caption_query, config.caption_top_k)
            caption_hits = self.hit_mapper.map_hits(scores, faiss_ids, self.mappings.caption_mapping)

        subtitle_hits = []
        if subtitle_encoder is not None:
            subtitle_query = subtitle_encoder.encode(query["subtitle_query"])
            scores, faiss_ids = self.full_searcher.search(self.subtitle_index, subtitle_query, config.subtitle_top_k)
            subtitle_hits = self.hit_mapper.map_hits(scores, faiss_ids, self.mappings.subtitle_mapping_ids)

        ocr_query = OCRQueryExtractor().extract(raw_query=query["raw_query"], ocr_query=query["ocr_query"])
        ocr_hits = self.ocr_searcher.search(ocr_query, config.ocr_top_k) if ocr_query else []

        event_fusion = EventLevelFusionService(self.mappings, rrf_k=config.rrf_k).fuse(
            event_hits=event_hits,
            caption_hits=caption_hits,
            subtitle_hits=subtitle_hits,
            ocr_hits=ocr_hits,
            event_weight=config.event_weight,
            caption_weight=config.caption_weight,
            subtitle_weight=config.subtitle_weight,
            ocr_weight=config.ocr_weight,
        )

        ranked_events = event_fusion.sorted_items()
        candidate_event_ids = [event_id for event_id, _score in ranked_events[: config.candidate_event_top_k]]
        candidate_video_ids = self._rank_videos_from_events(ranked_events)[: config.candidate_video_top_k]

        shot_results: list[ShotResult] = []
        if temporal_encoder is not None and candidate_event_ids:
            shot_query = temporal_encoder.encode_shot_query(query["translated_query"])
            candidate_shot_ids = ShotCandidateBuilder(self.mappings).build(
                event_ids=candidate_event_ids,
                video_ids=candidate_video_ids,
            )
            allowed_faiss_ids = self.mappings.shot_mapping.faiss_ids_from_item_ids(candidate_shot_ids)
            scores, faiss_ids = self.subset_searcher.search(
                self.shot_index,
                shot_query,
                allowed_faiss_ids,
                config.shot_top_k,
            )
            shot_hits = self.hit_mapper.map_hits(scores, faiss_ids, self.mappings.shot_mapping)
            shot_fusion = ShotLevelFusionService(self.mappings, rrf_k=config.rrf_k).fuse(
                shot_hits=shot_hits,
                subtitle_hits=subtitle_hits,
                ocr_hits=ocr_hits,
                parent_event_scores=dict(ranked_events),
                shot_weight=config.shot_weight,
                subtitle_weight=config.subtitle_weight,
                ocr_weight=config.ocr_weight,
                parent_event_weight=config.parent_event_weight,
            )
            shot_results = self._format_shot_results(shot_fusion.sorted_items(), shot_fusion.evidence)
        else:
            shot_fusion = None

        final_events = self._aggregate_final_events(
            ranked_events=ranked_events,
            shot_results=shot_results,
            event_evidence=event_fusion.evidence,
        )[: config.final_top_k]

        payload = {
            "query": query,
            "event_level": {
                "event_embedding_hits": [hit.__dict__ for hit in event_hits[:20]],
                "caption_hits": [hit.__dict__ for hit in caption_hits[:20]],
                "subtitle_hits": [hit.__dict__ for hit in subtitle_hits[:20]],
                "ocr_hits": [hit.__dict__ for hit in ocr_hits[:20]],
            },
            "candidates": {
                "event_ids": candidate_event_ids,
                "video_ids": candidate_video_ids,
            },
            "shot_level": {
                "top_shots": [shot.__dict__ for shot in shot_results[:20]],
            },
            "final_events": [self._event_result_to_dict(result) for result in final_events],
        }
        if config.output_json is not None:
            save_json(payload, config.output_json)
        return payload

    def _load_query(self, config: SearchConfig) -> dict[str, str]:
        query = {
            "raw_query": config.raw_query.strip(),
            "translated_query": config.translated_query.strip(),
            "subtitle_query": config.subtitle_query.strip(),
            "ocr_query": config.ocr_query.strip(),
        }
        if config.query_json is not None:
            payload = load_json(config.query_json)
            query.update({key: str(payload.get(key, query[key])).strip() for key in query})
        return query

    def _rank_videos_from_events(self, ranked_events: list[tuple[str, float]]) -> list[str]:
        video_scores: dict[str, float] = {}
        for event_id, score in ranked_events:
            video_id = self.mappings.hierarchy.video_id_for_event(event_id)
            video_scores[video_id] = max(video_scores.get(video_id, 0.0), float(score))
        return [video_id for video_id, _score in sorted(video_scores.items(), key=lambda item: item[1], reverse=True)]

    def _format_shot_results(self, ranked_shots: list[tuple[str, float]], evidence: dict[str, dict[str, Any]]) -> list[ShotResult]:
        results: list[ShotResult] = []
        for shot_id, score in ranked_shots:
            shot = self.metadata.shots[shot_id]
            results.append(
                ShotResult(
                    shot_id=shot_id,
                    event_id=shot.event_id,
                    video_id=shot.video_id,
                    start_time_sec=shot.start_time_sec,
                    end_time_sec=shot.end_time_sec,
                    score=float(score),
                    evidence=evidence.get(shot_id, {}),
                )
            )
        return results

    def _aggregate_final_events(
        self,
        *,
        ranked_events: list[tuple[str, float]],
        shot_results: list[ShotResult],
        event_evidence: dict[str, dict[str, Any]],
    ) -> list[EventResult]:
        shot_by_event: dict[str, list[ShotResult]] = {}
        for shot in shot_results:
            shot_by_event.setdefault(shot.event_id, []).append(shot)

        final_scores: dict[str, float] = dict(ranked_events)
        for event_id, shots in shot_by_event.items():
            final_scores[event_id] = final_scores.get(event_id, 0.0) + max(shot.score for shot in shots)

        results: list[EventResult] = []
        for event_id, score in sorted(final_scores.items(), key=lambda item: item[1], reverse=True):
            event = self.metadata.events[event_id]
            results.append(
                EventResult(
                    event_id=event_id,
                    video_id=event.video_id,
                    start_time_sec=event.start_time_sec,
                    end_time_sec=event.end_time_sec,
                    score=float(score),
                    shot_ids=list(event.shot_ids),
                    evidence={
                        "event_level": event_evidence.get(event_id, {}),
                        "top_shots": [shot.__dict__ for shot in sorted(shot_by_event.get(event_id, []), key=lambda item: item.score, reverse=True)[:5]],
                    },
                )
            )
        return results

    @staticmethod
    def _event_result_to_dict(result: EventResult) -> dict[str, Any]:
        return {
            "event_id": result.event_id,
            "video_id": result.video_id,
            "start_time_sec": result.start_time_sec,
            "end_time_sec": result.end_time_sec,
            "score": result.score,
            "shot_ids": result.shot_ids,
            "evidence": result.evidence,
        }
