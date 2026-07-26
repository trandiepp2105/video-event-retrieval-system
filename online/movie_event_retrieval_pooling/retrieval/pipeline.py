from __future__ import annotations

from pathlib import Path
from typing import Any

from ..common import load_json, save_json
from ..indexes.faiss import (
    FaissFullSearcher,
    FaissIndexLoader,
    FaissSubsetSearcher,
    SearchHitMapper,
)
from ..mappings.serializer import MappingSerializer
from ..metadata import MetadataRepository
from ..query_analyzer import SoftTemporalShotQueryAnalyzer, StageQuery
from .event_level import (
    EventLevelFusionService,
    OCRQueryExtractor,
)
from .shot_level import (
    ShotCandidateBuilder,
    StageShotSearchService,
    ShotLevelFusionService,
)
from .shot_temporal import ShotTemporalChainService
from ..schemas import EventResult, ShotResult

from ..config import SearchConfig
from ..embeddings import OpenClipQueryEncoder, SentenceTransformerQueryEncoder
from ..indexes.ocr import MeiliSearchClient, OCRSearcher, SubtitleSearcher


class PoolingMovieEventRetriever:
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

    def search(self, config: SearchConfig) -> dict[str, Any]:
        query = self._load_query(config)
        visual_encoder = None
        caption_encoder = None

        try:
            visual_query = query["translated_query"] or query["raw_query"]
            if visual_query:
                if config.clip_model_path is None:
                    raise ValueError("clip_model_path is required when translated_query/raw_query is provided for visual search")
                visual_encoder = OpenClipQueryEncoder(
                    config.clip_model_path,
                    model_name=config.clip_model_name,
                    device=config.visual_device,
                )
            if query["raw_query"]:
                if config.caption_model_path is None:
                    raise ValueError("caption_model_path is required when raw_query is provided")
                caption_encoder = SentenceTransformerQueryEncoder(config.caption_model_path, device=config.caption_device)
            subtitle_encoder = self._build_subtitle_encoder(config, query["subtitle_query"])
            ocr_searcher, subtitle_searcher = self._build_text_searchers(config)

            event_hits = []
            shot_query_vector = None
            if visual_encoder is not None:
                visual_query_vector = visual_encoder.encode(visual_query)
                shot_query_vector = visual_query_vector
                scores, faiss_ids = self.full_searcher.search(self.event_index, visual_query_vector, config.event_top_k)
                event_hits = self.hit_mapper.map_hits(scores, faiss_ids, self.mappings.event_mapping)

            caption_hits = []
            if caption_encoder is not None:
                caption_query = caption_encoder.encode(query["raw_query"])
                scores, faiss_ids = self.full_searcher.search(self.caption_index, caption_query, config.caption_top_k)
                caption_hits = self.hit_mapper.map_hits(scores, faiss_ids, self.mappings.caption_mapping)

            subtitle_hits = []
            if query["subtitle_query"] and subtitle_searcher is not None:
                subtitle_hits = subtitle_searcher.search(query["subtitle_query"], config.subtitle_top_k)
            elif subtitle_encoder is not None:
                subtitle_query = subtitle_encoder.encode(query["subtitle_query"])
                scores, faiss_ids = self.full_searcher.search(self.subtitle_index, subtitle_query, config.subtitle_top_k)
                subtitle_hits = self.hit_mapper.map_hits(scores, faiss_ids, self.mappings.subtitle_mapping_ids)

            ocr_query = OCRQueryExtractor().extract(raw_query=query["raw_query"], ocr_query=query["ocr_query"])
            ocr_hits = ocr_searcher.search(ocr_query, config.ocr_top_k) if ocr_query else []

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
            top_event_candidates = self._format_event_candidates(ranked_events)

            shot_results: list[ShotResult] = []
            temporal_payload: dict[str, Any] = {"enabled": bool(config.enable_shot_temporal), "query_analysis": None, "stage_results": [], "top_chains": []}
            if shot_query_vector is not None and candidate_event_ids:
                candidate_shot_ids = ShotCandidateBuilder(self.mappings).build(
                    event_ids=candidate_event_ids,
                    video_ids=candidate_video_ids,
                )
                allowed_faiss_ids = self.mappings.shot_mapping.faiss_ids_from_item_ids(candidate_shot_ids)
                scores, faiss_ids = self.subset_searcher.search(
                    self.shot_index,
                    shot_query_vector,
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
                if config.enable_shot_temporal and query["raw_query"]:
                    temporal_payload = self._run_shot_temporal_search(
                        config=config,
                        raw_query=query["raw_query"],
                        candidate_shot_ids=candidate_shot_ids,
                        visual_encoder=visual_encoder,
                        subtitle_searcher=subtitle_searcher,
                        ocr_searcher=ocr_searcher,
                    )

            final_events = self._aggregate_final_events(
                ranked_events=ranked_events,
                shot_results=shot_results,
                event_evidence=event_fusion.evidence,
                temporal_payload=temporal_payload,
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
                    "top_events": top_event_candidates[:20],
                },
                "shot_level": {
                    "top_shots": [shot.__dict__ for shot in shot_results[:20]],
                },
                "shot_temporal": temporal_payload,
                "final_events": [self._event_result_to_dict(result) for result in final_events],
            }
            if config.output_json is not None:
                save_json(payload, config.output_json)
            return payload
        finally:
            pass

    def _build_text_searchers(self, config: SearchConfig) -> tuple[OCRSearcher, SubtitleSearcher | None]:
        ocr_config_path = self.store_dir / "indexes" / "ocr" / "config.json"
        ocr_config = load_json(ocr_config_path)
        meilisearch_url = config.meilisearch_url or str(ocr_config["url"])
        meilisearch_index_name = config.meilisearch_index_name or str(ocr_config["index_name"])
        subtitle_searcher: SubtitleSearcher | None = None
        client = MeiliSearchClient(
            base_url=meilisearch_url,
            api_key=config.meilisearch_api_key,
        )
        if config.subtitle_backend == "meilisearch":
            subtitle_config = load_json(self.store_dir / "indexes" / "subtitle_text" / "config.json")
            subtitle_index_name = config.subtitle_meilisearch_index_name or str(subtitle_config["index_name"])
            subtitle_searcher = SubtitleSearcher(client=client, index_uid=subtitle_index_name)
        return OCRSearcher(client=client, index_uid=meilisearch_index_name), subtitle_searcher

    @staticmethod
    def _build_subtitle_encoder(config: SearchConfig, subtitle_query: str) -> SentenceTransformerQueryEncoder | None:
        if not subtitle_query:
            return None
        if config.subtitle_backend == "meilisearch":
            return None
        if config.subtitle_model_path is None:
            raise ValueError("subtitle_model_path is required when subtitle_backend=embedding and subtitle_query is provided")
        return SentenceTransformerQueryEncoder(config.subtitle_model_path, device=config.subtitle_device)

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
                    shot_order=int(shot.shot_order),
                    start_time_sec=shot.start_time_sec,
                    end_time_sec=shot.end_time_sec,
                    score=float(score),
                    evidence=evidence.get(shot_id, {}),
                )
            )
        return results

    def _format_event_candidates(self, ranked_events: list[tuple[str, float]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for event_id, score in ranked_events:
            event = self.metadata.events[event_id]
            results.append(
                {
                    "event_id": event_id,
                    "video_id": event.video_id,
                    "start_time_sec": float(event.start_time_sec),
                    "end_time_sec": float(event.end_time_sec),
                    "score": float(score),
                    "shot_ids": list(event.shot_ids),
                }
            )
        return results

    def _aggregate_final_events(
        self,
        *,
        ranked_events: list[tuple[str, float]],
        shot_results: list[ShotResult],
        event_evidence: dict[str, dict[str, Any]],
        temporal_payload: dict[str, Any] | None = None,
    ) -> list[EventResult]:
        shot_by_event: dict[str, list[ShotResult]] = {}
        for shot in shot_results:
            shot_by_event.setdefault(shot.event_id, []).append(shot)

        final_scores: dict[str, float] = dict(ranked_events)
        for event_id, shots in shot_by_event.items():
            final_scores[event_id] = final_scores.get(event_id, 0.0) + max(shot.score for shot in shots)

        for chain in (temporal_payload or {}).get("top_chains", []):
            for chain_item in chain.get("chain", []):
                event_id = str(chain_item["event_id"])
                final_scores[event_id] = final_scores.get(event_id, 0.0) + float(chain.get("score", 0.0))

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
                        "temporal_chains": [
                            chain
                            for chain in (temporal_payload or {}).get("top_chains", [])
                            if any(str(item["event_id"]) == event_id for item in chain.get("chain", []))
                        ][:3],
                    },
                )
            )
        return results

    def _run_shot_temporal_search(
        self,
        *,
        config: SearchConfig,
        raw_query: str,
        candidate_shot_ids: list[str],
        visual_encoder: OpenClipQueryEncoder | None,
        subtitle_searcher: SubtitleSearcher | None,
        ocr_searcher: OCRSearcher,
    ) -> dict[str, Any]:
        if not config.temporal_query_model_path:
            raise ValueError("temporal_query_model_path is required when enable_shot_temporal=True")
        analyzer = SoftTemporalShotQueryAnalyzer(
            config.temporal_query_model_path,
            device_map=config.temporal_query_device_map,
            torch_dtype=config.temporal_query_torch_dtype,
            max_new_tokens=config.temporal_query_max_new_tokens,
        )
        analyzed = analyzer.analyze(raw_query)
        if not analyzed:
            return {"enabled": True, "query_analysis": None, "stage_results": [], "top_chains": []}

        stages = [StageQuery.from_dict(item) for item in analyzed.get("stages", [])]
        stages = [stage for stage in stages if not stage.is_empty()]
        if not stages:
            return {"enabled": True, "query_analysis": analyzed, "stage_results": [], "top_chains": []}

        allowed_shot_ids = set(candidate_shot_ids)
        allowed_faiss_ids = self.mappings.shot_mapping.faiss_ids_from_item_ids(candidate_shot_ids)
        stage_service = StageShotSearchService(self.mappings)
        per_stage_results: list[list[ShotResult]] = []

        for stage_index, stage in enumerate(stages):
            shot_hits = []
            if stage.visual and visual_encoder is not None:
                query_vector = visual_encoder.encode(stage.visual)
                scores, faiss_ids = self.subset_searcher.search(
                    self.shot_index,
                    query_vector,
                    allowed_faiss_ids,
                    config.stage_shot_top_k,
                )
                shot_hits = self.hit_mapper.map_hits(scores, faiss_ids, self.mappings.shot_mapping)

            subtitle_hits = []
            if stage.subtitle:
                if subtitle_searcher is not None:
                    subtitle_hits = subtitle_searcher.search(stage.subtitle, config.subtitle_top_k)
                elif config.subtitle_backend != "meilisearch":
                    subtitle_encoder = self._build_subtitle_encoder(config, stage.subtitle)
                    if subtitle_encoder is not None:
                        subtitle_query = subtitle_encoder.encode(stage.subtitle)
                        scores, faiss_ids = self.full_searcher.search(self.subtitle_index, subtitle_query, config.subtitle_top_k)
                        subtitle_hits = self.hit_mapper.map_hits(scores, faiss_ids, self.mappings.subtitle_mapping_ids)

            ocr_hits = []
            if stage.ocr:
                ocr_hits = ocr_searcher.search(stage.ocr, config.ocr_top_k)

            fused = stage_service.fuse_stage_hits(
                shot_hits=shot_hits,
                subtitle_hits=subtitle_hits,
                ocr_hits=ocr_hits,
                allowed_shot_ids=allowed_shot_ids,
                shot_weight=config.stage_visual_weight,
                subtitle_weight=config.stage_subtitle_weight,
                ocr_weight=config.stage_ocr_weight,
                rrf_k=config.rrf_k,
            )
            stage_results = stage_service.to_shot_results(
                ranked_shots=fused.sorted_items(),
                evidence=fused.evidence,
                metadata=self.metadata,
                top_k=config.stage_shot_top_k,
            )
            per_stage_results.append(stage_results)

        chains = ShotTemporalChainService(self.metadata).search(
            stage_results=per_stage_results,
            top_k=config.temporal_chain_top_k,
            window_size_shots=config.temporal_window_shots,
            lambda_skip=config.temporal_lambda_skip,
            min_stage_gap_shots=config.temporal_min_stage_gap_shots,
            group_gap_shots=config.temporal_group_gap_shots,
        )
        return {
            "enabled": True,
            "query_analysis": analyzed,
            "stage_results": [
                {
                    "stage_index": stage_index,
                    "query": {
                        "visual": stages[stage_index].visual,
                        "ocr": stages[stage_index].ocr,
                        "subtitle": stages[stage_index].subtitle,
                    },
                    "top_shots": [shot.__dict__ for shot in results[:10]],
                }
                for stage_index, results in enumerate(per_stage_results)
            ],
            "top_chains": chains,
        }

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
