from __future__ import annotations

import numpy as np

from ..mappings import MappingBundle
from ..scoring import ScoreAccumulator, reciprocal_rank_score
from ..schemas import OCRSearchHit, SearchHit


class ShotCandidateBuilder:
    def __init__(self, mappings: MappingBundle) -> None:
        self.mappings = mappings

    def build(self, *, event_ids: list[str], video_ids: list[str]) -> list[str]:
        candidate_shots: set[str] = set()
        for event_id in event_ids:
            candidate_shots.update(self.mappings.hierarchy.shot_ids_for_event(event_id))
        for video_id in video_ids:
            candidate_shots.update(self.mappings.hierarchy.shot_ids_for_video(video_id))
        return sorted(candidate_shots)


class ShotLevelFusionService:
    def __init__(self, mappings: MappingBundle, *, rrf_k: int) -> None:
        self.mappings = mappings
        self.rrf_k = int(rrf_k)

    def fuse(
        self,
        *,
        shot_hits: list[SearchHit],
        subtitle_hits: list[SearchHit],
        ocr_hits: list[OCRSearchHit],
        parent_event_scores: dict[str, float],
        shot_weight: float,
        subtitle_weight: float,
        ocr_weight: float,
        parent_event_weight: float,
    ) -> ScoreAccumulator:
        accumulator = ScoreAccumulator()

        for hit in shot_hits:
            accumulator.add(
                hit.item_id,
                shot_weight * reciprocal_rank_score(hit.rank, self.rrf_k),
                source="shot_embedding",
                payload={"rank": hit.rank, "score": hit.score},
            )

        for hit in subtitle_hits:
            for ref in self.mappings.subtitle_mapping.shots_for_subtitle(hit.item_id):
                accumulator.add(
                    ref.shot_id,
                    subtitle_weight * reciprocal_rank_score(hit.rank, self.rrf_k) * ref.weight,
                    source=f"subtitle:{hit.item_id}",
                    payload={"rank": hit.rank, "score": hit.score, "weight": ref.weight},
                )

        for hit in ocr_hits:
            shot_id = self.mappings.ocr_mapping.shot_id_for_ocr(hit.ocr_id)
            accumulator.add(
                shot_id,
                ocr_weight * reciprocal_rank_score(hit.rank, self.rrf_k),
                source=f"ocr:{hit.ocr_id}",
                payload={"rank": hit.rank, "score": hit.score, "text": hit.text},
            )

        for shot_id in list(accumulator.scores):
            event_id = self.mappings.hierarchy.event_id_for_shot(shot_id)
            prior = parent_event_scores.get(event_id, 0.0)
            if prior > 0.0:
                accumulator.add(
                    shot_id,
                    parent_event_weight * prior,
                    source="parent_event",
                    payload={"event_id": event_id, "score": prior},
                )
        return accumulator

    def allowed_faiss_ids(self, shot_item_ids: list[str], shot_mapping_ids) -> np.ndarray:
        return shot_mapping_ids.faiss_ids_from_item_ids(shot_item_ids)
