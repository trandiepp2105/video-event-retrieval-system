from __future__ import annotations

from ..indexes.ocr import OCRSearcher
from ..mappings import MappingBundle
from ..scoring import ScoreAccumulator, reciprocal_rank_score
from ..schemas import OCRSearchHit, SearchHit


class EventLevelFusionService:
    def __init__(self, mappings: MappingBundle, *, rrf_k: int) -> None:
        self.mappings = mappings
        self.rrf_k = int(rrf_k)

    def fuse(
        self,
        *,
        event_hits: list[SearchHit],
        caption_hits: list[SearchHit],
        subtitle_hits: list[SearchHit],
        ocr_hits: list[OCRSearchHit],
        event_weight: float,
        caption_weight: float,
        subtitle_weight: float,
        ocr_weight: float,
    ) -> ScoreAccumulator:
        accumulator = ScoreAccumulator()

        for hit in event_hits:
            accumulator.add(
                hit.item_id,
                event_weight * reciprocal_rank_score(hit.rank, self.rrf_k),
                source="event_embedding",
                payload={"rank": hit.rank, "score": hit.score},
            )

        for hit in caption_hits:
            accumulator.add(
                hit.item_id,
                caption_weight * reciprocal_rank_score(hit.rank, self.rrf_k),
                source="caption_embedding",
                payload={"rank": hit.rank, "score": hit.score},
            )

        for hit in subtitle_hits:
            for ref in self.mappings.subtitle_mapping.events_for_subtitle(hit.item_id):
                accumulator.add(
                    ref.event_id,
                    subtitle_weight * reciprocal_rank_score(hit.rank, self.rrf_k) * ref.weight,
                    source=f"subtitle:{hit.item_id}",
                    payload={"rank": hit.rank, "score": hit.score, "weight": ref.weight},
                )

        for hit in ocr_hits:
            event_id = self.mappings.ocr_mapping.event_id_for_ocr(hit.ocr_id)
            accumulator.add(
                event_id,
                ocr_weight * reciprocal_rank_score(hit.rank, self.rrf_k),
                source=f"ocr:{hit.ocr_id}",
                payload={"rank": hit.rank, "score": hit.score, "text": hit.text},
            )

        return accumulator


class OCRQueryExtractor:
    def extract(self, *, raw_query: str, ocr_query: str) -> str:
        if ocr_query.strip():
            return ocr_query.strip()
        return raw_query.strip()
