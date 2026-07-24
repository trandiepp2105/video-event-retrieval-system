from __future__ import annotations

from ...schemas import OCRSearchHit
from .meilisearch_client import MeiliSearchClient


class OCRSearcher:
    def __init__(self, client: MeiliSearchClient, index_uid: str) -> None:
        self.client = client
        self.index_uid = index_uid

    def search(self, query: str, top_k: int) -> list[OCRSearchHit]:
        query = " ".join(query.split())
        if not query:
            return []
        payload = self.client.search(
            index_uid=self.index_uid,
            query=query,
            limit=top_k,
            matching_strategy="last",
            attributes_to_retrieve=["*"],
            show_ranking_score=True,
        )
        hits: list[OCRSearchHit] = []
        for rank, item in enumerate(payload.get("hits", []), start=1):
            ranking_score = item.get("_rankingScore")
            if ranking_score is None:
                ranking_score = max(float(top_k - rank + 1), 1.0) / float(max(top_k, 1))
            hits.append(
                OCRSearchHit(
                    ocr_id=str(item["ocr_id"]),
                    score=float(ranking_score),
                    rank=rank,
                    text=str(item.get("text_raw", "")),
                )
            )
        return hits
