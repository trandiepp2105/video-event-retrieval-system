from __future__ import annotations

from ...schemas import SearchHit
from .meilisearch_client import MeiliSearchClient


class SubtitleSearcher:
    def __init__(self, client: MeiliSearchClient, index_uid: str) -> None:
        self.client = client
        self.index_uid = index_uid

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        query = " ".join(query.split())
        if not query:
            return []
        payload = self.client.search(index_uid=self.index_uid, query=query, limit=top_k)
        hits: list[SearchHit] = []
        for rank, item in enumerate(payload.get("hits", []), start=1):
            ranking_score = item.get("_rankingScore")
            if ranking_score is None:
                ranking_score = max(float(top_k - rank + 1), 1.0) / float(max(top_k, 1))
            hits.append(
                SearchHit(
                    item_id=str(item["subtitle_id"]),
                    faiss_id=-1,
                    score=float(ranking_score),
                    rank=rank,
                    text=str(item.get("text", "")),
                )
            )
        return hits
