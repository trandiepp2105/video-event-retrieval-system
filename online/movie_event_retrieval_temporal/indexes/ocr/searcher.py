from __future__ import annotations

from collections import Counter

from ...schemas import OCRSearchHit
from .store import OCRStore


class OCRSearcher:
    def __init__(self, store: OCRStore) -> None:
        self.store = store

    def search(self, query: str, top_k: int) -> list[OCRSearchHit]:
        normalized_query = " ".join(query.lower().split())
        if not normalized_query:
            return []
        query_terms = normalized_query.split()
        hits: list[OCRSearchHit] = []
        for document in self.store.documents:
            text = str(document.get("text_clean", "")).lower()
            if not text:
                continue
            score = self._token_overlap_score(query_terms, text.split())
            if score <= 0.0:
                continue
            hits.append(
                OCRSearchHit(
                    ocr_id=str(document["ocr_id"]),
                    score=score,
                    rank=0,
                    text=str(document.get("text_raw", "")),
                )
            )
        hits.sort(key=lambda item: item.score, reverse=True)
        return [OCRSearchHit(hit.ocr_id, hit.score, index + 1, hit.text) for index, hit in enumerate(hits[:top_k])]

    @staticmethod
    def _token_overlap_score(query_terms: list[str], document_terms: list[str]) -> float:
        doc_counter = Counter(document_terms)
        match_count = 0
        for term in query_terms:
            if doc_counter[term] > 0:
                doc_counter[term] -= 1
                match_count += 1
        return float(match_count) / float(max(len(query_terms), 1))
