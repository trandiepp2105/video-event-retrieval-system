from __future__ import annotations

from rapidfuzz import fuzz

from ...schemas import SearchHit

from .meilisearch_client import MeiliSearchClient


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).lower()


class SubtitleFuzzyScorer:
    def __init__(self, w_partial: float = 1.0, w_ngrams: float = 1.0, w_token: float = 1.0) -> None:
        self.w_partial = float(w_partial)
        self.w_ngrams = float(w_ngrams)
        self.w_token = float(w_token)
        if self.w_partial < 0 or self.w_ngrams < 0 or self.w_token < 0:
            raise ValueError("Subtitle fuzzy weights must be non-negative")
        if (self.w_partial + self.w_ngrams + self.w_token) == 0:
            raise ValueError("At least one subtitle fuzzy weight must be positive")

    def score(self, query: str, document: str) -> float:
        query_norm = normalize_text(query)
        doc_norm = normalize_text(document)
        if not query_norm or not doc_norm:
            return 0.0
        if query_norm in doc_norm:
            return 1.0

        q_split = query_norm.split()
        d_split = doc_norm.split()
        q_no_space = "".join(q_split)
        d_no_space = "".join(d_split)

        partial = self._partial_ratio(q_no_space, d_no_space)
        ngrams = self._ngrams_ratio(q_split, d_split)
        token = fuzz.token_set_ratio(query_norm, doc_norm) / 100.0

        total = self.w_partial + self.w_ngrams + self.w_token
        return (
            self.w_partial * partial
            + self.w_ngrams * ngrams
            + self.w_token * token
        ) / total

    @staticmethod
    def _partial_ratio(query_no_space: str, doc_no_space: str) -> float:
        if not query_no_space:
            return 0.0
        if len(query_no_space) >= len(doc_no_space):
            return fuzz.ratio(query_no_space, doc_no_space) / 100.0
        return fuzz.partial_ratio(query_no_space, doc_no_space) / 100.0

    def _ngrams_ratio(self, q_split: list[str], d_split: list[str]) -> float:
        if not q_split:
            return 0.0
        query_text = " ".join(q_split)
        query_len = len(q_split)
        n_values = (query_len,) if query_len <= 1 else (query_len - 1, query_len)
        best = 0.0
        for n in n_values:
            for i in range(len(d_split) - n + 1):
                candidate = " ".join(d_split[i : i + n])
                best = max(best, fuzz.ratio(query_text, candidate))
        return best / 100.0


class SubtitleSearcher:
    def __init__(self, client: MeiliSearchClient, index_uid: str, *, candidate_multiplier: int = 5) -> None:
        self.client = client
        self.index_uid = index_uid
        self.candidate_multiplier = max(int(candidate_multiplier), 1)
        self.scorer = SubtitleFuzzyScorer()

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        query_norm = normalize_text(query)
        if not query_norm:
            return []

        retrieve_k = max(int(top_k), min(int(top_k) * self.candidate_multiplier, 1000))
        payload = self.client.search(index_uid=self.index_uid, query=query_norm, limit=retrieve_k)

        rescored: list[tuple[float, str, str]] = []
        for item in payload.get("hits", []):
            subtitle_id = str(item["subtitle_id"])
            text = str(item.get("text", ""))
            score = self.scorer.score(query_norm, text)
            if score > 0:
                rescored.append((score, subtitle_id, text))

        rescored.sort(key=lambda value: value[0], reverse=True)

        hits: list[SearchHit] = []
        for rank, (score, subtitle_id, text) in enumerate(rescored[: int(top_k)], start=1):
            hits.append(
                SearchHit(
                    item_id=subtitle_id,
                    faiss_id=-1,
                    score=float(score),
                    rank=rank,
                    text=text,
                )
            )
        return hits
