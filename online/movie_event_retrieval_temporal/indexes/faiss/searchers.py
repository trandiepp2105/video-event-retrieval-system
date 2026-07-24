from __future__ import annotations

import faiss
import numpy as np

from ...common import to_contiguous_float32
from ...mappings import FaissIdMapping
from ...schemas import SearchHit


class FaissFullSearcher:
    def search(self, index: faiss.Index, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        query = to_contiguous_float32(query)
        return index.search(query, int(top_k))


class FaissSubsetSearcher:
    def search(
        self,
        index: faiss.Index,
        query: np.ndarray,
        allowed_faiss_ids: np.ndarray,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        query = to_contiguous_float32(query)
        allowed_ids = np.asarray(allowed_faiss_ids, dtype=np.int64)
        allowed_ids = np.unique(allowed_ids)
        allowed_ids = allowed_ids[(allowed_ids >= 0) & (allowed_ids < index.ntotal)]
        allowed_ids = np.ascontiguousarray(allowed_ids)
        if allowed_ids.size == 0:
            return (
                np.empty((query.shape[0], 0), dtype=np.float32),
                np.empty((query.shape[0], 0), dtype=np.int64),
            )
        selector = faiss.IDSelectorArray(allowed_ids.size, faiss.swig_ptr(allowed_ids))
        params = faiss.SearchParameters()
        params.sel = selector
        actual_k = min(int(top_k), int(allowed_ids.size))
        return index.search(query, actual_k, params=params)


class SearchHitMapper:
    def map_hits(
        self,
        scores: np.ndarray,
        faiss_ids: np.ndarray,
        id_mapping: FaissIdMapping,
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        if scores.size == 0:
            return hits
        for rank, (score, faiss_id) in enumerate(zip(scores[0].tolist(), faiss_ids[0].tolist()), start=1):
            if int(faiss_id) < 0:
                continue
            hits.append(
                SearchHit(
                    item_id=id_mapping.item_id_from_faiss_id(int(faiss_id)),
                    faiss_id=int(faiss_id),
                    score=float(score),
                    rank=rank,
                )
            )
        return hits
