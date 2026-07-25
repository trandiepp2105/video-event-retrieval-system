from __future__ import annotations

from typing import Any, Optional

import numpy as np


class Reranker:
    """Rank-based fusion copied from the notebook logic."""

    def _get_rank_order(
        self,
        indices: np.ndarray,
        unique_index: np.ndarray,
        out_top_order: Optional[int] = None,
    ) -> np.ndarray:
        if out_top_order is None:
            out_top_order = indices.shape[0] * 2
        rank_order = np.full(unique_index.shape, out_top_order)
        in_top_mask = np.isin(unique_index, indices)
        in_top = unique_index[in_top_mask]
        in_top_rank_order = [np.where(indices == item)[0][0].item() for item in in_top]
        rank_order[in_top_mask] = np.array(in_top_rank_order)
        return rank_order

    def _calculate_rank_score(
        self,
        rank_order: np.ndarray,
        initial_k: int,
        alpha: float = 1.0,
        beta: float = 1.5,
        cutoff: Optional[int] = None,
    ) -> np.ndarray:
        if cutoff is None:
            cutoff = rank_order.shape[0]
        return np.where(
            rank_order >= cutoff,
            np.exp(-alpha * cutoff / initial_k) * np.exp(-beta * (rank_order - cutoff) / initial_k),
            np.exp(-alpha * rank_order / initial_k),
        )

    def _rerank_by_rank_order(
        self,
        list_indices: list[np.ndarray],
        top_k: Optional[int] = None,
        alpha: float = 1.0,
        beta: float = 0.5,
        cutoff: int = 0,
        out_top_order: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        num_queries = list_indices[0].shape[0]
        num_models = len(list_indices)
        initial_k = list_indices[0].shape[1]
        if top_k is None:
            top_k = initial_k

        all_candidates = np.empty((num_queries, 0))
        for indices in list_indices:
            all_candidates = np.concatenate((all_candidates, indices), axis=1)

        scores = []
        indices = []
        for query_idx in range(num_queries):
            unique_idx = np.unique(all_candidates[query_idx]).astype(int)
            rank_scores = []
            for model_idx in range(num_models):
                rank_order = self._get_rank_order(
                    list_indices[model_idx][query_idx],
                    unique_idx,
                    out_top_order,
                )
                rank_score = self._calculate_rank_score(
                    rank_order,
                    initial_k=initial_k,
                    alpha=alpha,
                    beta=beta,
                    cutoff=cutoff,
                )
                rank_scores.append(rank_score)

            numerator = np.prod(rank_scores, axis=0)
            denominator = np.sum(rank_scores, axis=0)
            fusion_score = num_models * numerator / denominator
            rerank_indices = np.argsort(-fusion_score)[:top_k]
            rerank_scores = fusion_score[rerank_indices]
            scores.append(rerank_scores)
            indices.append(unique_idx[rerank_indices])

        return np.array(scores), np.array(indices)

    def _get_unique_metas(self, per_model_results: list[list[tuple[Any, float]]]) -> set[Any]:
        metas: set[Any] = set()
        for result in per_model_results:
            for meta, _score in result:
                metas.add(meta)
        return metas

    def _create_global_mapping(self, unique_meta_sets: list[set[Any]]) -> tuple[dict[Any, int], dict[int, Any]]:
        all_metas = list(unique_meta_sets)[0]
        for idx in range(1, len(unique_meta_sets)):
            all_metas = all_metas | unique_meta_sets[idx]
        meta2id = {meta: i for i, meta in enumerate(all_metas)}
        id2meta = {i: meta for meta, i in meta2id.items()}
        return meta2id, id2meta

    def _reconstruct_batch_result_into_scores_indices(
        self,
        per_model_results: list[list[tuple[Any, float]]],
        meta2id: dict[Any, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_scores = []
        batch_indices = []
        for result in per_model_results:
            scores = [score for _, score in result]
            indices = [meta2id[meta] for meta, _ in result]
            batch_scores.append(scores)
            batch_indices.append(indices)
        return np.array(batch_scores), np.array(batch_indices).astype(int)

    def _reconstruct_scores_indices_into_batch_result(
        self,
        scores_per_query: np.ndarray,
        indices_per_query: np.ndarray,
        id2meta: dict[int, Any],
    ) -> list[list[tuple[Any, float]]]:
        batch_results = []
        for scores, indices in zip(scores_per_query, indices_per_query):
            single = []
            for score, idx in zip(scores, indices):
                single.append((id2meta[idx], float(score)))
            batch_results.append(single)
        return batch_results

    def __call__(
        self,
        batch_results_per_model: Optional[list[list[list[tuple[Any, float]]]]] = None,
        indices_per_model: Optional[list[np.ndarray]] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ):
        if batch_results_per_model is None and indices_per_model is None:
            raise ValueError("Cần cung cấp batch_results_per_model hoặc indices_per_model.")

        if batch_results_per_model is not None:
            unique_meta_sets = [
                self._get_unique_metas(per_model_results)
                for per_model_results in batch_results_per_model
            ]
            meta2id, id2meta = self._create_global_mapping(unique_meta_sets)
            indices_per_model = []
            for per_model_results in batch_results_per_model:
                _scores_per_query, indices_per_query = self._reconstruct_batch_result_into_scores_indices(
                    per_model_results,
                    meta2id,
                )
                indices_per_model.append(indices_per_query)

        assert indices_per_model is not None
        scores, indices = self._rerank_by_rank_order(
            list_indices=indices_per_model,
            top_k=top_k,
            **kwargs,
        )
        if batch_results_per_model is not None:
            return self._reconstruct_scores_indices_into_batch_result(scores, indices, id2meta)
        return scores, indices
