from __future__ import annotations

import numpy as np

from ..common import l2_normalize, to_contiguous_float32


class EmbeddingValidator:
    def validate(self, embeddings: np.ndarray, item_ids: list[str]) -> None:
        if embeddings.dtype != np.float32:
            raise ValueError(f"Embeddings must be float32, got {embeddings.dtype}")
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape={embeddings.shape}")
        if embeddings.shape[0] != len(item_ids):
            raise ValueError(
                f"Embedding count mismatch: vectors={embeddings.shape[0]}, item_ids={len(item_ids)}"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain NaN or Inf")


class EmbeddingNormalizer:
    def normalize(self, embeddings: np.ndarray) -> np.ndarray:
        return to_contiguous_float32(l2_normalize(embeddings))
