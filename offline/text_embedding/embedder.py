from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from .common import l2_normalize
from .config import TextEmbeddingConfig


class VietnameseTextEmbedder:
    def __init__(self, config: TextEmbeddingConfig) -> None:
        self.config = config
        self.model = SentenceTransformer(
            config.model_name_or_path,
            device=config.device,
        )

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False,
        ).astype(np.float32)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if not self.config.normalize_embeddings and embeddings.size > 0:
            embeddings = l2_normalize(embeddings)
        return embeddings
