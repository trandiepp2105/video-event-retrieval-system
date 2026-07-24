from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .faiss_index import l2_normalize


class OpenClipTextEncoder:
    def __init__(
        self,
        model_path: str | Path,
        *,
        model_name: str = "ViT-H-14-quickgelu",
        device: str = "cpu",
    ) -> None:
        try:
            import open_clip
            import torch
        except ImportError as error:  # pragma: no cover
            raise ImportError("Can cai open_clip_torch de encode visual query.") from error

        self.device = str(device)
        self.torch = torch
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=str(model_path),
            device=self.device,
        )
        self.model.eval()

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        with self.torch.inference_mode():
            tokens = self.tokenizer(texts).to(self.device)
            embeddings = self.model.encode_text(tokens).float().cpu().numpy().astype(np.float32)
        return l2_normalize(embeddings)


class E5TextEncoder:
    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:  # pragma: no cover
            raise ImportError("Can cai sentence-transformers de encode E5 query.") from error

        self.batch_size = int(batch_size)
        self.model = SentenceTransformer(str(model_path), device=device)

    @staticmethod
    def _as_prefixed_queries(texts: List[str]) -> List[str]:
        return [f"query: {text.strip()}" for text in texts]

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            self._as_prefixed_queries(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        return embeddings
