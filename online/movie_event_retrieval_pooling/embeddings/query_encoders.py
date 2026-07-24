from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from online.movie_event_retrieval_temporal.common import l2_normalize


class SentenceTransformerQueryEncoder:
    def __init__(self, model_name_or_path: str | Path, *, device: str = "cpu") -> None:
        self.model = SentenceTransformer(str(model_name_or_path), device=device)

    def encode(self, query: str) -> np.ndarray:
        vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        return l2_normalize(vector)


class OpenClipQueryEncoder:
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
            raise ImportError("Can cai open_clip_torch de encode CLIP visual query.") from error

        self.device = str(device)
        self.torch = torch
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=str(model_path),
            device=self.device,
        )
        self.model.eval()

    def encode(self, query: str) -> np.ndarray:
        with self.torch.inference_mode():
            tokens = self.tokenizer([query]).to(self.device)
            embeddings = self.model.encode_text(tokens).float().cpu().numpy().astype(np.float32)
        return l2_normalize(embeddings)
