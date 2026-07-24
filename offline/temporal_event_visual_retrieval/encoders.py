from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .io_utils import l2_normalize


class OpenClipTextEncoder:
    def __init__(
        self,
        model_path: str | Path,
        *,
        model_name: str = "ViT-H-14-quickgelu",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        import open_clip
        import torch

        self.device = device
        self.torch = torch
        self.batch_size = int(batch_size)
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=str(model_path),
            device=device,
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        outputs: List[np.ndarray] = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                tokens = self.tokenizer(batch).to(self.device)
                embeddings = self.model.encode_text(tokens)
                outputs.append(embeddings.float().cpu().numpy().astype(np.float32))
        return l2_normalize(np.concatenate(outputs, axis=0))

    def encode_texts_tensor(self, texts: List[str]):
        with self.torch.no_grad():
            tokens = self.tokenizer(texts).to(self.device)
            embeddings = self.model.encode_text(tokens)
            embeddings = embeddings.float()
            norms = embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            return (embeddings / norms).clone()
