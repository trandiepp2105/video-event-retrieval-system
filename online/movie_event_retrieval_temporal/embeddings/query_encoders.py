from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..common import l2_normalize


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


class TemporalQueryEncoder:
    def __init__(
        self,
        *,
        checkpoint_path: Path,
        clip_model_path_override: Path | None = None,
        device: str = "cpu",
    ) -> None:
        from offline.temporal_event_visual_retrieval.model import (
            HierarchicalTemporalRetrievalModel,
        )

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        clip_model_path = clip_model_path_override or config["clip_model_path"]
        self.model = HierarchicalTemporalRetrievalModel(
            keyframe_dim=1,
            keyframe_metadata_dim=4,
            shot_metadata_dim=4,
            clip_model_path=str(clip_model_path),
            clip_model_name=config.get("clip_model_name", "ViT-H-14-quickgelu"),
            device=device,
            keyframe_hidden_dim=int(config["keyframe_hidden_dim"]),
            shot_hidden_dim=int(config["shot_hidden_dim"]),
            projection_dim=int(config["projection_dim"]),
            keyframe_transformer_layers=int(config["keyframe_transformer_layers"]),
            shot_transformer_layers=int(config["shot_transformer_layers"]),
            keyframe_transformer_heads=int(config["keyframe_transformer_heads"]),
            shot_transformer_heads=int(config["shot_transformer_heads"]),
            dropout=float(config["dropout"]),
        )
        model_state = checkpoint["model_state_dict"]
        filtered_state = {
            key: value
            for key, value in model_state.items()
            if not key.startswith("keyframe_input.0.weight")
        }
        self.model.load_state_dict(filtered_state, strict=False)
        self.model.eval()
        self.model.to(device)
        self.device = device

    def encode_event_query(self, query: str) -> np.ndarray:
        with torch.no_grad():
            outputs = self.model.encode_text([query])
        return outputs["event_text_embeddings"].detach().cpu().numpy().astype(np.float32)

    def encode_shot_query(self, query: str) -> np.ndarray:
        with torch.no_grad():
            outputs = self.model.encode_text([query])
        return outputs["shot_text_embeddings"].detach().cpu().numpy().astype(np.float32)
