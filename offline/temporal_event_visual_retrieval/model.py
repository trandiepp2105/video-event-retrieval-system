from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .encoders import OpenClipTextEncoder


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(x).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        attention = torch.softmax(logits, dim=-1)
        pooled = torch.sum(x * attention.unsqueeze(-1), dim=1)
        return pooled, attention


class TemporalTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=~mask)


class HierarchicalTemporalRetrievalModel(nn.Module):
    def __init__(
        self,
        keyframe_dim: int,
        keyframe_metadata_dim: int,
        shot_metadata_dim: int,
        clip_model_path: str,
        clip_model_name: str = "ViT-H-14-quickgelu",
        device: str = "cpu",
        keyframe_hidden_dim: int = 768,
        shot_hidden_dim: int = 768,
        projection_dim: int = 768,
        keyframe_transformer_layers: int = 2,
        shot_transformer_layers: int = 2,
        keyframe_transformer_heads: int = 8,
        shot_transformer_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.keyframe_input = nn.Sequential(
            nn.Linear(keyframe_dim + keyframe_metadata_dim, keyframe_hidden_dim),
            nn.LayerNorm(keyframe_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.keyframe_transformer = TemporalTransformerBlock(
            hidden_dim=keyframe_hidden_dim,
            num_layers=keyframe_transformer_layers,
            num_heads=keyframe_transformer_heads,
            dropout=dropout,
        )
        self.keyframe_attention_pool = AttentionPooling(keyframe_hidden_dim)
        self.shot_pre_event_head = nn.Linear(keyframe_hidden_dim, projection_dim)

        self.shot_input = nn.Sequential(
            nn.Linear(keyframe_hidden_dim + shot_metadata_dim, shot_hidden_dim),
            nn.LayerNorm(shot_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.shot_transformer = TemporalTransformerBlock(
            hidden_dim=shot_hidden_dim,
            num_layers=shot_transformer_layers,
            num_heads=shot_transformer_heads,
            dropout=dropout,
        )
        self.shot_attention_pool = AttentionPooling(shot_hidden_dim)
        self.event_output_head = nn.Linear(shot_hidden_dim, projection_dim)
        self.shot_output_head = nn.Linear(shot_hidden_dim, projection_dim)

        self.text_encoder = OpenClipTextEncoder(
            clip_model_path,
            model_name=clip_model_name,
            device=device,
        )
        sample_dim = int(self.text_encoder.encode_texts(["sample"]).shape[-1])
        self.text_to_event_head = nn.Linear(sample_dim, projection_dim)
        self.text_to_shot_head = nn.Linear(sample_dim, projection_dim)

    def encode_visual(
        self,
        keyframe_vectors: torch.Tensor,
        keyframe_metadata: torch.Tensor,
        keyframe_mask: torch.Tensor,
        shot_metadata: torch.Tensor,
        shot_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_shots, num_keyframes, keyframe_dim = keyframe_vectors.shape
        key_meta_dim = int(keyframe_metadata.shape[-1])

        flat_keyframe_vectors = keyframe_vectors.reshape(batch_size * num_shots, num_keyframes, keyframe_dim)
        flat_keyframe_metadata = keyframe_metadata.reshape(batch_size * num_shots, num_keyframes, key_meta_dim)
        flat_keyframe_mask = keyframe_mask.reshape(batch_size * num_shots, num_keyframes)
        flat_shot_mask = shot_mask.reshape(batch_size * num_shots)
        safe_keyframe_mask = flat_keyframe_mask.clone()
        safe_keyframe_mask[~flat_shot_mask, 0] = True

        keyframe_input = torch.cat([flat_keyframe_vectors, flat_keyframe_metadata], dim=-1)
        keyframe_hidden = self.keyframe_input(keyframe_input)
        keyframe_hidden = self.keyframe_transformer(keyframe_hidden, safe_keyframe_mask)
        shot_base_hidden, flat_keyframe_attention = self.keyframe_attention_pool(keyframe_hidden, safe_keyframe_mask)
        shot_base_hidden = shot_base_hidden.reshape(batch_size, num_shots, -1)
        keyframe_attention = flat_keyframe_attention.reshape(batch_size, num_shots, num_keyframes)

        shot_pre_event_embeddings = F.normalize(self.shot_pre_event_head(shot_base_hidden), dim=-1)

        flat_keyframe_vectors_norm = F.normalize(flat_keyframe_vectors, dim=-1)
        shot_clip_embeddings = torch.sum(
            flat_keyframe_vectors_norm * flat_keyframe_attention.unsqueeze(-1),
            dim=1,
        )
        shot_clip_embeddings = F.normalize(shot_clip_embeddings, dim=-1)
        shot_clip_embeddings = shot_clip_embeddings.reshape(batch_size, num_shots, -1)

        shot_input = torch.cat([shot_base_hidden, shot_metadata], dim=-1)
        shot_hidden = self.shot_input(shot_input)
        shot_hidden = self.shot_transformer(shot_hidden, shot_mask)
        event_hidden, shot_attention = self.shot_attention_pool(shot_hidden, shot_mask)

        event_embeddings = F.normalize(self.event_output_head(event_hidden), dim=-1)
        shot_embeddings = F.normalize(self.shot_output_head(shot_hidden), dim=-1)
        weighted_shot_summary = F.normalize(
            torch.sum(shot_embeddings * shot_attention.unsqueeze(-1), dim=1),
            dim=-1,
        )

        return {
            "event_embeddings": event_embeddings,
            "shot_embeddings": shot_embeddings,
            "shot_pre_event_embeddings": shot_pre_event_embeddings,
            "shot_clip_embeddings": shot_clip_embeddings,
            "keyframe_attention_weights": keyframe_attention,
            "shot_attention_weights": shot_attention,
            "weighted_shot_summary": weighted_shot_summary,
        }

    def encode_text(self, texts: list[str]) -> dict[str, torch.Tensor]:
        text_embedding_base = self.text_encoder.encode_texts_tensor(texts)
        event_text = F.normalize(self.text_to_event_head(text_embedding_base), dim=-1)
        shot_text = F.normalize(self.text_to_shot_head(text_embedding_base), dim=-1)
        return {
            "text_embedding_base": text_embedding_base,
            "event_text_embeddings": event_text,
            "shot_text_embeddings": shot_text,
        }

    def forward(
        self,
        keyframe_vectors: torch.Tensor,
        keyframe_metadata: torch.Tensor,
        keyframe_mask: torch.Tensor,
        shot_metadata: torch.Tensor,
        shot_mask: torch.Tensor,
        texts: list[str],
    ) -> dict[str, torch.Tensor]:
        visual_outputs = self.encode_visual(
            keyframe_vectors=keyframe_vectors,
            keyframe_metadata=keyframe_metadata,
            keyframe_mask=keyframe_mask,
            shot_metadata=shot_metadata,
            shot_mask=shot_mask,
        )
        text_outputs = self.encode_text(texts)
        return {**visual_outputs, **text_outputs}
