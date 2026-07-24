from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommonConfig:
    event_dir: Path
    shot_keyframe_dir: Path
    event_captions_dir: Path | None
    clip_model_path: Path
    output_dir: Path
    raw_subtitle_dir: Path | None = Path("./features/raw_subtitle")
    translated_captions_dir: Path | None = None
    clip_model_name: str = "ViT-H-14-quickgelu"
    intro_padding_sec: float = 120.0
    outro_padding_sec: float = 120.0
    min_event_shots: int = 1
    max_event_shots: int = 64
    min_keyframes_per_shot: int = 1
    max_keyframes_per_shot: int = 16
    keyframe_hidden_dim: int = 768
    shot_hidden_dim: int = 768
    keyframe_transformer_layers: int = 2
    shot_transformer_layers: int = 2
    keyframe_transformer_heads: int = 8
    shot_transformer_heads: int = 8
    dropout: float = 0.1
    projection_dim: int = 768
    keyframe_metadata_time_norm_sec: float = 4.0
    shot_metadata_duration_norm_sec: float = 30.0
    caption_shot_similarity_weight: float = 0.80
    subtitle_overlap_weight: float = 0.20
    soft_positive_temperature: float = 0.07
    salient_subtitle_norm_sec: float = 4.0


@dataclass
class TrainConfig(CommonConfig):
    batch_size: int = 4
    num_epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    event_loss_weight: float = 1.0
    shot_loss_weight: float = 0.5
    event_consistency_loss_weight: float = 0.1
    shot_consistency_loss_weight: float = 0.1
    temperature: float = 0.07
    device: str = "cuda"
    num_workers: int = 0
    seed: int = 42


@dataclass
class EncodeConfig(CommonConfig):
    checkpoint_path: Path | None = None
    device: str = "cuda"
    num_workers: int = 0


@dataclass
class PrepareTranslationsConfig:
    event_captions_dir: Path
    output_dir: Path
    translation_model_path: Path
    device: str = "cuda"
    translation_batch_size: int = 8
    start_index: int | None = None
    end_index: int | None = None
    overwrite: bool = False
