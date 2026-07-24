from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TextEmbeddingConfig:
    input_dir: Path
    output_dir: Path
    model_name_or_path: str
    input_type: str
    device: str | None = None
    batch_size: int = 32
    normalize_embeddings: bool = True
    save_dtype: str = "float32"
    start_index: int = 0
    end_index: int | None = None
    video_ids: list[str] | None = None
    overwrite: bool = False
    text_field: str = "text"
    caption_field: str = "caption"
