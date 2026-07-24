from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BuildConfig:
    event_dir: Path
    event_embedding_dir: Path
    caption_embedding_dir: Path
    shot_embedding_dir: Path
    subtitle_embedding_dir: Path
    ocr_dir: Path
    output_dir: Path
    overwrite: bool = False
