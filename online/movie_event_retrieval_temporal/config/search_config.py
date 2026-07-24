from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SearchConfig:
    store_dir: Path
    query_json: Path | None = None
    raw_query: str = ""
    translated_query: str = ""
    subtitle_query: str = ""
    ocr_query: str = ""
    temporal_checkpoint_path: Path | None = None
    clip_model_path_override: Path | None = None
    caption_model_path: str | None = None
    subtitle_model_path: str | None = None
    event_top_k: int = 200
    caption_top_k: int = 200
    subtitle_top_k: int = 200
    ocr_top_k: int = 200
    candidate_event_top_k: int = 100
    candidate_video_top_k: int = 30
    shot_top_k: int = 100
    final_top_k: int = 30
    rrf_k: int = 60
    event_weight: float = 1.0
    caption_weight: float = 0.8
    subtitle_weight: float = 0.6
    ocr_weight: float = 0.4
    shot_weight: float = 0.8
    parent_event_weight: float = 0.2
    event_device: str = "cpu"
    caption_device: str = "cpu"
    subtitle_device: str = "cpu"
    output_json: Path | None = None
