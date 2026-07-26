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
    clip_model_path: Path | None = None
    clip_model_name: str = "ViT-H-14-quickgelu"
    caption_model_path: str | None = None
    subtitle_model_path: str | None = None
    subtitle_backend: str = "meilisearch"
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
    visual_device: str = "cpu"
    caption_device: str = "cpu"
    subtitle_device: str = "cpu"
    enable_shot_temporal: bool = False
    temporal_query_model_path: str | None = None
    temporal_query_device_map: str = "auto"
    temporal_query_torch_dtype: str = "auto"
    temporal_query_max_new_tokens: int = 768
    stage_shot_top_k: int = 100
    temporal_chain_top_k: int = 30
    stage_visual_weight: float = 0.45
    stage_ocr_weight: float = 0.35
    stage_subtitle_weight: float = 0.20
    temporal_window_shots: int = 3
    temporal_group_gap_shots: int = 12
    temporal_min_stage_gap_shots: int = 1
    temporal_lambda_skip: float = 0.7
    meilisearch_url: str | None = None
    meilisearch_index_name: str | None = None
    subtitle_meilisearch_index_name: str | None = None
    meilisearch_api_key: str | None = None
    output_json: Path | None = None
