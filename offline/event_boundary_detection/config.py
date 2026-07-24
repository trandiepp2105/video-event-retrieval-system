from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EventGroupingDatasetConfig:
    features_dir: str = "./features"
    output_root_dir: str = "./event_output"
    video_ids: Optional[List[str]] = None
    start_index: int = 0
    end_index: Optional[int] = None
    skip_missing_modalities: bool = True
    context_window: int = 3
    subtitle_use_recency_weight: bool = True
    subtitle_recency_tau: float = 2.0
    subtitle_bridge_penalty_weight: float = 0.40
    subtitle_bridge_norm_sec: float = 0.50
    use_face_recency_weight: bool = True
    face_recency_tau: float = 2.0
    visual_weight: float = 0.30
    action_weight: float = 0.15
    subtitle_weight: float = 0.40
    face_weight: float = 0.15
    boundary_percentile: float = 85.0
    use_local_peak: bool = True
    min_event_duration_sec: float = 3.0
    max_event_duration_sec: float = 30.0
    cut_penalty: float = 0.55
    non_candidate_penalty: float = 0.25
    event_softmax_temperature: float = 15.0
    overwrite: bool = False
