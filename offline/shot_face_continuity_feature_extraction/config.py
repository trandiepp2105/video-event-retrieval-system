from dataclasses import dataclass
from typing import Optional


@dataclass
class FaceContinuityConfig:
    video_dataset_dir: str
    shots_json_dir: str
    output_dir: str
    start_index: int = 0
    end_index: Optional[int] = None
    video_ids: Optional[list[str]] = None

    insightface_model_name: str = "buffalo_l"
    root_dir: Optional[str] = None
    ctx_id: int = 0
    det_size: tuple[int, int] = (640, 640)

    det_score_thresh: float = 0.50
    min_face_size: int = 40

    max_frames_per_shot: int = 5
    seconds_per_sample: float = 2.0
    recognition_batch_size: int = 64

    same_face_threshold: float = 0.45
    min_face_count_per_shot: int = 1
    face_window_size: int = 3

    save_debug_crops: bool = True
    max_debug_crops_per_face_id: int = 24
    overwrite: bool = False
