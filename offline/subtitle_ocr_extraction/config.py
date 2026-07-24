from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    videos_dir: Path
    subtitle_output_dir: Path
    ocr_output_dir: Path
    paddle_text_detection_model_dir: Path
    paddle_text_recognition_model_dir: Path
    vietocr_weights: Path
    roi_x_min: float = 0.0
    roi_x_max: float = 1.0
    roi_y_min: float = 0.65
    roi_y_max: float = 1.0
    padding: int = 3
    max_subtitle_angle_deg: float = 8.0
    detector_score_threshold: float = 0.75
    detector_max_boxes: int = 2
    detector_min_scene_score: float = 0.5
    subtitle_similarity_threshold: float = 0.8
    subtitle_max_gap: int = 12
    frame_step: int = 6
    frame_batch_size: int = 64
    start_index: int = 0
    end_index: int | None = None
    video_ids: list[str] | None = None
    overwrite: bool = False
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    paddle_device: str = "gpu:0"
    vietocr_device: str = "cuda"
    vietocr_repo_path: Path | None = None
