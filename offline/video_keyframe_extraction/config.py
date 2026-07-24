from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineConfig:
    videos_dir: str
    output_dir: str

    frame_step: int = 6
    similarity_threshold: float = 0.95

    start_index: int = 0
    end_index: Optional[int] = None
    video_ids: Optional[list[str]] = None

    clip_model_name: str = "ViT-H-14-quickgelu"
    clip_pretrained: str = ""

    frame_load_batch_size: int = 256
    batch_size: int = 256
    device: str = "cuda"

    save_dtype: str = "float16"
    overwrite: bool = False
