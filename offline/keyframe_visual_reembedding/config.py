from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineConfig:
    videos_dir: str
    reference_keyframe_dir: str
    output_dir: str

    start_index: int = 0
    end_index: Optional[int] = None
    video_ids: Optional[list[str]] = None

    clip_model_name: str = "ViT-gopt-16-SigLIP2-384"
    clip_pretrained: str = "webli"
    batch_size: int = 128
    device: str = "cuda"

    save_dtype: str = "float16"
    overwrite: bool = False
