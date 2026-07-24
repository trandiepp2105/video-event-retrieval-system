from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CaptionConfig:
    videos_dir: Path
    event_output_dir: Path
    subtitles_dir: Path
    model_path: Path
    output_dir: Path

    start_index: int = 0
    end_index: Optional[int] = None
    video_ids: Optional[list[str]] = None

    video_sample_fps: float = 0.5

    min_pixels: int = 128 * 128
    max_pixels: int = 256 * 256
    image_patch_size: int = 16

    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None

    torch_dtype: str = "float16"
    device_map: str = "auto"
    local_files_only: bool = True
    trust_remote_code: bool = True

    tmp_dir: Optional[Path] = None
    clip_codec: str = "copy"
    ffmpeg_loglevel: str = "error"

    overwrite: bool = False
    save_every_event: bool = True
    continue_on_error: bool = True
