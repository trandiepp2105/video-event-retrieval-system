import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2


def get_video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video with OpenCV: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if fps <= 0 or fps != fps:
        raise RuntimeError(f"Invalid FPS for video {video_path}: {fps}")

    return fps


def make_tmp_dir(base_tmp_dir: Optional[Path] = None) -> tempfile.TemporaryDirectory:
    if base_tmp_dir is not None:
        base_tmp_dir.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=str(base_tmp_dir))
    return tempfile.TemporaryDirectory()


def cut_event_clip(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    output_path: Path,
    codec: str = "copy",
    loglevel: str = "error",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.05, float(end_sec) - float(start_sec))

    if codec not in {"copy", "h264"}:
        raise ValueError(f"Unsupported clip codec: {codec}")

    if codec == "copy":
        cmd = [
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", loglevel,
            "-ss", f"{start_sec:.3f}",
            "-i", str(video_path),
            "-t", f"{duration:.3f}",
            "-map", "0:v:0",
            "-an",
            "-c:v", "copy",
            "-avoid_negative_ts", "make_zero",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", loglevel,
            "-ss", f"{start_sec:.3f}",
            "-i", str(video_path),
            "-t", f"{duration:.3f}",
            "-map", "0:v:0",
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        if codec == "copy":
            return cut_event_clip(
                video_path=video_path,
                start_sec=start_sec,
                end_sec=end_sec,
                output_path=output_path,
                codec="h264",
                loglevel=loglevel,
            )

        raise RuntimeError(
            "ffmpeg failed while cutting event clip.\n"
            f"Video: {video_path}\n"
            f"Start: {start_sec}, End: {end_sec}\n"
            f"STDERR:\n{result.stderr}"
        )

    return output_path
