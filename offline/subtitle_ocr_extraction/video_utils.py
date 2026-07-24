import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class VideoMetadata:
    fps: float
    width: int
    height: int
    frame_count: int | None
    duration_sec: float | None


@dataclass
class SampledFrame:
    frame_id: int
    time_sec: float
    image: np.ndarray


def parse_ffprobe_fps(raw: str) -> float:
    raw = str(raw).strip()
    if not raw or raw == "0/0":
        return 30.0
    if "/" in raw:
        numerator, denominator = raw.split("/", 1)
        denominator_value = float(denominator)
        if denominator_value == 0:
            return 30.0
        return float(numerator) / denominator_value
    return float(raw)


class FFprobeMetadataReader:
    def __init__(self, ffprobe_path: str = "ffprobe") -> None:
        self.ffprobe_path = ffprobe_path

    def read(self, video_path: str | Path) -> VideoMetadata:
        command = [
            self.ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,width,height,nb_frames",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        fps = parse_ffprobe_fps(stream.get("avg_frame_rate", "30/1"))
        width = int(stream["width"])
        height = int(stream["height"])
        nb_frames_raw = stream.get("nb_frames")
        frame_count = None
        if nb_frames_raw not in (None, "N/A", ""):
            frame_count = int(nb_frames_raw)
        duration_raw = payload.get("format", {}).get("duration")
        duration_sec = float(duration_raw) if duration_raw not in (None, "N/A", "") else None
        return VideoMetadata(
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
            duration_sec=duration_sec,
        )


class FFmpegFramePipeReader:
    def __init__(
        self,
        video_path: str | Path,
        metadata: VideoMetadata,
        *,
        frame_step: int = 6,
        ffmpeg_path: str = "ffmpeg",
    ) -> None:
        self.video_path = Path(video_path)
        self.metadata = metadata
        self.frame_step = max(int(frame_step), 1)
        self.ffmpeg_path = ffmpeg_path

    def iter_sampled_frames(self) -> Iterator[SampledFrame]:
        frame_size = int(self.metadata.width) * int(self.metadata.height) * 3
        select_expr = f"not(mod(n\\,{self.frame_step}))"
        command = [
            self.ffmpeg_path,
            "-v",
            "error",
            "-i",
            str(self.video_path),
            "-vf",
            f"select='{select_expr}'",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.stdout is None:
            raise RuntimeError("Khong mo duoc stdout cua ffmpeg process.")

        sample_index = 0
        try:
            while True:
                raw = process.stdout.read(frame_size)
                if not raw:
                    break
                if len(raw) != frame_size:
                    raise RuntimeError(
                        f"Frame raw bytes khong day du: nhan {len(raw)} bytes, ky vong {frame_size} bytes."
                    )
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.metadata.height, self.metadata.width, 3)
                frame_id = sample_index * self.frame_step
                time_sec = frame_id / max(float(self.metadata.fps), 1e-6)
                yield SampledFrame(
                    frame_id=frame_id,
                    time_sec=time_sec,
                    image=frame.copy(),
                )
                sample_index += 1
        finally:
            if process.stdout is not None:
                process.stdout.close()
            stderr_text = ""
            if process.stderr is not None:
                stderr_text = process.stderr.read().decode("utf-8", errors="replace")
                process.stderr.close()
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"ffmpeg pipe that bai voi return code {return_code}: {stderr_text.strip()}")
