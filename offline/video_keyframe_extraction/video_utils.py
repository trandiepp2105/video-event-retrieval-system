import json
import subprocess
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from PIL import Image


@dataclass
class VideoMetadata:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float | None


def _parse_ffprobe_rate(raw_value: str) -> float:
    raw_value = str(raw_value).strip()
    if not raw_value or raw_value == "0/0":
        return 0.0
    if "/" in raw_value:
        numerator, denominator = raw_value.split("/", 1)
        denominator_value = float(denominator)
        if denominator_value == 0:
            return 0.0
        return float(numerator) / denominator_value
    return float(raw_value)


def _probe_video_metadata(video_path: str) -> VideoMetadata:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"Khong tim thay video stream trong file: {video_path}")

    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fps = _parse_ffprobe_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/0")

    duration_sec = None
    duration_raw = stream.get("duration")
    if duration_raw in (None, "", "N/A"):
        duration_raw = (payload.get("format") or {}).get("duration")
    if duration_raw not in (None, "", "N/A"):
        duration_sec = float(duration_raw)

    nb_frames_raw = stream.get("nb_frames")
    if nb_frames_raw not in (None, "", "N/A"):
        frame_count = int(nb_frames_raw)
    elif duration_sec is not None and fps > 0:
        frame_count = max(int(round(duration_sec * fps)), 1)
    else:
        frame_count = 0

    return VideoMetadata(
        fps=float(fps),
        frame_count=int(frame_count),
        width=width,
        height=height,
        duration_sec=duration_sec,
    )


class VideoFrameReader:
    def __init__(self, video_path: str):
        self.video_path = str(video_path)
        metadata = _probe_video_metadata(self.video_path)
        self.fps = metadata.fps
        self.frame_count = metadata.frame_count
        self.width = metadata.width
        self.height = metadata.height
        self.duration_sec = metadata.duration_sec

    def sample_frame_indices(self, frame_step: int) -> list[int]:
        step = max(int(frame_step), 1)
        if self.frame_count <= 0:
            return []

        sampled = list(range(0, self.frame_count, step))
        if not sampled:
            sampled = [0]
        return sampled

    def iter_sampled_frames(self, frame_step: int) -> Iterator[tuple[int, Image.Image]]:
        sampled_indices = self.sample_frame_indices(frame_step)
        if not sampled_indices:
            return

        frame_bytes = self.width * self.height * 3
        if frame_bytes <= 0:
            raise RuntimeError(f"Metadata video khong hop le: {self.video_path}")

        select_filter = f"select=not(mod(n\\,{max(int(frame_step), 1)}))"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            self.video_path,
            "-vf",
            select_filter,
            "-vsync",
            "0",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-an",
            "-sn",
            "-",
        ]

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            assert process.stdout is not None
            for frame_index in sampled_indices:
                chunk = process.stdout.read(frame_bytes)
                if len(chunk) != frame_bytes:
                    break

                frame_array = np.frombuffer(chunk, dtype=np.uint8).reshape(
                    self.height,
                    self.width,
                    3,
                )
                yield int(frame_index), Image.fromarray(frame_array, mode="RGB")
        finally:
            stderr_output = b""
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                stderr_output = process.stderr.read()
                process.stderr.close()
            return_code = process.wait()
            if return_code != 0:
                message = stderr_output.decode("utf-8", errors="ignore").strip()
                raise RuntimeError(
                    f"ffmpeg decode that bai cho video {self.video_path}: {message or return_code}"
                )

    def close(self):
        return None
