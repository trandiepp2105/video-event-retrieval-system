import math
from typing import Optional

import cv2
import numpy as np


class VideoFrameReader:
    def __init__(self, video_path: str):
        self.video_path = str(video_path)
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = self.total_frames / max(self.fps, 1e-8)

    def read_frame_at_time(self, time_sec: float) -> Optional[np.ndarray]:
        time_sec = float(np.clip(time_sec, 0.0, max(0.0, self.duration_sec - 1e-3)))
        frame_idx = int(round(time_sec * self.fps))
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = self.cap.read()
        if not ok or frame_bgr is None:
            return None

        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class ShotFrameSampler:
    def __init__(self, max_frames_per_shot: int = 5, seconds_per_sample: float = 2.0):
        self.max_frames_per_shot = int(max_frames_per_shot)
        self.seconds_per_sample = float(seconds_per_sample)

    def sample_times(self, start_sec: float, end_sec: float) -> list[float]:
        start_sec = float(start_sec)
        end_sec = float(end_sec)
        duration = max(0.0, end_sec - start_sec)

        if duration <= 0:
            return []

        n_by_duration = max(1, int(math.ceil(duration / self.seconds_per_sample)))
        n = min(self.max_frames_per_shot, n_by_duration)

        if n == 1:
            return [0.5 * (start_sec + end_sec)]

        margin = min(0.15 * duration, 0.25)
        left = start_sec + margin
        right = end_sec - margin
        if right <= left:
            left = start_sec
            right = end_sec

        return np.linspace(left, right, n).astype(float).tolist()
