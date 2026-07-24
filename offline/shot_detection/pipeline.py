from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from scenedetect import SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector, ContentDetector
from tqdm.auto import tqdm


@dataclass
class ShotDetectionConfig:
    input_dir: Path
    output_dir: Path
    start_index: int = 0
    end_index: int | None = None
    video_ids: list[str] | None = None
    detector: str = "adaptive"
    adaptive_threshold: float = 3.0
    min_content_val: float = 15.0
    window_width: int = 2
    content_threshold: float = 27.0
    min_shot_len_sec: float = 0.25
    overwrite: bool = False


class DatasetShotScanner:
    def __init__(self, config: ShotDetectionConfig) -> None:
        self.config = config

    def get_video_paths(self) -> list[Path]:
        video_paths = sorted(Path(self.config.input_dir).rglob("*.mp4"))
        if not video_paths:
            raise ValueError(f"Khong tim thay file .mp4 nao trong: {self.config.input_dir}")

        if self.config.video_ids:
            by_id = {path.stem: path for path in video_paths}
            selected: list[Path] = []
            missing: list[str] = []
            for video_id in self.config.video_ids:
                path = by_id.get(str(video_id))
                if path is None:
                    missing.append(str(video_id))
                    continue
                selected.append(path)
            for video_id in missing:
                print(f"[WARN] Missing requested video id: {video_id}")
            return selected

        start = max(0, int(self.config.start_index))
        if start >= len(video_paths):
            return []
        if self.config.end_index is None:
            return video_paths[start:]
        return video_paths[start : self.config.end_index + 1]


class ShotDetectorPipeline:
    def __init__(self, config: ShotDetectionConfig) -> None:
        self.config = config
        self.scanner = DatasetShotScanner(config)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_detector(self) -> AdaptiveDetector | ContentDetector:
        min_scene_len = f"{self.config.min_shot_len_sec}s"
        if self.config.detector == "adaptive":
            return AdaptiveDetector(
                adaptive_threshold=self.config.adaptive_threshold,
                min_scene_len=min_scene_len,
                window_width=self.config.window_width,
                min_content_val=self.config.min_content_val,
            )
        if self.config.detector == "content":
            return ContentDetector(
                threshold=self.config.content_threshold,
                min_scene_len=min_scene_len,
            )
        raise ValueError("detector chi duoc la 'adaptive' hoac 'content'.")

    @staticmethod
    def _get_video_info(video_path: Path) -> tuple[float, int]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Khong the mo video: {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()

        if fps <= 0:
            raise ValueError(f"Khong doc duoc FPS hop le tu video: {video_path}")
        if total_frames <= 0:
            raise ValueError(f"Video khong co frame hop le: {video_path}")
        return fps, total_frames

    def _detect_shots_for_video(self, video_path: Path) -> list[dict[str, Any]]:
        fps, total_frames = self._get_video_info(video_path)
        detector = self._build_detector()
        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(detector)
        scene_manager.detect_scenes(video=video, show_progress=True)
        scene_list = scene_manager.get_scene_list(start_in_scene=True)

        shots: list[dict[str, Any]] = []
        for shot_id, (start_tc, end_tc) in enumerate(scene_list):
            start_frame = max(0, min(int(start_tc.frame_num), total_frames - 1))
            end_frame_exclusive = max(start_frame + 1, min(int(end_tc.frame_num), total_frames))
            end_frame = end_frame_exclusive - 1
            shots.append(
                {
                    "shot_id": shot_id,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time_sec": round(start_frame / fps, 3),
                    "end_time_sec": round(end_frame_exclusive / fps, 3),
                    "duration_sec": round((end_frame_exclusive - start_frame) / fps, 3),
                }
            )
        return shots

    def _save_json(self, video_path: Path, shots: list[dict[str, Any]]) -> Path:
        output_path = self.config.output_dir / f"{video_path.stem}.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(shots, file, ensure_ascii=False, indent=2)
        return output_path

    def run(self) -> dict[str, Any]:
        video_paths = self.scanner.get_video_paths()
        summary: dict[str, Any] = {
            "done": [],
            "skipped": [],
            "failed": [],
        }

        for video_path in tqdm(video_paths, desc="Shot detection videos"):
            output_path = self.config.output_dir / f"{video_path.stem}.json"
            if output_path.exists() and not self.config.overwrite:
                summary["skipped"].append(
                    {
                        "video_id": video_path.stem,
                        "output_path": str(output_path),
                        "reason": "output_exists",
                    }
                )
                continue

            try:
                shots = self._detect_shots_for_video(video_path)
                json_path = self._save_json(video_path, shots)
                summary["done"].append(
                    {
                        "video_id": video_path.stem,
                        "num_shots": len(shots),
                        "output_path": str(json_path),
                    }
                )
            except Exception as error:  # pragma: no cover
                summary["failed"].append(
                    {
                        "video_id": video_path.stem,
                        "video_path": str(video_path),
                        "error": repr(error),
                    }
                )

        manifest_path = self.config.output_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        return summary
