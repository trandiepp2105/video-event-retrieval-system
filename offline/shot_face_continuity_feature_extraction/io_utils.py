import json
from pathlib import Path
from typing import Any

from .config import FaceContinuityConfig


class FileIO:
    @staticmethod
    def ensure_dir(path: str | Path):
        Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_json(path: str | Path):
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def save_json(data, path: str | Path):
        path = Path(path)
        FileIO.ensure_dir(path.parent)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)


class ShotNormalizer:
    @staticmethod
    def normalize_shots(raw_shots: Any) -> list[dict[str, Any]]:
        if isinstance(raw_shots, dict):
            if "shots" in raw_shots:
                shots = raw_shots["shots"]
            elif "data" in raw_shots:
                shots = raw_shots["data"]
            else:
                raise ValueError("Khong tim thay key shots hoac data trong shots JSON.")
        elif isinstance(raw_shots, list):
            shots = raw_shots
        else:
            raise ValueError("shots JSON phai la list hoac dict chua key shots.")

        normalized = []
        for idx, shot in enumerate(shots):
            shot = dict(shot)

            if "shot_id" not in shot:
                shot["shot_id"] = idx

            if "start_time_sec" not in shot:
                if "start_time" in shot:
                    shot["start_time_sec"] = shot["start_time"]
                elif "start" in shot:
                    shot["start_time_sec"] = shot["start"]
                else:
                    raise ValueError(f"Shot {idx} thieu start_time_sec/start_time/start.")

            if "end_time_sec" not in shot:
                if "end_time" in shot:
                    shot["end_time_sec"] = shot["end_time"]
                elif "end" in shot:
                    shot["end_time_sec"] = shot["end"]
                else:
                    raise ValueError(f"Shot {idx} thieu end_time_sec/end_time/end.")

            shot["shot_id"] = int(shot["shot_id"])
            shot["start_time_sec"] = float(shot["start_time_sec"])
            shot["end_time_sec"] = float(shot["end_time_sec"])

            if "duration_sec" not in shot:
                shot["duration_sec"] = max(0.0, shot["end_time_sec"] - shot["start_time_sec"])
            else:
                shot["duration_sec"] = float(shot["duration_sec"])

            if "start_frame" in shot:
                shot["start_frame"] = int(shot["start_frame"])
            if "end_frame" in shot:
                shot["end_frame"] = int(shot["end_frame"])

            normalized.append(shot)

        normalized.sort(key=lambda item: (float(item["start_time_sec"]), int(item["shot_id"])))
        return normalized


class DatasetScanner:
    def __init__(self, config: FaceContinuityConfig):
        self.config = config

    def get_video_items(self) -> list[dict[str, str]]:
        video_paths = sorted(Path(self.config.video_dataset_dir).glob("*.mp4"))
        items: list[dict[str, str]] = []

        for video_path in video_paths:
            video_name = video_path.stem
            shots_path = Path(self.config.shots_json_dir) / f"{video_name}.json"
            output_dir = Path(self.config.output_dir) / video_name

            if not shots_path.exists():
                print(f"[WARN] Missing shots JSON for {video_name}: {shots_path}")
                continue

            items.append(
                {
                    "video_name": video_name,
                    "video_path": str(video_path),
                    "shots_json_path": str(shots_path),
                    "output_dir": str(output_dir),
                }
            )

        if self.config.video_ids:
            items_by_video_name = {item["video_name"]: item for item in items}
            filtered_items = []
            missing_video_ids = []

            for video_id in self.config.video_ids:
                normalized_video_id = str(video_id)
                item = items_by_video_name.get(normalized_video_id)
                if item is None:
                    missing_video_ids.append(normalized_video_id)
                    continue
                filtered_items.append(item)

            for video_id in missing_video_ids:
                print(f"[WARN] Missing video or shots JSON for requested video id: {video_id}")

            return filtered_items

        start = max(0, int(self.config.start_index))
        end = self.config.end_index

        if start >= len(items):
            return []
        if end is None:
            return items[start:]
        return items[start:end + 1]
