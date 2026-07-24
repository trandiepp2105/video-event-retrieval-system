import json
from pathlib import Path
from typing import Any


VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)
    tmp.replace(path)


def list_event_video_ids(event_output_dir: Path) -> list[str]:
    if not event_output_dir.exists():
        raise FileNotFoundError(f"event_output_dir does not exist: {event_output_dir}")

    video_ids = [
        path.name for path in event_output_dir.iterdir()
        if path.is_dir() and (path / "events.json").exists()
    ]
    return sorted(video_ids)


def resolve_video_path(videos_dir: Path, video_id: str) -> Path | None:
    for ext in sorted(VIDEO_EXTS):
        path = videos_dir / f"{video_id}{ext}"
        if path.exists() and path.is_file():
            return path
    return None


def get_event_json_from_output(event_output_dir: Path, video_id: str) -> Path:
    return event_output_dir / video_id / "events.json"


def get_sidecar_json(json_dir: Path, video_path: Path) -> Path:
    return json_dir / f"{video_path.stem}.json"


def validate_event_item(event: dict[str, Any], event_path: Path, index: int) -> None:
    if "start_time_sec" not in event or "end_time_sec" not in event:
        raise ValueError(
            f"Event item #{index} in {event_path} must contain start_time_sec and end_time_sec"
        )
