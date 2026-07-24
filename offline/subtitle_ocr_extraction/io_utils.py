import json
from pathlib import Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class DatasetLoader:
    def __init__(self, videos_dir: str | Path) -> None:
        self.videos_dir = Path(videos_dir)

    def list_videos(self) -> list[Path]:
        videos = [path for path in self.videos_dir.iterdir() if path.is_file() and path.suffix.lower() == ".mp4"]
        videos.sort(key=lambda path: path.stem)
        return videos


class ProgressManager:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            self.save(-1)

    def load(self) -> int:
        with self.path.open("r", encoding="utf-8") as file:
            return int(json.load(file)["last_index"])

    def save(self, index: int) -> None:
        ensure_dir(self.path.parent)
        with self.path.open("w", encoding="utf-8") as file:
            json.dump({"last_index": int(index)}, file)
