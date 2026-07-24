from __future__ import annotations

from pathlib import Path


class JsonDatasetScanner:
    def __init__(
        self,
        input_dir: Path,
        start_index: int = 0,
        end_index: int | None = None,
        video_ids: list[str] | None = None,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.start_index = int(start_index)
        self.end_index = end_index
        self.video_ids = [str(video_id) for video_id in video_ids] if video_ids else None

    def get_items(self) -> list[Path]:
        if self.video_ids:
            items: list[Path] = []
            for video_id in self.video_ids:
                path = self.input_dir / f"{video_id}.json"
                if not path.exists():
                    raise FileNotFoundError(f"Missing input file for video_id={video_id}: {path}")
                items.append(path)
            return items

        json_paths = sorted(self.input_dir.glob("*.json"))
        start = max(0, self.start_index)
        if start >= len(json_paths):
            return []
        if self.end_index is None:
            return json_paths[start:]
        return json_paths[start : self.end_index + 1]
