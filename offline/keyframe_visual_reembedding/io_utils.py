import pickle
from pathlib import Path
from typing import Any

from .config import PipelineConfig


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as file:
        return pickle.load(file)


def save_pickle(payload: Any, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as file:
        pickle.dump(payload, file)


class DatasetScanner:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.videos_dir = Path(config.videos_dir)
        self.reference_keyframe_dir = Path(config.reference_keyframe_dir)
        self.output_dir = Path(config.output_dir)

    def _all_reference_items(self) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        for reference_path in sorted(self.reference_keyframe_dir.glob("*.pkl"), key=lambda path: int(path.stem)):
            video_id = reference_path.stem
            items.append(
                {
                    "video_name": video_id,
                    "video_path": str(self.videos_dir / f"{video_id}.mp4"),
                    "reference_path": str(reference_path),
                    "output_path": str(self.output_dir / f"{video_id}.pkl"),
                }
            )
        return items

    def get_video_items(self) -> list[dict[str, str]]:
        items = self._all_reference_items()

        if self.config.video_ids:
            selected_ids = {str(video_id) for video_id in self.config.video_ids}
            items = [item for item in items if item["video_name"] in selected_ids]
            return items

        start_index = max(int(self.config.start_index), 0)
        end_index = self.config.end_index
        if end_index is None:
            return items[start_index:]
        return items[start_index : int(end_index) + 1]
