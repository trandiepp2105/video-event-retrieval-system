from pathlib import Path

from .config import PipelineConfig


class DatasetScanner:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def get_video_items(self) -> list[dict[str, str]]:
        video_paths = sorted(Path(self.config.videos_dir).glob("*.mp4"))
        items: list[dict[str, str]] = []

        for video_path in video_paths:
            video_name = video_path.stem
            output_path = Path(self.config.output_dir) / f"{video_name}.pkl"
            items.append(
                {
                    "video_name": video_name,
                    "video_path": str(video_path),
                    "output_path": str(output_path),
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
                print(f"[WARN] Missing requested video id: {video_id}")

            return filtered_items

        start = max(0, int(self.config.start_index))
        end = self.config.end_index

        if start >= len(items):
            return []
        if end is None:
            return items[start:]
        return items[start:end + 1]
