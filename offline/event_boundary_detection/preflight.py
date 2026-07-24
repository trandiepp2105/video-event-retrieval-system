from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .config import EventGroupingDatasetConfig


class DatasetPreflight:
    def __init__(self, config: EventGroupingDatasetConfig):
        self.config = config
        self.features_dir = Path(config.features_dir)

    def _list_ids_from_files(self, directory: Path, suffix: str = ".pkl") -> set[str]:
        if not directory.exists():
            return set()
        return {path.stem for path in directory.glob(f"*{suffix}")}

    def _list_ids_from_dirs(self, directory: Path) -> set[str]:
        if not directory.exists():
            return set()
        return {path.name for path in directory.iterdir() if path.is_dir()}

    def build_manifest(self) -> Dict[str, Any]:
        visual_ids = self._list_ids_from_files(self.features_dir / "visual_embeddings")
        action_ids = self._list_ids_from_files(self.features_dir / "action_features")
        subtitle_ids = self._list_ids_from_files(self.features_dir / "subtitle_embeddings")
        face_ids = self._list_ids_from_dirs(self.features_dir / "face_detection")

        all_ids = sorted(visual_ids | action_ids | subtitle_ids | face_ids, key=lambda x: int(x))
        if self.config.video_ids is not None:
            requested = {str(x) for x in self.config.video_ids}
            all_ids = [video_id for video_id in all_ids if video_id in requested]
        else:
            start = max(0, int(self.config.start_index))
            if start >= len(all_ids):
                all_ids = []
            elif self.config.end_index is None:
                all_ids = all_ids[start:]
            else:
                all_ids = all_ids[start : self.config.end_index + 1]

        missing_rows = []
        eligible_ids = []
        for video_id in all_ids:
            missing = []
            if video_id not in visual_ids:
                missing.append("visual")
            if video_id not in action_ids:
                missing.append("action")
            if video_id not in subtitle_ids:
                missing.append("subtitle")
            if video_id not in face_ids:
                missing.append("face")
            if len(missing) == 0:
                eligible_ids.append(video_id)
            else:
                missing_rows.append({"video_id": video_id, "missing": ", ".join(missing)})

        return {
            "visual_count": len(visual_ids),
            "action_count": len(action_ids),
            "subtitle_count": len(subtitle_ids),
            "face_count": len(face_ids),
            "all_ids": all_ids,
            "eligible_ids": eligible_ids,
            "missing_rows": missing_rows,
        }

    def summarize(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "visual_count": manifest["visual_count"],
            "action_count": manifest["action_count"],
            "subtitle_count": manifest["subtitle_count"],
            "face_count": manifest["face_count"],
            "eligible_count": len(manifest["eligible_ids"]),
            "missing_count": len(manifest["missing_rows"]),
        }
