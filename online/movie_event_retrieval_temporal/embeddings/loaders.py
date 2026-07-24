from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import load_pickle
from ..metadata.loader import DatasetMetadataLoader


class EventEmbeddingLoader:
    def load(self, input_dir: Path) -> tuple[np.ndarray, list[str]]:
        vectors: list[np.ndarray] = []
        item_ids: list[str] = []
        for path in sorted(input_dir.glob("*.pkl"), key=lambda value: int(value.stem)):
            payload = load_pickle(path)
            video_id = str(payload.get("video_id", path.stem))
            for item in payload["events"]:
                item_ids.append(DatasetMetadataLoader.make_event_id(video_id, item["event_id"]))
                vectors.append(np.asarray(item["embedding"], dtype=np.float32))
        return np.stack(vectors).astype(np.float32), item_ids


class ShotEmbeddingLoader:
    def load(self, input_dir: Path) -> tuple[np.ndarray, list[str]]:
        vectors: list[np.ndarray] = []
        item_ids: list[str] = []
        for path in sorted(input_dir.glob("*.pkl"), key=lambda value: int(value.stem)):
            payload = load_pickle(path)
            video_id = str(payload.get("video_id", path.stem))
            for item in payload["shots"]:
                item_ids.append(DatasetMetadataLoader.make_shot_id(video_id, item["shot_id"]))
                vectors.append(np.asarray(item["embedding"], dtype=np.float32))
        return np.stack(vectors).astype(np.float32), item_ids


class CaptionEmbeddingLoader:
    def load(self, input_dir: Path) -> tuple[np.ndarray, list[str]]:
        vectors: list[np.ndarray] = []
        item_ids: list[str] = []
        for path in sorted(input_dir.glob("*.pkl"), key=lambda value: int(value.stem)):
            payload = load_pickle(path)
            video_id = str(payload.get("video_id", path.stem))
            items = payload.get("captions", [])
            embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
            if len(items) != len(embeddings):
                raise ValueError(f"Caption count mismatch in {path}")
            for index, item in enumerate(items):
                item_ids.append(DatasetMetadataLoader.make_event_id(video_id, item["event_id"]))
                vectors.append(embeddings[index])
        return np.stack(vectors).astype(np.float32), item_ids


class SubtitleEmbeddingLoader:
    def load(self, input_dir: Path) -> tuple[np.ndarray, list[str]]:
        vectors: list[np.ndarray] = []
        item_ids: list[str] = []
        for path in sorted(input_dir.glob("*.pkl"), key=lambda value: int(value.stem)):
            payload = load_pickle(path)
            video_id = str(payload.get("video_id", path.stem))
            items = payload.get("subtitles", payload.get("captions", []))
            embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
            if len(items) != len(embeddings):
                raise ValueError(f"Subtitle count mismatch in {path}")
            for index, _item in enumerate(items):
                item_ids.append(DatasetMetadataLoader.make_subtitle_id(video_id, index))
                vectors.append(embeddings[index])
        return np.stack(vectors).astype(np.float32), item_ids
