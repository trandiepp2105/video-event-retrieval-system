from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .io_utils import load_json, save_json
from .schemas import FrameRangeMetadata


class MetadataStore:
    def __init__(self) -> None:
        self.id_to_meta: dict[int, FrameRangeMetadata] = {}
        self.video_to_ids: dict[str, list[int]] = defaultdict(list)

    def add(self, meta: FrameRangeMetadata) -> None:
        index_id = int(meta.index_id)
        self.id_to_meta[index_id] = meta
        self.video_to_ids[meta.video_id].append(index_id)

    def get(self, index_id: int) -> FrameRangeMetadata:
        return self.id_to_meta[int(index_id)]

    def get_many(self, index_ids: Iterable[int]) -> list[FrameRangeMetadata]:
        return [self.get(index_id) for index_id in index_ids]

    def get_ids_by_video(self, video_id: str) -> list[int]:
        return list(self.video_to_ids.get(str(video_id), []))

    def save(self, path: Path) -> None:
        payload = [meta.to_dict() for meta in sorted(self.id_to_meta.values(), key=lambda item: item.index_id)]
        save_json(payload, path)

    @classmethod
    def load(cls, path: Path) -> "MetadataStore":
        payload = load_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"Metadata file must contain a list: {path}")
        store = cls()
        for item in payload:
            store.add(FrameRangeMetadata.from_dict(item))
        return store
