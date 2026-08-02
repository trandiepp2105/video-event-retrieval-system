from __future__ import annotations

from pathlib import Path

from ..common import load_json
from ..schemas import EventRecord, OCRRecord, ShotRecord, SubtitleRecord, VideoRecord
from .repository import MetadataRepository


class StoredMetadataLoader:
    def load(self, store_dir: Path) -> MetadataRepository:
        payload = load_json(Path(store_dir) / "metadata.json")
        return MetadataRepository(
            videos={key: VideoRecord(**value) for key, value in payload["videos"].items()},
            events={
                key: EventRecord(
                    event_id=value["event_id"],
                    video_id=value["video_id"],
                    event_order=int(value["event_order"]),
                    start_time_sec=float(value["start_time_sec"]),
                    end_time_sec=float(value["end_time_sec"]),
                    shot_ids=tuple(value["shot_ids"]),
                )
                for key, value in payload["events"].items()
            },
            shots={key: ShotRecord(**value) for key, value in payload["shots"].items()},
            subtitles={key: SubtitleRecord(**value) for key, value in payload["subtitles"].items()},
            ocr_items={key: OCRRecord(**value) for key, value in payload["ocr_items"].items()},
        )
