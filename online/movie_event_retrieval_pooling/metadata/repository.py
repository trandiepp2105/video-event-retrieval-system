from __future__ import annotations

from dataclasses import dataclass

from ..schemas import EventRecord, OCRRecord, ShotRecord, SubtitleRecord, VideoRecord


@dataclass(frozen=True)
class MetadataRepository:
    videos: dict[str, VideoRecord]
    events: dict[str, EventRecord]
    shots: dict[str, ShotRecord]
    subtitles: dict[str, SubtitleRecord]
    ocr_items: dict[str, OCRRecord]
