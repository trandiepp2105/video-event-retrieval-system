from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    duration_sec: float


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    video_id: str
    event_order: int
    start_time_sec: float
    end_time_sec: float
    shot_ids: tuple[str, ...]


@dataclass(frozen=True)
class ShotRecord:
    shot_id: str
    event_id: str
    video_id: str
    shot_order: int
    start_time_sec: float
    end_time_sec: float


@dataclass(frozen=True)
class SubtitleRecord:
    subtitle_id: str
    video_id: str
    start_time_sec: float
    end_time_sec: float
    text: str
    frame_start: int | None = None
    frame_end: int | None = None


@dataclass(frozen=True)
class OCRRecord:
    ocr_id: str
    video_id: str
    shot_id: str
    event_id: str
    timestamp_sec: float
    text_raw: str
    text_clean: str
    confidence: float | None = None


@dataclass(frozen=True)
class WeightedShotReference:
    shot_id: str
    weight: float


@dataclass(frozen=True)
class WeightedEventReference:
    event_id: str
    weight: float
