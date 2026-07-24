from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class FrameRangeMetadata:
    index_id: int
    video_id: str
    frame_start: int
    frame_end: int
    item_id: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FrameRangeMetadata":
        return cls(
            index_id=int(payload["index_id"]),
            video_id=str(payload["video_id"]),
            frame_start=int(payload["frame_start"]),
            frame_end=int(payload["frame_end"]),
            item_id=None if payload.get("item_id") is None else int(payload["item_id"]),
        )


@dataclass
class SearchResult:
    index_id: int
    video_id: str
    frame_start: int
    frame_end: int
    score: float
    item_id: Optional[int] = None


@dataclass
class StageQuery:
    visual: str = ""
    subtitle: str = ""
    ocr: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StageQuery":
        return cls(
            visual=str(payload.get("visual", "")).strip(),
            subtitle=str(payload.get("subtitle", "")).strip(),
            ocr=str(payload.get("ocr", "")).strip(),
        )
