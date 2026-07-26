from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ShotResult:
    shot_id: str
    event_id: str
    video_id: str
    shot_order: int
    start_time_sec: float
    end_time_sec: float
    score: float
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventResult:
    event_id: str
    video_id: str
    start_time_sec: float
    end_time_sec: float
    score: float
    shot_ids: list[str]
    evidence: dict[str, Any] = field(default_factory=dict)
