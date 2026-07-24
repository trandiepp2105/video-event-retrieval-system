from __future__ import annotations

from dataclasses import dataclass

from ..schemas import WeightedEventReference, WeightedShotReference


@dataclass(frozen=True)
class SubtitleMapping:
    subtitle_to_shots: dict[str, tuple[WeightedShotReference, ...]]
    subtitle_to_events: dict[str, tuple[WeightedEventReference, ...]]

    def shots_for_subtitle(self, subtitle_id: str) -> tuple[WeightedShotReference, ...]:
        return self.subtitle_to_shots.get(str(subtitle_id), ())

    def events_for_subtitle(self, subtitle_id: str) -> tuple[WeightedEventReference, ...]:
        return self.subtitle_to_events.get(str(subtitle_id), ())
