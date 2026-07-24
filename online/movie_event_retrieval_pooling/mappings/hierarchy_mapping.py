from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HierarchyMapping:
    video_to_events: dict[str, tuple[str, ...]]
    video_to_shots: dict[str, tuple[str, ...]]
    event_to_shots: dict[str, tuple[str, ...]]
    shot_to_event: dict[str, str]
    event_to_video: dict[str, str]
    shot_to_video: dict[str, str]

    def event_ids_for_video(self, video_id: str) -> tuple[str, ...]:
        return self.video_to_events.get(str(video_id), ())

    def shot_ids_for_video(self, video_id: str) -> tuple[str, ...]:
        return self.video_to_shots.get(str(video_id), ())

    def shot_ids_for_event(self, event_id: str) -> tuple[str, ...]:
        return self.event_to_shots.get(str(event_id), ())

    def event_id_for_shot(self, shot_id: str) -> str:
        return self.shot_to_event[str(shot_id)]

    def video_id_for_event(self, event_id: str) -> str:
        return self.event_to_video[str(event_id)]

    def video_id_for_shot(self, shot_id: str) -> str:
        return self.shot_to_video[str(shot_id)]
