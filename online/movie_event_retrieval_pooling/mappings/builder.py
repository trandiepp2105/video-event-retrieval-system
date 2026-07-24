from __future__ import annotations

from dataclasses import dataclass

from ..metadata import MetadataRepository
from ..schemas import WeightedEventReference, WeightedShotReference
from .faiss_id_mapping import FaissIdMapping
from .hierarchy_mapping import HierarchyMapping
from .ocr_mapping import OCRMapping
from .subtitle_mapping import SubtitleMapping


@dataclass(frozen=True)
class MappingBundle:
    event_mapping: FaissIdMapping
    caption_mapping: FaissIdMapping
    shot_mapping: FaissIdMapping
    subtitle_mapping_ids: FaissIdMapping
    hierarchy: HierarchyMapping
    subtitle_mapping: SubtitleMapping
    ocr_mapping: OCRMapping


class MappingBundleBuilder:
    def build(
        self,
        *,
        metadata: MetadataRepository,
        event_item_ids: list[str],
        caption_item_ids: list[str],
        shot_item_ids: list[str],
        subtitle_item_ids: list[str],
    ) -> MappingBundle:
        return MappingBundle(
            event_mapping=FaissIdMapping.from_item_ids(event_item_ids),
            caption_mapping=FaissIdMapping.from_item_ids(caption_item_ids),
            shot_mapping=FaissIdMapping.from_item_ids(shot_item_ids),
            subtitle_mapping_ids=FaissIdMapping.from_item_ids(subtitle_item_ids),
            hierarchy=self._build_hierarchy(metadata),
            subtitle_mapping=self._build_subtitle_mapping(metadata),
            ocr_mapping=self._build_ocr_mapping(metadata),
        )

    def _build_hierarchy(self, metadata: MetadataRepository) -> HierarchyMapping:
        video_to_events: dict[str, list[str]] = {}
        video_to_shots: dict[str, list[str]] = {}
        event_to_shots: dict[str, list[str]] = {}
        shot_to_event: dict[str, str] = {}
        event_to_video: dict[str, str] = {}
        shot_to_video: dict[str, str] = {}

        for event_id, event in sorted(metadata.events.items(), key=lambda item: (int(item[1].video_id), item[1].event_order)):
            video_to_events.setdefault(event.video_id, []).append(event_id)
            event_to_shots[event_id] = list(event.shot_ids)
            event_to_video[event_id] = event.video_id

        for shot_id, shot in sorted(metadata.shots.items(), key=lambda item: (int(item[1].video_id), item[1].shot_order)):
            video_to_shots.setdefault(shot.video_id, []).append(shot_id)
            shot_to_event[shot_id] = shot.event_id
            shot_to_video[shot_id] = shot.video_id

        return HierarchyMapping(
            video_to_events={key: tuple(value) for key, value in video_to_events.items()},
            video_to_shots={key: tuple(value) for key, value in video_to_shots.items()},
            event_to_shots={key: tuple(value) for key, value in event_to_shots.items()},
            shot_to_event=shot_to_event,
            event_to_video=event_to_video,
            shot_to_video=shot_to_video,
        )

    def _build_subtitle_mapping(self, metadata: MetadataRepository) -> SubtitleMapping:
        subtitle_to_shots: dict[str, tuple[WeightedShotReference, ...]] = {}
        subtitle_to_events: dict[str, tuple[WeightedEventReference, ...]] = {}

        shots_by_video: dict[str, list] = {}
        events_by_video: dict[str, list] = {}
        for shot in metadata.shots.values():
            shots_by_video.setdefault(shot.video_id, []).append(shot)
        for event in metadata.events.values():
            events_by_video.setdefault(event.video_id, []).append(event)

        for subtitle_id, subtitle in metadata.subtitles.items():
            subtitle_duration = max(subtitle.end_time_sec - subtitle.start_time_sec, 1e-6)
            shot_refs: list[WeightedShotReference] = []
            for shot in shots_by_video.get(subtitle.video_id, []):
                overlap = self._overlap(
                    subtitle.start_time_sec,
                    subtitle.end_time_sec,
                    shot.start_time_sec,
                    shot.end_time_sec,
                )
                if overlap > 0.0:
                    shot_refs.append(WeightedShotReference(shot_id=shot.shot_id, weight=overlap / subtitle_duration))
            subtitle_to_shots[subtitle_id] = tuple(shot_refs)

            event_scores: dict[str, float] = {}
            for event in events_by_video.get(subtitle.video_id, []):
                overlap = self._overlap(
                    subtitle.start_time_sec,
                    subtitle.end_time_sec,
                    event.start_time_sec,
                    event.end_time_sec,
                )
                if overlap > 0.0:
                    event_scores[event.event_id] = overlap / subtitle_duration
            subtitle_to_events[subtitle_id] = tuple(
                WeightedEventReference(event_id=event_id, weight=weight)
                for event_id, weight in sorted(event_scores.items())
            )

        return SubtitleMapping(
            subtitle_to_shots=subtitle_to_shots,
            subtitle_to_events=subtitle_to_events,
        )

    def _build_ocr_mapping(self, metadata: MetadataRepository) -> OCRMapping:
        return OCRMapping(
            ocr_to_shot={ocr_id: record.shot_id for ocr_id, record in metadata.ocr_items.items()},
            ocr_to_event={ocr_id: record.event_id for ocr_id, record in metadata.ocr_items.items()},
            ocr_to_video={ocr_id: record.video_id for ocr_id, record in metadata.ocr_items.items()},
        )

    @staticmethod
    def _overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
        return max(0.0, min(end_a, end_b) - max(start_a, start_b))
