from __future__ import annotations

from pathlib import Path

from ..common import load_json, save_json
from ..schemas import WeightedEventReference, WeightedShotReference
from .builder import MappingBundle
from .faiss_id_mapping import FaissIdMapping
from .hierarchy_mapping import HierarchyMapping
from .ocr_mapping import OCRMapping
from .subtitle_mapping import SubtitleMapping


class MappingSerializer:
    def save(self, bundle: MappingBundle, output_dir: Path) -> None:
        save_json(
            {
                "event_mapping": list(bundle.event_mapping.faiss_id_to_item_id),
                "caption_mapping": list(bundle.caption_mapping.faiss_id_to_item_id),
                "shot_mapping": list(bundle.shot_mapping.faiss_id_to_item_id),
                "subtitle_mapping_ids": list(bundle.subtitle_mapping_ids.faiss_id_to_item_id),
                "hierarchy": {
                    "video_to_events": bundle.hierarchy.video_to_events,
                    "video_to_shots": bundle.hierarchy.video_to_shots,
                    "event_to_shots": bundle.hierarchy.event_to_shots,
                    "shot_to_event": bundle.hierarchy.shot_to_event,
                    "event_to_video": bundle.hierarchy.event_to_video,
                    "shot_to_video": bundle.hierarchy.shot_to_video,
                },
                "subtitle_mapping": {
                    "subtitle_to_shots": {
                        key: [{"shot_id": item.shot_id, "weight": item.weight} for item in value]
                        for key, value in bundle.subtitle_mapping.subtitle_to_shots.items()
                    },
                    "subtitle_to_events": {
                        key: [{"event_id": item.event_id, "weight": item.weight} for item in value]
                        for key, value in bundle.subtitle_mapping.subtitle_to_events.items()
                    },
                },
                "ocr_mapping": {
                    "ocr_to_shot": bundle.ocr_mapping.ocr_to_shot,
                    "ocr_to_event": bundle.ocr_mapping.ocr_to_event,
                    "ocr_to_video": bundle.ocr_mapping.ocr_to_video,
                },
            },
            output_dir / "mappings.json",
        )

    def load(self, input_dir: Path) -> MappingBundle:
        payload = load_json(input_dir / "mappings.json")
        return MappingBundle(
            event_mapping=FaissIdMapping.from_item_ids(payload["event_mapping"]),
            caption_mapping=FaissIdMapping.from_item_ids(payload["caption_mapping"]),
            shot_mapping=FaissIdMapping.from_item_ids(payload["shot_mapping"]),
            subtitle_mapping_ids=FaissIdMapping.from_item_ids(payload["subtitle_mapping_ids"]),
            hierarchy=HierarchyMapping(
                video_to_events={key: tuple(value) for key, value in payload["hierarchy"]["video_to_events"].items()},
                video_to_shots={key: tuple(value) for key, value in payload["hierarchy"]["video_to_shots"].items()},
                event_to_shots={key: tuple(value) for key, value in payload["hierarchy"]["event_to_shots"].items()},
                shot_to_event=dict(payload["hierarchy"]["shot_to_event"]),
                event_to_video=dict(payload["hierarchy"]["event_to_video"]),
                shot_to_video=dict(payload["hierarchy"]["shot_to_video"]),
            ),
            subtitle_mapping=SubtitleMapping(
                subtitle_to_shots={
                    key: tuple(WeightedShotReference(shot_id=item["shot_id"], weight=float(item["weight"])) for item in value)
                    for key, value in payload["subtitle_mapping"]["subtitle_to_shots"].items()
                },
                subtitle_to_events={
                    key: tuple(WeightedEventReference(event_id=item["event_id"], weight=float(item["weight"])) for item in value)
                    for key, value in payload["subtitle_mapping"]["subtitle_to_events"].items()
                },
            ),
            ocr_mapping=OCRMapping(
                ocr_to_shot=dict(payload["ocr_mapping"]["ocr_to_shot"]),
                ocr_to_event=dict(payload["ocr_mapping"]["ocr_to_event"]),
                ocr_to_video=dict(payload["ocr_mapping"]["ocr_to_video"]),
            ),
        )
