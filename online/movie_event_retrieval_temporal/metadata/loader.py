from __future__ import annotations

from pathlib import Path

from ..common import load_json, load_pickle
from ..schemas import EventRecord, OCRRecord, ShotRecord, SubtitleRecord, VideoRecord
from .repository import MetadataRepository


class DatasetMetadataLoader:
    def load(
        self,
        *,
        event_dir: Path,
        shot_embedding_dir: Path,
        subtitle_embedding_dir: Path,
        ocr_dir: Path,
    ) -> MetadataRepository:
        videos: dict[str, VideoRecord] = {}
        events: dict[str, EventRecord] = {}
        shots: dict[str, ShotRecord] = {}
        subtitles: dict[str, SubtitleRecord] = {}
        ocr_items: dict[str, OCRRecord] = {}

        video_ids = sorted(
            [path.name for path in event_dir.iterdir() if path.is_dir() and (path / "events.json").exists()],
            key=lambda value: int(value),
        )

        for video_id in video_ids:
            event_items = load_json(event_dir / video_id / "events.json")
            max_end_sec = 0.0
            for event_order, item in enumerate(event_items):
                event_key = self.make_event_id(video_id, item["event_id"])
                shot_ids = tuple(self.make_shot_id(video_id, shot_id) for shot_id in item.get("shot_ids", []))
                start_time_sec = float(item["start_time_sec"])
                end_time_sec = float(item["end_time_sec"])
                max_end_sec = max(max_end_sec, end_time_sec)
                events[event_key] = EventRecord(
                    event_id=event_key,
                    video_id=str(video_id),
                    event_order=event_order,
                    start_time_sec=start_time_sec,
                    end_time_sec=end_time_sec,
                    shot_ids=shot_ids,
                )

            shot_payload = load_pickle(shot_embedding_dir / f"{video_id}.pkl")
            for shot_order, item in enumerate(shot_payload["shots"]):
                shot_key = self.make_shot_id(video_id, item["shot_id"])
                event_key = self.make_event_id(video_id, item["event_id"])
                shots[shot_key] = ShotRecord(
                    shot_id=shot_key,
                    event_id=event_key,
                    video_id=str(video_id),
                    shot_order=shot_order,
                    start_time_sec=float(item["start_time_sec"]),
                    end_time_sec=float(item["end_time_sec"]),
                )
                max_end_sec = max(max_end_sec, float(item["end_time_sec"]))

            subtitle_payload = load_pickle(subtitle_embedding_dir / f"{video_id}.pkl")
            subtitle_items = subtitle_payload.get("subtitles")
            if subtitle_items is None:
                subtitle_items = subtitle_payload.get("items")
            if subtitle_items is None:
                subtitle_items = subtitle_payload.get("captions", [])
            for subtitle_index, item in enumerate(subtitle_items):
                subtitle_key = self.make_subtitle_id(video_id, subtitle_index)
                subtitles[subtitle_key] = SubtitleRecord(
                    subtitle_id=subtitle_key,
                    video_id=str(video_id),
                    start_time_sec=float(item["start_time_sec"]),
                    end_time_sec=float(item["end_time_sec"]),
                    text=str(item.get("text", "")).strip(),
                    frame_start=item.get("frame_start"),
                    frame_end=item.get("frame_end"),
                )
                max_end_sec = max(max_end_sec, float(item["end_time_sec"]))

            ocr_items_raw = load_json(ocr_dir / f"{video_id}.json")
            for ocr_index, item in enumerate(ocr_items_raw):
                ocr_key = self.make_ocr_id(video_id, ocr_index)
                shot_key = self._find_shot_id_for_timestamp(str(video_id), float(item["time_sec"]), shots)
                event_key = shots[shot_key].event_id
                ocr_items[ocr_key] = OCRRecord(
                    ocr_id=ocr_key,
                    video_id=str(video_id),
                    shot_id=shot_key,
                    event_id=event_key,
                    timestamp_sec=float(item["time_sec"]),
                    text_raw=str(item.get("text", "")).strip(),
                    text_clean=self._clean_text(str(item.get("text", "")).strip()),
                    confidence=item.get("confidence"),
                )
                max_end_sec = max(max_end_sec, float(item["time_sec"]))

            videos[str(video_id)] = VideoRecord(video_id=str(video_id), duration_sec=max_end_sec)

        return MetadataRepository(
            videos=videos,
            events=events,
            shots=shots,
            subtitles=subtitles,
            ocr_items=ocr_items,
        )

    @staticmethod
    def make_event_id(video_id: str | int, event_id: str | int) -> str:
        return f"{video_id}:{event_id}"

    @staticmethod
    def make_shot_id(video_id: str | int, shot_id: str | int) -> str:
        return f"{video_id}:{shot_id}"

    @staticmethod
    def make_subtitle_id(video_id: str | int, subtitle_index: int) -> str:
        return f"{video_id}:{subtitle_index}"

    @staticmethod
    def make_ocr_id(video_id: str | int, ocr_index: int) -> str:
        return f"{video_id}:{ocr_index}"

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join(text.split())

    @staticmethod
    def _find_shot_id_for_timestamp(video_id: str, timestamp_sec: float, shots: dict[str, ShotRecord]) -> str:
        best_match: str | None = None
        fallback: str | None = None
        for shot_id, shot in shots.items():
            if shot.video_id != video_id:
                continue
            fallback = shot_id
            if shot.start_time_sec <= timestamp_sec < shot.end_time_sec:
                return shot_id
            if best_match is None and abs(shot.end_time_sec - timestamp_sec) < 1e-6:
                best_match = shot_id
        if best_match is not None:
            return best_match
        if fallback is None:
            raise KeyError(f"Khong tim thay shot nao cho video_id={video_id}, timestamp={timestamp_sec}")
        return fallback
