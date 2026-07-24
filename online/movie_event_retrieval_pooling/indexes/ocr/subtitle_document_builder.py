from __future__ import annotations

from ...schemas import SubtitleRecord


class SubtitleDocumentBuilder:
    def build(self, record: SubtitleRecord) -> dict:
        return {
            "subtitle_id": record.subtitle_id,
            "video_id": record.video_id,
            "start_time_sec": record.start_time_sec,
            "end_time_sec": record.end_time_sec,
            "frame_start": record.frame_start,
            "frame_end": record.frame_end,
            "text": record.text,
        }
