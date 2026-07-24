from __future__ import annotations

from ...schemas import OCRRecord


class OCRDocumentBuilder:
    def build(self, record: OCRRecord) -> dict:
        return {
            "ocr_id": record.ocr_id,
            "video_id": record.video_id,
            "event_id": record.event_id,
            "shot_id": record.shot_id,
            "timestamp_sec": record.timestamp_sec,
            "text_raw": record.text_raw,
            "text_clean": record.text_clean,
            "confidence": record.confidence,
        }
