from __future__ import annotations

from ...schemas import OCRRecord


def make_meilisearch_safe_id(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "empty_id"
    safe_chars = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    safe = "".join(safe_chars).strip("_")
    return safe or "empty_id"


class OCRDocumentBuilder:
    def build(self, record: OCRRecord) -> dict:
        return {
            "id": make_meilisearch_safe_id(record.ocr_id),
            "ocr_id": record.ocr_id,
            "video_id": record.video_id,
            "event_id": record.event_id,
            "shot_id": record.shot_id,
            "timestamp_sec": record.timestamp_sec,
            "text_raw": record.text_raw,
            "text_clean": record.text_clean,
            "confidence": record.confidence,
        }
