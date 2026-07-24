from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OCRMapping:
    ocr_to_shot: dict[str, str]
    ocr_to_event: dict[str, str]
    ocr_to_video: dict[str, str]

    def shot_id_for_ocr(self, ocr_id: str) -> str:
        return self.ocr_to_shot[str(ocr_id)]

    def event_id_for_ocr(self, ocr_id: str) -> str:
        return self.ocr_to_event[str(ocr_id)]

    def video_id_for_ocr(self, ocr_id: str) -> str:
        return self.ocr_to_video[str(ocr_id)]
