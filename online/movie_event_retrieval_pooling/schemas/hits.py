from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchHit:
    item_id: str
    faiss_id: int
    score: float
    rank: int
    text: str = ""


@dataclass(frozen=True)
class OCRSearchHit:
    ocr_id: str
    score: float
    rank: int
    text: str
