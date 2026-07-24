from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...common import load_json, save_json


@dataclass(frozen=True)
class OCRStore:
    documents: list[dict]

    def save(self, output_path: Path) -> None:
        save_json(self.documents, output_path)

    @classmethod
    def load(cls, input_path: Path) -> "OCRStore":
        return cls(documents=load_json(input_path))
