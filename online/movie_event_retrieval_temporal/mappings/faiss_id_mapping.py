from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FaissIdMapping:
    faiss_id_to_item_id: tuple[str, ...]
    item_id_to_faiss_id: dict[str, int]

    @classmethod
    def from_item_ids(cls, item_ids: Sequence[str]) -> "FaissIdMapping":
        normalized = tuple(str(item_id) for item_id in item_ids)
        return cls(
            faiss_id_to_item_id=normalized,
            item_id_to_faiss_id={item_id: index for index, item_id in enumerate(normalized)},
        )

    def item_id_from_faiss_id(self, faiss_id: int) -> str:
        return self.faiss_id_to_item_id[int(faiss_id)]

    def faiss_id_from_item_id(self, item_id: str) -> int:
        return self.item_id_to_faiss_id[str(item_id)]

    def item_ids_from_faiss_ids(self, faiss_ids: np.ndarray) -> list[str]:
        return [self.item_id_from_faiss_id(int(faiss_id)) for faiss_id in faiss_ids.tolist() if int(faiss_id) >= 0]

    def faiss_ids_from_item_ids(self, item_ids: Sequence[str]) -> np.ndarray:
        return np.asarray([self.faiss_id_from_item_id(item_id) for item_id in item_ids], dtype=np.int64)
