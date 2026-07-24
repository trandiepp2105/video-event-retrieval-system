from __future__ import annotations

from pathlib import Path

from .bm25_index import BM25Index
from .faiss_index import FaissIndex


class IndexRegistry:
    def __init__(self) -> None:
        self.indices: dict[str, object] = {}

    def register(self, name: str, index: object) -> None:
        self.indices[str(name)] = index

    def load_faiss(self, name: str, input_dir: Path) -> FaissIndex:
        index = FaissIndex.load(input_dir)
        self.register(name, index)
        return index

    def load_bm25(self, name: str, input_dir: Path) -> BM25Index:
        index = BM25Index.load(input_dir)
        self.register(name, index)
        return index

    def get(self, name: str):
        if name not in self.indices:
            raise KeyError(f"Index not loaded: {name}")
        return self.indices[name]
