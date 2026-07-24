from __future__ import annotations

from pathlib import Path

import faiss


class FaissIndexLoader:
    def load(self, path: Path) -> faiss.Index:
        return faiss.read_index(str(path))
