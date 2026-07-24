from __future__ import annotations

from pathlib import Path

import faiss

from ...common import ensure_dir


class FaissIndexSaver:
    def save(self, index: faiss.Index, output_path: Path) -> None:
        ensure_dir(output_path.parent)
        faiss.write_index(index, str(output_path))
