from __future__ import annotations

import faiss
import numpy as np


class FlatIPIndexBuilder:
    def build(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        index = faiss.IndexFlatIP(int(embeddings.shape[1]))
        index.add(embeddings)
        return index
