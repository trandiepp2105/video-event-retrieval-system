from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .io_utils import load_json, save_json
from .metadata import MetadataStore
from .schemas import SearchResult

try:
    import faiss  # type: ignore
except ImportError as error:  # pragma: no cover
    raise ImportError("Can cai faiss truoc khi dung movie event retrieval system.") from error


def to_float32_matrix(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Vectors must be 2D, got shape={vectors.shape}")
    return vectors


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vectors = to_float32_matrix(vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, eps, None)


class FaissIndex:
    def __init__(
        self,
        *,
        index,
        vectors: np.ndarray,
        ids: np.ndarray,
        metadata_store: MetadataStore,
        index_name: str,
        config: dict,
    ) -> None:
        self.index = index
        self.vectors = to_float32_matrix(vectors)
        self.ids = np.asarray(ids, dtype=np.int64)
        self.metadata_store = metadata_store
        self.index_name = str(index_name)
        self.config = dict(config)
        self.id_to_row = {int(index_id): row for row, index_id in enumerate(self.ids.tolist())}

    @classmethod
    def build(
        cls,
        *,
        vectors: np.ndarray,
        ids: np.ndarray,
        metadata_store: MetadataStore,
        index_name: str,
        config: dict,
    ) -> "FaissIndex":
        vectors = l2_normalize(vectors)
        ids = np.asarray(ids, dtype=np.int64)
        if vectors.shape[0] != ids.shape[0]:
            raise ValueError(f"Vector/id size mismatch: {vectors.shape[0]} vs {ids.shape[0]}")
        base_index = faiss.IndexFlatIP(int(vectors.shape[1]))
        index = faiss.IndexIDMap2(base_index)
        index.add_with_ids(vectors, ids)
        return cls(
            index=index,
            vectors=vectors,
            ids=ids,
            metadata_store=metadata_store,
            index_name=index_name,
            config=config,
        )

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(output_dir / "index.faiss"))
        np.save(output_dir / "vectors.npy", self.vectors)
        np.save(output_dir / "ids.npy", self.ids)
        self.metadata_store.save(output_dir / "metadata.json")
        save_json(self.config, output_dir / "config.json")

    @classmethod
    def load(cls, input_dir: Path) -> "FaissIndex":
        index = faiss.read_index(str(input_dir / "index.faiss"))
        vectors = np.load(input_dir / "vectors.npy")
        ids = np.load(input_dir / "ids.npy")
        metadata_store = MetadataStore.load(input_dir / "metadata.json")
        config = load_json(input_dir / "config.json")
        return cls(
            index=index,
            vectors=vectors,
            ids=ids,
            metadata_store=metadata_store,
            index_name=config.get("index_name", input_dir.name),
            config=config,
        )

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 10,
        allowed_ids: Optional[list[int]] = None,
    ) -> list[SearchResult]:
        query_vector = np.asarray(query_vector, dtype=np.float32)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = l2_normalize(query_vector)

        if allowed_ids is None:
            scores, result_ids = self.index.search(query_vector, int(top_k))
            rows = [
                (int(index_id), float(score))
                for index_id, score in zip(result_ids[0].tolist(), scores[0].tolist())
                if int(index_id) != -1
            ]
        else:
            allowed_rows = [self.id_to_row[index_id] for index_id in allowed_ids if int(index_id) in self.id_to_row]
            if not allowed_rows:
                return []
            unique_rows = np.asarray(sorted(set(allowed_rows)), dtype=np.int64)
            subset_vectors = self.vectors[unique_rows]
            scores = subset_vectors @ query_vector[0]
            order = np.argsort(-scores)[: int(top_k)]
            rows = []
            for subset_index in order.tolist():
                row = int(unique_rows[subset_index])
                index_id = int(self.ids[row])
                rows.append((index_id, float(scores[subset_index])))

        results: list[SearchResult] = []
        for index_id, score in rows:
            meta = self.metadata_store.get(index_id)
            results.append(
                SearchResult(
                    index_id=index_id,
                    video_id=meta.video_id,
                    frame_start=meta.frame_start,
                    frame_end=meta.frame_end,
                    score=float(score),
                    item_id=meta.item_id,
                )
            )
        return results
