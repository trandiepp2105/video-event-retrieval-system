from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .bm25_index import BM25Index, OCRDocument
from .faiss_index import FaissIndex, l2_normalize
from .io_utils import load_json, load_pickle, utc_timestamp
from .metadata import MetadataStore
from .schemas import FrameRangeMetadata


def _finalize_faiss_index(
    *,
    index_name: str,
    vectors: List[np.ndarray],
    ids: List[int],
    metadata_store: MetadataStore,
    config: dict,
) -> FaissIndex:
    if not vectors:
        raise RuntimeError(f"No vectors found for index={index_name}")
    matrix = np.stack(vectors).astype(np.float32)
    matrix = l2_normalize(matrix)
    return FaissIndex.build(
        vectors=matrix,
        ids=np.asarray(ids, dtype=np.int64),
        metadata_store=metadata_store,
        index_name=index_name,
        config=config,
    )


@dataclass
class VisualIndexBuilder:
    keyframe_embedding_dir: Path
    index_name: str = "visual"

    def build(self) -> FaissIndex:
        vectors: List[np.ndarray] = []
        ids: List[int] = []
        store = MetadataStore()
        index_id = 0

        for pkl_path in sorted(self.keyframe_embedding_dir.glob("*.pkl")):
            video_id = pkl_path.stem
            payload = load_pickle(pkl_path)
            frame_indices = payload.get("keyframe_frame_indices")
            embeddings = np.asarray(payload.get("keyframe_embeddings"), dtype=np.float32)
            if frame_indices is None or embeddings.size == 0:
                raise ValueError(f"Missing keyframe data in file: {pkl_path}")
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            if len(frame_indices) != int(embeddings.shape[0]):
                raise ValueError(
                    f"Frame/embedding mismatch for video_id={video_id}: "
                    f"{len(frame_indices)} vs {embeddings.shape[0]}"
                )
            for row, frame_idx in enumerate(frame_indices):
                frame_idx = int(frame_idx)
                vectors.append(np.asarray(embeddings[row], dtype=np.float32).reshape(-1))
                ids.append(index_id)
                store.add(
                    FrameRangeMetadata(
                        index_id=index_id,
                        video_id=video_id,
                        frame_start=frame_idx,
                        frame_end=frame_idx,
                        item_id=frame_idx,
                    )
                )
                index_id += 1

        dim = int(vectors[0].shape[0])
        return _finalize_faiss_index(
            index_name=self.index_name,
            vectors=vectors,
            ids=ids,
            metadata_store=store,
            config={
                "index_name": self.index_name,
                "metric": "cosine",
                "index_type": "IndexFlatIP",
                "normalized": True,
                "dim": dim,
                "num_vectors": len(vectors),
                "input_dir": str(self.keyframe_embedding_dir),
                "created_at": utc_timestamp(),
            },
        )


@dataclass
class SubtitleIndexBuilder:
    subtitle_embedding_dir: Path

    def build(self) -> FaissIndex:
        vectors: List[np.ndarray] = []
        ids: List[int] = []
        store = MetadataStore()
        index_id = 0

        for pkl_path in sorted(self.subtitle_embedding_dir.glob("*.pkl")):
            video_id = pkl_path.stem
            payload = load_pickle(pkl_path)
            items = payload.get("items")
            embeddings = np.asarray(payload.get("embeddings"), dtype=np.float32)
            if items is None or embeddings.size == 0:
                raise ValueError(f"Missing subtitle items/embeddings in file: {pkl_path}")
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            if len(items) != int(embeddings.shape[0]):
                raise ValueError(
                    f"Subtitle item/embedding mismatch for video_id={video_id}: "
                    f"{len(items)} vs {embeddings.shape[0]}"
                )
            for row, item in enumerate(items):
                frame_start = int(item["frame_start"])
                frame_end = int(item["frame_end"])
                vectors.append(np.asarray(embeddings[row], dtype=np.float32).reshape(-1))
                ids.append(index_id)
                store.add(
                    FrameRangeMetadata(
                        index_id=index_id,
                        video_id=video_id,
                        frame_start=frame_start,
                        frame_end=frame_end,
                        item_id=int(item.get("subtitle_index", row)),
                    )
                )
                index_id += 1

        dim = int(vectors[0].shape[0])
        return _finalize_faiss_index(
            index_name="subtitle",
            vectors=vectors,
            ids=ids,
            metadata_store=store,
            config={
                "index_name": "subtitle",
                "metric": "cosine",
                "index_type": "IndexFlatIP",
                "normalized": True,
                "dim": dim,
                "num_vectors": len(vectors),
                "input_dir": str(self.subtitle_embedding_dir),
                "created_at": utc_timestamp(),
            },
        )


@dataclass
class OCRIndexBuilder:
    ocr_dir: Path

    def build(self) -> BM25Index:
        documents: list[OCRDocument] = []
        store = MetadataStore()
        index_id = 0

        for json_path in sorted(self.ocr_dir.glob("*.json")):
            video_id = json_path.stem
            payload = load_json(json_path)
            if not isinstance(payload, list):
                raise ValueError(f"OCR file must contain a list: {json_path}")
            for row, item in enumerate(payload):
                if "frame_id" not in item:
                    raise ValueError(f"Missing frame_id in OCR file: {json_path}")
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                frame_id = int(item["frame_id"])
                documents.append(
                    OCRDocument(
                        index_id=index_id,
                        video_id=video_id,
                        frame_id=frame_id,
                        text=text,
                    )
                )
                store.add(
                    FrameRangeMetadata(
                        index_id=index_id,
                        video_id=video_id,
                        frame_start=frame_id,
                        frame_end=frame_id,
                        item_id=frame_id,
                    )
                )
                index_id += 1

        if not documents:
            raise RuntimeError(f"No OCR documents found in dir: {self.ocr_dir}")
        return BM25Index.build(
            documents=documents,
            metadata_store=store,
            index_name="ocr",
            config={
                "index_name": "ocr",
                "index_type": "bm25",
                "num_documents": len(documents),
                "input_dir": str(self.ocr_dir),
                "created_at": utc_timestamp(),
            },
        )
