from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class FileIO:
    @staticmethod
    def ensure_dir(path: str | Path):
        Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_json(path: str | Path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_json(data: Any, path: str | Path):
        path = Path(path)
        FileIO.ensure_dir(path.parent)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_pickle(path: str | Path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def save_npy(array: np.ndarray, path: str | Path):
        path = Path(path)
        FileIO.ensure_dir(path.parent)
        np.save(path, array)


class MathUtils:
    @staticmethod
    def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        return float(np.dot(a, b))

    @staticmethod
    def overlap_len(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
        return max(0.0, min(end_a, end_b) - max(start_a, start_b))


class PoolingUtils:
    @staticmethod
    def mean_pool(vectors: List[np.ndarray]) -> Optional[np.ndarray]:
        vectors = [np.asarray(v, dtype=np.float32) for v in vectors if v is not None]
        if len(vectors) == 0:
            return None
        x = np.stack(vectors, axis=0)
        v = x.mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-8)
        return v.astype(np.float32)

    @staticmethod
    def max_pool(vectors: List[np.ndarray]) -> Optional[np.ndarray]:
        vectors = [np.asarray(v, dtype=np.float32) for v in vectors if v is not None]
        if len(vectors) == 0:
            return None
        x = np.stack(vectors, axis=0)
        v = x.max(axis=0)
        v = v / (np.linalg.norm(v) + 1e-8)
        return v.astype(np.float32)

    @staticmethod
    def softmax_center_pool(vectors: List[np.ndarray], temperature: float = 15.0) -> Optional[np.ndarray]:
        vectors = [np.asarray(v, dtype=np.float32) for v in vectors if v is not None]
        if len(vectors) == 0:
            return None
        E = np.stack(vectors, axis=0).astype(np.float32)
        E = MathUtils.l2_normalize(E, axis=1)
        center = E.mean(axis=0)
        center = center / (np.linalg.norm(center) + 1e-8)
        sims = E @ center
        logits = sims * float(temperature)
        logits = logits - np.max(logits)
        weights = np.exp(logits)
        weights = weights / (weights.sum() + 1e-8)
        pooled = (E * weights[:, None]).sum(axis=0)
        pooled = pooled / (np.linalg.norm(pooled) + 1e-8)
        return pooled.astype(np.float32)

    @staticmethod
    def multi_pool(vectors: List[np.ndarray], temperature: float = 15.0) -> Dict[str, Optional[np.ndarray]]:
        return {
            "mean": PoolingUtils.mean_pool(vectors),
            "max": PoolingUtils.max_pool(vectors),
            "softmax": PoolingUtils.softmax_center_pool(vectors, temperature=temperature),
        }
