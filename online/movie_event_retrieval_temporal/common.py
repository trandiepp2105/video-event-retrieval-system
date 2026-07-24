from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=_json_default)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def save_pickle(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def l2_normalize(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array for normalization, got shape={array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, eps, None)


def to_contiguous_float32(array: np.ndarray) -> np.ndarray:
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    return np.ascontiguousarray(array)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
