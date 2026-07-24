from __future__ import annotations

import json
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


def l2_normalize(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.clip(norms, eps, None)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
