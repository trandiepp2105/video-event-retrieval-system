from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


SUPPORTED_DTYPES = {"float16": np.float16, "float32": np.float32}


@dataclass
class TextEmbeddingConfig:
    input_dir: Path
    output_dir: Path
    model_name_or_path: str
    input_type: str
    device: str | None = None
    batch_size: int = 32
    normalize_embeddings: bool = True
    save_dtype: str = "float32"
    start_index: int = 0
    end_index: int | None = None
    video_ids: list[str] | None = None
    overwrite: bool = False
    text_field: str = "text"
    caption_field: str = "caption"


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


class JsonDatasetScanner:
    def __init__(
        self,
        input_dir: Path,
        start_index: int = 0,
        end_index: int | None = None,
        video_ids: list[str] | None = None,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.start_index = int(start_index)
        self.end_index = end_index
        self.video_ids = [str(video_id) for video_id in video_ids] if video_ids else None

    def get_items(self) -> list[Path]:
        if self.video_ids:
            items: list[Path] = []
            for video_id in self.video_ids:
                path = self.input_dir / f"{video_id}.json"
                if not path.exists():
                    raise FileNotFoundError(f"Missing input file for video_id={video_id}: {path}")
                items.append(path)
            return items

        json_paths = sorted(self.input_dir.glob("*.json"))
        start = max(0, self.start_index)
        if start >= len(json_paths):
            return []
        if self.end_index is None:
            return json_paths[start:]
        return json_paths[start : self.end_index + 1]


class SubtitleJsonLoader:
    OPTIONAL_FIELDS = ("frame_start", "frame_end", "start_time_sec", "end_time_sec")

    def __init__(self, text_field: str = "text") -> None:
        self.text_field = text_field

    def load(self, json_path: Path) -> list[dict[str, Any]]:
        with json_path.open("r", encoding="utf-8") as file:
            items = json.load(file)

        if not isinstance(items, list):
            raise ValueError(f"Subtitle JSON must be a list: {json_path}")

        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"Subtitle item {index} in {json_path} must be a dict")
            if self.text_field not in item:
                raise ValueError(f"Subtitle item {index} in {json_path} missing field: {self.text_field}")

            text = str(item[self.text_field]).strip()
            if not text:
                raise ValueError(f"Subtitle item {index} in {json_path} has empty text")

            payload = dict(item)
            payload[self.text_field] = text
            for field in self.OPTIONAL_FIELDS:
                if field in payload and payload[field] is not None:
                    payload[field] = payload[field]
            normalized.append(payload)
        return normalized


class CaptionJsonLoader:
    OPTIONAL_FIELDS = ("frame_start", "frame_end", "start_time_sec", "end_time_sec")

    def __init__(self, caption_field: str = "caption") -> None:
        self.caption_field = caption_field

    def load(self, json_path: Path) -> list[dict[str, Any]]:
        with json_path.open("r", encoding="utf-8") as file:
            items = json.load(file)

        if not isinstance(items, list):
            raise ValueError(f"Caption JSON must be a list: {json_path}")

        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"Caption item {index} in {json_path} must be a dict")
            if "event_id" not in item:
                raise ValueError(f"Caption item {index} in {json_path} missing field: event_id")
            if self.caption_field not in item:
                raise ValueError(f"Caption item {index} in {json_path} missing field: {self.caption_field}")

            caption = str(item[self.caption_field]).strip()
            if not caption:
                raise ValueError(f"Caption item {index} in {json_path} has empty caption")

            payload = dict(item)
            payload[self.caption_field] = caption
            normalized.append(payload)
        return normalized


class VietnameseTextEmbedder:
    def __init__(self, config: TextEmbeddingConfig) -> None:
        self.config = config
        self.model = SentenceTransformer(
            config.model_name_or_path,
            device=config.device,
        )

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False,
        ).astype(np.float32)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if not self.config.normalize_embeddings and embeddings.size > 0:
            embeddings = l2_normalize(embeddings)
        return embeddings


class TextEmbeddingPipeline:
    def __init__(self, config: TextEmbeddingConfig) -> None:
        self.config = config
        self.scanner = JsonDatasetScanner(
            input_dir=config.input_dir,
            start_index=config.start_index,
            end_index=config.end_index,
            video_ids=config.video_ids,
        )
        self.loader = self._build_loader()
        self.embedder = VietnameseTextEmbedder(config)

    def _build_loader(self) -> SubtitleJsonLoader | CaptionJsonLoader:
        if self.config.input_type == "subtitle":
            return SubtitleJsonLoader(text_field=self.config.text_field)
        if self.config.input_type == "caption":
            return CaptionJsonLoader(caption_field=self.config.caption_field)
        raise ValueError(f"Unsupported input_type: {self.config.input_type}")

    def _cast_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        dtype = SUPPORTED_DTYPES.get(self.config.save_dtype)
        if dtype is None:
            raise ValueError(f"Unsupported save_dtype: {self.config.save_dtype}")
        return embeddings.astype(dtype)

    def _get_texts(self, items: list[dict[str, Any]]) -> list[str]:
        if self.config.input_type == "subtitle":
            return [str(item[self.config.text_field]) for item in items]
        return [str(item[self.config.caption_field]) for item in items]

    def _build_payload(
        self,
        json_path: Path,
        items: list[dict[str, Any]],
        embeddings: np.ndarray,
    ) -> dict[str, Any]:
        video_id = json_path.stem
        kind = "subtitles" if self.config.input_type == "subtitle" else "captions"
        count_key = "num_subtitles" if self.config.input_type == "subtitle" else "num_captions"
        source_key = "subtitle_json_path" if self.config.input_type == "subtitle" else "caption_json_path"
        return {
            "video_id": video_id,
            source_key: str(json_path),
            "input_type": self.config.input_type,
            "model_name_or_path": self.config.model_name_or_path,
            "embedding_dtype": self.config.save_dtype,
            "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and len(embeddings) > 0 else 0,
            count_key: int(len(items)),
            kind: items,
            "embeddings": embeddings,
        }

    def _process_one_file(self, json_path: Path) -> dict[str, Any]:
        items = self.loader.load(json_path)
        texts = self._get_texts(items)
        embeddings = self._cast_embeddings(self.embedder.encode_texts(texts))

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.config.output_dir / f"{json_path.stem}.pkl"
        with output_path.open("wb") as file:
            pickle.dump(self._build_payload(json_path, items, embeddings), file, protocol=pickle.HIGHEST_PROTOCOL)

        return {
            "video_id": json_path.stem,
            "status": "done",
            "output_path": str(output_path),
            "num_items": len(items),
        }

    def run(self) -> dict[str, Any]:
        json_paths = self.scanner.get_items()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        done: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []

        desc = "Embedding subtitles" if self.config.input_type == "subtitle" else "Embedding captions"
        for json_path in tqdm(json_paths, desc=desc):
            output_path = self.config.output_dir / f"{json_path.stem}.pkl"
            if output_path.exists() and not self.config.overwrite:
                skipped.append(
                    {
                        "video_id": json_path.stem,
                        "status": "skipped",
                        "reason": "output_exists",
                        "output_path": str(output_path),
                    }
                )
                continue

            try:
                done.append(self._process_one_file(json_path))
            except Exception as error:  # pragma: no cover - defensive batch processing
                failed.append(
                    {
                        "video_id": json_path.stem,
                        "status": "failed",
                        "error": repr(error),
                        "input_path": str(json_path),
                    }
                )

        summary = {
            "input_type": self.config.input_type,
            "input_dir": str(self.config.input_dir),
            "output_dir": str(self.config.output_dir),
            "model_name_or_path": self.config.model_name_or_path,
            "total_files": len(json_paths),
            "done": done,
            "skipped": skipped,
            "failed": failed,
        }
        summary_path = self.config.output_dir / "manifest.json"
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        return summary
