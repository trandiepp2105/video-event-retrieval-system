from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .common import SUPPORTED_DTYPES
from .config import TextEmbeddingConfig
from .embedder import VietnameseTextEmbedder
from .loaders import CaptionJsonLoader, SubtitleJsonLoader
from .scanner import JsonDatasetScanner


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
