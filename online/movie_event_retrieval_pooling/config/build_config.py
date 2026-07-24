from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BuildConfig:
    event_dir: Path
    event_embedding_dir: Path
    caption_embedding_dir: Path
    shot_embedding_dir: Path
    subtitle_embedding_dir: Path
    ocr_dir: Path
    output_dir: Path
    meilisearch_url: str
    meilisearch_index_name: str
    subtitle_meilisearch_index_name: str | None = None
    meilisearch_api_key: str | None = None
    meilisearch_batch_size: int = 250
    auto_start_meilisearch: bool = False
    meilisearch_binary_path: Path | None = None
    meilisearch_db_path: Path | None = None
    overwrite: bool = False
