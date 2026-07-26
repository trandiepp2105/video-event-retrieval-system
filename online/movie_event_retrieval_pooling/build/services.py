from __future__ import annotations

from pathlib import Path
import time

from ..common import ensure_dir, load_json, save_json
from ..config import BuildConfig
from ..embeddings import (
    CaptionEmbeddingLoader,
    EmbeddingNormalizer,
    EmbeddingValidator,
    EventEmbeddingLoader,
    ShotEmbeddingLoader,
    SubtitleEmbeddingLoader,
)
from ..indexes.faiss import FaissIndexSaver, FlatIPIndexBuilder
from ..indexes.ocr import (
    MeiliSearchClient,
    OCRDocumentBuilder,
    OCRIndexConfigurator,
    OCRIndexWriter,
    OCRStore,
    SubtitleDocumentBuilder,
    SubtitleIndexConfigurator,
)
from ..mappings import MappingBundleBuilder
from ..mappings.serializer import MappingSerializer
from ..metadata import DatasetMetadataLoader, MetadataRepository
from ..schemas import OCRRecord, SubtitleRecord


def load_metadata_repository(config: BuildConfig) -> MetadataRepository:
    return DatasetMetadataLoader().load(
        event_dir=config.event_dir,
        shot_embedding_dir=config.shot_embedding_dir,
        subtitle_embedding_dir=config.subtitle_embedding_dir,
        ocr_dir=config.ocr_dir,
    )


def _save_meilisearch_configs(
    *,
    output_dir: Path,
    meilisearch_url: str,
    meilisearch_api_key: str | None,
    ocr_index_name: str,
    subtitle_index_name: str,
) -> None:
    save_json(
        {
            "backend": "meilisearch",
            "url": meilisearch_url,
            "index_name": ocr_index_name,
            "api_key_provided": bool(meilisearch_api_key),
            "documents_json_path": str(output_dir / "indexes" / "ocr" / "documents.json"),
        },
        output_dir / "indexes" / "ocr" / "config.json",
    )
    save_json(
        {
            "backend": "meilisearch",
            "url": meilisearch_url,
            "index_name": subtitle_index_name,
            "api_key_provided": bool(meilisearch_api_key),
            "documents_json_path": str(output_dir / "indexes" / "subtitle_text" / "documents.json"),
        },
        output_dir / "indexes" / "subtitle_text" / "config.json",
    )


def _build_ocr_documents_only(
    *,
    output_dir: Path,
    ocr_dir: Path,
    meilisearch_url: str,
    meilisearch_api_key: str | None,
    meilisearch_index_name: str,
    batch_size: int = 10000,
    wait_each_batch: bool = False,
) -> int:
    ensure_dir(output_dir)
    ensure_dir(output_dir / "indexes" / "ocr")
    ocr_documents: list[dict] = []
    builder = OCRDocumentBuilder()
    for json_path in sorted(ocr_dir.glob("*.json"), key=lambda path: int(path.stem)):
        video_id = json_path.stem
        items = load_json(json_path)
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items):
            ocr_id = f"{video_id}:{idx}"
            timestamp_sec = item.get("time_sec")
            if timestamp_sec is None:
                timestamp_sec = item.get("timestamp_sec", 0.0)
            text = str(item.get("text", item.get("text_raw", ""))).strip()
            ocr_documents.append(
                builder.build(
                    OCRRecord(
                        ocr_id=ocr_id,
                        video_id=str(video_id),
                        shot_id=str(item.get("shot_id", "")),
                        event_id=str(item.get("event_id", "")),
                        timestamp_sec=float(timestamp_sec),
                        text_raw=text,
                        text_clean=" ".join(text.split()),
                        confidence=item.get("confidence"),
                    )
                )
            )
    OCRStore(documents=ocr_documents).save(output_dir / "indexes" / "ocr" / "documents.json")
    client = MeiliSearchClient(base_url=meilisearch_url, api_key=meilisearch_api_key)
    OCRIndexConfigurator(client).configure(meilisearch_index_name)
    OCRIndexWriter(client, batch_size=batch_size, wait_each_batch=wait_each_batch).add_documents(
        meilisearch_index_name,
        ocr_documents,
    )
    return len(ocr_documents)


def _build_subtitle_documents_only(
    *,
    output_dir: Path,
    subtitle_dir: Path,
    meilisearch_url: str,
    meilisearch_api_key: str | None,
    subtitle_index_name: str,
    batch_size: int = 10000,
    wait_each_batch: bool = False,
) -> int:
    ensure_dir(output_dir)
    ensure_dir(output_dir / "indexes" / "subtitle_text")
    subtitle_documents: list[dict] = []
    builder = SubtitleDocumentBuilder()
    for json_path in sorted(subtitle_dir.glob("*.json"), key=lambda path: int(path.stem)):
        video_id = json_path.stem
        items = load_json(json_path)
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items):
            subtitle_documents.append(
                builder.build(
                    SubtitleRecord(
                        subtitle_id=f"{video_id}:{idx}",
                        video_id=str(video_id),
                        start_time_sec=float(item.get("start_time_sec", 0.0)),
                        end_time_sec=float(item.get("end_time_sec", 0.0)),
                        text=str(item.get("text", "")).strip(),
                        frame_start=item.get("frame_start"),
                        frame_end=item.get("frame_end"),
                    )
                )
            )
    OCRStore(documents=subtitle_documents).save(output_dir / "indexes" / "subtitle_text" / "documents.json")
    client = MeiliSearchClient(base_url=meilisearch_url, api_key=meilisearch_api_key)
    SubtitleIndexConfigurator(client).configure(subtitle_index_name)
    OCRIndexWriter(client, batch_size=batch_size, wait_each_batch=wait_each_batch).add_documents(
        subtitle_index_name,
        subtitle_documents,
    )
    return len(subtitle_documents)


class RetrievalStoreBuilder:
    def __init__(self) -> None:
        self.validator = EmbeddingValidator()
        self.normalizer = EmbeddingNormalizer()
        self.index_builder = FlatIPIndexBuilder()
        self.index_saver = FaissIndexSaver()
        self.mapping_serializer = MappingSerializer()
        self.ocr_document_builder = OCRDocumentBuilder()
        self.subtitle_document_builder = SubtitleDocumentBuilder()

    def build(self, config: BuildConfig) -> dict:
        print("[Build] Building pooling retrieval store")
        metadata = load_metadata_repository(config)
        try:
            ensure_dir(config.output_dir)
            ensure_dir(config.output_dir / "indexes" / "faiss")
            ensure_dir(config.output_dir / "indexes" / "ocr")
            ensure_dir(config.output_dir / "indexes" / "subtitle_text")

            event_embeddings, event_item_ids = EventEmbeddingLoader().load(config.event_embedding_dir)
            caption_embeddings, caption_item_ids = CaptionEmbeddingLoader().load(config.caption_embedding_dir)
            shot_embeddings, shot_item_ids = ShotEmbeddingLoader().load(config.shot_embedding_dir)
            subtitle_embeddings, subtitle_item_ids = SubtitleEmbeddingLoader().load(config.subtitle_embedding_dir)

            for embeddings, item_ids in (
                (event_embeddings, event_item_ids),
                (caption_embeddings, caption_item_ids),
                (shot_embeddings, shot_item_ids),
                (subtitle_embeddings, subtitle_item_ids),
            ):
                self.validator.validate(embeddings, item_ids)
            event_embeddings = self.normalizer.normalize(event_embeddings)
            caption_embeddings = self.normalizer.normalize(caption_embeddings)
            shot_embeddings = self.normalizer.normalize(shot_embeddings)
            subtitle_embeddings = self.normalizer.normalize(subtitle_embeddings)

            self.index_saver.save(self.index_builder.build(event_embeddings), config.output_dir / "indexes" / "faiss" / "event.faiss")
            self.index_saver.save(self.index_builder.build(caption_embeddings), config.output_dir / "indexes" / "faiss" / "caption.faiss")
            self.index_saver.save(self.index_builder.build(shot_embeddings), config.output_dir / "indexes" / "faiss" / "shot.faiss")
            self.index_saver.save(self.index_builder.build(subtitle_embeddings), config.output_dir / "indexes" / "faiss" / "subtitle.faiss")
            mappings = MappingBundleBuilder().build(
                metadata=metadata,
                event_item_ids=event_item_ids,
                caption_item_ids=caption_item_ids,
                shot_item_ids=shot_item_ids,
                subtitle_item_ids=subtitle_item_ids,
            )
            self.mapping_serializer.save(mappings, config.output_dir)
            ocr_documents = [self.ocr_document_builder.build(record) for record in metadata.ocr_items.values()]
            subtitle_documents = [self.subtitle_document_builder.build(record) for record in metadata.subtitles.values()]
            OCRStore(documents=ocr_documents).save(config.output_dir / "indexes" / "ocr" / "documents.json")
            OCRStore(documents=subtitle_documents).save(config.output_dir / "indexes" / "subtitle_text" / "documents.json")
            meili_client = MeiliSearchClient(
                base_url=config.meilisearch_url,
                api_key=config.meilisearch_api_key,
            )
            effective_batch_size = 10000
            effective_wait_each_batch = False
            OCRIndexConfigurator(meili_client).configure(config.meilisearch_index_name)
            OCRIndexWriter(
                meili_client,
                batch_size=effective_batch_size,
                wait_each_batch=effective_wait_each_batch,
            ).add_documents(
                config.meilisearch_index_name,
                ocr_documents,
            )
            subtitle_index_name = config.subtitle_meilisearch_index_name or f"{config.meilisearch_index_name}_subtitle"
            SubtitleIndexConfigurator(meili_client).configure(subtitle_index_name)
            OCRIndexWriter(
                meili_client,
                batch_size=effective_batch_size,
                wait_each_batch=effective_wait_each_batch,
            ).add_documents(
                subtitle_index_name,
                subtitle_documents,
            )
            time.sleep(15.0)
            _save_meilisearch_configs(
                output_dir=config.output_dir,
                meilisearch_url=config.meilisearch_url,
                meilisearch_api_key=config.meilisearch_api_key,
                ocr_index_name=config.meilisearch_index_name,
                subtitle_index_name=subtitle_index_name,
            )

            metadata_payload = {
                "videos": {key: value.__dict__ for key, value in metadata.videos.items()},
                "events": {key: value.__dict__ for key, value in metadata.events.items()},
                "shots": {key: value.__dict__ for key, value in metadata.shots.items()},
                "subtitles": {key: value.__dict__ for key, value in metadata.subtitles.items()},
                "ocr_items": {key: value.__dict__ for key, value in metadata.ocr_items.items()},
            }
            save_json(metadata_payload, config.output_dir / "metadata.json")

            manifest = {
                "event_index_path": str(config.output_dir / "indexes" / "faiss" / "event.faiss"),
                "caption_index_path": str(config.output_dir / "indexes" / "faiss" / "caption.faiss"),
                "shot_index_path": str(config.output_dir / "indexes" / "faiss" / "shot.faiss"),
                "subtitle_index_path": str(config.output_dir / "indexes" / "faiss" / "subtitle.faiss"),
                "ocr_documents_path": str(config.output_dir / "indexes" / "ocr" / "documents.json"),
                "subtitle_documents_path": str(config.output_dir / "indexes" / "subtitle_text" / "documents.json"),
                "ocr_backend": "meilisearch",
                "meilisearch_url": config.meilisearch_url,
                "meilisearch_index_name": config.meilisearch_index_name,
                "subtitle_meilisearch_index_name": subtitle_index_name,
                "num_videos": len(metadata.videos),
                "num_events": len(metadata.events),
                "num_shots": len(metadata.shots),
                "num_subtitles": len(metadata.subtitles),
                "num_ocr_items": len(metadata.ocr_items),
            }
            save_json(manifest, config.output_dir / "manifest.json")
            print(
                "[Build] Done | "
                f"videos={len(metadata.videos)} "
                f"events={len(metadata.events)} "
                f"shots={len(metadata.shots)} "
                f"subtitles={len(metadata.subtitles)} "
                f"ocr_items={len(metadata.ocr_items)}"
            )
            return manifest
        finally:
            pass


def build_ocr_index_only(
    *,
    output_dir: Path,
    ocr_dir: Path,
    meilisearch_url: str,
    meilisearch_api_key: str | None,
    meilisearch_index_name: str,
    batch_size: int = 10000,
    wait_each_batch: bool = False,
) -> dict:
    num_docs = _build_ocr_documents_only(
        output_dir=output_dir,
        ocr_dir=ocr_dir,
        meilisearch_url=meilisearch_url,
        meilisearch_api_key=meilisearch_api_key,
        meilisearch_index_name=meilisearch_index_name,
        batch_size=batch_size,
        wait_each_batch=wait_each_batch,
    )
    subtitle_config_path = output_dir / "indexes" / "subtitle_text" / "config.json"
    subtitle_index_name = f"{meilisearch_index_name}_subtitle"
    if subtitle_config_path.is_file():
        subtitle_index_name = str(load_json(subtitle_config_path).get("index_name", subtitle_index_name))
    _save_meilisearch_configs(
        output_dir=output_dir,
        meilisearch_url=meilisearch_url,
        meilisearch_api_key=meilisearch_api_key,
        ocr_index_name=meilisearch_index_name,
        subtitle_index_name=subtitle_index_name,
    )
    return {
        "index_type": "ocr",
        "index_name": meilisearch_index_name,
        "documents": num_docs,
        "output_dir": str(output_dir),
    }


def build_subtitle_index_only(
    *,
    output_dir: Path,
    subtitle_dir: Path,
    meilisearch_url: str,
    meilisearch_api_key: str | None,
    subtitle_index_name: str,
    batch_size: int = 10000,
    wait_each_batch: bool = False,
) -> dict:
    num_docs = _build_subtitle_documents_only(
        output_dir=output_dir,
        subtitle_dir=subtitle_dir,
        meilisearch_url=meilisearch_url,
        meilisearch_api_key=meilisearch_api_key,
        subtitle_index_name=subtitle_index_name,
        batch_size=batch_size,
        wait_each_batch=wait_each_batch,
    )
    ocr_config_path = output_dir / "indexes" / "ocr" / "config.json"
    ocr_index_name = "movie_event_pooling_ocr"
    if ocr_config_path.is_file():
        ocr_index_name = str(load_json(ocr_config_path).get("index_name", ocr_index_name))
    _save_meilisearch_configs(
        output_dir=output_dir,
        meilisearch_url=meilisearch_url,
        meilisearch_api_key=meilisearch_api_key,
        ocr_index_name=ocr_index_name,
        subtitle_index_name=subtitle_index_name,
    )
    return {
        "index_type": "subtitle",
        "index_name": subtitle_index_name,
        "documents": num_docs,
        "output_dir": str(output_dir),
    }
