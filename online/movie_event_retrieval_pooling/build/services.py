from __future__ import annotations

from pathlib import Path
import time

from ..common import ensure_dir, save_json
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


def load_metadata_repository(config: BuildConfig) -> MetadataRepository:
    return DatasetMetadataLoader().load(
        event_dir=config.event_dir,
        shot_embedding_dir=config.shot_embedding_dir,
        subtitle_embedding_dir=config.subtitle_embedding_dir,
        ocr_dir=config.ocr_dir,
    )


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
            save_json(
                {
                    "backend": "meilisearch",
                    "url": config.meilisearch_url,
                    "index_name": config.meilisearch_index_name,
                    "api_key_provided": bool(config.meilisearch_api_key),
                    "documents_json_path": str(config.output_dir / "indexes" / "ocr" / "documents.json"),
                },
                config.output_dir / "indexes" / "ocr" / "config.json",
            )
            save_json(
                {
                    "backend": "meilisearch",
                    "url": config.meilisearch_url,
                    "index_name": subtitle_index_name,
                    "api_key_provided": bool(config.meilisearch_api_key),
                    "documents_json_path": str(config.output_dir / "indexes" / "subtitle_text" / "documents.json"),
                },
                config.output_dir / "indexes" / "subtitle_text" / "config.json",
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
