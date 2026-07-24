from __future__ import annotations

from pathlib import Path

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
    MeiliSearchRuntimeManager,
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
        runtime = None
        print("[Build] Start build-all for temporal event retrieval store")
        print(f"[Build] Output dir: {config.output_dir}")
        if config.auto_start_meilisearch:
            print("[Build] Auto-start Meilisearch is enabled")
            runtime = MeiliSearchRuntimeManager().ensure_running(
                base_url=config.meilisearch_url,
                api_key=config.meilisearch_api_key,
                binary_path=config.meilisearch_binary_path,
                db_path=config.meilisearch_db_path,
            )
        else:
            print("[Build] Auto-start Meilisearch is disabled")
            print(f"[Build] Expect external Meilisearch at: {config.meilisearch_url}")
        print("[Build] Phase 1/6: load metadata repository")
        metadata = load_metadata_repository(config)
        print(
            f"[Build] Metadata loaded: videos={len(metadata.videos)}, "
            f"events={len(metadata.events)}, shots={len(metadata.shots)}, "
            f"subtitles={len(metadata.subtitles)}, ocr_items={len(metadata.ocr_items)}"
        )
        try:
            ensure_dir(config.output_dir)
            ensure_dir(config.output_dir / "indexes" / "faiss")
            ensure_dir(config.output_dir / "indexes" / "ocr")
            ensure_dir(config.output_dir / "indexes" / "subtitle_text")

            print("[Build] Phase 2/6: load embedding matrices")
            event_embeddings, event_item_ids = EventEmbeddingLoader().load(config.event_embedding_dir)
            caption_embeddings, caption_item_ids = CaptionEmbeddingLoader().load(config.caption_embedding_dir)
            shot_embeddings, shot_item_ids = ShotEmbeddingLoader().load(config.shot_embedding_dir)
            subtitle_embeddings, subtitle_item_ids = SubtitleEmbeddingLoader().load(config.subtitle_embedding_dir)
            print(
                "[Build] Embeddings loaded: "
                f"event_vectors={len(event_item_ids)} shape={event_embeddings.shape}, "
                f"caption_vectors={len(caption_item_ids)} shape={caption_embeddings.shape}, "
                f"shot_vectors={len(shot_item_ids)} shape={shot_embeddings.shape}, "
                f"subtitle_vectors={len(subtitle_item_ids)} shape={subtitle_embeddings.shape}"
            )

            print("[Build] Phase 3/6: validate embeddings")
            for embeddings, item_ids in (
                (event_embeddings, event_item_ids),
                (caption_embeddings, caption_item_ids),
                (shot_embeddings, shot_item_ids),
                (subtitle_embeddings, subtitle_item_ids),
            ):
                self.validator.validate(embeddings, item_ids)
            print("[Build] Embedding validation done")

            print("[Build] Phase 4/6: normalize embeddings and build FAISS indexes")
            event_embeddings = self.normalizer.normalize(event_embeddings)
            caption_embeddings = self.normalizer.normalize(caption_embeddings)
            shot_embeddings = self.normalizer.normalize(shot_embeddings)
            subtitle_embeddings = self.normalizer.normalize(subtitle_embeddings)

            self.index_saver.save(self.index_builder.build(event_embeddings), config.output_dir / "indexes" / "faiss" / "event.faiss")
            self.index_saver.save(self.index_builder.build(caption_embeddings), config.output_dir / "indexes" / "faiss" / "caption.faiss")
            self.index_saver.save(self.index_builder.build(shot_embeddings), config.output_dir / "indexes" / "faiss" / "shot.faiss")
            self.index_saver.save(self.index_builder.build(subtitle_embeddings), config.output_dir / "indexes" / "faiss" / "subtitle.faiss")
            print("[Build] Saved FAISS indexes: event, caption, shot, subtitle")

            print("[Build] Phase 5/6: build and save mappings")
            mappings = MappingBundleBuilder().build(
                metadata=metadata,
                event_item_ids=event_item_ids,
                caption_item_ids=caption_item_ids,
                shot_item_ids=shot_item_ids,
                subtitle_item_ids=subtitle_item_ids,
            )
            self.mapping_serializer.save(mappings, config.output_dir)
            print("[Build] Mappings saved")

            print("[Build] Phase 6/6: build OCR documents and index with Meilisearch")
            ocr_documents = [self.ocr_document_builder.build(record) for record in metadata.ocr_items.values()]
            subtitle_documents = [self.subtitle_document_builder.build(record) for record in metadata.subtitles.values()]
            print(f"[Build] OCR documents prepared: {len(ocr_documents)}")
            print(f"[Build] Subtitle documents prepared: {len(subtitle_documents)}")
            OCRStore(documents=ocr_documents).save(config.output_dir / "indexes" / "ocr" / "documents.json")
            OCRStore(documents=subtitle_documents).save(config.output_dir / "indexes" / "subtitle_text" / "documents.json")
            print("[Build] Saved OCR documents.json")
            meili_client = MeiliSearchClient(
                base_url=config.meilisearch_url,
                api_key=config.meilisearch_api_key,
            )
            print(f"[Build] Configuring Meilisearch index: {config.meilisearch_index_name}")
            OCRIndexConfigurator(meili_client).configure(config.meilisearch_index_name)
            OCRIndexWriter(meili_client, batch_size=config.meilisearch_batch_size).add_documents(
                config.meilisearch_index_name,
                ocr_documents,
            )
            print("[Build] OCR documents indexed into Meilisearch")
            subtitle_index_name = config.subtitle_meilisearch_index_name or f"{config.meilisearch_index_name}_subtitle"
            print(f"[Build] Configuring subtitle Meilisearch index: {subtitle_index_name}")
            SubtitleIndexConfigurator(meili_client).configure(subtitle_index_name)
            OCRIndexWriter(meili_client, batch_size=config.meilisearch_batch_size).add_documents(
                subtitle_index_name,
                subtitle_documents,
            )
            print("[Build] Subtitle documents indexed into Meilisearch")
            save_json(
                {
                    "backend": "meilisearch",
                    "url": config.meilisearch_url,
                    "index_name": config.meilisearch_index_name,
                    "api_key_provided": bool(config.meilisearch_api_key),
                    "documents_json_path": str(config.output_dir / "indexes" / "ocr" / "documents.json"),
                    "auto_start_meilisearch": config.auto_start_meilisearch,
                    "meilisearch_binary_path": None if config.meilisearch_binary_path is None else str(config.meilisearch_binary_path),
                    "meilisearch_db_path": None if config.meilisearch_db_path is None else str(config.meilisearch_db_path),
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
                    "auto_start_meilisearch": config.auto_start_meilisearch,
                    "meilisearch_binary_path": None if config.meilisearch_binary_path is None else str(config.meilisearch_binary_path),
                    "meilisearch_db_path": None if config.meilisearch_db_path is None else str(config.meilisearch_db_path),
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
            print("[Build] Manifest saved")
            print("[Build] build-all completed successfully")
            return manifest
        finally:
            if runtime is not None:
                runtime.shutdown()
