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
from ..indexes.ocr import OCRDocumentBuilder, OCRStore
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

    def build(self, config: BuildConfig) -> dict:
        metadata = load_metadata_repository(config)
        ensure_dir(config.output_dir)
        ensure_dir(config.output_dir / "indexes" / "faiss")
        ensure_dir(config.output_dir / "indexes" / "ocr")

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
        OCRStore(documents=ocr_documents).save(config.output_dir / "indexes" / "ocr" / "documents.json")

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
            "num_videos": len(metadata.videos),
            "num_events": len(metadata.events),
            "num_shots": len(metadata.shots),
            "num_subtitles": len(metadata.subtitles),
            "num_ocr_items": len(metadata.ocr_items),
        }
        save_json(manifest, config.output_dir / "manifest.json")
        return manifest
