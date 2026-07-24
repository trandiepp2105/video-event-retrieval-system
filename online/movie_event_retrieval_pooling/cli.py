from __future__ import annotations

import argparse
from pathlib import Path

from .config import BuildConfig, SearchConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pooling movie event retrieval CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_all = subparsers.add_parser("build-all")
    build_all.add_argument("--event_dir", type=Path, required=True)
    build_all.add_argument("--event_embedding_dir", type=Path, required=True)
    build_all.add_argument("--caption_embedding_dir", type=Path, required=True)
    build_all.add_argument("--shot_embedding_dir", type=Path, required=True)
    build_all.add_argument("--subtitle_embedding_dir", type=Path, required=True)
    build_all.add_argument("--ocr_dir", type=Path, required=True)
    build_all.add_argument("--output_dir", type=Path, required=True)
    build_all.add_argument("--meilisearch_url", type=str, required=True)
    build_all.add_argument("--meilisearch_index_name", type=str, required=True)
    build_all.add_argument("--meilisearch_api_key", type=str, default=None)
    build_all.add_argument("--meilisearch_batch_size", type=int, default=1000)
    build_all.add_argument("--auto_start_meilisearch", action="store_true")
    build_all.add_argument("--meilisearch_binary_path", type=Path, default=None)
    build_all.add_argument("--meilisearch_db_path", type=Path, default=None)
    build_all.add_argument("--overwrite", action="store_true")

    search = subparsers.add_parser("search")
    search.add_argument("--store_dir", type=Path, required=True)
    search.add_argument("--query_json", type=Path, default=None)
    search.add_argument("--raw_query", type=str, default="")
    search.add_argument("--translated_query", type=str, default="")
    search.add_argument("--subtitle_query", type=str, default="")
    search.add_argument("--ocr_query", type=str, default="")
    search.add_argument("--clip_model_path", type=Path, default=None)
    search.add_argument("--clip_model_name", type=str, default="ViT-H-14-quickgelu")
    search.add_argument("--caption_model_path", type=str, default=None)
    search.add_argument("--subtitle_model_path", type=str, default=None)
    search.add_argument("--event_top_k", type=int, default=200)
    search.add_argument("--caption_top_k", type=int, default=200)
    search.add_argument("--subtitle_top_k", type=int, default=200)
    search.add_argument("--ocr_top_k", type=int, default=200)
    search.add_argument("--candidate_event_top_k", type=int, default=100)
    search.add_argument("--candidate_video_top_k", type=int, default=30)
    search.add_argument("--shot_top_k", type=int, default=100)
    search.add_argument("--final_top_k", type=int, default=30)
    search.add_argument("--rrf_k", type=int, default=60)
    search.add_argument("--event_weight", type=float, default=1.0)
    search.add_argument("--caption_weight", type=float, default=0.8)
    search.add_argument("--subtitle_weight", type=float, default=0.6)
    search.add_argument("--ocr_weight", type=float, default=0.4)
    search.add_argument("--shot_weight", type=float, default=0.8)
    search.add_argument("--parent_event_weight", type=float, default=0.2)
    search.add_argument("--visual_device", type=str, default="cpu")
    search.add_argument("--caption_device", type=str, default="cpu")
    search.add_argument("--subtitle_device", type=str, default="cpu")
    search.add_argument("--meilisearch_url", type=str, default=None)
    search.add_argument("--meilisearch_index_name", type=str, default=None)
    search.add_argument("--meilisearch_api_key", type=str, default=None)
    search.add_argument("--auto_start_meilisearch", action="store_true")
    search.add_argument("--meilisearch_binary_path", type=Path, default=None)
    search.add_argument("--meilisearch_db_path", type=Path, default=None)
    search.add_argument("--output_json", type=Path, default=None)
    return parser


def _load_metadata(store_dir: Path):
    from online.movie_event_retrieval_temporal.common import load_json
    from online.movie_event_retrieval_temporal.metadata import MetadataRepository
    from online.movie_event_retrieval_temporal.schemas import (
        EventRecord,
        OCRRecord,
        ShotRecord,
        SubtitleRecord,
        VideoRecord,
    )

    payload = load_json(store_dir / "metadata.json")
    return MetadataRepository(
        videos={key: VideoRecord(**value) for key, value in payload["videos"].items()},
        events={
            key: EventRecord(
                event_id=value["event_id"],
                video_id=value["video_id"],
                event_order=int(value["event_order"]),
                start_time_sec=float(value["start_time_sec"]),
                end_time_sec=float(value["end_time_sec"]),
                shot_ids=tuple(value["shot_ids"]),
            )
            for key, value in payload["events"].items()
        },
        shots={key: ShotRecord(**value) for key, value in payload["shots"].items()},
        subtitles={key: SubtitleRecord(**value) for key, value in payload["subtitles"].items()},
        ocr_items={key: OCRRecord(**value) for key, value in payload["ocr_items"].items()},
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build-all":
        from online.movie_event_retrieval_temporal.build import RetrievalStoreBuilder

        manifest = RetrievalStoreBuilder().build(
            BuildConfig(
                event_dir=args.event_dir,
                event_embedding_dir=args.event_embedding_dir,
                caption_embedding_dir=args.caption_embedding_dir,
                shot_embedding_dir=args.shot_embedding_dir,
                subtitle_embedding_dir=args.subtitle_embedding_dir,
                ocr_dir=args.ocr_dir,
                output_dir=args.output_dir,
                meilisearch_url=args.meilisearch_url,
                meilisearch_index_name=args.meilisearch_index_name,
                meilisearch_api_key=args.meilisearch_api_key,
                meilisearch_batch_size=args.meilisearch_batch_size,
                auto_start_meilisearch=args.auto_start_meilisearch,
                meilisearch_binary_path=args.meilisearch_binary_path,
                meilisearch_db_path=args.meilisearch_db_path,
                overwrite=args.overwrite,
            )
        )
        print(manifest)
        return

    if args.command == "search":
        from .retrieval import PoolingMovieEventRetriever

        retriever = PoolingMovieEventRetriever(_load_metadata(args.store_dir), args.store_dir)
        result = retriever.search(
            SearchConfig(
                store_dir=args.store_dir,
                query_json=args.query_json,
                raw_query=args.raw_query,
                translated_query=args.translated_query,
                subtitle_query=args.subtitle_query,
                ocr_query=args.ocr_query,
                clip_model_path=args.clip_model_path,
                clip_model_name=args.clip_model_name,
                caption_model_path=args.caption_model_path,
                subtitle_model_path=args.subtitle_model_path,
                event_top_k=args.event_top_k,
                caption_top_k=args.caption_top_k,
                subtitle_top_k=args.subtitle_top_k,
                ocr_top_k=args.ocr_top_k,
                candidate_event_top_k=args.candidate_event_top_k,
                candidate_video_top_k=args.candidate_video_top_k,
                shot_top_k=args.shot_top_k,
                final_top_k=args.final_top_k,
                rrf_k=args.rrf_k,
                event_weight=args.event_weight,
                caption_weight=args.caption_weight,
                subtitle_weight=args.subtitle_weight,
                ocr_weight=args.ocr_weight,
                shot_weight=args.shot_weight,
                parent_event_weight=args.parent_event_weight,
                visual_device=args.visual_device,
                caption_device=args.caption_device,
                subtitle_device=args.subtitle_device,
                meilisearch_url=args.meilisearch_url,
                meilisearch_index_name=args.meilisearch_index_name,
                meilisearch_api_key=args.meilisearch_api_key,
                auto_start_meilisearch=args.auto_start_meilisearch,
                meilisearch_binary_path=args.meilisearch_binary_path,
                meilisearch_db_path=args.meilisearch_db_path,
                output_json=args.output_json,
            )
        )
        print(result)
