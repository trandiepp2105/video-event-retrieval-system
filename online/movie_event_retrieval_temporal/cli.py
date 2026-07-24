from __future__ import annotations

import argparse
from pathlib import Path

from .build import RetrievalStoreBuilder
from .common import load_json
from .config import BuildConfig, SearchConfig
from .metadata import MetadataRepository
from .retrieval import TemporalMovieEventRetriever
from .schemas import EventRecord, OCRRecord, ShotRecord, SubtitleRecord, VideoRecord


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Temporal movie event retrieval CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_all = subparsers.add_parser("build-all")
    build_all.add_argument("--event_dir", type=Path, required=True)
    build_all.add_argument("--event_embedding_dir", type=Path, required=True)
    build_all.add_argument("--caption_embedding_dir", type=Path, required=True)
    build_all.add_argument("--shot_embedding_dir", type=Path, required=True)
    build_all.add_argument("--subtitle_embedding_dir", type=Path, required=True)
    build_all.add_argument("--ocr_dir", type=Path, required=True)
    build_all.add_argument("--output_dir", type=Path, required=True)
    build_all.add_argument("--overwrite", action="store_true")

    search = subparsers.add_parser("search")
    search.add_argument("--store_dir", type=Path, required=True)
    search.add_argument("--query_json", type=Path, default=None)
    search.add_argument("--raw_query", type=str, default="")
    search.add_argument("--translated_query", type=str, default="")
    search.add_argument("--subtitle_query", type=str, default="")
    search.add_argument("--ocr_query", type=str, default="")
    search.add_argument("--temporal_checkpoint_path", type=Path, default=None)
    search.add_argument("--clip_model_path_override", type=Path, default=None)
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
    search.add_argument("--event_device", type=str, default="cpu")
    search.add_argument("--caption_device", type=str, default="cpu")
    search.add_argument("--subtitle_device", type=str, default="cpu")
    search.add_argument("--output_json", type=Path, default=None)
    return parser


def _load_metadata(store_dir: Path) -> MetadataRepository:
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
        manifest = RetrievalStoreBuilder().build(
            BuildConfig(
                event_dir=args.event_dir,
                event_embedding_dir=args.event_embedding_dir,
                caption_embedding_dir=args.caption_embedding_dir,
                shot_embedding_dir=args.shot_embedding_dir,
                subtitle_embedding_dir=args.subtitle_embedding_dir,
                ocr_dir=args.ocr_dir,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
            )
        )
        print(manifest)
        return

    if args.command == "search":
        retriever = TemporalMovieEventRetriever(_load_metadata(args.store_dir), args.store_dir)
        result = retriever.search(
            SearchConfig(
                store_dir=args.store_dir,
                query_json=args.query_json,
                raw_query=args.raw_query,
                translated_query=args.translated_query,
                subtitle_query=args.subtitle_query,
                ocr_query=args.ocr_query,
                temporal_checkpoint_path=args.temporal_checkpoint_path,
                clip_model_path_override=args.clip_model_path_override,
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
                event_device=args.event_device,
                caption_device=args.caption_device,
                subtitle_device=args.subtitle_device,
                output_json=args.output_json,
            )
        )
        print(result)
