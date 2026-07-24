from __future__ import annotations

import argparse
from pathlib import Path

from .builders import OCRIndexBuilder, SubtitleIndexBuilder, VisualIndexBuilder
from .encoders import E5TextEncoder, OpenClipTextEncoder
from .io_utils import load_json, save_json
from .registry import IndexRegistry
from .schemas import StageQuery
from .search import SearchEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Movie event retrieval system CLI (current mode: keyframe retrieval)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_visual = subparsers.add_parser("build-visual-index")
    build_visual.add_argument("--keyframe_embedding_dir", type=Path, required=True)
    build_visual.add_argument("--output_dir", type=Path, required=True)

    build_subtitle = subparsers.add_parser("build-subtitle-index")
    build_subtitle.add_argument("--subtitle_embedding_dir", type=Path, required=True)
    build_subtitle.add_argument("--output_dir", type=Path, required=True)

    build_ocr = subparsers.add_parser("build-ocr-index")
    build_ocr.add_argument("--ocr_dir", type=Path, required=True)
    build_ocr.add_argument("--output_dir", type=Path, required=True)

    build_all = subparsers.add_parser("build-all")
    build_all.add_argument("--keyframe_embedding_dir", type=Path, required=True)
    build_all.add_argument("--subtitle_embedding_dir", type=Path, required=True)
    build_all.add_argument("--ocr_dir", type=Path, required=True)
    build_all.add_argument("--output_dir", type=Path, required=True)

    search_stage = subparsers.add_parser("search-stage")
    search_stage.add_argument("--visual_index_dir", type=Path, required=True)
    search_stage.add_argument("--subtitle_index_dir", type=Path, required=True)
    search_stage.add_argument("--ocr_index_dir", type=Path, required=True)
    search_stage.add_argument("--clip_model_path", type=Path, required=True)
    search_stage.add_argument("--subtitle_model_path", type=Path, required=True)
    search_stage.add_argument("--visual_query", type=str, default="")
    search_stage.add_argument("--subtitle_query", type=str, default="")
    search_stage.add_argument("--ocr_query", type=str, default="")
    search_stage.add_argument("--top_k", type=int, default=100)
    search_stage.add_argument("--visual_top_k", type=int, default=300)
    search_stage.add_argument("--subtitle_top_k", type=int, default=300)
    search_stage.add_argument("--ocr_top_k", type=int, default=300)
    search_stage.add_argument("--visual_weight", type=float, default=0.45)
    search_stage.add_argument("--ocr_weight", type=float, default=0.35)
    search_stage.add_argument("--subtitle_weight", type=float, default=0.20)
    search_stage.add_argument("--visual_device", type=str, default="cpu")
    search_stage.add_argument("--subtitle_device", type=str, default="cpu")
    search_stage.add_argument("--output_json", type=Path, default=None)

    search_temporal = subparsers.add_parser("search-temporal")
    search_temporal.add_argument("--visual_index_dir", type=Path, required=True)
    search_temporal.add_argument("--subtitle_index_dir", type=Path, required=True)
    search_temporal.add_argument("--ocr_index_dir", type=Path, required=True)
    search_temporal.add_argument("--clip_model_path", type=Path, required=True)
    search_temporal.add_argument("--subtitle_model_path", type=Path, required=True)
    search_temporal.add_argument("--stages_json", type=Path, required=True)
    search_temporal.add_argument("--top_k", type=int, default=20)
    search_temporal.add_argument("--per_stage_top_k", type=int, default=100)
    search_temporal.add_argument("--visual_top_k", type=int, default=300)
    search_temporal.add_argument("--subtitle_top_k", type=int, default=300)
    search_temporal.add_argument("--ocr_top_k", type=int, default=300)
    search_temporal.add_argument("--visual_weight", type=float, default=0.45)
    search_temporal.add_argument("--ocr_weight", type=float, default=0.35)
    search_temporal.add_argument("--subtitle_weight", type=float, default=0.20)
    search_temporal.add_argument("--visual_device", type=str, default="cpu")
    search_temporal.add_argument("--subtitle_device", type=str, default="cpu")
    search_temporal.add_argument("--output_json", type=Path, default=None)

    return parser


def _save_result_if_needed(payload, output_json: Path | None) -> None:
    if output_json is not None:
        save_json(payload, output_json)


def _build_search_engine(args) -> SearchEngine:
    registry = IndexRegistry()
    registry.load_faiss("visual", args.visual_index_dir)
    registry.load_faiss("subtitle", args.subtitle_index_dir)
    registry.load_bm25("ocr", args.ocr_index_dir)
    visual_encoder = OpenClipTextEncoder(args.clip_model_path, device=args.visual_device)
    subtitle_encoder = E5TextEncoder(args.subtitle_model_path, device=args.subtitle_device)
    return SearchEngine(
        registry=registry,
        visual_encoder=visual_encoder,
        subtitle_encoder=subtitle_encoder,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build-visual-index":
        index = VisualIndexBuilder(args.keyframe_embedding_dir).build()
        index.save(args.output_dir)
        print(f"Saved visual index to: {args.output_dir}")
        return

    if args.command == "build-subtitle-index":
        index = SubtitleIndexBuilder(args.subtitle_embedding_dir).build()
        index.save(args.output_dir)
        print(f"Saved subtitle index to: {args.output_dir}")
        return

    if args.command == "build-ocr-index":
        index = OCRIndexBuilder(args.ocr_dir).build()
        index.save(args.output_dir)
        print(f"Saved OCR index to: {args.output_dir}")
        return

    if args.command == "build-all":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        visual_index = VisualIndexBuilder(args.keyframe_embedding_dir).build()
        subtitle_index = SubtitleIndexBuilder(args.subtitle_embedding_dir).build()
        ocr_index = OCRIndexBuilder(args.ocr_dir).build()
        visual_index.save(args.output_dir / "visual_index")
        subtitle_index.save(args.output_dir / "subtitle_index")
        ocr_index.save(args.output_dir / "ocr_index")
        print(f"Saved all indices to: {args.output_dir}")
        return

    if args.command == "search-stage":
        engine = _build_search_engine(args)
        stage_query = StageQuery(
            visual=args.visual_query,
            subtitle=args.subtitle_query,
            ocr=args.ocr_query,
        )
        results = engine.stage_search(
            stage_query=stage_query,
            top_k=args.top_k,
            visual_top_k=args.visual_top_k,
            subtitle_top_k=args.subtitle_top_k,
            ocr_top_k=args.ocr_top_k,
            visual_weight=args.visual_weight,
            ocr_weight=args.ocr_weight,
            subtitle_weight=args.subtitle_weight,
        )
        payload = [item.to_dict() for item in results]
        _save_result_if_needed(payload, args.output_json)
        print(payload[:10])
        return

    if args.command == "search-temporal":
        engine = _build_search_engine(args)
        payload = load_json(args.stages_json)
        if not isinstance(payload, list):
            raise ValueError("stages_json must contain a list of stage objects")
        stage_queries = [StageQuery.from_dict(item) for item in payload]
        results = engine.temporal_search(
            stage_queries=stage_queries,
            top_k=args.top_k,
            per_stage_top_k=args.per_stage_top_k,
            visual_top_k=args.visual_top_k,
            subtitle_top_k=args.subtitle_top_k,
            ocr_top_k=args.ocr_top_k,
            visual_weight=args.visual_weight,
            ocr_weight=args.ocr_weight,
            subtitle_weight=args.subtitle_weight,
        )
        _save_result_if_needed(results, args.output_json)
        print(results[:5])
        return
