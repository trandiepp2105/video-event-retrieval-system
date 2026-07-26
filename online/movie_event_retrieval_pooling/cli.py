from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .common import load_json, save_json
from .config import BuildConfig, SearchConfig


def _select_query_items(payload: list[Any], *, start_index: int | None, end_index: int | None) -> list[tuple[int, dict[str, Any]]]:
    indexed_items: list[tuple[int, dict[str, Any]]] = [
        (idx, item) for idx, item in enumerate(payload) if isinstance(item, dict)
    ]
    if start_index is None and end_index is None:
        return indexed_items

    start = 0 if start_index is None else int(start_index)
    end = (len(payload) - 1) if end_index is None else int(end_index)
    if start < 0 or end < 0:
        raise ValueError("start_index and end_index must be >= 0")
    if end < start:
        raise ValueError("end_index must be >= start_index")
    return [(idx, item) for idx, item in indexed_items if start <= idx <= end]


def _format_timecode(seconds: float) -> str:
    total_seconds = max(float(seconds), 0.0)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = total_seconds % 60.0
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def _print_pooling_summary(result: dict) -> None:
    query = result.get("query", {})
    print("Raw query")
    print(f"  {query.get('raw_query', '')}")
    if query.get("translated_query"):
        print(f"  translated_query: {query.get('translated_query', '')}")
    if query.get("subtitle_query"):
        print(f"  subtitle_query: {query.get('subtitle_query', '')}")
    if query.get("ocr_query"):
        print(f"  ocr_query: {query.get('ocr_query', '')}")

    shot_temporal = result.get("shot_temporal", {})
    analysis = shot_temporal.get("query_analysis") or {}
    print("\nQuery analyzer")
    if analysis:
        if analysis.get("en_query"):
            print(f"  en_query: {analysis['en_query']}")
        stages = analysis.get("stages", [])
        print(f"  num_stages: {len(stages)}")
        for idx, stage in enumerate(stages, start=1):
            print(f"  {idx:>2}. visual={stage.get('visual', '')}")
            print(f"      ocr={stage.get('ocr', '')}")
            print(f"      subtitle={stage.get('subtitle', '')}")
    else:
        print("  <empty>")

    subtitle_hits = result.get("event_level", {}).get("subtitle_hits", [])
    has_subtitle_signal = bool(query.get("subtitle_query")) or any(
        bool(str(stage.get("subtitle", "")).strip())
        for stage in (analysis.get("stages", []) if isinstance(analysis, dict) else [])
    )
    if has_subtitle_signal:
        print(f"\nTop subtitle result: {len(subtitle_hits)}")
        for idx, item in enumerate(subtitle_hits[:10], start=1):
            text = str(item.get("text", "")).replace("\n", " ").strip()
            if len(text) > 120:
                text = text[:117] + "..."
            print(
                f"{idx:>2}. subtitle_id={item.get('item_id', '')} "
                f"score={float(item.get('score', 0.0)):.4f} "
                f"rank={int(item.get('rank', 0))} "
                f"text={text}"
            )

    ocr_hits = result.get("event_level", {}).get("ocr_hits", [])
    has_ocr_signal = bool(query.get("ocr_query")) or any(
        bool(str(stage.get("ocr", "")).strip())
        for stage in (analysis.get("stages", []) if isinstance(analysis, dict) else [])
    )
    if has_ocr_signal:
        print(f"\nTop ocr result: {len(ocr_hits)}")
        for idx, item in enumerate(ocr_hits[:10], start=1):
            text = str(item.get("text", "")).replace("\n", " ").strip()
            if len(text) > 120:
                text = text[:117] + "..."
            print(
                f"{idx:>2}. ocr_id={item.get('ocr_id', '')} "
                f"score={float(item.get('score', 0.0)):.4f} "
                f"rank={int(item.get('rank', 0))} "
                f"text={text}"
            )

    top_event_candidates = result.get("candidates", {}).get("top_events", [])
    print(f"\nTop event candidate: {len(top_event_candidates)}")
    for idx, item in enumerate(top_event_candidates[:10], start=1):
        print(
            f"{idx:>2}. video={item['video_id']} event={item['event_id']} "
            f"time={_format_timecode(item['start_time_sec'])}-{_format_timecode(item['end_time_sec'])} "
            f"score={float(item['score']):.4f} shots={len(item.get('shot_ids', []))}"
        )

    top_shots = result.get('shot_level', {}).get('top_shots', [])
    print(f"\nTop shot candidate: {len(top_shots)}")
    for idx, item in enumerate(top_shots[:10], start=1):
        print(
            f"{idx:>2}. video={item['video_id']} shot={item['shot_id']} event={item['event_id']} "
            f"time={_format_timecode(item['start_time_sec'])}-{_format_timecode(item['end_time_sec'])} "
            f"score={float(item['score']):.4f}"
        )

    top_chains = shot_temporal.get("top_chains", [])
    print(f"\nTop shot chain: {len(top_chains)}")
    for idx, chain in enumerate(top_chains[:5], start=1):
        print(
            f"{idx:>2}. video={chain['video_id']} score={float(chain['score']):.4f} "
            f"matched={chain['num_stages_matched']} skipped={chain['num_stages_skipped']}"
        )
        for item in chain.get("chain", []):
            print(
                f"    - stage={item['stage_index']} shot={item['shot_id']} event={item['event_id']} "
                f"time={_format_timecode(item['start_time_sec'])}-{_format_timecode(item['end_time_sec'])} "
                f"score={float(item['score']):.4f}"
            )


def _build_search_config_from_args(args, *, raw_query: str, output_json: Path | None = None) -> SearchConfig:
    return SearchConfig(
        store_dir=args.store_dir,
        query_json=None,
        raw_query=raw_query,
        translated_query=args.translated_query,
        subtitle_query=args.subtitle_query,
        ocr_query=args.ocr_query,
        clip_model_path=args.clip_model_path,
        clip_model_name=args.clip_model_name,
        caption_model_path=args.caption_model_path,
        subtitle_model_path=args.subtitle_model_path,
        subtitle_backend=args.subtitle_backend,
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
        enable_shot_temporal=args.enable_shot_temporal,
        temporal_query_model_path=args.temporal_query_model_path,
        temporal_query_device_map=args.temporal_query_device_map,
        temporal_query_torch_dtype=args.temporal_query_torch_dtype,
        temporal_query_max_new_tokens=args.temporal_query_max_new_tokens,
        stage_shot_top_k=args.stage_shot_top_k,
        temporal_chain_top_k=args.temporal_chain_top_k,
        stage_visual_weight=args.stage_visual_weight,
        stage_ocr_weight=args.stage_ocr_weight,
        stage_subtitle_weight=args.stage_subtitle_weight,
        temporal_window_shots=args.temporal_window_shots,
        temporal_group_gap_shots=args.temporal_group_gap_shots,
        temporal_min_stage_gap_shots=args.temporal_min_stage_gap_shots,
        temporal_lambda_skip=args.temporal_lambda_skip,
        meilisearch_url=args.meilisearch_url,
        meilisearch_index_name=args.meilisearch_index_name,
        subtitle_meilisearch_index_name=args.subtitle_meilisearch_index_name,
        meilisearch_api_key=args.meilisearch_api_key,
        output_json=output_json,
    )


def _summarize_pooling_batch_result(
    *,
    query_item: dict[str, Any],
    result: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    return {
        "video_id": str(query_item.get("video_id", "")),
        "query": str(query_item.get("query", "")),
        "start_time_sec": query_item.get("start_time_sec"),
        "end_time_sec": query_item.get("end_time_sec"),
        "query_analysis": result.get("shot_temporal", {}).get("query_analysis"),
        "top_event_candidates": result.get("candidates", {}).get("top_events", [])[:top_k],
        "top_shot_candidates": result.get("shot_level", {}).get("top_shots", [])[:top_k],
        "top_shot_chains": result.get("shot_temporal", {}).get("top_chains", [])[:top_k],
        "final_events": result.get("final_events", [])[:top_k],
    }


def _run_pooling_batch_search(args) -> dict[str, Any]:
    from .retrieval import PoolingMovieEventRetriever

    retriever = PoolingMovieEventRetriever(_load_metadata(args.store_dir), args.store_dir)
    payload = load_json(args.queries_json)
    if not isinstance(payload, list):
        raise ValueError("queries_json must contain a list of query objects")
    selected_items = _select_query_items(
        payload,
        start_index=args.start_index,
        end_index=args.end_index,
    )

    results: list[dict[str, Any]] = []
    total_selected = len(selected_items)
    for order, (index, item) in enumerate(selected_items, start=1):
        raw_query = str(item.get("query", "")).strip()
        if not raw_query:
            continue
        print(
            f"[{order}/{total_selected}] Searching pooling query "
            f"(query_index={index}, video_id={item.get('video_id', '')})"
        )
        result = retriever.search(_build_search_config_from_args(args, raw_query=raw_query, output_json=None))
        results.append(
            _summarize_pooling_batch_result(
                query_item=item,
                result=result,
                top_k=args.batch_result_top_k,
            )
        )
    output_payload = {
        "num_queries": len(results),
        "top_k": int(args.batch_result_top_k),
        "results": results,
    }
    save_json(output_payload, args.output_json)
    return output_payload


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
    build_all.add_argument("--subtitle_meilisearch_index_name", type=str, default=None)
    build_all.add_argument("--meilisearch_api_key", type=str, default=None)
    build_all.add_argument("--meilisearch_batch_size", type=int, default=10000)
    build_all.add_argument("--overwrite", action="store_true")

    build_ocr = subparsers.add_parser("build-ocr-index")
    build_ocr.add_argument("--ocr_dir", type=Path, required=True)
    build_ocr.add_argument("--output_dir", type=Path, required=True)
    build_ocr.add_argument("--meilisearch_url", type=str, required=True)
    build_ocr.add_argument("--meilisearch_index_name", type=str, required=True)
    build_ocr.add_argument("--meilisearch_api_key", type=str, default=None)
    build_ocr.add_argument("--meilisearch_batch_size", type=int, default=10000)

    build_subtitle = subparsers.add_parser("build-subtitle-index")
    build_subtitle.add_argument("--subtitle_dir", type=Path, required=True)
    build_subtitle.add_argument("--output_dir", type=Path, required=True)
    build_subtitle.add_argument("--meilisearch_url", type=str, required=True)
    build_subtitle.add_argument("--subtitle_meilisearch_index_name", type=str, required=True)
    build_subtitle.add_argument("--meilisearch_api_key", type=str, default=None)
    build_subtitle.add_argument("--meilisearch_batch_size", type=int, default=10000)

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
    search.add_argument("--subtitle_backend", type=str, default="meilisearch", choices=["meilisearch", "embedding"])
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
    search.add_argument("--enable_shot_temporal", action="store_true")
    search.add_argument("--temporal_query_model_path", type=str, default=None)
    search.add_argument("--temporal_query_device_map", type=str, default="auto")
    search.add_argument("--temporal_query_torch_dtype", type=str, default="auto")
    search.add_argument("--temporal_query_max_new_tokens", type=int, default=768)
    search.add_argument("--stage_shot_top_k", type=int, default=100)
    search.add_argument("--temporal_chain_top_k", type=int, default=30)
    search.add_argument("--stage_visual_weight", type=float, default=0.45)
    search.add_argument("--stage_ocr_weight", type=float, default=0.35)
    search.add_argument("--stage_subtitle_weight", type=float, default=0.20)
    search.add_argument("--temporal_window_shots", type=int, default=3)
    search.add_argument("--temporal_group_gap_shots", type=int, default=12)
    search.add_argument("--temporal_min_stage_gap_shots", type=int, default=1)
    search.add_argument("--temporal_lambda_skip", type=float, default=0.7)
    search.add_argument("--meilisearch_url", type=str, default=None)
    search.add_argument("--meilisearch_index_name", type=str, default=None)
    search.add_argument("--subtitle_meilisearch_index_name", type=str, default=None)
    search.add_argument("--meilisearch_api_key", type=str, default=None)
    search.add_argument("--output_json", type=Path, default=None)

    search_batch = subparsers.add_parser("search-batch")
    search_batch.add_argument("--store_dir", type=Path, required=True)
    search_batch.add_argument("--queries_json", type=Path, required=True)
    search_batch.add_argument("--start_index", type=int, default=None)
    search_batch.add_argument("--end_index", type=int, default=None)
    search_batch.add_argument("--translated_query", type=str, default="")
    search_batch.add_argument("--subtitle_query", type=str, default="")
    search_batch.add_argument("--ocr_query", type=str, default="")
    search_batch.add_argument("--clip_model_path", type=Path, default=None)
    search_batch.add_argument("--clip_model_name", type=str, default="ViT-H-14-quickgelu")
    search_batch.add_argument("--caption_model_path", type=str, default=None)
    search_batch.add_argument("--subtitle_model_path", type=str, default=None)
    search_batch.add_argument("--subtitle_backend", type=str, default="meilisearch", choices=["meilisearch", "embedding"])
    search_batch.add_argument("--event_top_k", type=int, default=200)
    search_batch.add_argument("--caption_top_k", type=int, default=200)
    search_batch.add_argument("--subtitle_top_k", type=int, default=200)
    search_batch.add_argument("--ocr_top_k", type=int, default=200)
    search_batch.add_argument("--candidate_event_top_k", type=int, default=100)
    search_batch.add_argument("--candidate_video_top_k", type=int, default=30)
    search_batch.add_argument("--shot_top_k", type=int, default=100)
    search_batch.add_argument("--final_top_k", type=int, default=30)
    search_batch.add_argument("--rrf_k", type=int, default=60)
    search_batch.add_argument("--event_weight", type=float, default=1.0)
    search_batch.add_argument("--caption_weight", type=float, default=0.8)
    search_batch.add_argument("--subtitle_weight", type=float, default=0.6)
    search_batch.add_argument("--ocr_weight", type=float, default=0.4)
    search_batch.add_argument("--shot_weight", type=float, default=0.8)
    search_batch.add_argument("--parent_event_weight", type=float, default=0.2)
    search_batch.add_argument("--visual_device", type=str, default="cpu")
    search_batch.add_argument("--caption_device", type=str, default="cpu")
    search_batch.add_argument("--subtitle_device", type=str, default="cpu")
    search_batch.add_argument("--enable_shot_temporal", action="store_true")
    search_batch.add_argument("--temporal_query_model_path", type=str, default=None)
    search_batch.add_argument("--temporal_query_device_map", type=str, default="auto")
    search_batch.add_argument("--temporal_query_torch_dtype", type=str, default="auto")
    search_batch.add_argument("--temporal_query_max_new_tokens", type=int, default=768)
    search_batch.add_argument("--stage_shot_top_k", type=int, default=100)
    search_batch.add_argument("--temporal_chain_top_k", type=int, default=100)
    search_batch.add_argument("--stage_visual_weight", type=float, default=0.45)
    search_batch.add_argument("--stage_ocr_weight", type=float, default=0.35)
    search_batch.add_argument("--stage_subtitle_weight", type=float, default=0.20)
    search_batch.add_argument("--temporal_window_shots", type=int, default=3)
    search_batch.add_argument("--temporal_group_gap_shots", type=int, default=12)
    search_batch.add_argument("--temporal_min_stage_gap_shots", type=int, default=1)
    search_batch.add_argument("--temporal_lambda_skip", type=float, default=0.7)
    search_batch.add_argument("--meilisearch_url", type=str, default=None)
    search_batch.add_argument("--meilisearch_index_name", type=str, default=None)
    search_batch.add_argument("--subtitle_meilisearch_index_name", type=str, default=None)
    search_batch.add_argument("--meilisearch_api_key", type=str, default=None)
    search_batch.add_argument("--batch_result_top_k", type=int, default=100)
    search_batch.add_argument("--output_json", type=Path, required=True)
    return parser


def _load_metadata(store_dir: Path):
    from .metadata import MetadataRepository
    from .schemas import (
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
        from .build import RetrievalStoreBuilder

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
                subtitle_meilisearch_index_name=args.subtitle_meilisearch_index_name,
                meilisearch_api_key=args.meilisearch_api_key,
                meilisearch_batch_size=args.meilisearch_batch_size,
                overwrite=args.overwrite,
            )
        )
        print(
            "Build complete\n"
            f"  output_dir: {args.output_dir}\n"
            f"  videos: {manifest.get('num_videos', 0)}\n"
            f"  events: {manifest.get('num_events', 0)}\n"
            f"  shots: {manifest.get('num_shots', 0)}\n"
            f"  subtitles: {manifest.get('num_subtitles', 0)}\n"
            f"  ocr_items: {manifest.get('num_ocr_items', 0)}"
        )
        return

    if args.command == "build-ocr-index":
        from .build import build_ocr_index_only

        result = build_ocr_index_only(
            output_dir=args.output_dir,
            ocr_dir=args.ocr_dir,
            meilisearch_url=args.meilisearch_url,
            meilisearch_api_key=args.meilisearch_api_key,
            meilisearch_index_name=args.meilisearch_index_name,
            batch_size=args.meilisearch_batch_size,
            wait_each_batch=False,
        )
        print(
            "OCR index complete\n"
            f"  output_dir: {result['output_dir']}\n"
            f"  index_name: {result['index_name']}\n"
            f"  documents: {result['documents']}"
        )
        return

    if args.command == "build-subtitle-index":
        from .build import build_subtitle_index_only

        result = build_subtitle_index_only(
            output_dir=args.output_dir,
            subtitle_dir=args.subtitle_dir,
            meilisearch_url=args.meilisearch_url,
            meilisearch_api_key=args.meilisearch_api_key,
            subtitle_index_name=args.subtitle_meilisearch_index_name,
            batch_size=args.meilisearch_batch_size,
            wait_each_batch=False,
        )
        print(
            "Subtitle index complete\n"
            f"  output_dir: {result['output_dir']}\n"
            f"  index_name: {result['index_name']}\n"
            f"  documents: {result['documents']}"
        )
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
                subtitle_backend=args.subtitle_backend,
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
                enable_shot_temporal=args.enable_shot_temporal,
                temporal_query_model_path=args.temporal_query_model_path,
                temporal_query_device_map=args.temporal_query_device_map,
                temporal_query_torch_dtype=args.temporal_query_torch_dtype,
                temporal_query_max_new_tokens=args.temporal_query_max_new_tokens,
                stage_shot_top_k=args.stage_shot_top_k,
                temporal_chain_top_k=args.temporal_chain_top_k,
                stage_visual_weight=args.stage_visual_weight,
                stage_ocr_weight=args.stage_ocr_weight,
                stage_subtitle_weight=args.stage_subtitle_weight,
                temporal_window_shots=args.temporal_window_shots,
                temporal_group_gap_shots=args.temporal_group_gap_shots,
                temporal_min_stage_gap_shots=args.temporal_min_stage_gap_shots,
                temporal_lambda_skip=args.temporal_lambda_skip,
                meilisearch_url=args.meilisearch_url,
                meilisearch_index_name=args.meilisearch_index_name,
                subtitle_meilisearch_index_name=args.subtitle_meilisearch_index_name,
                meilisearch_api_key=args.meilisearch_api_key,
                output_json=args.output_json,
            )
        )
        _print_pooling_summary(result)
        return

    if args.command == "search-batch":
        output_payload = _run_pooling_batch_search(args)
        print(
            "Batch search complete\n"
            f"  queries: {output_payload['num_queries']}\n"
            f"  top_k: {output_payload['top_k']}\n"
            f"  output_json: {args.output_json}"
        )
