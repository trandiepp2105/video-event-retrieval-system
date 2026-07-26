from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .encoders import OpenClipTextEncoder
from .io_utils import load_json, save_json
from .meili_search import MeiliSearchService
from .query_analyzer import DEFAULT_SYSTEM_PROMPT, MovieQueryAnalyzer
from .registry import IndexRegistry
from .schemas import StageQuery
from .search import FAISSSearchEngine, SearchEngine


class KeyframeQuerySearchSession:
    def __init__(self, engine: SearchEngine, analyzer: MovieQueryAnalyzer | None = None) -> None:
        self.engine = engine
        self.analyzer = analyzer

    def search_raw_query(self, args, raw_query: str) -> dict[str, Any]:
        if self.analyzer is None:
            raise ValueError("MovieQueryAnalyzer has not been initialized for raw query search.")
        return _run_keyframe_query_search(self.engine, self.analyzer, args, raw_query)


def _format_timecode_from_frame(frame_idx: int, fps: float = 25.0) -> str:
    total_seconds = max(float(frame_idx) / float(fps), 0.0)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60.0
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def _print_keyframe_stage_results(results: list[dict], limit: int = 10) -> None:
    if not results:
        print("No results.")
        return
    print("Top results")
    for idx, item in enumerate(results[:limit], start=1):
        print(
            f"{idx:>2}. video={item['video_id']} "
            f"frame={item['frame_idx']} "
            f"time={_format_timecode_from_frame(int(item['frame_idx']))} "
            f"score={float(item['fused_score']):.4f} "
            f"(visual={float(item.get('visual_score', 0.0)):.4f}, "
            f"subtitle={float(item.get('subtitle_score', 0.0)):.4f}, "
            f"ocr={float(item.get('ocr_score', 0.0)):.4f})"
        )


def _print_keyframe_temporal_results(results: list, limit: int = 5) -> None:
    if not results:
        print("No temporal chains.")
        return
    print("Top temporal chains")
    for chain_idx, chain in enumerate(results[:limit], start=1):
        total_score = float(chain[0][1][1]) if chain else 0.0
        print(f"{chain_idx:>2}. total_score={total_score:.4f} | steps={len(chain)}")
        for step_idx, ((video_id, frame_idx), (_stage_scores, _chain_score, matched_stage_idx)) in enumerate(chain, start=1):
            print(
                f"    step={step_idx} stage={matched_stage_idx + 1} "
                f"video={video_id} frame={frame_idx} time={_format_timecode_from_frame(int(frame_idx))}"
            )


def _truncate_text(text: str, limit: int = 120) -> str:
    value = str(text).replace("\n", " ").strip()
    if len(value) > limit:
        return value[: limit - 3] + "..."
    return value


def _print_keyframe_text_hits(title: str, hits: list[dict], *, id_key: str, limit: int = 10) -> None:
    print(f"{title}: {len(hits)}")
    for idx, item in enumerate(hits[:limit], start=1):
        print(
            f"{idx:>2}. {id_key}={item.get(id_key, '')} "
            f"score={float(item.get('_rankingScore', item.get('score', 0.0))):.4f} "
            f"text={_truncate_text(item.get('text', ''))}"
        )


def _print_keyframe_aux_hits(engine: SearchEngine, *, subtitle_query: str = "", ocr_query: str = "") -> None:
    if subtitle_query.strip():
        subtitle_hits = engine.ocr_engine.search_subtitle(subtitle_query.strip(), 10)
        print()
        _print_keyframe_text_hits("Top subtitle result", subtitle_hits, id_key="subtitle_id", limit=10)
    if ocr_query.strip():
        ocr_hits = engine.ocr_engine.search_ocr(ocr_query.strip(), 10)
        print()
        _print_keyframe_text_hits("Top ocr result", ocr_hits, id_key="ocr_id", limit=10)


def _print_keyframe_query_payload(payload: dict) -> None:
    print(f"Mode: {payload.get('mode', '')}")
    analyzed = payload.get("analyzed") or {}
    if analyzed.get("en_query"):
        print(f"English query: {analyzed.get('en_query', '')}")
    stage_queries = payload.get("stage_queries") or []
    print(f"Stages: {len(stage_queries)}")
    for idx, stage in enumerate(stage_queries, start=1):
        print(f"  Stage {idx}: {stage}")
    subtitle_queries = [str(stage.get("subtitle", "")).strip() for stage in stage_queries if str(stage.get("subtitle", "")).strip()]
    ocr_queries = [str(stage.get("ocr", "")).strip() for stage in stage_queries if str(stage.get("ocr", "")).strip()]
    subtitle_hit_groups = payload.get("subtitle_hits_by_stage") or []
    ocr_hit_groups = payload.get("ocr_hits_by_stage") or []
    if subtitle_queries:
        print("\nTop subtitle result")
        if subtitle_hit_groups:
            for idx, (query_text, hits) in enumerate(zip(subtitle_queries, subtitle_hit_groups), start=1):
                print(f"  Stage {idx} query: {query_text}")
                _print_keyframe_text_hits("  Hits", hits, id_key="subtitle_id", limit=5)
        else:
            print("  <empty>")
    if ocr_queries:
        print("\nTop ocr result")
        if ocr_hit_groups:
            for idx, (query_text, hits) in enumerate(zip(ocr_queries, ocr_hit_groups), start=1):
                print(f"  Stage {idx} query: {query_text}")
                _print_keyframe_text_hits("  Hits", hits, id_key="ocr_id", limit=5)
        else:
            print("  <empty>")
    results = payload.get("results") or []
    if payload.get("mode") == "multi-stage":
        _print_keyframe_temporal_results(results, limit=5)
    else:
        formatted = [
            {
                "video_id": video_id,
                "frame_idx": frame_idx,
                "fused_score": score,
                "visual_score": score,
                "subtitle_score": 0.0,
                "ocr_score": 0.0,
            }
            for (video_id, frame_idx), score in results[:10]
        ]
        _print_keyframe_stage_results(formatted, limit=10)


def _run_keyframe_query_search(engine: SearchEngine, analyzer: MovieQueryAnalyzer, args, raw_query: str) -> dict[str, Any]:
    analyzed = analyzer.analyze(raw_query, return_raw=False)
    if analyzed:
        stage_queries = analyzer.to_search_queries(analyzed)
    else:
        stage_queries = [{"visual": raw_query, "ocr": "", "subtitle": ""}]
    subtitle_hits_by_stage = []
    ocr_hits_by_stage = []
    for stage_query in stage_queries:
        subtitle_text = str(stage_query.get("subtitle", "")).strip()
        ocr_text = str(stage_query.get("ocr", "")).strip()
        if subtitle_text:
            subtitle_hits_by_stage.append(engine.ocr_engine.search_subtitle(subtitle_text, 10))
        if ocr_text:
            ocr_hits_by_stage.append(engine.ocr_engine.search_ocr(ocr_text, 10))
    if len(stage_queries) <= 1:
        stage_query = stage_queries[0] if stage_queries else {"visual": raw_query, "ocr": "", "subtitle": ""}
        retrieval_stage_query = {
            "text": stage_query.get("visual"),
            "ocr": stage_query.get("ocr"),
            "subtitle": stage_query.get("subtitle"),
        }
        weights = engine._build_stage_weights(retrieval_stage_query)
        results = engine.hybrid_search(
            text_query=retrieval_stage_query.get("text"),
            ocr_query=retrieval_stage_query.get("ocr"),
            subtitle_query=retrieval_stage_query.get("subtitle"),
            k=args.top_k,
            weights=weights,
        )
        return {
            "query": raw_query,
            "analyzed": analyzed,
            "stage_queries": stage_queries,
            "subtitle_hits_by_stage": subtitle_hits_by_stage,
            "ocr_hits_by_stage": ocr_hits_by_stage,
            "mode": "single-stage",
            "results": results,
        }
    results = engine.temporal_search(
        queries=[
            {
                "text": stage_query.get("visual", ""),
                "ocr": stage_query.get("ocr", ""),
                "subtitle": stage_query.get("subtitle", ""),
            }
            for stage_query in stage_queries
        ],
        k=args.top_k,
        initial_search_k=args.initial_search_k,
        frame_distance=args.frame_distance,
        window_size_frames=args.window_size_frames,
        lambda_skip=args.lambda_skip,
        min_stage_gap=args.min_stage_gap,
    )
    return {
        "query": raw_query,
        "analyzed": analyzed,
        "stage_queries": stage_queries,
        "subtitle_hits_by_stage": subtitle_hits_by_stage,
        "ocr_hits_by_stage": ocr_hits_by_stage,
        "mode": "multi-stage",
        "results": results,
    }


def _summarize_keyframe_batch_result(
    *,
    query_item: dict[str, Any],
    payload: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    return {
        "video_id": str(query_item.get("video_id", "")),
        "query": str(query_item.get("query", "")),
        "start_time_sec": query_item.get("start_time_sec"),
        "end_time_sec": query_item.get("end_time_sec"),
        "mode": payload.get("mode", ""),
        "analyzed": payload.get("analyzed"),
        "stage_queries": payload.get("stage_queries", []),
        "results": (payload.get("results") or [])[:top_k],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Movie event retrieval system CLI (current mode: keyframe retrieval)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_visual = subparsers.add_parser("build-visual-index")
    build_visual.add_argument("--keyframe_embedding_dir", type=Path, nargs="+", required=True)
    build_visual.add_argument("--output_dir", type=Path, required=True)
    build_visual.add_argument("--visual_index_names", type=str, nargs="*", default=None)

    build_subtitle = subparsers.add_parser("build-subtitle-index")
    build_subtitle.add_argument("--subtitle_dir", type=Path, required=True)
    build_subtitle.add_argument("--meilisearch_url", type=str, required=True)
    build_subtitle.add_argument("--meilisearch_api_key", type=str, required=True)
    build_subtitle.add_argument("--subtitle_index_name", type=str, required=True)

    build_ocr = subparsers.add_parser("build-ocr-index")
    build_ocr.add_argument("--ocr_dir", type=Path, required=True)
    build_ocr.add_argument("--meilisearch_url", type=str, required=True)
    build_ocr.add_argument("--meilisearch_api_key", type=str, required=True)
    build_ocr.add_argument("--ocr_index_name", type=str, required=True)

    build_all = subparsers.add_parser("build-all")
    build_all.add_argument("--keyframe_embedding_dir", type=Path, nargs="+", required=True)
    build_all.add_argument("--subtitle_dir", type=Path, required=True)
    build_all.add_argument("--ocr_dir", type=Path, required=True)
    build_all.add_argument("--output_dir", type=Path, required=True)
    build_all.add_argument("--visual_index_names", type=str, nargs="*", default=None)
    build_all.add_argument("--meilisearch_url", type=str, required=True)
    build_all.add_argument("--meilisearch_api_key", type=str, required=True)
    build_all.add_argument("--ocr_index_name", type=str, default="movie_keyframe_ocr")
    build_all.add_argument("--subtitle_index_name", type=str, default="movie_keyframe_subtitle")

    search_stage = subparsers.add_parser("search-stage")
    search_stage.add_argument("--visual_index_dir", type=Path, nargs="+", required=True)
    search_stage.add_argument("--clip_model_path", type=Path, nargs="+", required=True)
    search_stage.add_argument("--clip_model_name", type=str, nargs="*", default=None)
    search_stage.add_argument("--visual_index_names", type=str, nargs="*", default=None)
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
    search_stage.add_argument("--visual_device", type=str, nargs="*", default=None)
    search_stage.add_argument("--meilisearch_url", type=str, required=True)
    search_stage.add_argument("--meilisearch_api_key", type=str, required=True)
    search_stage.add_argument("--ocr_index_name", type=str, required=True)
    search_stage.add_argument("--subtitle_index_name", type=str, required=True)
    search_stage.add_argument("--output_json", type=Path, default=None)

    search_temporal = subparsers.add_parser("search-temporal")
    search_temporal.add_argument("--visual_index_dir", type=Path, nargs="+", required=True)
    search_temporal.add_argument("--clip_model_path", type=Path, nargs="+", required=True)
    search_temporal.add_argument("--clip_model_name", type=str, nargs="*", default=None)
    search_temporal.add_argument("--visual_index_names", type=str, nargs="*", default=None)
    search_temporal.add_argument("--stages_json", type=Path, required=True)
    search_temporal.add_argument("--top_k", type=int, default=20)
    search_temporal.add_argument("--per_stage_top_k", type=int, default=100)
    search_temporal.add_argument("--visual_top_k", type=int, default=300)
    search_temporal.add_argument("--subtitle_top_k", type=int, default=300)
    search_temporal.add_argument("--ocr_top_k", type=int, default=300)
    search_temporal.add_argument("--visual_weight", type=float, default=0.45)
    search_temporal.add_argument("--ocr_weight", type=float, default=0.35)
    search_temporal.add_argument("--subtitle_weight", type=float, default=0.20)
    search_temporal.add_argument("--visual_device", type=str, nargs="*", default=None)
    search_temporal.add_argument("--meilisearch_url", type=str, required=True)
    search_temporal.add_argument("--meilisearch_api_key", type=str, required=True)
    search_temporal.add_argument("--ocr_index_name", type=str, required=True)
    search_temporal.add_argument("--subtitle_index_name", type=str, required=True)
    search_temporal.add_argument("--frame_distance", type=int, default=1500)
    search_temporal.add_argument("--window_size_frames", type=int, default=100)
    search_temporal.add_argument("--lambda_skip", type=float, default=0.7)
    search_temporal.add_argument("--min_stage_gap", type=int, default=30)
    search_temporal.add_argument("--output_json", type=Path, default=None)

    search_query = subparsers.add_parser("search-query")
    search_query.add_argument("--visual_index_dir", type=Path, nargs="+", required=True)
    search_query.add_argument("--clip_model_path", type=Path, nargs="+", required=True)
    search_query.add_argument("--clip_model_name", type=str, nargs="*", default=None)
    search_query.add_argument("--visual_index_names", type=str, nargs="*", default=None)
    search_query.add_argument("--raw_query", type=str, required=True)
    search_query.add_argument("--llm_model_path", type=str, required=True)
    search_query.add_argument("--llm_system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    search_query.add_argument("--llm_device_map", type=str, default="auto")
    search_query.add_argument("--llm_torch_dtype", type=str, default="auto")
    search_query.add_argument("--llm_max_new_tokens", type=int, default=768)
    search_query.add_argument("--llm_load_in_4bit", action="store_true")
    search_query.add_argument("--llm_load_in_8bit", action="store_true")
    search_query.add_argument("--llm_bnb_8bit_cpu_offload", action="store_true")
    search_query.add_argument("--visual_device", type=str, nargs="*", default=None)
    search_query.add_argument("--meilisearch_url", type=str, required=True)
    search_query.add_argument("--meilisearch_api_key", type=str, required=True)
    search_query.add_argument("--ocr_index_name", type=str, required=True)
    search_query.add_argument("--subtitle_index_name", type=str, required=True)
    search_query.add_argument("--top_k", type=int, default=100)
    search_query.add_argument("--initial_search_k", type=int, default=2048)
    search_query.add_argument("--frame_distance", type=int, default=1500)
    search_query.add_argument("--window_size_frames", type=int, default=100)
    search_query.add_argument("--lambda_skip", type=float, default=0.7)
    search_query.add_argument("--min_stage_gap", type=int, default=30)
    search_query.add_argument("--output_json", type=Path, default=None)

    search_batch = subparsers.add_parser("search-batch")
    search_batch.add_argument("--visual_index_dir", type=Path, nargs="+", required=True)
    search_batch.add_argument("--clip_model_path", type=Path, nargs="+", required=True)
    search_batch.add_argument("--clip_model_name", type=str, nargs="*", default=None)
    search_batch.add_argument("--visual_index_names", type=str, nargs="*", default=None)
    search_batch.add_argument("--queries_json", type=Path, required=True)
    search_batch.add_argument("--llm_model_path", type=str, required=True)
    search_batch.add_argument("--llm_system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    search_batch.add_argument("--llm_device_map", type=str, default="auto")
    search_batch.add_argument("--llm_torch_dtype", type=str, default="auto")
    search_batch.add_argument("--llm_max_new_tokens", type=int, default=768)
    search_batch.add_argument("--llm_load_in_4bit", action="store_true")
    search_batch.add_argument("--llm_load_in_8bit", action="store_true")
    search_batch.add_argument("--llm_bnb_8bit_cpu_offload", action="store_true")
    search_batch.add_argument("--visual_device", type=str, nargs="*", default=None)
    search_batch.add_argument("--meilisearch_url", type=str, required=True)
    search_batch.add_argument("--meilisearch_api_key", type=str, required=True)
    search_batch.add_argument("--ocr_index_name", type=str, required=True)
    search_batch.add_argument("--subtitle_index_name", type=str, required=True)
    search_batch.add_argument("--top_k", type=int, default=100)
    search_batch.add_argument("--initial_search_k", type=int, default=2048)
    search_batch.add_argument("--frame_distance", type=int, default=1500)
    search_batch.add_argument("--window_size_frames", type=int, default=100)
    search_batch.add_argument("--lambda_skip", type=float, default=0.7)
    search_batch.add_argument("--min_stage_gap", type=int, default=30)
    search_batch.add_argument("--batch_result_top_k", type=int, default=100)
    search_batch.add_argument("--output_json", type=Path, required=True)

    return parser


def _save_result_if_needed(payload, output_json: Path | None) -> None:
    if output_json is not None:
        save_json(payload, output_json)


def _resolve_visual_names(args) -> list[str]:
    dirs = [Path(item) for item in args.visual_index_dir] if hasattr(args, "visual_index_dir") else [Path(item) for item in args.keyframe_embedding_dir]
    provided = getattr(args, "visual_index_names", None)
    if provided:
        if len(provided) != len(dirs):
            raise ValueError("So luong visual_index_names phai khop voi so luong visual index/embedding dirs.")
        return [str(item) for item in provided]
    if len(dirs) == 1:
        return ["visual"]
    return [path.stem.replace("-", "_") for path in dirs]


def _resolve_clip_model_names(args, count: int) -> list[str]:
    provided = getattr(args, "clip_model_name", None)
    if provided:
        if len(provided) == 1 and count > 1:
            return [str(provided[0])] * count
        if len(provided) != count:
            raise ValueError("So luong clip_model_name phai khop voi so luong visual indices.")
        return [str(item) for item in provided]
    return ["ViT-H-14-quickgelu"] * count


def _resolve_visual_devices(args, count: int) -> list[str]:
    provided = getattr(args, "visual_device", None)
    if not provided:
        return ["cpu"] * count
    if len(provided) == 1 and count > 1:
        return [str(provided[0])] * count
    if len(provided) != count:
        raise ValueError("So luong visual_device phai khop voi so luong visual indices.")
    return [str(item) for item in provided]


def _build_search_engine(args) -> SearchEngine:
    registry = IndexRegistry()
    visual_index_dirs = [Path(item) for item in args.visual_index_dir]
    visual_index_names = _resolve_visual_names(args)
    clip_model_paths = [Path(item) for item in args.clip_model_path]
    if len(clip_model_paths) == 1 and len(visual_index_dirs) > 1:
        clip_model_paths = clip_model_paths * len(visual_index_dirs)
    if len(clip_model_paths) != len(visual_index_dirs):
        raise ValueError("So luong clip_model_path phai khop voi so luong visual_index_dir.")
    clip_model_names = _resolve_clip_model_names(args, len(visual_index_dirs))
    visual_devices = _resolve_visual_devices(args, len(visual_index_dirs))

    visual_configs = []
    for index_name, index_dir, model_path, model_name, device in zip(
        visual_index_names,
        visual_index_dirs,
        clip_model_paths,
        clip_model_names,
        visual_devices,
    ):
        visual_index = registry.load_faiss(index_name, index_dir)
        visual_encoder = OpenClipTextEncoder(model_path, model_name=model_name, device=device)
        visual_configs.append(
            {
                "model_name": index_name,
                "index": visual_index,
                "embedder": visual_encoder,
            }
        )
    vector_engine = FAISSSearchEngine(visual_configs)
    meili_service = MeiliSearchService(
        url=args.meilisearch_url,
        api_key=args.meilisearch_api_key,
        ocr_index_name=args.ocr_index_name,
        subtitle_index_name=args.subtitle_index_name,
    )
    return SearchEngine(
        vector_engine=vector_engine,
        ocr_engine=meili_service,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build-visual-index":
        input_dirs = [Path(item) for item in args.keyframe_embedding_dir]
        index_names = _resolve_visual_names(args)
        faiss_configs = []
        for input_dir, index_name in zip(input_dirs, index_names):
            save_dir = args.output_dir / f"{index_name}_index" if len(input_dirs) > 1 else args.output_dir
            faiss_configs.append(
                {
                    "model_name": index_name,
                    "embedding_path": input_dir,
                    "output_index_path": save_dir,
                }
            )
        engine = FAISSSearchEngine(faiss_configs)
        engine.build_all_indexes()
        engine.save_all_indexes()
        print(f"Saved {len(index_names)} visual index(es) to: {args.output_dir}")
        return

    if args.command == "build-subtitle-index":
        service = MeiliSearchService(
            url=args.meilisearch_url,
            api_key=args.meilisearch_api_key,
            ocr_index_name="unused_ocr",
            subtitle_index_name=args.subtitle_index_name,
        )
        service.create_indices()
        service.index_subtitle_dataset(args.subtitle_dir)
        print(f"Subtitle index ready: {args.subtitle_index_name}")
        return

    if args.command == "build-ocr-index":
        service = MeiliSearchService(
            url=args.meilisearch_url,
            api_key=args.meilisearch_api_key,
            ocr_index_name=args.ocr_index_name,
            subtitle_index_name="unused_subtitle",
        )
        service.create_indices()
        service.index_ocr_dataset(args.ocr_dir)
        print(f"OCR index ready: {args.ocr_index_name}")
        return

    if args.command == "build-all":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        input_dirs = [Path(item) for item in args.keyframe_embedding_dir]
        index_names = _resolve_visual_names(args)
        faiss_configs = []
        for input_dir, index_name in zip(input_dirs, index_names):
            save_dir = args.output_dir / f"{index_name}_index" if len(input_dirs) > 1 else args.output_dir / "visual_index"
            faiss_configs.append(
                {
                    "model_name": index_name,
                    "embedding_path": input_dir,
                    "output_index_path": save_dir,
                }
            )
        engine = FAISSSearchEngine(faiss_configs)
        engine.build_all_indexes()
        engine.save_all_indexes()
        service = MeiliSearchService(
            url=args.meilisearch_url,
            api_key=args.meilisearch_api_key,
            ocr_index_name=args.ocr_index_name,
            subtitle_index_name=args.subtitle_index_name,
        )
        service.create_indices()
        service.index_ocr_dataset(args.ocr_dir)
        service.index_subtitle_dataset(args.subtitle_dir)
        print(f"All indices ready at: {args.output_dir}")
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
        _print_keyframe_aux_hits(
            engine,
            subtitle_query=args.subtitle_query,
            ocr_query=args.ocr_query,
        )
        if args.subtitle_query or args.ocr_query:
            print()
        _print_keyframe_stage_results(payload, limit=10)
        return

    if args.command == "search-temporal":
        engine = _build_search_engine(args)
        payload = load_json(args.stages_json)
        if not isinstance(payload, list):
            raise ValueError("stages_json must contain a list of stage objects")
        stage_queries = [StageQuery.from_dict(item) for item in payload]
        results = engine.temporal_search(
            top_k=args.top_k,
            queries=[{"text": item.visual, "subtitle": item.subtitle, "ocr": item.ocr} for item in stage_queries],
            initial_search_k=args.per_stage_top_k,
            weights={"text": args.visual_weight, "ocr": args.ocr_weight, "subtitle": args.subtitle_weight},
            frame_distance=args.frame_distance,
            window_size_frames=args.window_size_frames,
            lambda_skip=args.lambda_skip,
            min_stage_gap=args.min_stage_gap,
        )
        _save_result_if_needed(results, args.output_json)
        _print_keyframe_temporal_results(results, limit=5)
        return

    if args.command == "search-query":
        engine = _build_search_engine(args)
        analyzer = MovieQueryAnalyzer(
            model_id=args.llm_model_path,
            system_prompt=args.llm_system_prompt,
            device_map=args.llm_device_map,
            torch_dtype=args.llm_torch_dtype,
            max_new_tokens=args.llm_max_new_tokens,
            load_in_4bit=args.llm_load_in_4bit,
            load_in_8bit=args.llm_load_in_8bit,
            bnb_8bit_cpu_offload=args.llm_bnb_8bit_cpu_offload,
        )
        session = KeyframeQuerySearchSession(engine=engine, analyzer=analyzer)
        payload = session.search_raw_query(args, args.raw_query)
        _save_result_if_needed(payload, args.output_json)
        _print_keyframe_query_payload(payload)
        return

    if args.command == "search-batch":
        engine = _build_search_engine(args)
        analyzer = MovieQueryAnalyzer(
            model_id=args.llm_model_path,
            system_prompt=args.llm_system_prompt,
            device_map=args.llm_device_map,
            torch_dtype=args.llm_torch_dtype,
            max_new_tokens=args.llm_max_new_tokens,
            load_in_4bit=args.llm_load_in_4bit,
            load_in_8bit=args.llm_load_in_8bit,
            bnb_8bit_cpu_offload=args.llm_bnb_8bit_cpu_offload,
        )
        session = KeyframeQuerySearchSession(engine=engine, analyzer=analyzer)
        raw_payload = load_json(args.queries_json)
        if not isinstance(raw_payload, list):
            raise ValueError("queries_json must contain a list of query objects")
        results = []
        for index, item in enumerate(raw_payload):
            if not isinstance(item, dict):
                continue
            raw_query = str(item.get("query", "")).strip()
            if not raw_query:
                continue
            print(f"[{index + 1}/{len(raw_payload)}] Searching keyframe query for video_id={item.get('video_id', '')}")
            payload = session.search_raw_query(args, raw_query)
            results.append(
                _summarize_keyframe_batch_result(
                    query_item=item,
                    payload=payload,
                    top_k=args.batch_result_top_k,
                )
            )
        output_payload = {
            "num_queries": len(results),
            "top_k": int(args.batch_result_top_k),
            "results": results,
        }
        save_json(output_payload, args.output_json)
        print(
            "Batch search complete\n"
            f"  queries: {output_payload['num_queries']}\n"
            f"  top_k: {output_payload['top_k']}\n"
            f"  output_json: {args.output_json}"
        )
        return
