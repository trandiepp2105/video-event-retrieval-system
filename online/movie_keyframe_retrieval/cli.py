from __future__ import annotations

import argparse
from pathlib import Path

from .encoders import OpenClipTextEncoder
from .io_utils import load_json, save_json
from .meili_search import MeiliSearchService
from .query_analyzer import DEFAULT_SYSTEM_PROMPT, MovieQueryAnalyzer
from .registry import IndexRegistry
from .schemas import StageQuery
from .search import FAISSSearchEngine, SearchEngine


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
    search_stage.add_argument("--debug", action="store_true")
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
        for index_name, config in zip(index_names, faiss_configs):
            print(f"Saved visual index '{index_name}' to: {config['output_index_path']}")
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
        print(f"Indexed subtitle documents to Meilisearch index: {args.subtitle_index_name}")
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
        print(f"Indexed OCR documents to Meilisearch index: {args.ocr_index_name}")
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
            debug=args.debug,
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
        print(results[:5])
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
        analyzed = analyzer.analyze(args.raw_query, return_raw=False)
        if analyzed:
            stage_queries = analyzer.to_search_queries(analyzed)
        else:
            stage_queries = [{"text": args.raw_query}]
        if len(stage_queries) <= 1:
            stage_query = stage_queries[0] if stage_queries else {"text": args.raw_query}
            weights = engine._build_stage_weights(stage_query)
            results = engine.hybrid_search(
                text_query=stage_query.get("text"),
                ocr_query=stage_query.get("ocr"),
                subtitle_query=stage_query.get("subtitle"),
                k=args.top_k,
                weights=weights,
            )
            payload = {
                "query": args.raw_query,
                "analyzed": analyzed,
                "stage_queries": stage_queries,
                "mode": "single-stage",
                "results": results,
            }
        else:
            results = engine.temporal_search(
                queries=stage_queries,
                k=args.top_k,
                initial_search_k=args.initial_search_k,
                frame_distance=args.frame_distance,
                window_size_frames=args.window_size_frames,
                lambda_skip=args.lambda_skip,
                min_stage_gap=args.min_stage_gap,
            )
            payload = {
                "query": args.raw_query,
                "analyzed": analyzed,
                "stage_queries": stage_queries,
                "mode": "multi-stage",
                "results": results,
            }
        _save_result_if_needed(payload, args.output_json)
        print(payload)
        return
