from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from online.movie_event_retrieval_pooling.api import create_app
from online.movie_event_retrieval_pooling.config import SearchConfig


def add_retrieval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store_dir", type=Path, required=True)
    parser.add_argument("--clip_model_path", type=Path, required=True)
    parser.add_argument("--clip_model_name", type=str, default="ViT-H-14-quickgelu")
    parser.add_argument("--caption_model_path", type=str, required=True)
    parser.add_argument("--subtitle_model_path", type=str, default=None)
    parser.add_argument("--subtitle_backend", choices=["meilisearch", "embedding"], default="meilisearch")
    parser.add_argument("--visual_device", type=str, default="cpu")
    parser.add_argument("--caption_device", type=str, default="cpu")
    parser.add_argument("--subtitle_device", type=str, default="cpu")

    parser.add_argument("--event_top_k", type=int, default=200)
    parser.add_argument("--caption_top_k", type=int, default=200)
    parser.add_argument("--subtitle_top_k", type=int, default=200)
    parser.add_argument("--ocr_top_k", type=int, default=200)
    parser.add_argument("--candidate_event_top_k", type=int, default=100)
    parser.add_argument("--candidate_video_top_k", type=int, default=30)
    parser.add_argument("--shot_top_k", type=int, default=100)
    parser.add_argument("--final_top_k", type=int, default=30)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--event_weight", type=float, default=1.0)
    parser.add_argument("--caption_weight", type=float, default=0.8)
    parser.add_argument("--subtitle_weight", type=float, default=0.6)
    parser.add_argument("--ocr_weight", type=float, default=0.4)
    parser.add_argument("--shot_weight", type=float, default=0.8)
    parser.add_argument("--parent_event_weight", type=float, default=0.2)

    parser.add_argument("--enable_shot_temporal", action="store_true")
    parser.add_argument("--temporal_query_model_path", type=str, default=None)
    parser.add_argument("--temporal_query_device_map", type=str, default="auto")
    parser.add_argument("--temporal_query_torch_dtype", type=str, default="auto")
    parser.add_argument("--temporal_query_max_new_tokens", type=int, default=768)
    parser.add_argument("--stage_shot_top_k", type=int, default=100)
    parser.add_argument("--temporal_chain_top_k", type=int, default=30)
    parser.add_argument("--stage_visual_weight", type=float, default=0.45)
    parser.add_argument("--stage_ocr_weight", type=float, default=0.35)
    parser.add_argument("--stage_subtitle_weight", type=float, default=0.20)
    parser.add_argument("--temporal_window_shots", type=int, default=3)
    parser.add_argument("--temporal_group_gap_shots", type=int, default=12)
    parser.add_argument("--temporal_min_stage_gap_shots", type=int, default=1)
    parser.add_argument("--temporal_lambda_skip", type=float, default=0.7)

    parser.add_argument("--meilisearch_url", type=str, default=None)
    parser.add_argument("--meilisearch_index_name", type=str, default=None)
    parser.add_argument("--subtitle_meilisearch_index_name", type=str, default=None)
    parser.add_argument("--meilisearch_api_key", type=str, default=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve pooling movie event retrieval with FastAPI")
    add_retrieval_arguments(parser)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log_level", type=str, default="info")
    parser.add_argument("--no_ngrok", action="store_true")
    parser.add_argument("--ngrok_authtoken", type=str, default=None)
    parser.add_argument("--ngrok_secret_name", type=str, default="NGROK_AUTHTOKEN")
    return parser.parse_args()


def build_search_config(args: argparse.Namespace) -> SearchConfig:
    return SearchConfig(
        store_dir=args.store_dir,
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
    )


def resolve_ngrok_token(args: argparse.Namespace) -> str:
    if args.ngrok_authtoken:
        return str(args.ngrok_authtoken)

    environment_token = os.environ.get(args.ngrok_secret_name)
    if environment_token:
        return environment_token

    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret(args.ngrok_secret_name)
    except Exception as error:
        raise RuntimeError(
            f"Ngrok token not found. Add Kaggle Secret {args.ngrok_secret_name!r}, "
            f"set environment variable {args.ngrok_secret_name!r}, or pass --ngrok_authtoken."
        ) from error
    if not token:
        raise RuntimeError(f"Kaggle Secret {args.ngrok_secret_name!r} is empty")
    return str(token)


def open_ngrok_tunnel(args: argparse.Namespace) -> str | None:
    if args.no_ngrok:
        return None

    from pyngrok import ngrok

    ngrok.set_auth_token(resolve_ngrok_token(args))
    tunnel = ngrok.connect(addr=args.port, proto="http", bind_tls=True)
    public_url = str(tunnel.public_url)
    print(f"Public URL: {public_url}", flush=True)
    print(f"Search endpoint: {public_url}/search", flush=True)
    print(f"Health endpoint: {public_url}/health", flush=True)
    return public_url


def main() -> None:
    args = parse_args()
    config = build_search_config(args)
    app = create_app(config)
    public_url = open_ngrok_tunnel(args)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    finally:
        if public_url is not None:
            from pyngrok import ngrok

            ngrok.disconnect(public_url)
            ngrok.kill()


if __name__ == "__main__":
    main()
