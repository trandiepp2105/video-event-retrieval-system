from __future__ import annotations

import argparse
from pathlib import Path

from .config import TextEmbeddingConfig
from .pipeline import TextEmbeddingPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline text embedding for subtitles and event captions"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed_subtitle = subparsers.add_parser("embed-subtitle")
    _add_common_args(embed_subtitle)
    embed_subtitle.add_argument("--subtitle_json_dir", type=Path, required=True)
    embed_subtitle.add_argument("--text_field", type=str, default="text")

    embed_caption = subparsers.add_parser("embed-caption")
    _add_common_args(embed_caption)
    embed_caption.add_argument("--caption_json_dir", type=Path, required=True)
    embed_caption.add_argument("--caption_field", type=str, default="caption")

    return parser


def _add_common_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--output_dir", type=Path, required=True)
    subparser.add_argument(
        "--model_name_or_path",
        type=str,
        default="AITeamVN/Vietnamese_Embedding",
    )
    subparser.add_argument("--device", type=str, default=None)
    subparser.add_argument("--batch_size", type=int, default=32)
    subparser.add_argument(
        "--normalize_embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    subparser.add_argument("--save_dtype", type=str, default="float32", choices=["float16", "float32"])
    subparser.add_argument("--start_index", type=int, default=0)
    subparser.add_argument("--end_index", type=int, default=None)
    subparser.add_argument("--video_ids", nargs="+", default=None)
    subparser.add_argument("--overwrite", action="store_true")


def _build_config(args: argparse.Namespace) -> TextEmbeddingConfig:
    if args.command == "embed-subtitle":
        return TextEmbeddingConfig(
            input_dir=args.subtitle_json_dir,
            output_dir=args.output_dir,
            model_name_or_path=args.model_name_or_path,
            input_type="subtitle",
            device=args.device,
            batch_size=args.batch_size,
            normalize_embeddings=args.normalize_embeddings,
            save_dtype=args.save_dtype,
            start_index=args.start_index,
            end_index=args.end_index,
            video_ids=args.video_ids,
            overwrite=args.overwrite,
            text_field=args.text_field,
        )

    return TextEmbeddingConfig(
        input_dir=args.caption_json_dir,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        input_type="caption",
        device=args.device,
        batch_size=args.batch_size,
        normalize_embeddings=args.normalize_embeddings,
        save_dtype=args.save_dtype,
        start_index=args.start_index,
        end_index=args.end_index,
        video_ids=args.video_ids,
        overwrite=args.overwrite,
        caption_field=args.caption_field,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = _build_config(args)
    result = TextEmbeddingPipeline(config).run()
    print(result)
