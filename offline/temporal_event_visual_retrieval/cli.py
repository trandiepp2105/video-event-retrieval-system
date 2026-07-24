from __future__ import annotations

import argparse
from pathlib import Path

from .config import EncodeConfig, PrepareTranslationsConfig, TrainConfig
from .encode import encode
from .prepare_translations import prepare_translations
from .train import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Temporal visual event retrieval training and encoding with "
            "keyframe-to-shot and shot-to-event transformers."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    encode_parser = subparsers.add_parser("encode")

    def add_shared_args(sub, *, require_captions: bool) -> None:
        sub.add_argument("--event_dir", type=Path, required=True)
        sub.add_argument("--shot_keyframe_dir", type=Path, required=True)
        sub.add_argument("--event_captions_dir", type=Path, required=require_captions, default=None)
        sub.add_argument("--translated_captions_dir", type=Path, default=None)
        sub.add_argument("--clip_model_path", type=Path, required=True)
        sub.add_argument("--output_dir", type=Path, required=True)
        sub.add_argument("--raw_subtitle_dir", type=Path, default=Path("./features/raw_subtitle"))
        sub.add_argument("--clip_model_name", type=str, default="ViT-H-14-quickgelu")
        sub.add_argument("--intro_padding_sec", type=float, default=120.0)
        sub.add_argument("--outro_padding_sec", type=float, default=120.0)
        sub.add_argument("--max_event_shots", type=int, default=64)
        sub.add_argument("--max_keyframes_per_shot", type=int, default=16)
        sub.add_argument("--keyframe_hidden_dim", type=int, default=768)
        sub.add_argument("--shot_hidden_dim", type=int, default=768)
        sub.add_argument("--keyframe_transformer_layers", type=int, default=2)
        sub.add_argument("--shot_transformer_layers", type=int, default=2)
        sub.add_argument("--keyframe_transformer_heads", type=int, default=8)
        sub.add_argument("--shot_transformer_heads", type=int, default=8)
        sub.add_argument("--dropout", type=float, default=0.1)
        sub.add_argument("--projection_dim", type=int, default=768)
        sub.add_argument("--keyframe_metadata_time_norm_sec", type=float, default=4.0)
        sub.add_argument("--shot_metadata_duration_norm_sec", type=float, default=30.0)
        sub.add_argument("--caption_shot_similarity_weight", type=float, default=0.80)
        sub.add_argument("--subtitle_overlap_weight", type=float, default=0.20)
        sub.add_argument("--soft_positive_temperature", type=float, default=0.07)
        sub.add_argument("--salient_subtitle_norm_sec", type=float, default=4.0)
        sub.add_argument("--device", type=str, default="cuda")
        sub.add_argument("--num_workers", type=int, default=0)

    add_shared_args(train_parser, require_captions=True)
    add_shared_args(encode_parser, require_captions=False)

    train_parser.add_argument("--batch_size", type=int, default=4)
    train_parser.add_argument("--num_epochs", type=int, default=3)
    train_parser.add_argument("--learning_rate", type=float, default=1e-4)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--event_loss_weight", type=float, default=1.0)
    train_parser.add_argument("--shot_loss_weight", type=float, default=0.5)
    train_parser.add_argument("--event_consistency_loss_weight", type=float, default=0.1)
    train_parser.add_argument("--shot_consistency_loss_weight", type=float, default=0.1)
    train_parser.add_argument("--temperature", type=float, default=0.07)
    train_parser.add_argument("--seed", type=int, default=42)

    encode_parser.add_argument("--checkpoint_path", type=Path, default=None)

    translate_parser = subparsers.add_parser("prepare_translations")
    translate_parser.add_argument("--event_captions_dir", type=Path, required=True)
    translate_parser.add_argument("--translation_model_path", type=Path, required=True)
    translate_parser.add_argument("--output_dir", type=Path, required=True)
    translate_parser.add_argument("--device", type=str, default="cuda")
    translate_parser.add_argument("--translation_batch_size", type=int, default=8)
    translate_parser.add_argument("--start_index", type=int, default=None)
    translate_parser.add_argument("--end_index", type=int, default=None)
    translate_parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare_translations":
        output_dir = prepare_translations(
            PrepareTranslationsConfig(
                event_captions_dir=args.event_captions_dir,
                translation_model_path=args.translation_model_path,
                output_dir=args.output_dir,
                device=args.device,
                translation_batch_size=args.translation_batch_size,
                start_index=args.start_index,
                end_index=args.end_index,
                overwrite=args.overwrite,
            )
        )
        print(f"Saved translated captions to: {output_dir}")
        return

    common_kwargs = dict(
        event_dir=args.event_dir,
        shot_keyframe_dir=args.shot_keyframe_dir,
        event_captions_dir=args.event_captions_dir,
        translated_captions_dir=args.translated_captions_dir,
        clip_model_path=args.clip_model_path,
        output_dir=args.output_dir,
        raw_subtitle_dir=args.raw_subtitle_dir,
        clip_model_name=args.clip_model_name,
        intro_padding_sec=args.intro_padding_sec,
        outro_padding_sec=args.outro_padding_sec,
        max_event_shots=args.max_event_shots,
        max_keyframes_per_shot=args.max_keyframes_per_shot,
        keyframe_hidden_dim=args.keyframe_hidden_dim,
        shot_hidden_dim=args.shot_hidden_dim,
        keyframe_transformer_layers=args.keyframe_transformer_layers,
        shot_transformer_layers=args.shot_transformer_layers,
        keyframe_transformer_heads=args.keyframe_transformer_heads,
        shot_transformer_heads=args.shot_transformer_heads,
        dropout=args.dropout,
        projection_dim=args.projection_dim,
        keyframe_metadata_time_norm_sec=args.keyframe_metadata_time_norm_sec,
        shot_metadata_duration_norm_sec=args.shot_metadata_duration_norm_sec,
        caption_shot_similarity_weight=args.caption_shot_similarity_weight,
        subtitle_overlap_weight=args.subtitle_overlap_weight,
        soft_positive_temperature=args.soft_positive_temperature,
        salient_subtitle_norm_sec=args.salient_subtitle_norm_sec,
        device=args.device,
        num_workers=args.num_workers,
    )

    if args.command == "train":
        config = TrainConfig(
            **common_kwargs,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            event_loss_weight=args.event_loss_weight,
            shot_loss_weight=args.shot_loss_weight,
            event_consistency_loss_weight=args.event_consistency_loss_weight,
            shot_consistency_loss_weight=args.shot_consistency_loss_weight,
            temperature=args.temperature,
            seed=args.seed,
        )
        checkpoint_path = train(config)
        print(f"Saved best checkpoint to: {checkpoint_path}")
        return

    if args.command == "encode":
        config = EncodeConfig(
            **common_kwargs,
            checkpoint_path=args.checkpoint_path,
        )
        output_dir = encode(config)
        print(f"Saved temporal visual embeddings to: {output_dir}")
        return

    raise ValueError(f"Unknown command: {args.command}")
