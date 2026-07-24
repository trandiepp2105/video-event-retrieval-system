from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import EventGroupingDatasetConfig, EventGroupingPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline event boundary detection from shot-level features"
    )
    parser.add_argument("--features_dir", type=Path, required=True)
    parser.add_argument("--output_root_dir", type=Path, required=True)
    parser.add_argument("--video_ids", nargs="+", default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--skip_missing_modalities", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--context_window", type=int, default=3)
    parser.add_argument("--subtitle_use_recency_weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subtitle_recency_tau", type=float, default=2.0)
    parser.add_argument("--subtitle_bridge_penalty_weight", type=float, default=0.40)
    parser.add_argument("--subtitle_bridge_norm_sec", type=float, default=0.50)
    parser.add_argument("--use_face_recency_weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--face_recency_tau", type=float, default=2.0)
    parser.add_argument("--visual_weight", type=float, default=0.30)
    parser.add_argument("--action_weight", type=float, default=0.15)
    parser.add_argument("--subtitle_weight", type=float, default=0.40)
    parser.add_argument("--face_weight", type=float, default=0.15)
    parser.add_argument("--boundary_percentile", type=float, default=85.0)
    parser.add_argument("--use_local_peak", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_event_duration_sec", type=float, default=3.0)
    parser.add_argument("--max_event_duration_sec", type=float, default=30.0)
    parser.add_argument("--cut_penalty", type=float, default=0.55)
    parser.add_argument("--non_candidate_penalty", type=float, default=0.25)
    parser.add_argument("--event_softmax_temperature", type=float, default=15.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight_only", action="store_true")
    return parser


def _build_config(args: argparse.Namespace) -> EventGroupingDatasetConfig:
    return EventGroupingDatasetConfig(
        features_dir=str(args.features_dir),
        output_root_dir=str(args.output_root_dir),
        video_ids=args.video_ids,
        start_index=args.start_index,
        end_index=args.end_index,
        skip_missing_modalities=args.skip_missing_modalities,
        context_window=args.context_window,
        subtitle_use_recency_weight=args.subtitle_use_recency_weight,
        subtitle_recency_tau=args.subtitle_recency_tau,
        subtitle_bridge_penalty_weight=args.subtitle_bridge_penalty_weight,
        subtitle_bridge_norm_sec=args.subtitle_bridge_norm_sec,
        use_face_recency_weight=args.use_face_recency_weight,
        face_recency_tau=args.face_recency_tau,
        visual_weight=args.visual_weight,
        action_weight=args.action_weight,
        subtitle_weight=args.subtitle_weight,
        face_weight=args.face_weight,
        boundary_percentile=args.boundary_percentile,
        use_local_peak=args.use_local_peak,
        min_event_duration_sec=args.min_event_duration_sec,
        max_event_duration_sec=args.max_event_duration_sec,
        cut_penalty=args.cut_penalty,
        non_candidate_penalty=args.non_candidate_penalty,
        event_softmax_temperature=args.event_softmax_temperature,
        overwrite=args.overwrite,
    )


def main() -> None:
    args = build_parser().parse_args()
    config = _build_config(args)
    pipeline = EventGroupingPipeline(config)

    if args.preflight_only:
        preflight = pipeline.run_preflight()
        print(json.dumps(preflight["summary"], ensure_ascii=False, indent=2))
        return

    output = pipeline.run()
    print(f"Saved dataset summary to: {output['summary_path']}")
    print(json.dumps(output["results"][:10], ensure_ascii=False, indent=2))
