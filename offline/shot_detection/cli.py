from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import ShotDetectionConfig, ShotDetectorPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline shot detection for dataset videos"
    )
    parser.add_argument("--videos_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--video_ids", nargs="+", default=None)
    parser.add_argument("--detector", type=str, default="adaptive", choices=["adaptive", "content"])
    parser.add_argument("--adaptive_threshold", type=float, default=3.0)
    parser.add_argument("--min_content_val", type=float, default=15.0)
    parser.add_argument("--window_width", type=int, default=2)
    parser.add_argument("--content_threshold", type=float, default=27.0)
    parser.add_argument("--min_shot_len_sec", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ShotDetectionConfig(
        input_dir=args.videos_dir,
        output_dir=args.output_dir,
        start_index=args.start_index,
        end_index=args.end_index,
        video_ids=args.video_ids,
        detector=args.detector,
        adaptive_threshold=args.adaptive_threshold,
        min_content_val=args.min_content_val,
        window_width=args.window_width,
        content_threshold=args.content_threshold,
        min_shot_len_sec=args.min_shot_len_sec,
        overwrite=args.overwrite,
    )
    result = ShotDetectorPipeline(config).run()
    print(result)
