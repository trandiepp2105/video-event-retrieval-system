from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-embed reference keyframes using a visual embedding model via OpenCLIP, "
            "while preserving the reference keyframe output structure."
        )
    )
    parser.add_argument("--videos_dir", type=str, required=True)
    parser.add_argument("--reference_keyframe_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--video_ids", type=str, nargs="+", default=None)

    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-gopt-16-SigLIP2-384",
        help="OpenCLIP model name.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="webli",
        help=(
            "OpenCLIP pretrained tag or local weights path. "
            "If this is a local path, weights are loaded locally. "
            "If this is a tag, OpenCLIP may resolve/download it via internet."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    module_root = Path(__file__).resolve().parents[3] / "keyframe_siglip2_embedding_module"
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

    from keyframe_siglip2_embedding import BatchProcessor, PipelineConfig

    config = PipelineConfig(
        videos_dir=args.videos_dir,
        reference_keyframe_dir=args.reference_keyframe_dir,
        output_dir=args.output_dir,
        start_index=args.start_index,
        end_index=args.end_index,
        video_ids=args.video_ids,
        clip_model_name=args.model_name,
        clip_pretrained=args.pretrained,
        batch_size=args.batch_size,
        device=args.device,
        save_dtype=args.save_dtype,
        overwrite=args.overwrite,
    )
    summary = BatchProcessor(config).run()
    print(summary)
