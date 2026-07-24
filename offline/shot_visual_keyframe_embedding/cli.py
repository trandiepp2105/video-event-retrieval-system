import argparse

from .config import PipelineConfig
from .pipeline import BatchProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build keyframe embeddings for each shot using OpenCLIP, without pooling into shot embeddings."
    )

    parser.add_argument("--videos_dir", type=str, required=True)
    parser.add_argument("--shots_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--frame_step", type=int, default=6)
    parser.add_argument("--similarity_threshold", type=float, default=0.88)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument(
        "--video_ids",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of video ids. If provided, these video ids are used instead of start/end index.",
    )
    parser.add_argument("--clip_pretrained", type=str, default="dfn5b")
    parser.add_argument("--batch_size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = PipelineConfig(
        videos_dir=args.videos_dir,
        shots_dir=args.shots_dir,
        output_dir=args.output_dir,
        frame_step=args.frame_step,
        similarity_threshold=args.similarity_threshold,
        start_index=args.start_index,
        end_index=args.end_index,
        video_ids=args.video_ids,
        clip_model_name="ViT-H-14-quickgelu",
        clip_pretrained=args.clip_pretrained,
        batch_size=args.batch_size,
        device="cuda",
        save_dtype="float16",
        overwrite=False,
    )

    processor = BatchProcessor(config)
    summary = processor.run()
    print(summary)
