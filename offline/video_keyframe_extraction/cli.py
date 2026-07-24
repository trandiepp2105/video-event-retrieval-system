import argparse

from .config import PipelineConfig
from .pipeline import BatchProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract video-level keyframe embeddings from mp4 videos using OpenCLIP."
    )

    parser.add_argument("--videos_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--frame_step", type=int, default=6)
    parser.add_argument("--similarity_threshold", type=float, default=0.95)

    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument(
        "--video_ids",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of video ids. If provided, these video ids are used instead of start/end index.",
    )

    parser.add_argument("--clip_model_name", type=str, default="ViT-H-14-quickgelu")
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        required=True,
        help="Path to local OpenCLIP pretrained weights or a local tag already available in the environment.",
    )
    parser.add_argument(
        "--frame_load_batch_size",
        type=int,
        default=256,
        help="Number of sampled frames kept in RAM before each streaming encode step.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    config = PipelineConfig(
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        frame_step=args.frame_step,
        similarity_threshold=args.similarity_threshold,
        start_index=args.start_index,
        end_index=args.end_index,
        video_ids=args.video_ids,
        clip_model_name=args.clip_model_name,
        clip_pretrained=args.clip_pretrained,
        frame_load_batch_size=args.frame_load_batch_size,
        batch_size=args.batch_size,
        device=args.device,
        save_dtype=args.save_dtype,
        overwrite=args.overwrite,
    )

    processor = BatchProcessor(config)
    summary = processor.run()
    print(summary)
