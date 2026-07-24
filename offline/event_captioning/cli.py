from pathlib import Path

from .config import CaptionConfig
from .pipeline import EventCaptionPipeline


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate one Vietnamese retrieval caption per event using Qwen3-VL."
    )

    parser.add_argument("--videos_dir", type=Path, required=True)
    parser.add_argument("--event_output_dir", type=Path, required=True)
    parser.add_argument("--subtitles_dir", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="Exclusive end index. Example: 0 100 processes video indices 0..99.",
    )
    parser.add_argument(
        "--video_ids",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of video IDs. If provided, only these videos are processed and start/end index are ignored.",
    )

    parser.add_argument("--video_sample_fps", type=float, default=0.5)
    parser.add_argument("--min_pixels", type=int, default=128 * 128)
    parser.add_argument("--max_pixels", type=int, default=256 * 256)
    parser.add_argument("--image_patch_size", type=int, default=16)

    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)

    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--device_map", type=str, default="auto")

    parser.add_argument("--tmp_dir", type=Path, default=None)
    parser.add_argument(
        "--clip_codec",
        type=str,
        default="copy",
        choices=["copy", "h264"],
        help="copy is faster; h264 re-encodes and is safer.",
    )
    parser.add_argument("--ffmpeg_loglevel", type=str, default="error")

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_save_every_event", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CaptionConfig(
        videos_dir=args.videos_dir,
        event_output_dir=args.event_output_dir,
        subtitles_dir=args.subtitles_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        start_index=args.start_index,
        end_index=args.end_index,
        video_ids=args.video_ids,
        video_sample_fps=args.video_sample_fps,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        image_patch_size=args.image_patch_size,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        tmp_dir=args.tmp_dir,
        clip_codec=args.clip_codec,
        ffmpeg_loglevel=args.ffmpeg_loglevel,
        overwrite=args.overwrite,
        save_every_event=not args.no_save_every_event,
        continue_on_error=not args.stop_on_error,
    )
    EventCaptionPipeline(config).run()
