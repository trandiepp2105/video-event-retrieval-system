import argparse

from .config import FaceContinuityConfig
from .pipeline import FaceContinuityPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract face continuity features and boundary scores using InsightFace."
    )

    parser.add_argument("--video_dataset_dir", type=str, required=True)
    parser.add_argument("--shots_json_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument(
        "--video_ids",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of video ids. If provided, these video ids are used instead of start/end index.",
    )

    parser.add_argument("--insightface_model_name", type=str, default="buffalo_l")
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--ctx_id", type=int, default=0)
    parser.add_argument("--det_width", type=int, default=640)
    parser.add_argument("--det_height", type=int, default=640)

    parser.add_argument("--det_score_thresh", type=float, default=0.50)
    parser.add_argument("--min_face_size", type=int, default=40)

    parser.add_argument("--max_frames_per_shot", type=int, default=5)
    parser.add_argument("--seconds_per_sample", type=float, default=2.0)
    parser.add_argument("--recognition_batch_size", type=int, default=64)

    parser.add_argument("--same_face_threshold", type=float, default=0.45)
    parser.add_argument("--min_face_count_per_shot", type=int, default=1)
    parser.add_argument("--face_window_size", type=int, default=3)

    parser.add_argument("--save_debug_crops", action="store_true")
    parser.add_argument("--no_save_debug_crops", action="store_true")
    parser.add_argument("--max_debug_crops_per_face_id", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    save_debug_crops = True
    if args.no_save_debug_crops:
        save_debug_crops = False
    elif args.save_debug_crops:
        save_debug_crops = True

    config = FaceContinuityConfig(
        video_dataset_dir=args.video_dataset_dir,
        shots_json_dir=args.shots_json_dir,
        output_dir=args.output_dir,
        start_index=args.start_index,
        end_index=args.end_index,
        video_ids=args.video_ids,
        insightface_model_name=args.insightface_model_name,
        root_dir=args.root_dir,
        ctx_id=args.ctx_id,
        det_size=(args.det_width, args.det_height),
        det_score_thresh=args.det_score_thresh,
        min_face_size=args.min_face_size,
        max_frames_per_shot=args.max_frames_per_shot,
        seconds_per_sample=args.seconds_per_sample,
        recognition_batch_size=args.recognition_batch_size,
        same_face_threshold=args.same_face_threshold,
        min_face_count_per_shot=args.min_face_count_per_shot,
        face_window_size=args.face_window_size,
        save_debug_crops=save_debug_crops,
        max_debug_crops_per_face_id=args.max_debug_crops_per_face_id,
        overwrite=args.overwrite,
    )

    pipeline = FaceContinuityPipeline(config)
    run_summary = pipeline.run()
    print(run_summary)
