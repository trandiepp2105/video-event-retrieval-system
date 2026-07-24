import argparse
from pathlib import Path

from .config import PipelineConfig
from .pipeline import DatasetPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract hardcoded subtitles and scene text directly from mp4 videos using ffmpeg/ffprobe + PaddleOCR + VietOCR."
    )
    parser.add_argument("--videos_dir", type=Path, required=True)
    parser.add_argument("--subtitle_output_dir", type=Path, required=True)
    parser.add_argument("--ocr_output_dir", type=Path, required=True)
    parser.add_argument("--paddle_text_detection_model_dir", type=Path, required=True)
    parser.add_argument("--paddle_text_recognition_model_dir", type=Path, required=True)
    parser.add_argument("--vietocr_weights", type=Path, required=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--video_ids", nargs="+", default=None)
    parser.add_argument("--roi_x_min", type=float, default=0.0)
    parser.add_argument("--roi_x_max", type=float, default=1.0)
    parser.add_argument("--roi_y_min", type=float, default=0.65)
    parser.add_argument("--roi_y_max", type=float, default=1.0)
    parser.add_argument("--padding", type=int, default=3)
    parser.add_argument("--max_subtitle_angle_deg", type=float, default=8.0)
    parser.add_argument("--detector_score_threshold", type=float, default=0.75)
    parser.add_argument("--detector_max_boxes", type=int, default=2)
    parser.add_argument("--detector_min_scene_score", type=float, default=0.5)
    parser.add_argument("--subtitle_similarity_threshold", type=float, default=0.8)
    parser.add_argument("--subtitle_max_gap", type=int, default=12)
    parser.add_argument("--frame_step", type=int, default=6)
    parser.add_argument("--frame_batch_size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ffmpeg_path", type=str, default="ffmpeg")
    parser.add_argument("--ffprobe_path", type=str, default="ffprobe")
    parser.add_argument("--paddle_device", type=str, default="gpu:0")
    parser.add_argument("--vietocr_device", type=str, default="cuda")
    parser.add_argument("--vietocr_repo_path", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig(
        videos_dir=args.videos_dir,
        subtitle_output_dir=args.subtitle_output_dir,
        ocr_output_dir=args.ocr_output_dir,
        paddle_text_detection_model_dir=args.paddle_text_detection_model_dir,
        paddle_text_recognition_model_dir=args.paddle_text_recognition_model_dir,
        vietocr_weights=args.vietocr_weights,
        roi_x_min=args.roi_x_min,
        roi_x_max=args.roi_x_max,
        roi_y_min=args.roi_y_min,
        roi_y_max=args.roi_y_max,
        padding=args.padding,
        max_subtitle_angle_deg=args.max_subtitle_angle_deg,
        detector_score_threshold=args.detector_score_threshold,
        detector_max_boxes=args.detector_max_boxes,
        detector_min_scene_score=args.detector_min_scene_score,
        subtitle_similarity_threshold=args.subtitle_similarity_threshold,
        subtitle_max_gap=args.subtitle_max_gap,
        frame_step=args.frame_step,
        frame_batch_size=args.frame_batch_size,
        start_index=args.start_index,
        end_index=args.end_index,
        video_ids=args.video_ids,
        overwrite=args.overwrite,
        ffmpeg_path=args.ffmpeg_path,
        ffprobe_path=args.ffprobe_path,
        paddle_device=args.paddle_device,
        vietocr_device=args.vietocr_device,
        vietocr_repo_path=args.vietocr_repo_path,
    )
    summary = DatasetPipeline(config).run()
    print(summary)
