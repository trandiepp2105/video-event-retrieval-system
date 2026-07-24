from collections import defaultdict
from pathlib import Path

from tqdm.auto import tqdm

from .config import PipelineConfig
from .io_utils import DatasetLoader, ProgressManager, ensure_dir
from .ocr_components import (
    CropProcessor,
    SceneTextWriter,
    SubtitleDetector,
    SubtitleJSONWriter,
    SubtitleMerger,
    VietOCRBatch,
    find_logo_positions,
    is_logo_bbox,
)
from .video_utils import FFmpegFramePipeReader, FFprobeMetadataReader, SampledFrame


def batched(items: list[SampledFrame], batch_size: int):
    batch_size = max(int(batch_size), 1)
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


class SubtitleScenePipeline:
    def __init__(self, video_path: str | Path, config: PipelineConfig, detector: SubtitleDetector, ocr: VietOCRBatch) -> None:
        self.video_path = Path(video_path)
        self.config = config
        self.detector = detector
        self.cropper = CropProcessor(padding=config.padding)
        self.ocr = ocr
        self.merger = SubtitleMerger(
            similarity_threshold=config.subtitle_similarity_threshold,
            max_gap=config.subtitle_max_gap,
        )
        self.writer = SubtitleJSONWriter()
        self.scene_writer = SceneTextWriter()
        self.metadata_reader = FFprobeMetadataReader(config.ffprobe_path)

    def run(self, output_subtitle: str | Path, output_ocr: str | Path) -> None:
        metadata = self.metadata_reader.read(self.video_path)
        reader = FFmpegFramePipeReader(
            self.video_path,
            metadata,
            frame_step=self.config.frame_step,
            ffmpeg_path=self.config.ffmpeg_path,
        )
        sampled_frames = list(reader.iter_sampled_frames())

        detections: list[dict] = []
        for frame_batch in tqdm(
            list(batched(sampled_frames, self.config.frame_batch_size)),
            desc=f"Detect {self.video_path.stem}",
            leave=False,
        ):
            images = [item.image for item in frame_batch]
            subtitle_boxes_batch, scene_boxes_batch = self.detector.detect_batch(
                images,
                roi_x=(self.config.roi_x_min, self.config.roi_x_max),
                roi_y=(self.config.roi_y_min, self.config.roi_y_max),
            )
            for frame, subtitle_boxes, scene_boxes in zip(frame_batch, subtitle_boxes_batch, scene_boxes_batch):
                detections.append(
                    {
                        "frame_id": frame.frame_id,
                        "time_sec": frame.time_sec,
                        "subtitle_boxes": subtitle_boxes,
                        "scene_boxes": scene_boxes,
                    }
                )

        all_scene_boxes = [item["scene_boxes"] for item in detections]
        logo_centers = find_logo_positions(all_scene_boxes, len(sampled_frames))
        detection_by_frame_id = {item["frame_id"]: item for item in detections}

        subtitle_crops = []
        subtitle_meta = []
        scene_crops = []
        scene_meta = []

        for frame in tqdm(sampled_frames, desc=f"Crop {self.video_path.stem}", leave=False):
            detection = detection_by_frame_id.get(frame.frame_id)
            if detection is None:
                continue
            sub_boxes = detection["subtitle_boxes"]
            scn_boxes = detection["scene_boxes"]
            if not sub_boxes and not scn_boxes:
                continue
            img = frame.image

            for box_index, item in enumerate(sub_boxes):
                crop = self.cropper.crop(img, item["box"])
                if crop is None:
                    continue
                subtitle_crops.append(crop)
                subtitle_meta.append(
                    {
                        "frame_id": frame.frame_id,
                        "time_sec": frame.time_sec,
                        "box_idx": box_index,
                        "box": item["box"],
                        "paddle_text": item["text"],
                        "paddle_score": item["score"],
                    }
                )

            for box_index, item in enumerate(scn_boxes):
                if is_logo_bbox(item["box"], logo_centers):
                    continue
                crop = self.cropper.crop(img, item["box"])
                if crop is None:
                    continue
                scene_crops.append(crop)
                scene_meta.append(
                    {
                        "frame_id": frame.frame_id,
                        "time_sec": frame.time_sec,
                        "box_idx": box_index,
                        "box": item["box"],
                        "paddle_text": item["text"],
                        "paddle_score": item["score"],
                    }
                )

        frame_lines: dict[int, list[dict]] = defaultdict(list)
        if subtitle_crops:
            subtitle_texts = self.ocr.predict(subtitle_crops)
            for meta, text in zip(subtitle_meta, subtitle_texts):
                frame_lines[meta["frame_id"]].append(
                    {
                        "text": text,
                        "box": meta["box"],
                        "box_idx": meta["box_idx"],
                        "paddle_text": meta["paddle_text"],
                        "paddle_score": meta["paddle_score"],
                    }
                )

        frame_texts: dict[int, str] = {}
        for frame_id, items in frame_lines.items():
            items = sorted(items, key=lambda x: (x["box"][1], x["box"][0]))
            frame_texts[frame_id] = "\n".join(item["text"] for item in items)

        subs = self.merger.merge(frame_texts)
        self.writer.write(subs, output_subtitle, metadata.fps)

        scene_records = []
        if scene_crops:
            scene_texts = self.ocr.predict(scene_crops)
            for meta, text in zip(scene_meta, scene_texts):
                scene_records.append(
                    {
                        "frame_id": meta["frame_id"],
                        "time_sec": meta["time_sec"],
                        "box": meta["box"],
                        "paddle_text": meta["paddle_text"],
                        "paddle_score": meta["paddle_score"],
                        "vietocr_text": text,
                    }
                )
        self.scene_writer.write(scene_records, output_ocr)


class DatasetPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        ensure_dir(config.subtitle_output_dir)
        ensure_dir(config.ocr_output_dir)
        self.loader = DatasetLoader(config.videos_dir)
        self.progress = ProgressManager(config.subtitle_output_dir / "progress.json")
        print("Loading detector once...")
        self.detector = SubtitleDetector(config)
        print("Loading VietOCR once...")
        self.ocr = VietOCRBatch(config)

    def _select_videos(self) -> list[Path]:
        videos = self.loader.list_videos()
        if self.config.video_ids:
            wanted = {str(video_id) for video_id in self.config.video_ids}
            videos = [video for video in videos if video.stem in wanted or video.name in wanted]
        else:
            start_index = max(int(self.config.start_index), 0)
            if self.config.end_index is None:
                videos = videos[start_index:]
            else:
                videos = videos[start_index : int(self.config.end_index) + 1]
        return videos

    def run(self) -> dict[str, list[str]]:
        all_videos = self.loader.list_videos()
        selected_videos = self._select_videos()
        selected_names = {video.stem for video in selected_videos}
        last_done = self.progress.load()
        summary = {"done": [], "skipped": [], "failed": []}

        for index, video in enumerate(tqdm(all_videos, desc="Processing videos")):
            if video.stem not in selected_names:
                continue
            if not self.config.video_ids and index <= last_done and index >= self.config.start_index:
                continue

            output_subtitle = self.config.subtitle_output_dir / f"{video.stem}.json"
            output_ocr = self.config.ocr_output_dir / f"{video.stem}.json"
            if output_subtitle.exists() and output_ocr.exists() and not self.config.overwrite:
                print(f"[SKIP] {video.stem}")
                summary["skipped"].append(video.stem)
                continue

            try:
                print(f"Processing {index} {video.stem}")
                pipeline = SubtitleScenePipeline(
                    video_path=video,
                    config=self.config,
                    detector=self.detector,
                    ocr=self.ocr,
                )
                pipeline.run(output_subtitle, output_ocr)
                self.progress.save(index)
                summary["done"].append(video.stem)
            except Exception as exc:
                print(f"[FAILED] {video.stem}: {exc!r}")
                summary["failed"].append(video.stem)

        return summary
