from pathlib import Path
from typing import Any

import numpy as np

from .boundary import FaceBoundaryScorer
from .clustering import FaceClusterer
from .config import FaceContinuityConfig
from .extractor import FaceDetectionProcessor, InsightFaceExtractor
from .io_utils import DatasetScanner, FileIO, ShotNormalizer
from .output_utils import FaceDebugCropSaver, FaceOutputWriter
from .video_utils import ShotFrameSampler


class FaceContinuityPipeline:
    def __init__(self, config: FaceContinuityConfig):
        self.config = config
        self.scanner = DatasetScanner(config)
        self.frame_sampler = ShotFrameSampler(
            max_frames_per_shot=config.max_frames_per_shot,
            seconds_per_sample=config.seconds_per_sample,
        )
        self.face_extractor = InsightFaceExtractor(
            model_name=config.insightface_model_name,
            root_dir=config.root_dir,
            ctx_id=config.ctx_id,
            det_size=config.det_size,
            det_score_thresh=config.det_score_thresh,
            min_face_size=config.min_face_size,
        )
        self.detection_processor = FaceDetectionProcessor(
            self.frame_sampler,
            self.face_extractor,
            recognition_batch_size=config.recognition_batch_size,
        )
        self.clusterer = FaceClusterer(config.same_face_threshold)
        self.boundary_scorer = FaceBoundaryScorer(config.face_window_size)
        self.output_writer = FaceOutputWriter()
        self.debug_crop_saver = FaceDebugCropSaver()

    def _load_shots(self, shots_json_path: str) -> list[dict[str, Any]]:
        raw_shots = FileIO.load_json(shots_json_path)
        return ShotNormalizer.normalize_shots(raw_shots)

    def _build_summary(
        self,
        shots: list[dict[str, Any]],
        all_detections: list[dict[str, Any]],
        face_clusters: list[list[int]],
        face_boundary_scores: list[dict[str, Any]],
        extraction_stats: dict[str, Any],
    ) -> dict[str, Any]:
        num_shots = len(shots)
        num_shots_with_face = sum(len(shot.get("face_ids", [])) > 0 for shot in shots)
        valid_boundaries = sum(row["valid"] for row in face_boundary_scores)
        valid_face_changes = [row["face_change"] for row in face_boundary_scores if row["valid"]]
        avg_face_change = float(np.mean(valid_face_changes)) if len(valid_face_changes) > 0 else None

        return {
            "num_shots": int(num_shots),
            "num_shots_with_face": int(num_shots_with_face),
            "shot_face_coverage": float(num_shots_with_face / max(num_shots, 1)),
            "num_sampled_frames": int(extraction_stats["num_sampled_frames"]),
            "num_face_detections": int(len(all_detections)),
            "num_face_clusters": int(len(face_clusters)),
            "valid_face_boundaries": int(valid_boundaries),
            "valid_face_boundary_ratio": float(valid_boundaries / max(len(face_boundary_scores), 1)),
            "avg_face_change_on_valid_boundaries": avg_face_change,
            "same_face_threshold": float(self.config.same_face_threshold),
            "face_window_size": int(self.config.face_window_size),
            "det_score_thresh": float(self.config.det_score_thresh),
            "min_face_size": int(self.config.min_face_size),
            "max_frames_per_shot": int(self.config.max_frames_per_shot),
            "seconds_per_sample": float(self.config.seconds_per_sample),
            "recognition_batch_size": int(self.config.recognition_batch_size),
        }

    def process_video_item(self, item: dict[str, str]) -> dict[str, Any]:
        shots = self._load_shots(item["shots_json_path"])
        shots, all_detections, extraction_stats = self.detection_processor.extract_for_shots(
            video_path=item["video_path"],
            shots=shots,
        )

        all_detections, face_clusters, face_centroids = self.clusterer.cluster_online(all_detections)
        shots = self.clusterer.attach_to_shots(shots, all_detections)
        shots = self.clusterer.assign_face_ids_to_shots(shots, min_count=self.config.min_face_count_per_shot)

        face_boundary_scores = self.boundary_scorer.compute(shots)
        summary = self._build_summary(shots, all_detections, face_clusters, face_boundary_scores, extraction_stats)

        output_paths = self.output_writer.write_video_outputs(
            output_dir=item["output_dir"],
            shots=shots,
            face_boundary_scores=face_boundary_scores,
            face_centroids=face_centroids,
            summary=summary,
        )

        if self.config.save_debug_crops:
            self.debug_crop_saver.save(
                video_path=item["video_path"],
                detections=all_detections,
                output_dir=item["output_dir"],
                max_crops_per_face_id=self.config.max_debug_crops_per_face_id,
            )

        return {
            "video_name": item["video_name"],
            "video_path": item["video_path"],
            "shots_json_path": item["shots_json_path"],
            "output_dir": item["output_dir"],
            "summary": summary,
            "output_paths": output_paths,
        }

    def run(self) -> dict[str, Any]:
        items = self.scanner.get_video_items()
        print(f"Found {len(items)} videos to process")

        run_summary: dict[str, Any] = {
            "done": [],
            "skipped": [],
            "failed": [],
        }

        for item in items:
            output_dir = Path(item["output_dir"])
            summary_path = output_dir / "summary.json"

            try:
                if summary_path.exists() and not self.config.overwrite:
                    print(f"[SKIP] {item['video_name']}")
                    run_summary["skipped"].append(item["video_name"])
                    continue

                result = self.process_video_item(item)
                print(
                    f"[DONE] {item['video_name']} | num_shots={result['summary']['num_shots']} | "
                    f"num_faces={result['summary']['num_face_clusters']} | "
                    f"coverage={result['summary']['shot_face_coverage']:.3f}"
                )
                run_summary["done"].append(result)
            except Exception as error:
                print(f"[FAILED] {item['video_name']}: {error}")
                run_summary["failed"].append(
                    {
                        "video_name": item["video_name"],
                        "video_path": item["video_path"],
                        "shots_json_path": item["shots_json_path"],
                        "output_dir": item["output_dir"],
                        "error": repr(error),
                    }
                )

        FileIO.ensure_dir(self.config.output_dir)
        FileIO.save_json(run_summary, Path(self.config.output_dir) / "run_summary.json")
        return run_summary
