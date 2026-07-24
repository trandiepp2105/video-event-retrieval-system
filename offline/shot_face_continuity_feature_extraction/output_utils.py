import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm.auto import tqdm

from .io_utils import FileIO
from .video_utils import VideoFrameReader


class FaceOutputWriter:
    @staticmethod
    def strip_embeddings_from_shots_for_json(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
        clean_shots = []
        for shot in shots:
            shot_out = dict(shot)
            dets_out = []
            for det in shot.get("face_detections", []):
                det_out = {}
                for key, value in det.items():
                    if key == "embedding":
                        continue
                    if isinstance(value, np.generic):
                        value = value.item()
                    det_out[key] = value
                dets_out.append(det_out)
            shot_out["face_detections"] = dets_out
            clean_shots.append(shot_out)
        return clean_shots

    @staticmethod
    def save_face_centroids(centroids: list[np.ndarray], output_dir: str | Path):
        output_dir = Path(output_dir)
        FileIO.ensure_dir(output_dir)
        arr = np.stack(centroids).astype(np.float32) if len(centroids) > 0 else np.empty((0, 0), dtype=np.float32)
        np.save(output_dir / "face_centroids.npy", arr)

    def write_video_outputs(
        self,
        output_dir: str | Path,
        shots: list[dict[str, Any]],
        face_boundary_scores: list[dict[str, Any]],
        face_centroids: list[np.ndarray],
        summary: dict[str, Any],
    ) -> dict[str, str]:
        output_dir = Path(output_dir)
        FileIO.ensure_dir(output_dir)

        shots_path = output_dir / "shots_with_faces.json"
        boundaries_path = output_dir / "face_boundary_scores.json"
        centroids_path = output_dir / "face_centroids.npy"
        summary_path = output_dir / "summary.json"

        FileIO.save_json(self.strip_embeddings_from_shots_for_json(shots), shots_path)
        FileIO.save_json(face_boundary_scores, boundaries_path)
        self.save_face_centroids(face_centroids, output_dir)
        FileIO.save_json(summary, summary_path)

        return {
            "shots_with_faces_json": str(shots_path),
            "face_boundary_scores_json": str(boundaries_path),
            "face_centroids_npy": str(centroids_path),
            "summary_json": str(summary_path),
        }


class FaceDebugCropSaver:
    @staticmethod
    def expand_bbox(bbox, image_w, image_h, scale=1.25):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = (x2 - x1) * scale
        h = (y2 - y1) * scale
        nx1 = int(max(0, round(cx - w / 2.0)))
        ny1 = int(max(0, round(cy - h / 2.0)))
        nx2 = int(min(image_w - 1, round(cx + w / 2.0)))
        ny2 = int(min(image_h - 1, round(cy + h / 2.0)))
        return nx1, ny1, nx2, ny2

    def save(
        self,
        video_path: str,
        detections: list[dict[str, Any]],
        output_dir: str | Path,
        max_crops_per_face_id: int = 24,
    ):
        output_dir = Path(output_dir)
        debug_dir = output_dir / "debug_faces"

        if debug_dir.exists():
            shutil.rmtree(debug_dir)
        FileIO.ensure_dir(debug_dir)

        face_to_dets = defaultdict(list)
        for det in detections:
            if "face_id" in det:
                face_to_dets[int(det["face_id"])].append(det)

        reader = VideoFrameReader(video_path)
        try:
            for face_id, dets in tqdm(face_to_dets.items(), desc="Save debug face crops"):
                face_dir = debug_dir / f"face_id_{face_id:04d}"
                FileIO.ensure_dir(face_dir)

                if len(dets) > max_crops_per_face_id:
                    indices = np.linspace(0, len(dets) - 1, max_crops_per_face_id).astype(int).tolist()
                    selected = [dets[i] for i in indices]
                else:
                    selected = dets

                for idx, det in enumerate(selected):
                    frame_rgb = reader.read_frame_at_time(det["time_sec"])
                    if frame_rgb is None:
                        continue

                    h, w = frame_rgb.shape[:2]
                    x1, y1, x2, y2 = self.expand_bbox(det["bbox"], w, h, scale=1.35)
                    crop_rgb = frame_rgb[y1:y2, x1:x2]
                    if crop_rgb.size == 0:
                        continue

                    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                    filename = f"{idx:03d}_shot_{int(det['shot_id']):04d}_t_{det['time_sec']:.2f}.jpg"
                    cv2.imwrite(str(face_dir / filename), crop_bgr)
        finally:
            reader.close()
