from typing import Any

import cv2
import numpy as np
from tqdm.auto import tqdm

from .video_utils import ShotFrameSampler, VideoFrameReader

try:
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
except Exception as error:
    FaceAnalysis = None
    _IMPORT_ERROR = error
else:
    _IMPORT_ERROR = None


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x) + eps)


class InsightFaceExtractor:
    def __init__(
        self,
        model_name: str = "buffalo_l",
        root_dir: str | None = None,
        ctx_id: int = 0,
        det_size: tuple[int, int] = (640, 640),
        det_score_thresh: float = 0.5,
        min_face_size: int = 40,
    ):
        if FaceAnalysis is None:
            raise RuntimeError(
                "InsightFace chua duoc cai hoac import loi. "
                f"Chi tiet: {repr(_IMPORT_ERROR)}"
            )

        kwargs = {
            "name": model_name,
            "allowed_modules": ["detection", "recognition"],
        }
        if root_dir is not None:
            kwargs["root"] = root_dir

        self.app = FaceAnalysis(**kwargs)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.det_model = self.app.models["detection"]
        self.rec_model = self.app.models["recognition"]
        self.det_score_thresh = float(det_score_thresh)
        self.min_face_size = int(min_face_size)

    def detect(self, frame_rgb: np.ndarray, time_sec: float) -> list[dict[str, Any]]:
        bboxes, kpss = self.det_model.detect(frame_rgb, max_num=0, metric="default")
        outputs = []
        h_img, w_img = frame_rgb.shape[:2]

        if bboxes.shape[0] == 0:
            return outputs

        for index in range(bboxes.shape[0]):
            x1, y1, x2, y2 = bboxes[index, 0:4].astype(float).tolist()
            w = x2 - x1
            h = y2 - y1
            det_score = float(bboxes[index, 4])

            if det_score < self.det_score_thresh:
                continue
            if min(w, h) < self.min_face_size:
                continue

            x1c = float(np.clip(x1, 0, w_img - 1))
            y1c = float(np.clip(y1, 0, h_img - 1))
            x2c = float(np.clip(x2, 0, w_img - 1))
            y2c = float(np.clip(y2, 0, h_img - 1))

            kps = None
            if kpss is not None:
                kps = np.asarray(kpss[index], dtype=np.float32)

            outputs.append(
                {
                    "time_sec": float(time_sec),
                    "bbox": [x1c, y1c, x2c, y2c],
                    "det_score": det_score,
                    "kps": None if kps is None else kps.tolist(),
                }
            )

        return outputs

    def build_face_crop(self, frame_rgb: np.ndarray, detection: dict[str, Any]) -> np.ndarray:
        kps = detection.get("kps")
        if kps is not None:
            kps_array = np.asarray(kps, dtype=np.float32)
            return face_align.norm_crop(
                frame_rgb,
                landmark=kps_array,
                image_size=self.rec_model.input_size[0],
            )

        x1, y1, x2, y2 = [int(round(value)) for value in detection["bbox"]]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = max(x1 + 1, x2)
        y2 = max(y1 + 1, y2)
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(
                (self.rec_model.input_size[1], self.rec_model.input_size[0], 3),
                dtype=np.uint8,
            )
        return cv2.resize(crop, self.rec_model.input_size)

    def recognize_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        if len(crops) == 0:
            return np.empty((0, 0), dtype=np.float32)
        embeddings = self.rec_model.get_feat(crops).astype(np.float32)
        return embeddings


class FaceDetectionProcessor:
    def __init__(
        self,
        frame_sampler: ShotFrameSampler,
        face_extractor: InsightFaceExtractor,
        recognition_batch_size: int = 64,
    ):
        self.frame_sampler = frame_sampler
        self.face_extractor = face_extractor
        self.recognition_batch_size = int(recognition_batch_size)

    def _flush_recognition_batch(
        self,
        pending_crops: list[np.ndarray],
        pending_detections: list[dict[str, Any]],
    ):
        if len(pending_crops) == 0:
            return
        embeddings = self.face_extractor.recognize_batch(pending_crops)
        for detection, embedding in zip(pending_detections, embeddings):
            detection["embedding"] = l2_normalize(np.asarray(embedding, dtype=np.float32))
        pending_crops.clear()
        pending_detections.clear()

    def extract_for_shots(
        self,
        video_path: str,
        shots: list[dict[str, Any]],
        show_progress: bool = True,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        reader = VideoFrameReader(video_path)
        all_detections: list[dict[str, Any]] = []
        num_sampled_frames = 0
        pending_crops: list[np.ndarray] = []
        pending_detections: list[dict[str, Any]] = []

        try:
            iterator = enumerate(shots)
            if show_progress:
                iterator = tqdm(iterator, total=len(shots), desc="Extract faces per shot")

            for shot_idx, shot in iterator:
                times = self.frame_sampler.sample_times(shot["start_time_sec"], shot["end_time_sec"])
                shot_detections = []
                num_sampled_frames += len(times)

                for time_sec in times:
                    frame_rgb = reader.read_frame_at_time(time_sec)
                    if frame_rgb is None:
                        continue

                    faces = self.face_extractor.detect(frame_rgb, time_sec=time_sec)
                    for det in faces:
                        det["shot_idx"] = int(shot_idx)
                        det["shot_id"] = int(shot["shot_id"])
                        crop = self.face_extractor.build_face_crop(frame_rgb, det)
                        pending_crops.append(crop)
                        pending_detections.append(det)

                        if len(pending_crops) >= self.recognition_batch_size:
                            self._flush_recognition_batch(pending_crops, pending_detections)

                    shot_detections.extend(faces)
                    all_detections.extend(faces)

                shot["face_detections"] = shot_detections
            self._flush_recognition_batch(pending_crops, pending_detections)
        finally:
            reader.close()

        stats = {
            "num_sampled_frames": int(num_sampled_frames),
            "num_face_detections": int(len(all_detections)),
            "num_shots_with_face": int(sum(len(s.get("face_detections", [])) > 0 for s in shots)),
        }
        return shots, all_detections, stats
