import importlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .config import PipelineConfig
from .text_utils import choose_better_text, is_vi_en_candidate, normalize_text_for_match, similarity


def bbox_angle_deg(poly: np.ndarray) -> float:
    poly = np.array(poly, dtype=np.float32)
    left_mid = (poly[0] + poly[3]) / 2.0
    right_mid = (poly[1] + poly[2]) / 2.0
    angle = float(np.degrees(np.arctan2(
        right_mid[1] - left_mid[1],
        right_mid[0] - left_mid[0],
    )))
    if angle <= -90:
        angle += 180
    elif angle > 90:
        angle -= 180
    return angle


class SubtitleDetector:
    def __init__(self, config: PipelineConfig) -> None:
        from paddleocr import PaddleOCR

        det_model_dir = Path(config.paddle_text_detection_model_dir)
        rec_model_dir = Path(config.paddle_text_recognition_model_dir)
        if not det_model_dir.exists():
            raise FileNotFoundError(f"Khong tim thay PaddleOCR detection model dir: {det_model_dir}")
        if not rec_model_dir.exists():
            raise FileNotFoundError(f"Khong tim thay PaddleOCR recognition model dir: {rec_model_dir}")

        self.detector = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="vi",
            device=config.paddle_device,
            text_detection_model_dir=str(det_model_dir),
            text_recognition_model_dir=str(rec_model_dir),
        )
        self.score_threshold = config.detector_score_threshold
        self.max_boxes = config.detector_max_boxes
        self.max_subtitle_angle_deg = config.max_subtitle_angle_deg
        self.min_scene_score = config.detector_min_scene_score

    def detect_batch(
        self,
        images: list[np.ndarray],
        *,
        roi_x: tuple[float, float],
        roi_y: tuple[float, float],
    ) -> tuple[list[list[dict]], list[list[dict]]]:
        if not images:
            return [], []

        raw_results = self.detector.predict(images)
        all_subtitle: list[list[dict]] = []
        all_scene: list[list[dict]] = []

        for idx, res in enumerate(raw_results):
            img = images[idx]
            h, w = img.shape[:2]
            x_min_roi = int(w * roi_x[0])
            x_max_roi = int(w * roi_x[1])
            y_min_roi = int(h * roi_y[0])
            y_max_roi = int(h * roi_y[1])

            rec_boxes = res.get("rec_boxes", [])
            rec_texts = res.get("rec_texts", [])
            rec_scores = res.get("rec_scores", [])
            dt_polys = res.get("dt_polys", None)

            subtitle_boxes = []
            scene_boxes = []
            for i in range(len(rec_boxes)):
                text = rec_texts[i]
                score = float(rec_scores[i])
                box = rec_boxes[i]
                x1, y1, x2, y2 = map(int, box)
                angle_deg = 0.0
                if dt_polys is not None and i < len(dt_polys):
                    angle_deg = bbox_angle_deg(dt_polys[i])
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                in_roi = x_min_roi <= cx <= x_max_roi and y_min_roi <= cy <= y_max_roi
                is_horizontal = abs(angle_deg) <= self.max_subtitle_angle_deg
                vi_en = is_vi_en_candidate(text, score, min_score=self.score_threshold)
                item = {
                    "box": (x1, y1, x2, y2),
                    "text": text,
                    "score": score,
                    "angle_deg": float(angle_deg),
                }
                if in_roi and is_horizontal and vi_en:
                    subtitle_boxes.append(item)
                elif score >= self.min_scene_score:
                    scene_boxes.append(item)

            subtitle_boxes.sort(key=lambda b: (b["box"][1], b["box"][0]))
            if self.max_boxes is not None and len(subtitle_boxes) > self.max_boxes:
                extra = subtitle_boxes[self.max_boxes:]
                scene_boxes.extend(extra)
                subtitle_boxes = subtitle_boxes[: self.max_boxes]

            scene_boxes.sort(key=lambda b: (b["box"][1], b["box"][0]))
            all_subtitle.append(subtitle_boxes)
            all_scene.append(scene_boxes)

        return all_subtitle, all_scene


class CropProcessor:
    def __init__(self, padding: int = 5) -> None:
        self.padding = int(padding)

    def crop(self, img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
        h_img, w_img = img.shape[:2]
        x1, y1, x2, y2 = box
        x1 = max(0, x1 - self.padding)
        y1 = max(0, y1 - self.padding)
        x2 = min(w_img, x2 + self.padding)
        y2 = min(h_img, y2 + self.padding)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop


class VietOCRBatch:
    def __init__(self, config: PipelineConfig) -> None:
        weights_path = Path(config.vietocr_weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"Khong tim thay VietOCR weights: {weights_path}")

        if config.vietocr_repo_path is not None:
            repo_path = str(config.vietocr_repo_path)
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)
            importlib.invalidate_caches()

        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        cfg = Cfg.load_config_from_name("vgg_seq2seq")
        cfg["device"] = config.vietocr_device if torch.cuda.is_available() else "cpu"
        cfg["weights"] = str(weights_path)
        self.predictor = Predictor(cfg)

    def predict(self, imgs: list[np.ndarray]) -> list[str]:
        pil_imgs = [
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            for img in imgs
        ]
        return self.predictor.predict_batch(pil_imgs)


class SubtitleMerger:
    def __init__(self, similarity_threshold: float = 0.8, max_gap: int = 12) -> None:
        self.similarity_threshold = float(similarity_threshold)
        self.max_gap = int(max_gap)

    def merge(self, frame_texts: dict[int, str]) -> list[tuple[int, int, str]]:
        frames = sorted(frame_texts.keys())
        results = []
        current_raw_text = None
        current_match_text = None
        start_frame = None
        prev_frame = None

        for frame_id in frames:
            raw_text = frame_texts[frame_id].strip()
            if not raw_text:
                continue
            match_text = normalize_text_for_match(raw_text)
            if current_raw_text is None:
                current_raw_text = raw_text
                current_match_text = match_text
                start_frame = frame_id
                prev_frame = frame_id
                continue
            sim = similarity(match_text, current_match_text)
            if sim >= self.similarity_threshold and (frame_id - prev_frame) <= self.max_gap:
                prev_frame = frame_id
                current_raw_text = choose_better_text(current_raw_text, raw_text)
                current_match_text = normalize_text_for_match(current_raw_text)
            else:
                results.append((start_frame, prev_frame, current_raw_text))
                start_frame = frame_id
                prev_frame = frame_id
                current_raw_text = raw_text
                current_match_text = match_text

        if current_raw_text is not None:
            results.append((start_frame, prev_frame, current_raw_text))
        return results


class SubtitleJSONWriter:
    def write(self, subs: list[tuple[int, int, str]], output: str | Path, fps: float) -> None:
        payload = [
            {
                "frame_start": int(start),
                "frame_end": int(end),
                "start_time_sec": float(start) / max(float(fps), 1e-6),
                "end_time_sec": float(end) / max(float(fps), 1e-6),
                "text": text,
            }
            for start, end, text in subs
        ]
        with Path(output).open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)


class SceneTextWriter:
    def write(self, records: list[dict], output: str | Path) -> None:
        grouped: dict[int, list[dict]] = defaultdict(list)
        frame_times: dict[int, float] = {}
        for rec in records:
            frame_id = rec["frame_id"]
            frame_times[frame_id] = float(rec["time_sec"])
            grouped[frame_id].append(
                {
                    "box": rec["box"],
                    "paddle_text": rec["paddle_text"],
                    "paddle_score": rec["paddle_score"],
                    "vietocr_text": rec["vietocr_text"],
                }
            )
        payload = [
            {
                "frame_id": frame_id,
                "time_sec": frame_times[frame_id],
                "items": grouped[frame_id],
            }
            for frame_id in sorted(grouped.keys())
        ]
        with Path(output).open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)


def find_logo_positions(
    all_scene_boxes: list[list[dict]],
    total_frames: int,
    pos_threshold: int = 30,
    min_frame_ratio: float = 0.3,
) -> list[tuple[float, float]]:
    if total_frames == 0:
        return []
    clusters: list[dict] = []
    for frame_idx, boxes in enumerate(all_scene_boxes):
        for item in boxes:
            x1, y1, x2, y2 = item["box"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            matched = False
            for cluster in clusters:
                dist = math.sqrt((cx - cluster["cx"]) ** 2 + (cy - cluster["cy"]) ** 2)
                if dist <= pos_threshold:
                    cluster["frame_ids"].add(frame_idx)
                    n = len(cluster["frame_ids"])
                    cluster["cx"] = cluster["cx"] + (cx - cluster["cx"]) / n
                    cluster["cy"] = cluster["cy"] + (cy - cluster["cy"]) / n
                    matched = True
                    break
            if not matched:
                clusters.append({"cx": cx, "cy": cy, "frame_ids": {frame_idx}})

    min_frames = total_frames * min_frame_ratio
    return [
        (cluster["cx"], cluster["cy"])
        for cluster in clusters
        if len(cluster["frame_ids"]) >= min_frames
    ]


def is_logo_bbox(box: tuple[int, int, int, int], logo_centers: list[tuple[float, float]], pos_threshold: int = 30) -> bool:
    if not logo_centers:
        return False
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    for lcx, lcy in logo_centers:
        dist = math.sqrt((cx - lcx) ** 2 + (cy - lcy) ** 2)
        if dist <= pos_threshold:
            return True
    return False
