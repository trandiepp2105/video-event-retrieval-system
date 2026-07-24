from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .common import FileIO, MathUtils, PoolingUtils
from .config import EventGroupingDatasetConfig


class FeatureRepository:
    def __init__(self, config: EventGroupingDatasetConfig):
        self.config = config
        self.features_dir = Path(config.features_dir)

    def load_video_features(self, video_id: str) -> Dict[str, Any]:
        return {
            "visual": FileIO.load_pickle(self.features_dir / "visual_embeddings" / f"{video_id}.pkl"),
            "action": FileIO.load_pickle(self.features_dir / "action_features" / f"{video_id}.pkl"),
            "subtitle": FileIO.load_pickle(self.features_dir / "subtitle_embeddings" / f"{video_id}.pkl"),
            "face": FileIO.load_json(self.features_dir / "face_detection" / video_id / "shots_with_faces.json"),
        }


class VideoFeatureAssembler:
    def __init__(self, config: EventGroupingDatasetConfig):
        self.config = config

    def _build_visual_embedding(self, shot: Dict[str, Any]) -> Tuple[Optional[np.ndarray], bool]:
        if "embedding" in shot and shot["embedding"] is not None:
            visual_emb = MathUtils.l2_normalize(np.asarray(shot["embedding"], dtype=np.float32), axis=-1)
            return visual_emb, True

        keyframe_embeddings = np.asarray(shot.get("keyframe_embeddings"), dtype=np.float32)
        if keyframe_embeddings.size == 0:
            return None, False
        if keyframe_embeddings.ndim == 1:
            keyframe_embeddings = keyframe_embeddings.reshape(1, -1)

        keyframe_vectors = [keyframe_embeddings[i] for i in range(keyframe_embeddings.shape[0])]
        pooled = PoolingUtils.softmax_center_pool(
            keyframe_vectors,
            temperature=self.config.event_softmax_temperature,
        )
        if pooled is None:
            return None, False
        return pooled.astype(np.float32), True

    def _build_action_by_id(self, action_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        action_by_id = {}
        for shot in action_data["shots"]:
            shot_id = int(shot["shot_id"])
            action_by_id[shot_id] = {
                "embedding": MathUtils.l2_normalize(np.asarray(shot["action_feature"], dtype=np.float32), axis=-1),
                "metadata": {
                    "num_subshots": int(shot.get("num_subshots", 1)),
                    "pooling": shot.get("pooling", None),
                },
            }
        return action_by_id

    def _build_face_by_id(self, face_data: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        face_by_id = {}
        for shot in face_data:
            shot_id = int(shot["shot_id"])
            face_by_id[shot_id] = {
                "face_ids": [int(x) for x in shot.get("face_ids", [])],
                "face_id_counts": {str(k): int(v) for k, v in shot.get("face_id_counts", {}).items()},
                "num_face_detections": len(shot.get("face_detections", [])),
            }
        return face_by_id

    def _build_subtitle_items(self, subtitle_data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
        subtitle_items_raw = subtitle_data.get("items", [])
        subtitle_embeddings = np.asarray(subtitle_data.get("embeddings"), dtype=np.float32)
        if subtitle_embeddings.ndim == 1:
            subtitle_embeddings = subtitle_embeddings.reshape(1, -1)
        subtitle_embeddings = MathUtils.l2_normalize(subtitle_embeddings, axis=1)

        subtitle_items = []
        for idx, item in enumerate(subtitle_items_raw):
            if idx >= len(subtitle_embeddings):
                break
            start_time_sec = float(item.get("start_time_sec", 0.0))
            end_time_sec = float(item.get("end_time_sec", start_time_sec))
            if end_time_sec < start_time_sec:
                end_time_sec = start_time_sec
            text = str(item.get("text", "")).strip()
            subtitle_items.append({
                "subtitle_index": int(item.get("subtitle_index", idx)),
                "start_time_sec": start_time_sec,
                "end_time_sec": end_time_sec,
                "text": text,
                "embedding": subtitle_embeddings[idx],
            })

        subtitle_dim = int(subtitle_embeddings.shape[1]) if subtitle_embeddings.size > 0 else 0
        return subtitle_items, subtitle_dim

    def _build_shot_subtitle_feature(self, subtitle_items: List[Dict[str, Any]], start_time_sec: float, end_time_sec: float) -> Tuple[Optional[np.ndarray], bool, str]:
        overlap_texts = []
        overlap_embeddings = []
        overlap_weights = []
        seen_text = set()
        for item in subtitle_items:
            ov = MathUtils.overlap_len(start_time_sec, end_time_sec, item["start_time_sec"], item["end_time_sec"])
            if ov <= 0:
                continue
            overlap_embeddings.append(item["embedding"])
            overlap_weights.append(float(ov))
            if item["text"] and item["text"] not in seen_text:
                overlap_texts.append(item["text"])
                seen_text.add(item["text"])

        if len(overlap_embeddings) == 0:
            return None, False, ""

        E = np.stack(overlap_embeddings, axis=0).astype(np.float32)
        w = np.asarray(overlap_weights, dtype=np.float32)
        w = w / (w.sum() + 1e-8)
        subtitle_emb = (E * w[:, None]).sum(axis=0)
        subtitle_emb = subtitle_emb / (np.linalg.norm(subtitle_emb) + 1e-8)
        return subtitle_emb.astype(np.float32), True, " ".join(overlap_texts).strip()

    def build(self, video_features: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, float]:
        visual_data = video_features["visual"]
        action_by_id = self._build_action_by_id(video_features["action"])
        face_by_id = self._build_face_by_id(video_features["face"])
        subtitle_items, subtitle_dim = self._build_subtitle_items(video_features["subtitle"])

        shot_table = []
        fps = float(visual_data.get("fps", 30.0))
        if fps <= 0:
            fps = 30.0

        for order_idx, shot in enumerate(visual_data["shots"]):
            shot_id = int(shot["shot_id"])
            start_frame = int(shot["start_frame"])
            end_frame = int(shot["end_frame"])
            start_time_sec = float(shot["start_time_sec"])
            end_time_sec = float(shot["end_time_sec"])
            subtitle_emb, subtitle_valid, subtitle_text = self._build_shot_subtitle_feature(subtitle_items, start_time_sec, end_time_sec)
            visual_emb, visual_valid = self._build_visual_embedding(shot)

            shot_table.append({
                "order_idx": int(order_idx),
                "shot_id": shot_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_sec": float(shot["start_time_sec"]),
                "end_time_sec": float(shot["end_time_sec"]),
                "duration_sec": float(shot["duration_sec"]),
                "visual_valid": visual_valid,
                "visual_emb": visual_emb,
                "action_valid": shot_id in action_by_id,
                "action_emb": action_by_id[shot_id]["embedding"] if shot_id in action_by_id else None,
                "subtitle_valid": subtitle_valid,
                "subtitle_emb": subtitle_emb,
                "subtitle_text": subtitle_text,
                "face_valid": shot_id in face_by_id and len(face_by_id[shot_id]["face_ids"]) > 0,
                "face_ids": face_by_id[shot_id]["face_ids"] if shot_id in face_by_id else [],
                "face_id_counts": face_by_id[shot_id]["face_id_counts"] if shot_id in face_by_id else {},
            })

        return shot_table, subtitle_items, subtitle_dim, fps
