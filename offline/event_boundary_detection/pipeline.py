import json
import math
import pickle
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


@dataclass
class EventGroupingDatasetConfig:
    features_dir: str = "./features"
    output_root_dir: str = "./event_output"
    video_ids: Optional[List[str]] = None
    start_index: int = 0
    end_index: Optional[int] = None
    skip_missing_modalities: bool = True
    context_window: int = 3
    subtitle_use_recency_weight: bool = True
    subtitle_recency_tau: float = 2.0
    subtitle_bridge_penalty_weight: float = 0.40
    subtitle_bridge_norm_sec: float = 0.50
    use_face_recency_weight: bool = True
    face_recency_tau: float = 2.0
    visual_weight: float = 0.30
    action_weight: float = 0.15
    subtitle_weight: float = 0.40
    face_weight: float = 0.15
    boundary_percentile: float = 85.0
    use_local_peak: bool = True
    min_event_duration_sec: float = 3.0
    max_event_duration_sec: float = 30.0
    cut_penalty: float = 0.55
    non_candidate_penalty: float = 0.25
    event_softmax_temperature: float = 15.0
    overwrite: bool = False


class FileIO:
    @staticmethod
    def ensure_dir(path: str | Path):
        Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_json(path: str | Path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_json(data: Any, path: str | Path):
        path = Path(path)
        FileIO.ensure_dir(path.parent)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_pickle(path: str | Path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def save_npy(array: np.ndarray, path: str | Path):
        path = Path(path)
        FileIO.ensure_dir(path.parent)
        np.save(path, array)


class MathUtils:
    @staticmethod
    def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        return float(np.dot(a, b))

    @staticmethod
    def overlap_len(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
        return max(0.0, min(end_a, end_b) - max(start_a, start_b))


class PoolingUtils:
    @staticmethod
    def mean_pool(vectors: List[np.ndarray]) -> Optional[np.ndarray]:
        vectors = [np.asarray(v, dtype=np.float32) for v in vectors if v is not None]
        if len(vectors) == 0:
            return None
        x = np.stack(vectors, axis=0)
        v = x.mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-8)
        return v.astype(np.float32)

    @staticmethod
    def max_pool(vectors: List[np.ndarray]) -> Optional[np.ndarray]:
        vectors = [np.asarray(v, dtype=np.float32) for v in vectors if v is not None]
        if len(vectors) == 0:
            return None
        x = np.stack(vectors, axis=0)
        v = x.max(axis=0)
        v = v / (np.linalg.norm(v) + 1e-8)
        return v.astype(np.float32)

    @staticmethod
    def softmax_center_pool(vectors: List[np.ndarray], temperature: float = 15.0) -> Optional[np.ndarray]:
        vectors = [np.asarray(v, dtype=np.float32) for v in vectors if v is not None]
        if len(vectors) == 0:
            return None
        E = np.stack(vectors, axis=0).astype(np.float32)
        E = MathUtils.l2_normalize(E, axis=1)
        center = E.mean(axis=0)
        center = center / (np.linalg.norm(center) + 1e-8)
        sims = E @ center
        logits = sims * float(temperature)
        logits = logits - np.max(logits)
        weights = np.exp(logits)
        weights = weights / (weights.sum() + 1e-8)
        pooled = (E * weights[:, None]).sum(axis=0)
        pooled = pooled / (np.linalg.norm(pooled) + 1e-8)
        return pooled.astype(np.float32)

    @staticmethod
    def multi_pool(vectors: List[np.ndarray], temperature: float = 15.0) -> Dict[str, Optional[np.ndarray]]:
        return {
            "mean": PoolingUtils.mean_pool(vectors),
            "max": PoolingUtils.max_pool(vectors),
            "softmax": PoolingUtils.softmax_center_pool(vectors, temperature=temperature),
        }


class DatasetPreflight:
    def __init__(self, config: EventGroupingDatasetConfig):
        self.config = config
        self.features_dir = Path(config.features_dir)

    def _list_ids_from_files(self, directory: Path, suffix: str = ".pkl") -> set[str]:
        if not directory.exists():
            return set()
        return {path.stem for path in directory.glob(f"*{suffix}")}

    def _list_ids_from_dirs(self, directory: Path) -> set[str]:
        if not directory.exists():
            return set()
        return {path.name for path in directory.iterdir() if path.is_dir()}

    def build_manifest(self) -> Dict[str, Any]:
        visual_ids = self._list_ids_from_files(self.features_dir / "visual_embeddings")
        action_ids = self._list_ids_from_files(self.features_dir / "action_features")
        subtitle_ids = self._list_ids_from_files(self.features_dir / "subtitle_embeddings")
        face_ids = self._list_ids_from_dirs(self.features_dir / "face_detection")

        all_ids = sorted(visual_ids | action_ids | subtitle_ids | face_ids, key=lambda x: int(x))
        if self.config.video_ids is not None:
            requested = {str(x) for x in self.config.video_ids}
            all_ids = [video_id for video_id in all_ids if video_id in requested]
        else:
            start = max(0, int(self.config.start_index))
            if start >= len(all_ids):
                all_ids = []
            elif self.config.end_index is None:
                all_ids = all_ids[start:]
            else:
                all_ids = all_ids[start : self.config.end_index + 1]

        missing_rows = []
        eligible_ids = []
        for video_id in all_ids:
            missing = []
            if video_id not in visual_ids:
                missing.append("visual")
            if video_id not in action_ids:
                missing.append("action")
            if video_id not in subtitle_ids:
                missing.append("subtitle")
            if video_id not in face_ids:
                missing.append("face")
            if len(missing) == 0:
                eligible_ids.append(video_id)
            else:
                missing_rows.append({"video_id": video_id, "missing": ", ".join(missing)})

        return {
            "visual_count": len(visual_ids),
            "action_count": len(action_ids),
            "subtitle_count": len(subtitle_ids),
            "face_count": len(face_ids),
            "all_ids": all_ids,
            "eligible_ids": eligible_ids,
            "missing_rows": missing_rows,
        }

    def summarize(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "visual_count": manifest["visual_count"],
            "action_count": manifest["action_count"],
            "subtitle_count": manifest["subtitle_count"],
            "face_count": manifest["face_count"],
            "eligible_count": len(manifest["eligible_ids"]),
            "missing_count": len(manifest["missing_rows"]),
        }


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


class BoundaryScorer:
    def __init__(self, config: EventGroupingDatasetConfig):
        self.config = config

    def collect_context_indices(self, n: int, boundary_idx: int) -> Tuple[List[int], List[int]]:
        window = self.config.context_window
        left_start = max(0, boundary_idx - window + 1)
        left_end = boundary_idx + 1
        right_start = boundary_idx + 1
        right_end = min(n, boundary_idx + 1 + window)
        return list(range(left_start, left_end)), list(range(right_start, right_end))

    def embedding_change_for_boundary(self, shot_table: List[Dict[str, Any]], boundary_idx: int, key: str, valid_key: str) -> Tuple[float, bool]:
        n = len(shot_table)
        left_indices, right_indices = self.collect_context_indices(n, boundary_idx)
        left_vecs = [shot_table[j][key] for j in left_indices if shot_table[j].get(valid_key, False) and shot_table[j][key] is not None]
        right_vecs = [shot_table[j][key] for j in right_indices if shot_table[j].get(valid_key, False) and shot_table[j][key] is not None]
        if len(left_vecs) == 0 or len(right_vecs) == 0:
            return 0.0, False
        left_emb = PoolingUtils.mean_pool(left_vecs)
        right_emb = PoolingUtils.mean_pool(right_vecs)
        sim = MathUtils.cosine(left_emb, right_emb)
        return float(1.0 - sim), True

    def collect_face_hist_from_context(self, shot_table: List[Dict[str, Any]], indices: List[int], side: str) -> Counter:
        hist = Counter()
        if len(indices) == 0:
            return hist
        for pos, j in enumerate(indices):
            counts = shot_table[j].get("face_id_counts", {})
            if not counts:
                continue
            if self.config.use_face_recency_weight:
                if side == "left":
                    dist = (len(indices) - 1) - pos
                else:
                    dist = pos
                shot_weight = float(np.exp(-dist / self.config.face_recency_tau))
            else:
                shot_weight = 1.0
            for face_id_str, count in counts.items():
                face_id = int(face_id_str)
                hist[face_id] += float(count) * shot_weight
        return hist

    def weighted_jaccard_hist(self, a: Counter, b: Counter) -> Optional[float]:
        ids = set(a.keys()) | set(b.keys())
        if len(ids) == 0:
            return None
        numerator = 0.0
        denominator = 0.0
        for face_id in ids:
            av = float(a.get(face_id, 0.0))
            bv = float(b.get(face_id, 0.0))
            numerator += min(av, bv)
            denominator += max(av, bv)
        if denominator <= 1e-8:
            return None
        return numerator / denominator

    def face_change_for_boundary(self, shot_table: List[Dict[str, Any]], boundary_idx: int) -> Tuple[float, bool, Dict[str, Any]]:
        n = len(shot_table)
        left_indices, right_indices = self.collect_context_indices(n, boundary_idx)
        left_hist = self.collect_face_hist_from_context(shot_table, left_indices, side="left")
        right_hist = self.collect_face_hist_from_context(shot_table, right_indices, side="right")
        overlap = self.weighted_jaccard_hist(left_hist, right_hist)
        if overlap is None:
            return 0.0, False, {"left_face_hist": {}, "right_face_hist": {}, "face_overlap": None}
        return float(1.0 - overlap), True, {
            "left_face_hist": {str(k): float(v) for k, v in sorted(left_hist.items())},
            "right_face_hist": {str(k): float(v) for k, v in sorted(right_hist.items())},
            "face_overlap": float(overlap),
        }

    def build_subtitle_context_embedding(self, subtitle_items: List[Dict[str, Any]], range_start_sec: float, range_end_sec: float, boundary_time_sec: float, side: str) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
        weighted_vectors = []
        debug_items = []
        for item in subtitle_items:
            item_start_sec = float(item["start_time_sec"])
            item_end_sec = float(item["end_time_sec"])
            ov = MathUtils.overlap_len(range_start_sec, range_end_sec, item_start_sec, item_end_sec)
            if ov <= 0:
                continue

            if self.config.subtitle_use_recency_weight:
                if side == "left":
                    anchor = min(item_end_sec, boundary_time_sec)
                    dist = max(0.0, boundary_time_sec - anchor)
                else:
                    anchor = max(item_start_sec, boundary_time_sec)
                    dist = max(0.0, anchor - boundary_time_sec)
                recency = float(np.exp(-dist / self.config.subtitle_recency_tau))
            else:
                recency = 1.0

            weight = float(ov * recency)
            if weight <= 0:
                continue
            weighted_vectors.append((item["embedding"], weight))
            debug_items.append({
                "subtitle_index": int(item["subtitle_index"]),
                "text": item["text"],
                "item_start_sec": round(item_start_sec, 3),
                "item_end_sec": round(item_end_sec, 3),
                "overlap_sec": round(ov, 3),
                "weight": round(weight, 6),
            })

        if len(weighted_vectors) == 0:
            return None, []

        E = np.stack([v for v, _ in weighted_vectors], axis=0).astype(np.float32)
        w = np.asarray([w for _, w in weighted_vectors], dtype=np.float32)
        w = w / (w.sum() + 1e-8)
        emb = (E * w[:, None]).sum(axis=0)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb.astype(np.float32), debug_items

    def subtitle_change_for_boundary(self, shot_table: List[Dict[str, Any]], subtitle_items: List[Dict[str, Any]], boundary_idx: int) -> Tuple[float, bool, Dict[str, Any]]:
        n = len(shot_table)
        left_indices, right_indices = self.collect_context_indices(n, boundary_idx)
        left_start_sec = float(shot_table[left_indices[0]]["start_time_sec"])
        left_end_sec = float(shot_table[left_indices[-1]]["end_time_sec"])
        right_start_sec = float(shot_table[right_indices[0]]["start_time_sec"])
        right_end_sec = float(shot_table[right_indices[-1]]["end_time_sec"])
        boundary_time_sec = float(shot_table[boundary_idx]["end_time_sec"])

        left_emb, left_debug = self.build_subtitle_context_embedding(subtitle_items, left_start_sec, left_end_sec, boundary_time_sec, "left")
        right_emb, right_debug = self.build_subtitle_context_embedding(subtitle_items, right_start_sec, right_end_sec, boundary_time_sec, "right")

        if left_emb is None or right_emb is None:
            return 0.0, False, {
                "left_subtitle_items": left_debug,
                "right_subtitle_items": right_debug,
                "left_subtitle_range_sec": [round(left_start_sec, 3), round(left_end_sec, 3)],
                "right_subtitle_range_sec": [round(right_start_sec, 3), round(right_end_sec, 3)],
            }

        sim = MathUtils.cosine(left_emb, right_emb)
        return float(1.0 - sim), True, {
            "left_subtitle_items": left_debug,
            "right_subtitle_items": right_debug,
            "left_subtitle_range_sec": [round(left_start_sec, 3), round(left_end_sec, 3)],
            "right_subtitle_range_sec": [round(right_start_sec, 3), round(right_end_sec, 3)],
        }

    def subtitle_bridge_penalty_for_boundary(self, shot_table: List[Dict[str, Any]], subtitle_items: List[Dict[str, Any]], boundary_idx: int) -> Dict[str, Any]:
        left_shot = shot_table[boundary_idx]
        right_shot = shot_table[boundary_idx + 1]
        left_start = float(left_shot["start_time_sec"])
        left_end = float(left_shot["end_time_sec"])
        right_start = float(right_shot["start_time_sec"])
        right_end = float(right_shot["end_time_sec"])

        bridge_items = []
        total_bridge_sec = 0.0
        for item in subtitle_items:
            item_start_sec = float(item["start_time_sec"])
            item_end_sec = float(item["end_time_sec"])
            overlap_left = MathUtils.overlap_len(left_start, left_end, item_start_sec, item_end_sec)
            overlap_right = MathUtils.overlap_len(right_start, right_end, item_start_sec, item_end_sec)
            if overlap_left <= 0 or overlap_right <= 0:
                continue
            bridge_strength_sec = min(overlap_left, overlap_right)
            total_bridge_sec += bridge_strength_sec
            bridge_items.append({
                "subtitle_index": int(item["subtitle_index"]),
                "text": item["text"],
                "item_start_sec": round(item_start_sec, 3),
                "item_end_sec": round(item_end_sec, 3),
                "overlap_left_sec": round(overlap_left, 3),
                "overlap_right_sec": round(overlap_right, 3),
                "bridge_strength_sec": round(bridge_strength_sec, 3),
            })

        normalized_bridge_strength = min(1.0, total_bridge_sec / max(self.config.subtitle_bridge_norm_sec, 1e-8))
        bridge_penalty = float(self.config.subtitle_bridge_penalty_weight * normalized_bridge_strength)
        return {
            "subtitle_bridge_items": bridge_items,
            "subtitle_bridge_total_sec": float(total_bridge_sec),
            "subtitle_bridge_strength": float(normalized_bridge_strength),
            "subtitle_bridge_penalty": float(bridge_penalty),
        }

    def compute_boundary_rows(self, shot_table: List[Dict[str, Any]], subtitle_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        n = len(shot_table)
        rows = []
        for i in tqdm(range(n - 1), desc="Compute boundary changes", leave=False):
            left_indices, right_indices = self.collect_context_indices(n, i)
            visual_change, visual_valid = self.embedding_change_for_boundary(shot_table, i, "visual_emb", "visual_valid")
            action_change, action_valid = self.embedding_change_for_boundary(shot_table, i, "action_emb", "action_valid")
            subtitle_change, subtitle_valid, subtitle_extra = self.subtitle_change_for_boundary(shot_table, subtitle_items, i)
            subtitle_bridge_extra = self.subtitle_bridge_penalty_for_boundary(shot_table, subtitle_items, i)
            face_change, face_valid, face_extra = self.face_change_for_boundary(shot_table, i)
            rows.append({
                "boundary_index": int(i),
                "left_shot_id": int(shot_table[i]["shot_id"]),
                "right_shot_id": int(shot_table[i + 1]["shot_id"]),
                "boundary_time_sec": float(shot_table[i]["end_time_sec"]),
                "left_context_range": [int(left_indices[0]), int(left_indices[-1])],
                "right_context_range": [int(right_indices[0]), int(right_indices[-1])],
                "visual_change": float(visual_change),
                "visual_valid": bool(visual_valid),
                "action_change": float(action_change),
                "action_valid": bool(action_valid),
                "subtitle_change": float(subtitle_change),
                "subtitle_valid": bool(subtitle_valid),
                "face_change": float(face_change),
                "face_valid": bool(face_valid),
                **subtitle_extra,
                **subtitle_bridge_extra,
                **face_extra,
            })
        return rows

    def normalize_boundary_rows(self, boundary_rows: List[Dict[str, Any]]) -> None:
        for modality in ["visual", "action", "subtitle", "face"]:
            key = f"{modality}_change"
            valid_key = f"{modality}_valid"
            norm_key = f"{modality}_change_norm"
            values = [float(row[key]) for row in boundary_rows if row.get(valid_key, False)]
            if len(values) == 0:
                for row in boundary_rows:
                    row[norm_key] = 0.0
                continue
            vmin = float(min(values))
            vmax = float(max(values))
            denom = max(vmax - vmin, 1e-8)
            for row in boundary_rows:
                if not row.get(valid_key, False):
                    row[norm_key] = 0.0
                else:
                    row[norm_key] = float((float(row[key]) - vmin) / denom)

    def fuse_boundary_score(self, row: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        weights = {
            "visual": self.config.visual_weight,
            "action": self.config.action_weight,
            "subtitle": self.config.subtitle_weight,
            "face": self.config.face_weight,
        }
        used_weight_sum = sum(w for m, w in weights.items() if row.get(f"{m}_valid", False))
        if used_weight_sum <= 1e-8:
            return 0.0, {}
        score = 0.0
        used_weights = {}
        for modality, w in weights.items():
            if not row.get(f"{modality}_valid", False):
                continue
            effective_w = w / used_weight_sum
            used_weights[modality] = float(effective_w)
            score += effective_w * float(row[f"{modality}_change_norm"])
        return float(score), used_weights

    def score_boundaries(self, boundary_rows: List[Dict[str, Any]]) -> None:
        self.normalize_boundary_rows(boundary_rows)
        for row in boundary_rows:
            score, used_weights = self.fuse_boundary_score(row)
            row["boundary_score"] = score
            row["used_weights"] = used_weights

    def select_candidate_boundaries(self, boundary_rows: List[Dict[str, Any]]) -> List[int]:
        if len(boundary_rows) == 0:
            return []
        scores = np.asarray([row["boundary_score"] for row in boundary_rows], dtype=np.float32)
        threshold = float(np.percentile(scores, self.config.boundary_percentile))
        selected = []
        for i, row in enumerate(boundary_rows):
            score = float(row["boundary_score"])
            is_candidate = score >= threshold
            if is_candidate and self.config.use_local_peak:
                left_score = float(boundary_rows[i - 1]["boundary_score"]) if i > 0 else -1.0
                right_score = float(boundary_rows[i + 1]["boundary_score"]) if i + 1 < len(boundary_rows) else -1.0
                is_candidate = score >= left_score and score >= right_score
            row["is_candidate"] = bool(is_candidate)
            row["candidate_threshold"] = threshold
            if is_candidate:
                selected.append(i)
        return selected


class DPSegmenter:
    def __init__(self, config: EventGroupingDatasetConfig):
        self.config = config

    def event_duration(self, shot_table: List[Dict[str, Any]], start: int, end_exclusive: int) -> float:
        return float(shot_table[end_exclusive - 1]["end_time_sec"] - shot_table[start]["start_time_sec"])

    def is_forced_singleton_long_shot(self, shot_table: List[Dict[str, Any]], shot_index: int) -> bool:
        return float(shot_table[shot_index]["duration_sec"]) > self.config.max_event_duration_sec

    def transition_valid(self, shot_table: List[Dict[str, Any]], start: int, end_exclusive: int) -> bool:
        num_shots = end_exclusive - start
        dur = self.event_duration(shot_table, start, end_exclusive)
        if num_shots == 1 and self.is_forced_singleton_long_shot(shot_table, start):
            return True
        if dur < self.config.min_event_duration_sec:
            return False
        if dur > self.config.max_event_duration_sec:
            return False
        return True

    def cut_reward_after_event(self, end_exclusive: int, n: int, boundary_rows: List[Dict[str, Any]]) -> float:
        if end_exclusive >= n:
            return 0.0
        b_idx = end_exclusive - 1
        row = boundary_rows[b_idx]
        reward = float(row["boundary_score"]) - self.config.cut_penalty
        if not row.get("is_candidate", False):
            reward -= self.config.non_candidate_penalty
        reward -= float(row.get("subtitle_bridge_penalty", 0.0))
        return float(reward)

    def _segment_chunk(self, shot_table: List[Dict[str, Any]], boundary_rows: List[Dict[str, Any]], start_offset: int = 0):
        n = len(shot_table)
        NEG_INF = -1e18
        dp = np.full(n + 1, NEG_INF, dtype=np.float64)
        backptr = [None] * (n + 1)
        dp[0] = 0.0

        for end in range(1, n + 1):
            best_score = NEG_INF
            best_start = None
            for start in range(0, end):
                if dp[start] <= NEG_INF / 2:
                    continue
                if not self.transition_valid(shot_table, start, end):
                    continue
                reward = self.cut_reward_after_event(end, n, boundary_rows)
                score = dp[start] + reward
                if score > best_score:
                    best_score = score
                    best_start = start
            dp[end] = best_score
            backptr[end] = best_start

        if backptr[n] is None:
            raise RuntimeError(
                f"DP không tìm được segmentation hợp lệ cho chunk bắt đầu tại shot index {start_offset}. "
                "Hãy giảm min_event_duration_sec hoặc tăng max_event_duration_sec."
            )

        ranges = []
        cur = n
        while cur > 0:
            prev = backptr[cur]
            if prev is None:
                raise RuntimeError("Backpointer bị lỗi trong lúc reconstruct.")
            ranges.append((prev + start_offset, cur + start_offset))
            cur = prev
        ranges.reverse()
        return ranges

    def segment(self, shot_table: List[Dict[str, Any]], boundary_rows: List[Dict[str, Any]]):
        n = len(shot_table)
        forced_indices = [idx for idx in range(n) if self.is_forced_singleton_long_shot(shot_table, idx)]
        if not forced_indices:
            ranges = self._segment_chunk(shot_table, boundary_rows, start_offset=0)
            return ranges, None, None

        ranges = []
        chunk_start = 0
        pending_prefix_start = None

        for forced_idx in forced_indices:
            if chunk_start < forced_idx:
                chunk_duration = self.event_duration(shot_table, chunk_start, forced_idx)
                if chunk_duration < self.config.min_event_duration_sec:
                    if ranges:
                        prev_start, _ = ranges[-1]
                        ranges[-1] = (prev_start, forced_idx)
                    else:
                        pending_prefix_start = chunk_start
                else:
                    chunk_ranges = self._segment_chunk(
                        shot_table[chunk_start:forced_idx],
                        boundary_rows[chunk_start:forced_idx - 1] if forced_idx - chunk_start >= 2 else [],
                        start_offset=chunk_start,
                    )
                    ranges.extend(chunk_ranges)

            event_start = pending_prefix_start if pending_prefix_start is not None else forced_idx
            ranges.append((event_start, forced_idx + 1))
            pending_prefix_start = None
            chunk_start = forced_idx + 1

        if chunk_start < n:
            chunk_duration = self.event_duration(shot_table, chunk_start, n)
            if chunk_duration < self.config.min_event_duration_sec and ranges:
                prev_start, _ = ranges[-1]
                ranges[-1] = (prev_start, n)
            else:
                chunk_ranges = self._segment_chunk(
                    shot_table[chunk_start:n],
                    boundary_rows[chunk_start:n - 1] if n - chunk_start >= 2 else [],
                    start_offset=chunk_start,
                )
                ranges.extend(chunk_ranges)

        return ranges, None, None


class EventBuilder:
    def __init__(self, config: EventGroupingDatasetConfig):
        self.config = config

    def pool_subtitle_event_embeddings(self, subtitle_items: List[Dict[str, Any]], event_start_sec: float, event_end_sec: float) -> Dict[str, Optional[np.ndarray]]:
        vectors = []
        for item in subtitle_items:
            item_start_sec = float(item["start_time_sec"])
            item_end_sec = float(item["end_time_sec"])
            if MathUtils.overlap_len(event_start_sec, event_end_sec, item_start_sec, item_end_sec) <= 0:
                continue
            vectors.append(item["embedding"])
        return PoolingUtils.multi_pool(vectors, temperature=self.config.event_softmax_temperature)

    def concat_subtitle_for_event(self, shot_table, start, end_exclusive) -> str:
        texts = []
        seen = set()
        for i in range(start, end_exclusive):
            t = shot_table[i].get("subtitle_text", "").strip()
            if not t or t in seen:
                continue
            seen.add(t)
            texts.append(t)
        return " ".join(texts).strip()

    def union_face_ids_for_event(self, shot_table, start, end_exclusive) -> List[int]:
        ids = set()
        for i in range(start, end_exclusive):
            ids.update(shot_table[i].get("face_ids", []))
        return sorted(int(x) for x in ids)

    def aggregate_face_counts_for_event(self, shot_table, start, end_exclusive) -> Dict[str, int]:
        counter = Counter()
        for i in range(start, end_exclusive):
            for k, v in shot_table[i].get("face_id_counts", {}).items():
                counter[str(k)] += int(v)
        return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda kv: int(kv[0]))}

    def collect_event_shot_vectors(self, shot_table, start, end_exclusive, key, valid_key):
        return [shot_table[i][key] for i in range(start, end_exclusive) if shot_table[i].get(valid_key, False) and shot_table[i].get(key) is not None]

    def build(self, shot_table: List[Dict[str, Any]], subtitle_items: List[Dict[str, Any]], subtitle_dim: int, event_ranges: List[Tuple[int, int]], boundary_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        visual_dim = int(len(shot_table[0]["visual_emb"]))
        action_dim = int(len(next(s["action_emb"] for s in shot_table if s["action_emb"] is not None)))

        events = []
        event_visual_embeddings_mean = []
        event_visual_embeddings_max = []
        event_visual_embeddings_softmax = []
        event_action_embeddings_mean = []
        event_action_embeddings_max = []
        event_action_embeddings_softmax = []
        event_subtitle_embeddings_mean = []
        event_subtitle_embeddings_max = []
        event_subtitle_embeddings_softmax = []
        event_subtitle_mask = []

        for event_id, (start, end) in enumerate(event_ranges):
            start_shot = shot_table[start]
            end_shot = shot_table[end - 1]
            end_boundary_idx = end - 1 if end < len(shot_table) else None

            visual_pool = PoolingUtils.multi_pool(self.collect_event_shot_vectors(shot_table, start, end, "visual_emb", "visual_valid"), temperature=self.config.event_softmax_temperature)
            action_pool = PoolingUtils.multi_pool(self.collect_event_shot_vectors(shot_table, start, end, "action_emb", "action_valid"), temperature=self.config.event_softmax_temperature)
            subtitle_pool = self.pool_subtitle_event_embeddings(subtitle_items, float(start_shot["start_time_sec"]), float(end_shot["end_time_sec"]))

            event_visual_embeddings_mean.append(visual_pool["mean"] if visual_pool["mean"] is not None else np.zeros((visual_dim,), dtype=np.float32))
            event_visual_embeddings_max.append(visual_pool["max"] if visual_pool["max"] is not None else np.zeros((visual_dim,), dtype=np.float32))
            event_visual_embeddings_softmax.append(visual_pool["softmax"] if visual_pool["softmax"] is not None else np.zeros((visual_dim,), dtype=np.float32))
            event_action_embeddings_mean.append(action_pool["mean"] if action_pool["mean"] is not None else np.zeros((action_dim,), dtype=np.float32))
            event_action_embeddings_max.append(action_pool["max"] if action_pool["max"] is not None else np.zeros((action_dim,), dtype=np.float32))
            event_action_embeddings_softmax.append(action_pool["softmax"] if action_pool["softmax"] is not None else np.zeros((action_dim,), dtype=np.float32))
            event_subtitle_embeddings_mean.append(subtitle_pool["mean"] if subtitle_pool["mean"] is not None else np.zeros((subtitle_dim,), dtype=np.float32))
            event_subtitle_embeddings_max.append(subtitle_pool["max"] if subtitle_pool["max"] is not None else np.zeros((subtitle_dim,), dtype=np.float32))
            event_subtitle_embeddings_softmax.append(subtitle_pool["softmax"] if subtitle_pool["softmax"] is not None else np.zeros((subtitle_dim,), dtype=np.float32))
            event_subtitle_mask.append(subtitle_pool["softmax"] is not None)

            events.append({
                "event_id": int(event_id),
                "start_shot_index": int(start),
                "end_shot_index": int(end - 1),
                "start_shot_id": int(start_shot["shot_id"]),
                "end_shot_id": int(end_shot["shot_id"]),
                "shot_ids": [int(shot_table[i]["shot_id"]) for i in range(start, end)],
                "start_time_sec": float(start_shot["start_time_sec"]),
                "end_time_sec": float(end_shot["end_time_sec"]),
                "duration_sec": float(end_shot["end_time_sec"] - start_shot["start_time_sec"]),
                "num_shots": int(end - start),
                "subtitle_text": self.concat_subtitle_for_event(shot_table, start, end),
                "face_ids": self.union_face_ids_for_event(shot_table, start, end),
                "face_id_counts": self.aggregate_face_counts_for_event(shot_table, start, end),
                "end_boundary_index": None if end_boundary_idx is None else int(end_boundary_idx),
                "end_boundary_score": None if end_boundary_idx is None else float(boundary_rows[end_boundary_idx]["boundary_score"]),
                "end_boundary_is_candidate": None if end_boundary_idx is None else bool(boundary_rows[end_boundary_idx]["is_candidate"]),
                "has_visual_embedding": visual_pool["softmax"] is not None,
                "has_action_embedding": action_pool["softmax"] is not None,
                "has_subtitle_embedding": subtitle_pool["softmax"] is not None,
            })

        return {
            "events": events,
            "event_visual_embeddings_mean": np.stack(event_visual_embeddings_mean).astype(np.float32),
            "event_visual_embeddings_max": np.stack(event_visual_embeddings_max).astype(np.float32),
            "event_visual_embeddings_softmax": np.stack(event_visual_embeddings_softmax).astype(np.float32),
            "event_action_embeddings_mean": np.stack(event_action_embeddings_mean).astype(np.float32),
            "event_action_embeddings_max": np.stack(event_action_embeddings_max).astype(np.float32),
            "event_action_embeddings_softmax": np.stack(event_action_embeddings_softmax).astype(np.float32),
            "event_subtitle_embeddings_mean": np.stack(event_subtitle_embeddings_mean).astype(np.float32),
            "event_subtitle_embeddings_max": np.stack(event_subtitle_embeddings_max).astype(np.float32),
            "event_subtitle_embeddings_softmax": np.stack(event_subtitle_embeddings_softmax).astype(np.float32),
            "event_subtitle_mask": np.asarray(event_subtitle_mask, dtype=np.uint8),
        }


class EventOutputWriter:
    def __init__(self, config: EventGroupingDatasetConfig):
        self.config = config

    def build_debug_dataframe(self, shot_table: List[Dict[str, Any]], boundary_rows: List[Dict[str, Any]], scorer: BoundaryScorer) -> pd.DataFrame:
        debug_rows = []
        for row in boundary_rows:
            i = row["boundary_index"]
            left_idx, right_idx = scorer.collect_context_indices(len(shot_table), i)
            left_text = " ".join([shot_table[j].get("subtitle_text", "") for j in left_idx]).strip()
            right_text = " ".join([shot_table[j].get("subtitle_text", "") for j in right_idx]).strip()
            debug_rows.append({
                "boundary_index": row["boundary_index"],
                "boundary_time_sec": row["boundary_time_sec"],
                "left_shot_id": row["left_shot_id"],
                "right_shot_id": row["right_shot_id"],
                "visual_change": row["visual_change"],
                "action_change": row["action_change"],
                "subtitle_change": row["subtitle_change"],
                "face_change": row["face_change"],
                "visual_change_norm": row["visual_change_norm"],
                "action_change_norm": row["action_change_norm"],
                "subtitle_change_norm": row["subtitle_change_norm"],
                "face_change_norm": row["face_change_norm"],
                "boundary_score": row["boundary_score"],
                "is_candidate": row["is_candidate"],
                "is_selected_boundary": row["is_selected_boundary"],
                "left_context_range": str(row["left_context_range"]),
                "right_context_range": str(row["right_context_range"]),
                "left_text": left_text[:500],
                "right_text": right_text[:500],
                "left_face_hist": json.dumps(row.get("left_face_hist", {}), ensure_ascii=False),
                "right_face_hist": json.dumps(row.get("right_face_hist", {}), ensure_ascii=False),
                "num_left_subtitle_items": len(row.get("left_subtitle_items", [])),
                "num_right_subtitle_items": len(row.get("right_subtitle_items", [])),
                "num_subtitle_bridge_items": len(row.get("subtitle_bridge_items", [])),
                "subtitle_bridge_total_sec": row.get("subtitle_bridge_total_sec", 0.0),
                "subtitle_bridge_strength": row.get("subtitle_bridge_strength", 0.0),
                "subtitle_bridge_penalty": row.get("subtitle_bridge_penalty", 0.0),
            })
        return pd.DataFrame(debug_rows)

    def write(self, video_id: str, shot_table: List[Dict[str, Any]], subtitle_items: List[Dict[str, Any]], boundary_rows: List[Dict[str, Any]], selected_boundary_indices: List[int], event_data: Dict[str, Any], scorer: BoundaryScorer) -> Dict[str, Any]:
        output_dir = Path(self.config.output_root_dir) / video_id
        if output_dir.exists() and not self.config.overwrite:
            existing_events = output_dir / "events.json"
            if existing_events.exists():
                return {
                    "video_id": video_id,
                    "status": "skipped",
                    "output_dir": str(output_dir),
                    "num_events": None,
                }

        FileIO.ensure_dir(output_dir)
        FileIO.save_json(event_data["events"], output_dir / "events.json")
        FileIO.save_json(boundary_rows, output_dir / "boundary_scores.json")
        FileIO.save_json(asdict(self.config), output_dir / "config.json")
        FileIO.save_npy(event_data["event_visual_embeddings_mean"], output_dir / "event_visual_embeddings_mean.npy")
        FileIO.save_npy(event_data["event_visual_embeddings_max"], output_dir / "event_visual_embeddings_max.npy")
        FileIO.save_npy(event_data["event_visual_embeddings_softmax"], output_dir / "event_visual_embeddings_softmax.npy")
        FileIO.save_npy(event_data["event_action_embeddings_mean"], output_dir / "event_action_embeddings_mean.npy")
        FileIO.save_npy(event_data["event_action_embeddings_max"], output_dir / "event_action_embeddings_max.npy")
        FileIO.save_npy(event_data["event_action_embeddings_softmax"], output_dir / "event_action_embeddings_softmax.npy")
        FileIO.save_npy(event_data["event_subtitle_embeddings_mean"], output_dir / "event_subtitle_embeddings_mean.npy")
        FileIO.save_npy(event_data["event_subtitle_embeddings_max"], output_dir / "event_subtitle_embeddings_max.npy")
        FileIO.save_npy(event_data["event_subtitle_embeddings_softmax"], output_dir / "event_subtitle_embeddings_softmax.npy")
        FileIO.save_npy(event_data["event_subtitle_mask"], output_dir / "event_subtitle_mask.npy")

        debug_df = self.build_debug_dataframe(shot_table, boundary_rows, scorer)
        debug_df.to_csv(output_dir / "debug_boundary_table.csv", index=False, encoding="utf-8")

        manifest = {
            "video_id": video_id,
            "num_shots": len(shot_table),
            "num_subtitle_items": len(subtitle_items),
            "num_boundaries": len(boundary_rows),
            "num_events": len(event_data["events"]),
            "selected_boundary_indices": sorted(int(x) for x in selected_boundary_indices),
            "events_path": str(output_dir / "events.json"),
            "boundary_scores_path": str(output_dir / "boundary_scores.json"),
            "debug_boundary_table_path": str(output_dir / "debug_boundary_table.csv"),
            "event_visual_embeddings_mean_path": str(output_dir / "event_visual_embeddings_mean.npy"),
            "event_visual_embeddings_max_path": str(output_dir / "event_visual_embeddings_max.npy"),
            "event_visual_embeddings_softmax_path": str(output_dir / "event_visual_embeddings_softmax.npy"),
            "event_action_embeddings_mean_path": str(output_dir / "event_action_embeddings_mean.npy"),
            "event_action_embeddings_max_path": str(output_dir / "event_action_embeddings_max.npy"),
            "event_action_embeddings_softmax_path": str(output_dir / "event_action_embeddings_softmax.npy"),
            "event_subtitle_embeddings_mean_path": str(output_dir / "event_subtitle_embeddings_mean.npy"),
            "event_subtitle_embeddings_max_path": str(output_dir / "event_subtitle_embeddings_max.npy"),
            "event_subtitle_embeddings_softmax_path": str(output_dir / "event_subtitle_embeddings_softmax.npy"),
            "event_subtitle_mask_path": str(output_dir / "event_subtitle_mask.npy"),
        }
        FileIO.save_json(manifest, output_dir / "manifest.json")

        return {
            "video_id": video_id,
            "status": "done",
            "output_dir": str(output_dir),
            "num_events": len(event_data["events"]),
        }


class EventGroupingPipeline:
    def __init__(self, config: EventGroupingDatasetConfig):
        self.config = config
        self.preflight = DatasetPreflight(config)
        self.repository = FeatureRepository(config)
        self.assembler = VideoFeatureAssembler(config)
        self.scorer = BoundaryScorer(config)
        self.segmenter = DPSegmenter(config)
        self.builder = EventBuilder(config)
        self.writer = EventOutputWriter(config)

    def run_preflight(self) -> Dict[str, Any]:
        manifest = self.preflight.build_manifest()
        summary = self.preflight.summarize(manifest)
        return {
            "manifest": manifest,
            "summary": summary,
            "missing_df": pd.DataFrame(manifest["missing_rows"]),
        }

    def process_one_video(self, video_id: str) -> Dict[str, Any]:
        video_features = self.repository.load_video_features(video_id)
        shot_table, subtitle_items, subtitle_dim, fps = self.assembler.build(video_features)
        if len(shot_table) == 0:
            raise RuntimeError(f"Video {video_id} không có shot nào trong visual embedding")

        boundary_rows = self.scorer.compute_boundary_rows(shot_table, subtitle_items)
        self.scorer.score_boundaries(boundary_rows)
        selected_boundary_indices = self.scorer.select_candidate_boundaries(boundary_rows)

        event_ranges, dp, backptr = self.segmenter.segment(shot_table, boundary_rows)
        selected_boundary_set = {end - 1 for _, end in event_ranges if end < len(shot_table)}
        for row in boundary_rows:
            row["is_selected_boundary"] = bool(row["boundary_index"] in selected_boundary_set)

        event_data = self.builder.build(shot_table, subtitle_items, subtitle_dim, event_ranges, boundary_rows)
        result = self.writer.write(video_id, shot_table, subtitle_items, boundary_rows, selected_boundary_indices, event_data, self.scorer)
        result["dp_final_score"] = float(dp[len(shot_table)]) if dp is not None else None
        return result

    def run(self) -> Dict[str, Any]:
        preflight_info = self.run_preflight()
        manifest = preflight_info["manifest"]
        if not self.config.skip_missing_modalities and len(manifest["missing_rows"]) > 0:
            raise RuntimeError("Dataset còn video thiếu modality. Hãy sửa preflight trước khi chạy batch.")

        results = []
        for video_id in tqdm(manifest["eligible_ids"], desc="Process dataset"):
            try:
                result = self.process_one_video(video_id)
            except Exception as error:
                result = {
                    "video_id": video_id,
                    "status": "failed",
                    "error": repr(error),
                }
            results.append(result)

        summary_path = Path(self.config.output_root_dir) / "dataset_summary.json"
        FileIO.save_json(results, summary_path)
        return {
            "preflight": preflight_info,
            "results": results,
            "summary_df": pd.DataFrame(results),
            "summary_path": str(summary_path),
        }
