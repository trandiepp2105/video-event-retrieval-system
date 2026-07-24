from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from .common import MathUtils, PoolingUtils
from .config import EventGroupingDatasetConfig


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
