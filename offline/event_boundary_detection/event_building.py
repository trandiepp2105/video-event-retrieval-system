from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .common import MathUtils, PoolingUtils
from .config import EventGroupingDatasetConfig


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
