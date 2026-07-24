from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .common import FileIO
from .config import EventGroupingDatasetConfig
from .boundary_scoring import BoundaryScorer


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
