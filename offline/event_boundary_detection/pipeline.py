from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .boundary_scoring import BoundaryScorer
from .config import EventGroupingDatasetConfig
from .event_building import EventBuilder
from .feature_assembly import FeatureRepository, VideoFeatureAssembler
from .output_writer import EventOutputWriter
from .preflight import DatasetPreflight
from .segmentation import DPSegmenter
from .common import FileIO


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
