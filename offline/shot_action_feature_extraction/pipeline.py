import gc
from pathlib import Path
from typing import Any

import torch

from .config import SlowFastShotFeatureConfig
from .ffmpeg_reader import ensure_ffmpeg_available
from .io_utils import DatasetScanner, ShotLoader
from .model_loader import load_slowfast_model
from .video_processor import VideoFeatureProcessor


class SlowFastShotFeatureExtractor:
    def __init__(self, config: SlowFastShotFeatureConfig):
        self.config = config
        self.device = torch.device(
            config.device if config.device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.model = load_slowfast_model(config, self.device)
        self.scanner = DatasetScanner(config)
        self.shot_loader = ShotLoader()
        self.video_processor = VideoFeatureProcessor(
            config=config,
            device=self.device,
            model=self.model,
            shot_loader=self.shot_loader,
        )
        ensure_ffmpeg_available()

    def _empty_cuda_cache(self):
        self.video_processor.empty_cuda_cache()

    def run(self) -> dict[str, Any]:
        items = self.scanner.get_video_items()
        print(f"Found {len(items)} videos to process")

        summary: dict[str, Any] = {
            "done": [],
            "skipped": [],
            "failed": [],
        }

        for item in items:
            output_path = Path(item["output_pkl_path"])
            try:
                if output_path.exists() and not self.config.overwrite:
                    print(f"[SKIP] {item['video_name']}")
                    summary["skipped"].append(item["video_name"])
                    continue

                output = self.video_processor.process_video_item(item)
                print(
                    f"[DONE] {item['video_name']} | num_shots={output['num_shots']} | "
                    f"num_subshots_total={output['num_subshots_total']} | feature_dim={output['feature_dim']}"
                )
                summary["done"].append(
                    {
                        "video_name": item["video_name"],
                        "output_pkl_path": item["output_pkl_path"],
                        "num_shots": output["num_shots"],
                        "num_subshots_total": output["num_subshots_total"],
                        "feature_dim": output["feature_dim"],
                    }
                )
            except Exception as error:
                print(f"[FAILED] {item['video_name']}: {error}")
                summary["failed"].append(
                    {
                        "video_name": item["video_name"],
                        "video_path": item["video_path"],
                        "shots_json_path": item["shots_json_path"],
                        "output_pkl_path": item["output_pkl_path"],
                        "error": repr(error),
                    }
                )
            finally:
                gc.collect()
                self._empty_cuda_cache()

        return summary
