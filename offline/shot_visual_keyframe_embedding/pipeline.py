from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .config import PipelineConfig
from .io_utils import DatasetScanner, ShotLoader
from .model import CLIPEmbedder
from .shot_builder import ShotKeyframeEmbeddingBuilder
from .video_processor import VideoProcessor


class BatchProcessor:
    def __init__(self, config: PipelineConfig):
        self.config = config
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        self.scanner = DatasetScanner(config)
        self.shot_loader = ShotLoader()
        self.embedder = CLIPEmbedder(config)
        self.embedding_builder = ShotKeyframeEmbeddingBuilder(config, self.embedder)
        self.video_processor = VideoProcessor(
            config=config,
            shot_loader=self.shot_loader,
            embedding_builder=self.embedding_builder,
        )

    def run(self) -> dict[str, Any]:
        items = self.scanner.get_video_items()
        print(f"Found {len(items)} videos to process")

        summary: dict[str, Any] = {
            "done": [],
            "skipped": [],
            "failed": [],
        }

        for item in tqdm(items, desc="Processing videos"):
            output_path = Path(item["output_path"])

            try:
                if output_path.exists() and not self.config.overwrite:
                    print(f"[SKIP] {item['video_name']}")
                    summary["skipped"].append(item["video_name"])
                    continue

                output = self.video_processor.process_video(item)
                print(
                    f"[DONE] {item['video_name']} | "
                    f"num_shots={output['num_shots']} | "
                    f"embedding_dim={output['embedding_dim']}"
                )
                summary["done"].append(
                    {
                        "video_name": item["video_name"],
                        "output_path": item["output_path"],
                        "num_shots": output["num_shots"],
                        "embedding_dim": output["embedding_dim"],
                    }
                )
            except Exception as error:
                print(f"[FAILED] {item['video_name']}: {error}")
                summary["failed"].append(
                    {
                        "video_name": item["video_name"],
                        "output_path": item["output_path"],
                        "error": repr(error),
                    }
                )

        return summary
