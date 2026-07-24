from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from .config import PipelineConfig
from .io_utils import DatasetScanner, load_pickle, save_pickle
from .model import CLIPImageEmbedder
from .video_utils import VideoFrameReader


class ReferenceKeyframeEmbedder:
    def __init__(self, config: PipelineConfig, embedder: CLIPImageEmbedder):
        self.config = config
        self.embedder = embedder

    def _build_output_payload(
        self,
        *,
        reference: dict[str, Any],
        video_name: str,
        video_path: str,
        reader: VideoFrameReader,
        keyframe_frame_indices: list[int],
        keyframe_embeddings: np.ndarray,
    ) -> dict[str, Any]:
        if self.config.save_dtype == "float16":
            embeddings_to_save = keyframe_embeddings.astype(np.float16)
        elif self.config.save_dtype == "float32":
            embeddings_to_save = keyframe_embeddings.astype(np.float32)
        else:
            raise ValueError(f"Unsupported save_dtype: {self.config.save_dtype}")

        payload = dict(reference)
        payload.update(
            {
                "video_name": video_name,
                "video_path": video_path,
                "fps": reader.fps,
                "frame_count": reader.frame_count,
                "width": reader.width,
                "height": reader.height,
                "duration_sec": reader.duration_sec,
                "clip_model_name": self.config.clip_model_name,
                "clip_pretrained": self.config.clip_pretrained,
                "embedding_dim": int(keyframe_embeddings.shape[1]),
                "embedding_dtype": self.config.save_dtype,
                "num_keyframes": int(embeddings_to_save.shape[0]),
                "keyframe_frame_indices": [int(frame_index) for frame_index in keyframe_frame_indices],
                "keyframe_embeddings": embeddings_to_save,
            }
        )
        return payload

    def process_video(self, item: dict[str, str]) -> dict[str, Any]:
        video_name = item["video_name"]
        video_path = item["video_path"]
        reference_path = item["reference_path"]
        output_path = item["output_path"]

        if Path(output_path).exists() and not self.config.overwrite:
            print(f"[SKIP] {video_name}: output already exists")
            return load_pickle(output_path)

        if not Path(video_path).exists():
            raise FileNotFoundError(video_path)

        reference = load_pickle(reference_path)
        keyframe_frame_indices = [int(frame_index) for frame_index in reference.get("keyframe_frame_indices", [])]
        if not keyframe_frame_indices:
            raise RuntimeError(f"Khong co keyframe_frame_indices trong file reference: {reference_path}")

        reader = VideoFrameReader(video_path)
        try:
            frame_images: list[Any] = []
            decoded_frame_indices: list[int] = []
            for frame_index, image in tqdm(
                reader.iter_selected_frames(keyframe_frame_indices),
                total=len(keyframe_frame_indices),
                desc=f"Frames: {video_name}",
                leave=False,
            ):
                decoded_frame_indices.append(int(frame_index))
                frame_images.append(image)

            if decoded_frame_indices != keyframe_frame_indices:
                raise RuntimeError(
                    f"Khong doc du cac keyframe cho video {video_name}: "
                    f"expected={len(keyframe_frame_indices)}, got={len(decoded_frame_indices)}"
                )

            keyframe_embeddings = self.embedder.encode_images(frame_images)
            if keyframe_embeddings.shape[0] != len(keyframe_frame_indices):
                raise RuntimeError(
                    f"So luong embeddings khong khop cho video {video_name}: "
                    f"{keyframe_embeddings.shape[0]} vs {len(keyframe_frame_indices)}"
                )

            payload = self._build_output_payload(
                reference=reference,
                video_name=video_name,
                video_path=video_path,
                reader=reader,
                keyframe_frame_indices=keyframe_frame_indices,
                keyframe_embeddings=keyframe_embeddings,
            )
        finally:
            reader.close()

        save_pickle(payload, output_path)
        return payload


class BatchProcessor:
    def __init__(self, config: PipelineConfig):
        self.config = config
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        self.scanner = DatasetScanner(config)
        self.embedder = CLIPImageEmbedder(config)
        self.processor = ReferenceKeyframeEmbedder(config, self.embedder)

    def run(self) -> dict[str, Any]:
        items = self.scanner.get_video_items()
        print(f"Found {len(items)} videos to process")

        summary: dict[str, Any] = {
            "done": [],
            "skipped": [],
            "failed": [],
        }

        for item in tqdm(items, desc="Processing videos"):
            try:
                already_exists = Path(item["output_path"]).exists() and not self.config.overwrite
                data = self.processor.process_video(item)
                if already_exists:
                    summary["skipped"].append(item["video_name"])
                else:
                    summary["done"].append(
                        {
                            "video_name": data["video_name"],
                            "num_keyframes": data["num_keyframes"],
                            "output_path": item["output_path"],
                        }
                    )
            except Exception as exc:
                print(f"[FAILED] {item['video_name']}: {exc!r}")
                summary["failed"].append(
                    {
                        "video_name": item["video_name"],
                        "video_path": item["video_path"],
                        "reference_path": item["reference_path"],
                        "output_path": item["output_path"],
                        "error": repr(exc),
                    }
                )

        return summary
