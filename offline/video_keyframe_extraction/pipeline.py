import pickle
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from .config import PipelineConfig
from .io_utils import DatasetScanner
from .model import CLIPEmbedder
from .video_utils import VideoFrameReader


class VideoKeyframeBuilder:
    def __init__(self, config: PipelineConfig, embedder: CLIPEmbedder):
        self.config = config
        self.embedder = embedder

    def _select_keyframes_in_batch(
        self,
        embeddings: np.ndarray,
        frame_indices: list[int],
        last_key_embedding: np.ndarray | None,
        is_first_batch: bool,
    ) -> tuple[list[np.ndarray], list[int], np.ndarray | None]:
        if embeddings.shape[0] == 0:
            return [], [], last_key_embedding

        selected_embeddings: list[np.ndarray] = []
        selected_frame_indices: list[int] = []
        start_index = 0

        if is_first_batch or last_key_embedding is None:
            selected_embeddings.append(embeddings[0].astype(np.float32))
            selected_frame_indices.append(int(frame_indices[0]))
            last_key_embedding = embeddings[0]
            start_index = 1

        for index in range(start_index, len(embeddings)):
            similarity = float(np.dot(last_key_embedding, embeddings[index]))
            if similarity < self.config.similarity_threshold:
                selected_embeddings.append(embeddings[index].astype(np.float32))
                selected_frame_indices.append(int(frame_indices[index]))
                last_key_embedding = embeddings[index]

        return selected_embeddings, selected_frame_indices, last_key_embedding

    def _flush_frame_batch(
        self,
        batch_images: list[Any],
        batch_frame_indices: list[int],
        sampled_frame_indices: list[int],
        selected_embeddings: list[np.ndarray],
        selected_frame_indices: list[int],
        last_key_embedding: np.ndarray | None,
        is_first_batch: bool,
    ) -> np.ndarray | None:
        if not batch_images:
            return last_key_embedding

        embeddings = self.embedder.encode_images(batch_images)
        sampled_frame_indices.extend(int(frame_idx) for frame_idx in batch_frame_indices)
        batch_selected_embeddings, batch_selected_frame_indices, last_key_embedding = (
            self._select_keyframes_in_batch(
                embeddings=embeddings,
                frame_indices=batch_frame_indices,
                last_key_embedding=last_key_embedding,
                is_first_batch=is_first_batch,
            )
        )
        selected_embeddings.extend(batch_selected_embeddings)
        selected_frame_indices.extend(batch_selected_frame_indices)
        batch_images.clear()
        batch_frame_indices.clear()
        return last_key_embedding

    def process_video(self, item: dict[str, str]) -> dict[str, Any]:
        video_name = item["video_name"]
        video_path = item["video_path"]
        output_path = item["output_path"]

        if Path(output_path).exists() and not self.config.overwrite:
            print(f"[SKIP] {video_name}: output already exists")
            with open(output_path, "rb") as file:
                return pickle.load(file)

        reader = VideoFrameReader(video_path)
        try:
            sampled_frame_indices = reader.sample_frame_indices(self.config.frame_step)
            frame_load_batch_size = max(int(self.config.frame_load_batch_size), 1)
            batch_images: list[Any] = []
            batch_frame_indices: list[int] = []
            valid_frame_indices: list[int] = []
            selected_frame_indices: list[int] = []
            selected_embeddings: list[np.ndarray] = []
            last_key_embedding: np.ndarray | None = None
            is_first_batch = True

            for frame_index, image in tqdm(
                reader.iter_sampled_frames(self.config.frame_step),
                total=len(sampled_frame_indices),
                desc=f"Frames: {video_name}",
                leave=False,
            ):
                batch_images.append(image)
                batch_frame_indices.append(int(frame_index))

                if len(batch_images) >= frame_load_batch_size:
                    last_key_embedding = self._flush_frame_batch(
                        batch_images=batch_images,
                        batch_frame_indices=batch_frame_indices,
                        sampled_frame_indices=valid_frame_indices,
                        selected_embeddings=selected_embeddings,
                        selected_frame_indices=selected_frame_indices,
                        last_key_embedding=last_key_embedding,
                        is_first_batch=is_first_batch,
                    )
                    is_first_batch = False

            self._flush_frame_batch(
                batch_images=batch_images,
                batch_frame_indices=batch_frame_indices,
                sampled_frame_indices=valid_frame_indices,
                selected_embeddings=selected_embeddings,
                selected_frame_indices=selected_frame_indices,
                last_key_embedding=last_key_embedding,
                is_first_batch=is_first_batch,
            )

            if len(valid_frame_indices) == 0:
                raise RuntimeError(f"Khong doc duoc frame nao cho video {video_name}")

            if len(selected_embeddings) == 0:
                raise RuntimeError(f"Khong chon duoc keyframe nao cho video {video_name}")

            keyframe_embeddings = np.stack(selected_embeddings, axis=0).astype(np.float32)
            keyframe_frame_indices = [int(frame_idx) for frame_idx in selected_frame_indices]

            if self.config.save_dtype == "float16":
                embeddings_to_save = keyframe_embeddings.astype(np.float16)
            elif self.config.save_dtype == "float32":
                embeddings_to_save = keyframe_embeddings.astype(np.float32)
            else:
                raise ValueError(f"Unsupported save_dtype: {self.config.save_dtype}")

            data = {
                "video_name": video_name,
                "video_path": video_path,
                "fps": reader.fps,
                "frame_count": reader.frame_count,
                "width": reader.width,
                "height": reader.height,
                "duration_sec": reader.duration_sec,
                "frame_step": self.config.frame_step,
                "similarity_threshold": self.config.similarity_threshold,
                "clip_model_name": self.config.clip_model_name,
                "clip_pretrained": self.config.clip_pretrained,
                "embedding_dim": int(keyframe_embeddings.shape[1]),
                "embedding_dtype": self.config.save_dtype,
                "num_sampled_frames": len(valid_frame_indices),
                "num_keyframes": int(embeddings_to_save.shape[0]),
                "sampled_frame_indices": [int(frame_idx) for frame_idx in valid_frame_indices],
                "keyframe_frame_indices": keyframe_frame_indices,
                "keyframe_embeddings": embeddings_to_save,
            }
        finally:
            reader.close()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as file:
            pickle.dump(data, file)

        return data


class BatchProcessor:
    def __init__(self, config: PipelineConfig):
        self.config = config
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        self.scanner = DatasetScanner(config)
        self.embedder = CLIPEmbedder(config)
        self.builder = VideoKeyframeBuilder(config, self.embedder)

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
                data = self.builder.process_video(item)
                if already_exists:
                    summary["skipped"].append(item["video_name"])
                else:
                    summary["done"].append(
                        {
                            "video_name": data["video_name"],
                            "num_sampled_frames": data["num_sampled_frames"],
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
                        "output_path": item["output_path"],
                        "error": repr(exc),
                    }
                )

        return summary
