import pickle
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from .config import PipelineConfig
from .io_utils import ShotLoader
from .shot_builder import ShotKeyframeEmbeddingBuilder
from .video_utils import VideoFrameReader


class VideoProcessor:
    def __init__(
        self,
        config: PipelineConfig,
        shot_loader: ShotLoader,
        embedding_builder: ShotKeyframeEmbeddingBuilder,
    ):
        self.config = config
        self.shot_loader = shot_loader
        self.embedding_builder = embedding_builder

    def _encode_sampled_shots(
        self,
        reader: VideoFrameReader,
        sampled_shots: list[dict[str, Any]],
        video_name: str,
    ) -> list[dict[str, Any]]:
        frame_refs: list[tuple[int, int]] = []
        embeddings_by_shot: list[list[np.ndarray]] = [list() for _ in sampled_shots]
        valid_frame_indices_by_shot: list[list[int]] = [list() for _ in sampled_shots]

        for shot_index, sampled_shot in enumerate(sampled_shots):
            for frame_index in sampled_shot["sampled_frame_indices"]:
                frame_refs.append((shot_index, frame_index))

        batch_starts = range(0, len(frame_refs), self.config.batch_size)
        for start in tqdm(
            batch_starts,
            total=(len(frame_refs) + self.config.batch_size - 1) // self.config.batch_size,
            desc=f"Encoding: {video_name}",
            leave=False,
        ):
            batch_refs = frame_refs[start:start + self.config.batch_size]
            batch_images: list[Image.Image] = []
            valid_batch_refs: list[tuple[int, int]] = []

            for shot_index, frame_index in batch_refs:
                image, actual_frame_index = reader.read_frame_with_fallback(frame_index)
                if image is not None:
                    batch_images.append(image)
                    valid_batch_refs.append((shot_index, int(actual_frame_index)))

            if not batch_images:
                continue

            batch_embeddings = self.embedding_builder.embedder.encode_images(batch_images)
            for (shot_index, actual_frame_index), embedding in zip(valid_batch_refs, batch_embeddings):
                embeddings_by_shot[shot_index].append(embedding)
                valid_frame_indices_by_shot[shot_index].append(actual_frame_index)

        outputs: list[dict[str, Any]] = []
        for shot_index, shot_embeddings in enumerate(embeddings_by_shot):
            if len(shot_embeddings) == 0:
                shot = sampled_shots[shot_index]["shot"]
                midpoint = (int(shot["start_frame"]) + int(shot["end_frame"])) // 2
                image, actual_frame_index = reader.read_frame_with_fallback(midpoint)
                if image is not None and actual_frame_index is not None:
                    fallback_embedding = self.embedding_builder.embedder.encode_images([image])[0]
                    shot_embeddings.append(fallback_embedding.astype(np.float32))
                    valid_frame_indices_by_shot[shot_index].append(int(actual_frame_index))

            if len(shot_embeddings) == 0:
                shot_id = sampled_shots[shot_index]["shot"]["shot_id"]
                raise RuntimeError(f"No embeddings collected for shot_id={shot_id}")

            outputs.append(
                {
                    "embeddings": np.stack(shot_embeddings, axis=0).astype(np.float32),
                    "valid_frame_indices": valid_frame_indices_by_shot[shot_index],
                }
            )

        return outputs

    def process_video(self, item: dict[str, str]) -> dict[str, Any]:
        video_name = item["video_name"]
        video_path = item["video_path"]
        shots_path = item["shots_path"]
        output_path = item["output_path"]

        shots = self.shot_loader.load(shots_path)
        reader = VideoFrameReader(video_path)
        sampled_shots: list[dict[str, Any]] = []
        processed_shots: list[dict[str, Any]] = []

        try:
            for shot in tqdm(shots, desc=f"Shots: {video_name}", leave=False):
                sampled_shots.append(self.embedding_builder.sample_shot_frames(reader, shot))

            embeddings_by_shot = self._encode_sampled_shots(reader, sampled_shots, video_name)

            for sampled_shot, shot_result in zip(sampled_shots, embeddings_by_shot):
                processed_shot = self.embedding_builder.build_shot_output(
                    shot=sampled_shot["shot"],
                    valid_frame_indices=shot_result["valid_frame_indices"],
                    embeddings=shot_result["embeddings"],
                )
                processed_shots.append(processed_shot)
        finally:
            reader.close()

        total_sampled_frames = sum(len(item["sampled_frame_indices"]) for item in sampled_shots)

        embedding_dim = 0
        for shot in processed_shots:
            embeddings = shot["keyframe_embeddings"]
            if hasattr(embeddings, "shape") and len(embeddings.shape) == 2:
                embedding_dim = int(embeddings.shape[1])
                break

        data = {
            "video_name": video_name,
            "video_path": video_path,
            "shots_path": shots_path,
            "fps": reader.fps,
            "frame_count": reader.frame_count,
            "width": reader.width,
            "height": reader.height,
            "frame_step": self.config.frame_step,
            "similarity_threshold": self.config.similarity_threshold,
            "clip_model_name": self.config.clip_model_name,
            "clip_pretrained": self.config.clip_pretrained,
            "embedding_dim": embedding_dim,
            "embedding_dtype": self.config.save_dtype,
            "num_shots": len(processed_shots),
            "total_sampled_frames": total_sampled_frames,
            "shots": processed_shots,
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as file:
            pickle.dump(data, file)

        return data
