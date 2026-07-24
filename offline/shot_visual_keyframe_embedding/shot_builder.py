from typing import Any

import numpy as np

from .config import PipelineConfig
from .model import CLIPEmbedder
from .video_utils import VideoFrameReader


class ShotKeyframeEmbeddingBuilder:
    def __init__(self, config: PipelineConfig, embedder: CLIPEmbedder):
        self.config = config
        self.embedder = embedder

    def sample_shot_frames(
        self,
        reader: VideoFrameReader,
        shot: dict[str, Any],
    ) -> dict[str, Any]:
        start_frame = int(shot["start_frame"])
        end_frame = int(shot["end_frame"])

        sampled_indices = list(range(start_frame, end_frame + 1, self.config.frame_step))
        if len(sampled_indices) == 0:
            sampled_indices = [(start_frame + end_frame) // 2]

        valid_indices = [
            frame_idx
            for frame_idx in sampled_indices
            if 0 <= frame_idx < reader.frame_count
        ]

        if len(valid_indices) == 0:
            midpoint = (start_frame + end_frame) // 2
            midpoint = int(max(0, min(midpoint, reader.frame_count - 1)))
            valid_indices = [midpoint]

        return {
            "shot": shot,
            "sampled_frame_indices": valid_indices,
        }

    def _select_keyframes(
        self,
        embeddings: np.ndarray,
        frame_indices: list[int],
    ) -> tuple[np.ndarray, list[int]]:
        selected_local_indices = [0]
        last_key_embedding = embeddings[0]

        for index in range(1, len(embeddings)):
            similarity = float(np.dot(last_key_embedding, embeddings[index]))
            if similarity < self.config.similarity_threshold:
                selected_local_indices.append(index)
                last_key_embedding = embeddings[index]

        selected_embeddings = embeddings[selected_local_indices].astype(np.float32)
        selected_frame_indices = [int(frame_indices[index]) for index in selected_local_indices]
        return selected_embeddings, selected_frame_indices

    def build_shot_output(
        self,
        shot: dict[str, Any],
        valid_frame_indices: list[int],
        embeddings: np.ndarray,
    ) -> dict[str, Any]:
        if embeddings.shape[0] != len(valid_frame_indices):
            raise ValueError(
                f"Embedding count mismatch for shot_id={shot['shot_id']}: "
                f"{embeddings.shape[0]} vs {len(valid_frame_indices)}"
            )

        keyframe_embeddings, keyframe_frame_indices = self._select_keyframes(
            embeddings=embeddings,
            frame_indices=valid_frame_indices,
        )

        if self.config.save_dtype == "float16":
            keyframe_embeddings = keyframe_embeddings.astype(np.float16)
        elif self.config.save_dtype == "float32":
            keyframe_embeddings = keyframe_embeddings.astype(np.float32)
        else:
            raise ValueError(f"Unsupported save_dtype: {self.config.save_dtype}")

        return {
            "shot_id": int(shot["shot_id"]),
            "start_time_sec": float(shot["start_time_sec"]),
            "end_time_sec": float(shot["end_time_sec"]),
            "start_frame": int(shot["start_frame"]),
            "end_frame": int(shot["end_frame"]),
            "duration_sec": float(shot["duration_sec"]),
            "sampled_frame_indices": [int(frame_idx) for frame_idx in valid_frame_indices],
            "keyframe_frame_indices": keyframe_frame_indices,
            "num_sampled_frames": len(valid_frame_indices),
            "num_keyframes": int(keyframe_embeddings.shape[0]),
            "keyframe_embeddings": keyframe_embeddings,
        }
