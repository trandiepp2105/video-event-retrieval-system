from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import CommonConfig
from .io_utils import load_json, load_pickle, save_json


@dataclass
class HierarchicalTemporalSample:
    video_id: str
    event_id: int
    caption: str
    translated_caption: str
    start_time_sec: float
    end_time_sec: float
    video_duration_sec: float
    shot_ids: List[int]
    shot_metadata: np.ndarray
    shot_start_times: List[float]
    shot_end_times: List[float]
    subtitle_overlap_scores: np.ndarray
    keyframe_vectors: np.ndarray
    keyframe_metadata: np.ndarray
    keyframe_mask: np.ndarray


class HierarchicalTemporalEventDataset(Dataset):
    def __init__(self, samples: List[HierarchicalTemporalSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        return {
            "video_id": sample.video_id,
            "event_id": sample.event_id,
            "caption": sample.caption,
            "translated_caption": sample.translated_caption,
            "start_time_sec": sample.start_time_sec,
            "end_time_sec": sample.end_time_sec,
            "video_duration_sec": sample.video_duration_sec,
            "shot_ids": sample.shot_ids,
            "shot_metadata": torch.from_numpy(sample.shot_metadata),
            "shot_start_times": sample.shot_start_times,
            "shot_end_times": sample.shot_end_times,
            "subtitle_overlap_scores": torch.from_numpy(sample.subtitle_overlap_scores),
            "keyframe_vectors": torch.from_numpy(sample.keyframe_vectors),
            "keyframe_metadata": torch.from_numpy(sample.keyframe_metadata),
            "keyframe_mask": torch.from_numpy(sample.keyframe_mask),
        }


class HierarchicalTemporalEventDatasetBuilder:
    def __init__(self, config: CommonConfig) -> None:
        self.config = config

    def _load_video_duration_and_shots(self, video_id: str) -> tuple[float, float, List[dict]]:
        path = self.config.shot_keyframe_dir / f"{video_id}.pkl"
        data = load_pickle(path)
        fps = float(data["fps"])
        frame_count = int(data["frame_count"])
        shots = data["shots"]
        video_duration_sec = frame_count / fps
        return video_duration_sec, fps, shots

    def _build_shot_lookup(self, shots: List[dict]) -> Dict[int, dict]:
        return {int(shot["shot_id"]): shot for shot in shots}

    def _load_raw_subtitles(self, video_id: str, fps: float) -> List[dict]:
        if self.config.raw_subtitle_dir is None:
            return []
        subtitle_path = self.config.raw_subtitle_dir / f"{video_id}.json"
        if not subtitle_path.exists():
            return []
        items = load_json(subtitle_path)
        subtitles: List[dict] = []
        for item in items:
            frame_start = float(item.get("frame_start", 0.0))
            frame_end = float(item.get("frame_end", frame_start))
            subtitles.append(
                {
                    "start_time_sec": frame_start / fps,
                    "end_time_sec": frame_end / fps,
                    "text": str(item.get("text", "")).strip(),
                }
            )
        return subtitles

    def _make_shot_metadata(
        self,
        shot: dict,
        *,
        local_index: int,
        total_shots: int,
        event_start: float,
        event_end: float,
    ) -> np.ndarray:
        shot_start = float(shot["start_time_sec"])
        shot_end = float(shot["end_time_sec"])
        shot_duration = float(shot["duration_sec"])
        event_duration = max(event_end - event_start, 1e-6)
        rel_start = (shot_start - event_start) / event_duration
        rel_end = (shot_end - event_start) / event_duration
        position = 0.0 if total_shots <= 1 else local_index / (total_shots - 1)
        duration_norm = min(shot_duration / self.config.shot_metadata_duration_norm_sec, 1.0)
        return np.asarray([position, duration_norm, rel_start, rel_end], dtype=np.float32)

    def _make_keyframe_metadata(
        self,
        shot: dict,
        *,
        local_index: int,
        total_keyframes: int,
        frame_index: int,
        previous_frame_index: int | None,
        next_frame_index: int | None,
        fps: float,
    ) -> np.ndarray:
        shot_start_sec = float(shot["start_time_sec"])
        shot_duration_sec = max(float(shot["duration_sec"]), 1e-6)
        frame_time_sec = float(frame_index) / fps
        relative_position = 0.0 if total_keyframes <= 1 else local_index / (total_keyframes - 1)
        relative_time = (frame_time_sec - shot_start_sec) / shot_duration_sec
        if previous_frame_index is None:
            delta_prev_sec = 0.0
        else:
            delta_prev_sec = max(frame_index - previous_frame_index, 0) / fps
        if next_frame_index is None:
            delta_next_sec = 0.0
        else:
            delta_next_sec = max(next_frame_index - frame_index, 0) / fps
        delta_prev_norm = min(delta_prev_sec / max(self.config.keyframe_metadata_time_norm_sec, 1e-6), 1.0)
        delta_next_norm = min(delta_next_sec / max(self.config.keyframe_metadata_time_norm_sec, 1e-6), 1.0)
        return np.asarray(
            [relative_position, relative_time, delta_prev_norm, delta_next_norm],
            dtype=np.float32,
        )

    @staticmethod
    def _segment_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
        return max(0.0, min(end_a, end_b) - max(start_a, start_b))

    def _compute_subtitle_overlap_scores(
        self,
        *,
        shot_ids: List[int],
        shot_lookup: Dict[int, dict],
        subtitles: List[dict],
    ) -> np.ndarray:
        scores = []
        for shot_id in shot_ids:
            shot = shot_lookup[shot_id]
            shot_start = float(shot["start_time_sec"])
            shot_end = float(shot["end_time_sec"])
            subtitle_overlap_sec = 0.0
            for subtitle_item in subtitles:
                subtitle_overlap_sec += self._segment_overlap(
                    shot_start,
                    shot_end,
                    float(subtitle_item["start_time_sec"]),
                    float(subtitle_item["end_time_sec"]),
                )
            subtitle_score = min(
                subtitle_overlap_sec / max(self.config.salient_subtitle_norm_sec, 1e-6),
                1.0,
            )
            scores.append(subtitle_score)
        return np.asarray(scores, dtype=np.float32)

    def _build_keyframe_arrays(
        self,
        shot: dict,
        *,
        fps: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        keyframe_embeddings = np.asarray(shot["keyframe_embeddings"], dtype=np.float32)
        keyframe_frame_indices = [int(x) for x in shot["keyframe_frame_indices"]]
        if keyframe_embeddings.ndim != 2 or len(keyframe_frame_indices) != keyframe_embeddings.shape[0]:
            return None
        if keyframe_embeddings.shape[0] < self.config.min_keyframes_per_shot:
            return None

        keyframe_embeddings = keyframe_embeddings[: self.config.max_keyframes_per_shot]
        keyframe_frame_indices = keyframe_frame_indices[: self.config.max_keyframes_per_shot]
        num_keyframes = keyframe_embeddings.shape[0]

        keyframe_metadata = []
        for local_index, frame_index in enumerate(keyframe_frame_indices):
            previous_frame_index = keyframe_frame_indices[local_index - 1] if local_index > 0 else None
            next_frame_index = keyframe_frame_indices[local_index + 1] if local_index + 1 < num_keyframes else None
            keyframe_metadata.append(
                self._make_keyframe_metadata(
                    shot,
                    local_index=local_index,
                    total_keyframes=num_keyframes,
                    frame_index=frame_index,
                    previous_frame_index=previous_frame_index,
                    next_frame_index=next_frame_index,
                    fps=fps,
                )
            )

        keyframe_mask = np.ones((num_keyframes,), dtype=bool)
        return (
            keyframe_embeddings.astype(np.float32),
            np.stack(keyframe_metadata).astype(np.float32),
            keyframe_mask,
        )

    def _build_sample(
        self,
        video_id: str,
        event: dict,
        caption_text: str,
        translated_caption: str,
        shot_lookup: Dict[int, dict],
        video_duration_sec: float,
        subtitles: List[dict],
        fps: float,
    ) -> HierarchicalTemporalSample | None:
        start_time_sec = float(event["start_time_sec"])
        end_time_sec = float(event["end_time_sec"])
        if start_time_sec <= float(self.config.intro_padding_sec):
            return None
        if end_time_sec >= (video_duration_sec - float(self.config.outro_padding_sec)):
            return None

        shot_ids = [int(shot_id) for shot_id in event["shot_ids"]]
        if len(shot_ids) < self.config.min_event_shots:
            return None
        shot_ids = shot_ids[: self.config.max_event_shots]

        shot_metadata = []
        shot_start_times: List[float] = []
        shot_end_times: List[float] = []
        keyframe_vectors = []
        keyframe_metadata = []
        keyframe_mask = []

        for local_index, shot_id in enumerate(shot_ids):
            shot = shot_lookup.get(shot_id)
            if shot is None:
                return None
            keyframe_arrays = self._build_keyframe_arrays(shot, fps=fps)
            if keyframe_arrays is None:
                return None
            shot_metadata.append(
                self._make_shot_metadata(
                    shot,
                    local_index=local_index,
                    total_shots=len(shot_ids),
                    event_start=start_time_sec,
                    event_end=end_time_sec,
                )
            )
            shot_start_times.append(float(shot["start_time_sec"]))
            shot_end_times.append(float(shot["end_time_sec"]))
            shot_keyframes, shot_keyframe_metadata, shot_keyframe_mask = keyframe_arrays
            keyframe_vectors.append(shot_keyframes)
            keyframe_metadata.append(shot_keyframe_metadata)
            keyframe_mask.append(shot_keyframe_mask)

        subtitle_overlap_scores = self._compute_subtitle_overlap_scores(
            shot_ids=shot_ids,
            shot_lookup=shot_lookup,
            subtitles=subtitles,
        )
        caption_text = str(caption_text).strip()
        translated_caption = str(translated_caption).strip() or caption_text

        num_shots = len(shot_ids)
        max_keyframes = max(item.shape[0] for item in keyframe_vectors)
        embedding_dim = int(keyframe_vectors[0].shape[-1])
        key_meta_dim = int(keyframe_metadata[0].shape[-1])

        keyframe_vectors_array = np.zeros((num_shots, max_keyframes, embedding_dim), dtype=np.float32)
        keyframe_metadata_array = np.zeros((num_shots, max_keyframes, key_meta_dim), dtype=np.float32)
        keyframe_mask_array = np.zeros((num_shots, max_keyframes), dtype=bool)

        for shot_index in range(num_shots):
            length = int(keyframe_vectors[shot_index].shape[0])
            keyframe_vectors_array[shot_index, :length] = keyframe_vectors[shot_index]
            keyframe_metadata_array[shot_index, :length] = keyframe_metadata[shot_index]
            keyframe_mask_array[shot_index, :length] = keyframe_mask[shot_index]

        return HierarchicalTemporalSample(
            video_id=video_id,
            event_id=int(event["event_id"]),
            caption=caption_text,
            translated_caption=translated_caption,
            start_time_sec=start_time_sec,
            end_time_sec=end_time_sec,
            video_duration_sec=video_duration_sec,
            shot_ids=shot_ids,
            shot_metadata=np.stack(shot_metadata).astype(np.float32),
            shot_start_times=shot_start_times,
            shot_end_times=shot_end_times,
            subtitle_overlap_scores=subtitle_overlap_scores.astype(np.float32),
            keyframe_vectors=keyframe_vectors_array,
            keyframe_metadata=keyframe_metadata_array,
            keyframe_mask=keyframe_mask_array,
        )

    def build_samples(self) -> List[HierarchicalTemporalSample]:
        samples: List[HierarchicalTemporalSample] = []
        summary = {
            "intro_padding_sec": float(self.config.intro_padding_sec),
            "outro_padding_sec": float(self.config.outro_padding_sec),
            "translated_captions_dir": self.config.translated_captions_dir,
            "videos": [],
        }

        captions_root = self.config.translated_captions_dir or self.config.event_captions_dir
        for caption_path in sorted(captions_root.glob("*.json")):
            video_id = caption_path.stem
            event_path = self.config.event_dir / video_id / "events.json"
            keyframe_path = self.config.shot_keyframe_dir / f"{video_id}.pkl"
            if not event_path.exists() or not keyframe_path.exists():
                continue

            event_items = load_json(event_path)
            caption_items = load_json(caption_path)
            if not isinstance(event_items, list) or not isinstance(caption_items, list):
                continue

            event_by_id = {int(item["event_id"]): item for item in event_items}
            video_duration_sec, fps, shots = self._load_video_duration_and_shots(video_id)
            subtitles = self._load_raw_subtitles(video_id, fps)
            shot_lookup = self._build_shot_lookup(shots)

            total_captions = 0
            kept_captions = 0
            kept_events = []
            for caption_item in caption_items:
                total_captions += 1
                event_id = int(caption_item["event_id"])
                event = event_by_id.get(event_id)
                caption_text = str(caption_item.get("caption", "")).strip()
                if event is None or not caption_text:
                    continue
                sample = self._build_sample(
                    video_id=video_id,
                    event=event,
                    caption_text=caption_text,
                    translated_caption=str(caption_item.get("translated_caption", "")).strip() or caption_text,
                    shot_lookup=shot_lookup,
                    video_duration_sec=video_duration_sec,
                    subtitles=subtitles,
                    fps=fps,
                )
                if sample is None:
                    continue
                samples.append(sample)
                kept_captions += 1
                kept_events.append(
                    {
                        "event_id": sample.event_id,
                        "translated_caption": sample.translated_caption,
                        "num_shots": len(sample.shot_ids),
                        "max_keyframes_per_shot": int(sample.keyframe_vectors.shape[1]),
                    }
                )

            summary["videos"].append(
                {
                    "video_id": video_id,
                    "video_duration_sec": video_duration_sec,
                    "num_event_items": len(event_items),
                    "num_caption_items": total_captions,
                    "num_kept_caption_items": kept_captions,
                    "kept_events_preview": kept_events[:10],
                }
            )

        save_json(summary, self.config.output_dir / "dataset_summary.json")
        return samples

    def build_dataset(self) -> HierarchicalTemporalEventDataset:
        samples = self.build_samples()
        if not samples:
            raise RuntimeError("Khong co sample nao sau khi filter event captions.")
        return HierarchicalTemporalEventDataset(samples=samples)


class HierarchicalTemporalEncodeDatasetBuilder(HierarchicalTemporalEventDatasetBuilder):
    def build_samples(self) -> List[HierarchicalTemporalSample]:
        samples: List[HierarchicalTemporalSample] = []
        summary = {
            "intro_padding_sec": float(self.config.intro_padding_sec),
            "outro_padding_sec": float(self.config.outro_padding_sec),
            "mode": "encode_visual_only",
            "videos": [],
        }

        for event_dir in sorted(self.config.event_dir.iterdir()):
            if not event_dir.is_dir():
                continue
            video_id = event_dir.name
            event_path = event_dir / "events.json"
            keyframe_path = self.config.shot_keyframe_dir / f"{video_id}.pkl"
            if not event_path.exists() or not keyframe_path.exists():
                continue

            event_items = load_json(event_path)
            if not isinstance(event_items, list):
                continue

            video_duration_sec, fps, shots = self._load_video_duration_and_shots(video_id)
            subtitles = self._load_raw_subtitles(video_id, fps)
            shot_lookup = self._build_shot_lookup(shots)

            kept_events = []
            kept_count = 0
            for event in event_items:
                sample = self._build_sample(
                    video_id=video_id,
                    event=event,
                    caption_text="",
                    translated_caption="",
                    shot_lookup=shot_lookup,
                    video_duration_sec=video_duration_sec,
                    subtitles=subtitles,
                    fps=fps,
                )
                if sample is None:
                    continue
                samples.append(sample)
                kept_count += 1
                kept_events.append(
                    {
                        "event_id": sample.event_id,
                        "num_shots": len(sample.shot_ids),
                        "max_keyframes_per_shot": int(sample.keyframe_vectors.shape[1]),
                    }
                )

            summary["videos"].append(
                {
                    "video_id": video_id,
                    "video_duration_sec": video_duration_sec,
                    "num_event_items": len(event_items),
                    "num_kept_event_items": kept_count,
                    "kept_events_preview": kept_events[:10],
                }
            )

        save_json(summary, self.config.output_dir / "dataset_summary.json")
        return samples

    def build_dataset(self) -> HierarchicalTemporalEventDataset:
        samples = self.build_samples()
        if not samples:
            raise RuntimeError("Khong co sample nao sau khi filter event boundaries.")
        return HierarchicalTemporalEventDataset(samples=samples)


def hierarchical_temporal_event_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    batch_size = len(batch)
    max_shots = max(int(item["shot_metadata"].shape[0]) for item in batch)
    max_keyframes = max(int(item["keyframe_vectors"].shape[1]) for item in batch)
    key_dim = int(batch[0]["keyframe_vectors"].shape[-1])
    shot_meta_dim = int(batch[0]["shot_metadata"].shape[-1])
    key_meta_dim = int(batch[0]["keyframe_metadata"].shape[-1])

    shot_metadata = torch.zeros(batch_size, max_shots, shot_meta_dim, dtype=torch.float32)
    shot_mask = torch.zeros(batch_size, max_shots, dtype=torch.bool)
    subtitle_overlap_scores = torch.zeros(batch_size, max_shots, dtype=torch.float32)
    keyframe_vectors = torch.zeros(batch_size, max_shots, max_keyframes, key_dim, dtype=torch.float32)
    keyframe_metadata = torch.zeros(batch_size, max_shots, max_keyframes, key_meta_dim, dtype=torch.float32)
    keyframe_mask = torch.zeros(batch_size, max_shots, max_keyframes, dtype=torch.bool)

    collated = {
        "video_ids": [],
        "event_ids": [],
        "captions": [],
        "translated_captions": [],
        "start_time_sec": [],
        "end_time_sec": [],
        "video_duration_sec": [],
        "shot_ids": [],
        "shot_start_times": [],
        "shot_end_times": [],
    }

    for batch_index, item in enumerate(batch):
        num_shots = int(item["shot_metadata"].shape[0])
        local_max_keyframes = int(item["keyframe_vectors"].shape[1])

        shot_metadata[batch_index, :num_shots] = item["shot_metadata"]
        shot_mask[batch_index, :num_shots] = True
        subtitle_overlap_scores[batch_index, :num_shots] = item["subtitle_overlap_scores"]
        keyframe_vectors[batch_index, :num_shots, :local_max_keyframes] = item["keyframe_vectors"]
        keyframe_metadata[batch_index, :num_shots, :local_max_keyframes] = item["keyframe_metadata"]
        keyframe_mask[batch_index, :num_shots, :local_max_keyframes] = item["keyframe_mask"]

        for key_src, key_dst in [
            ("video_id", "video_ids"),
            ("event_id", "event_ids"),
            ("caption", "captions"),
            ("translated_caption", "translated_captions"),
            ("start_time_sec", "start_time_sec"),
            ("end_time_sec", "end_time_sec"),
            ("video_duration_sec", "video_duration_sec"),
            ("shot_ids", "shot_ids"),
            ("shot_start_times", "shot_start_times"),
            ("shot_end_times", "shot_end_times"),
        ]:
            collated[key_dst].append(item[key_src])

    collated["shot_metadata"] = shot_metadata
    collated["shot_mask"] = shot_mask
    collated["subtitle_overlap_scores"] = subtitle_overlap_scores
    collated["keyframe_vectors"] = keyframe_vectors
    collated["keyframe_metadata"] = keyframe_metadata
    collated["keyframe_mask"] = keyframe_mask
    return collated
