from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .io_utils import ensure_dir, load_json, load_pickle, save_json, save_pickle
from .model import HierarchicalTemporalRetrievalModel


@dataclass
class VideoDatasetEncodeConfig:
    event_dir: Path
    visual_embedding_dir: Path
    checkpoint_path: Path
    event_output_dir: Path
    shot_output_dir: Path
    raw_subtitle_dir: Path | None = None
    clip_model_path_override: Path | None = None
    start_index: int = 0
    end_index: int | None = None
    video_ids: list[str] | None = None
    device: str = "cuda"
    save_dtype: str = "float32"
    overwrite: bool = False


def _to_numpy(tensor: torch.Tensor, save_dtype: str) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if save_dtype == "float16":
        return array.astype(np.float16)
    return array.astype(np.float32)


def _as_runtime_device(device: str) -> str:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return str(device)
    return "cpu"


def _load_checkpoint(path: Path, device: str) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def _infer_keyframe_dim(visual_embedding_path: Path) -> int:
    payload = load_pickle(visual_embedding_path)
    for shot in payload["shots"]:
        embeddings = np.asarray(shot["keyframe_embeddings"])
        if embeddings.ndim == 2 and embeddings.shape[0] > 0:
            return int(embeddings.shape[-1])
    raise RuntimeError(f"Khong suy ra duoc keyframe_dim tu file: {visual_embedding_path}")


def _build_model(
    *,
    checkpoint: dict[str, Any],
    visual_embedding_path: Path,
    device: str,
    clip_model_path_override: Path | None,
) -> tuple[HierarchicalTemporalRetrievalModel, dict[str, Any]]:
    config = checkpoint["config"]
    keyframe_dim = _infer_keyframe_dim(visual_embedding_path)
    clip_model_path = clip_model_path_override or config["clip_model_path"]
    model = HierarchicalTemporalRetrievalModel(
        keyframe_dim=keyframe_dim,
        keyframe_metadata_dim=4,
        shot_metadata_dim=4,
        clip_model_path=str(clip_model_path),
        clip_model_name=config.get("clip_model_name", "ViT-H-14-quickgelu"),
        device=device,
        keyframe_hidden_dim=int(config["keyframe_hidden_dim"]),
        shot_hidden_dim=int(config["shot_hidden_dim"]),
        projection_dim=int(config["projection_dim"]),
        keyframe_transformer_layers=int(config["keyframe_transformer_layers"]),
        shot_transformer_layers=int(config["shot_transformer_layers"]),
        keyframe_transformer_heads=int(config["keyframe_transformer_heads"]),
        shot_transformer_heads=int(config["shot_transformer_heads"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    return model, config


def _segment_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _load_raw_subtitles(video_id: str, raw_subtitle_dir: Path | None, fps: float) -> list[dict[str, Any]]:
    if raw_subtitle_dir is None:
        return []
    subtitle_path = raw_subtitle_dir / f"{video_id}.json"
    if not subtitle_path.exists():
        return []
    items = load_json(subtitle_path)
    subtitles: list[dict[str, Any]] = []
    for item in items:
        if "start_time_sec" in item and "end_time_sec" in item:
            start_time_sec = float(item["start_time_sec"])
            end_time_sec = float(item["end_time_sec"])
        else:
            frame_start = float(item.get("frame_start", 0.0))
            frame_end = float(item.get("frame_end", frame_start))
            start_time_sec = frame_start / max(fps, 1e-6)
            end_time_sec = frame_end / max(fps, 1e-6)
        subtitles.append(
            {
                "start_time_sec": start_time_sec,
                "end_time_sec": end_time_sec,
                "text": str(item.get("text", "")).strip(),
            }
        )
    return subtitles


def _make_shot_metadata(
    shot: dict[str, Any],
    *,
    local_index: int,
    total_shots: int,
    event_start: float,
    event_end: float,
    duration_norm_sec: float,
) -> np.ndarray:
    shot_start = float(shot["start_time_sec"])
    shot_end = float(shot["end_time_sec"])
    shot_duration = float(shot["duration_sec"])
    event_duration = max(float(event_end) - float(event_start), 1e-6)
    rel_start = (shot_start - float(event_start)) / event_duration
    rel_end = (shot_end - float(event_start)) / event_duration
    position = 0.0 if total_shots <= 1 else local_index / (total_shots - 1)
    duration_norm = min(shot_duration / max(float(duration_norm_sec), 1e-6), 1.0)
    return np.asarray([position, duration_norm, rel_start, rel_end], dtype=np.float32)


def _make_keyframe_metadata(
    shot: dict[str, Any],
    *,
    local_index: int,
    total_keyframes: int,
    frame_index: int,
    previous_frame_index: int | None,
    next_frame_index: int | None,
    fps: float,
    time_norm_sec: float,
) -> np.ndarray:
    shot_start_sec = float(shot["start_time_sec"])
    shot_duration_sec = max(float(shot["duration_sec"]), 1e-6)
    frame_time_sec = float(frame_index) / max(fps, 1e-6)
    relative_position = 0.0 if total_keyframes <= 1 else local_index / (total_keyframes - 1)
    relative_time = (frame_time_sec - shot_start_sec) / shot_duration_sec
    delta_prev_sec = 0.0 if previous_frame_index is None else max(frame_index - previous_frame_index, 0) / max(fps, 1e-6)
    delta_next_sec = 0.0 if next_frame_index is None else max(next_frame_index - frame_index, 0) / max(fps, 1e-6)
    delta_prev_norm = min(delta_prev_sec / max(time_norm_sec, 1e-6), 1.0)
    delta_next_norm = min(delta_next_sec / max(time_norm_sec, 1e-6), 1.0)
    return np.asarray(
        [relative_position, relative_time, delta_prev_norm, delta_next_norm],
        dtype=np.float32,
    )


def _compute_subtitle_overlap_scores(
    *,
    shots: list[dict[str, Any]],
    subtitles: list[dict[str, Any]],
    subtitle_norm_sec: float,
) -> np.ndarray:
    scores: list[float] = []
    for shot in shots:
        shot_start = float(shot["start_time_sec"])
        shot_end = float(shot["end_time_sec"])
        overlap_sec = 0.0
        for subtitle in subtitles:
            overlap_sec += _segment_overlap(
                shot_start,
                shot_end,
                float(subtitle["start_time_sec"]),
                float(subtitle["end_time_sec"]),
            )
        scores.append(min(overlap_sec / max(subtitle_norm_sec, 1e-6), 1.0))
    return np.asarray(scores, dtype=np.float32)


def _build_event_tensors(
    *,
    event: dict[str, Any],
    shot_lookup: dict[int, dict[str, Any]],
    fps: float,
    max_event_shots: int,
    max_keyframes_per_shot: int,
    shot_duration_norm_sec: float,
    keyframe_time_norm_sec: float,
    subtitles: list[dict[str, Any]],
    subtitle_norm_sec: float,
    device: str,
) -> dict[str, Any] | None:
    shot_ids = [int(shot_id) for shot_id in event["shot_ids"]][: int(max_event_shots)]
    if not shot_ids:
        return None

    shot_infos: list[dict[str, Any]] = []
    shot_metadata: list[np.ndarray] = []
    shot_start_times: list[float] = []
    shot_end_times: list[float] = []
    shot_keyframe_vectors: list[np.ndarray] = []
    shot_keyframe_metadata: list[np.ndarray] = []
    shot_keyframe_mask: list[np.ndarray] = []

    for local_shot_index, shot_id in enumerate(shot_ids):
        shot = shot_lookup.get(shot_id)
        if shot is None:
            return None
        keyframe_vectors = np.asarray(shot["keyframe_embeddings"], dtype=np.float32)[: int(max_keyframes_per_shot)]
        keyframe_frame_indices = [int(x) for x in shot["keyframe_frame_indices"][: int(max_keyframes_per_shot)]]
        if keyframe_vectors.ndim != 2 or keyframe_vectors.shape[0] == 0:
            return None

        shot_infos.append(shot)
        shot_start_times.append(float(shot["start_time_sec"]))
        shot_end_times.append(float(shot["end_time_sec"]))
        shot_metadata.append(
            _make_shot_metadata(
                shot,
                local_index=local_shot_index,
                total_shots=len(shot_ids),
                event_start=float(event["start_time_sec"]),
                event_end=float(event["end_time_sec"]),
                duration_norm_sec=shot_duration_norm_sec,
            )
        )

        key_meta: list[np.ndarray] = []
        for local_key_index, frame_index in enumerate(keyframe_frame_indices):
            prev_frame = keyframe_frame_indices[local_key_index - 1] if local_key_index > 0 else None
            next_frame = keyframe_frame_indices[local_key_index + 1] if local_key_index + 1 < len(keyframe_frame_indices) else None
            key_meta.append(
                _make_keyframe_metadata(
                    shot,
                    local_index=local_key_index,
                    total_keyframes=len(keyframe_frame_indices),
                    frame_index=frame_index,
                    previous_frame_index=prev_frame,
                    next_frame_index=next_frame,
                    fps=fps,
                    time_norm_sec=keyframe_time_norm_sec,
                )
            )

        shot_keyframe_vectors.append(keyframe_vectors.astype(np.float32))
        shot_keyframe_metadata.append(np.stack(key_meta).astype(np.float32))
        shot_keyframe_mask.append(np.ones((keyframe_vectors.shape[0],), dtype=bool))

    subtitle_overlap_scores = _compute_subtitle_overlap_scores(
        shots=shot_infos,
        subtitles=subtitles,
        subtitle_norm_sec=subtitle_norm_sec,
    )

    num_shots = len(shot_ids)
    max_keyframes = max(arr.shape[0] for arr in shot_keyframe_vectors)
    key_dim = int(shot_keyframe_vectors[0].shape[-1])
    key_meta_dim = int(shot_keyframe_metadata[0].shape[-1])

    keyframe_vectors_tensor = np.zeros((1, num_shots, max_keyframes, key_dim), dtype=np.float32)
    keyframe_metadata_tensor = np.zeros((1, num_shots, max_keyframes, key_meta_dim), dtype=np.float32)
    keyframe_mask_tensor = np.zeros((1, num_shots, max_keyframes), dtype=bool)

    for shot_index in range(num_shots):
        length = int(shot_keyframe_vectors[shot_index].shape[0])
        keyframe_vectors_tensor[0, shot_index, :length] = shot_keyframe_vectors[shot_index]
        keyframe_metadata_tensor[0, shot_index, :length] = shot_keyframe_metadata[shot_index]
        keyframe_mask_tensor[0, shot_index, :length] = shot_keyframe_mask[shot_index]

    shot_metadata_tensor = np.stack(shot_metadata).astype(np.float32)[None, ...]
    shot_mask_tensor = np.ones((1, num_shots), dtype=bool)

    return {
        "event": event,
        "shot_ids": shot_ids,
        "shot_infos": shot_infos,
        "shot_start_times": shot_start_times,
        "shot_end_times": shot_end_times,
        "subtitle_overlap_scores": subtitle_overlap_scores,
        "keyframe_vectors": torch.from_numpy(keyframe_vectors_tensor).to(device),
        "keyframe_metadata": torch.from_numpy(keyframe_metadata_tensor).to(device),
        "keyframe_mask": torch.from_numpy(keyframe_mask_tensor).to(device),
        "shot_metadata": torch.from_numpy(shot_metadata_tensor).to(device),
        "shot_mask": torch.from_numpy(shot_mask_tensor).to(device),
    }


def _encode_event(model: HierarchicalTemporalRetrievalModel, batch: dict[str, Any], save_dtype: str) -> dict[str, Any]:
    with torch.no_grad():
        outputs = model.encode_visual(
            keyframe_vectors=batch["keyframe_vectors"],
            keyframe_metadata=batch["keyframe_metadata"],
            keyframe_mask=batch["keyframe_mask"],
            shot_metadata=batch["shot_metadata"],
            shot_mask=batch["shot_mask"],
        )
    return {
        "event_embedding": _to_numpy(outputs["event_embeddings"][0], save_dtype),
        "shot_embeddings": _to_numpy(outputs["shot_embeddings"][0], save_dtype),
        "shot_clip_embeddings": _to_numpy(outputs["shot_clip_embeddings"][0], save_dtype),
        "shot_attention_weights": _to_numpy(outputs["shot_attention_weights"][0], "float32"),
        "keyframe_attention_weights": _to_numpy(outputs["keyframe_attention_weights"][0], "float32"),
    }


def _list_target_video_ids(config: VideoDatasetEncodeConfig) -> list[str]:
    event_video_ids = sorted(
        [
            path.name
            for path in config.event_dir.iterdir()
            if path.is_dir() and (path / "events.json").exists()
        ],
        key=lambda value: int(value),
    )

    if config.video_ids:
        selected_ids = {str(video_id) for video_id in config.video_ids}
        return [video_id for video_id in event_video_ids if video_id in selected_ids]

    start_index = max(int(config.start_index), 0)
    if config.end_index is None:
        return event_video_ids[start_index:]
    return event_video_ids[start_index : int(config.end_index) + 1]


def encode_video_dataset(config: VideoDatasetEncodeConfig) -> dict[str, Any]:
    runtime_device = _as_runtime_device(config.device)
    ensure_dir(config.event_output_dir)
    ensure_dir(config.shot_output_dir)

    target_video_ids = _list_target_video_ids(config)
    if not target_video_ids:
        raise RuntimeError("Khong tim thay video nao de encode.")

    sample_visual_path = None
    for video_id in target_video_ids:
        candidate = config.visual_embedding_dir / f"{video_id}.pkl"
        if candidate.exists():
            sample_visual_path = candidate
            break
    if sample_visual_path is None:
        raise RuntimeError("Khong tim thay file visual embedding nao tu danh sach video da chon.")

    checkpoint = _load_checkpoint(config.checkpoint_path, runtime_device)
    model, train_config = _build_model(
        checkpoint=checkpoint,
        visual_embedding_path=sample_visual_path,
        device=runtime_device,
        clip_model_path_override=config.clip_model_path_override,
    )

    summary: dict[str, Any] = {
        "done": [],
        "skipped": [],
        "failed": [],
    }

    for video_id in tqdm(target_video_ids, desc="Encoding videos"):
        event_path = config.event_dir / video_id / "events.json"
        visual_path = config.visual_embedding_dir / f"{video_id}.pkl"
        event_output_path = config.event_output_dir / f"{video_id}.pkl"
        shot_output_path = config.shot_output_dir / f"{video_id}.pkl"

        if not event_path.exists() or not visual_path.exists():
            summary["failed"].append(
                {
                    "video_id": video_id,
                    "error": "Missing events.json or visual embedding file.",
                }
            )
            continue

        if (
            event_output_path.exists()
            and shot_output_path.exists()
            and not config.overwrite
        ):
            summary["skipped"].append(video_id)
            continue

        try:
            event_items = load_json(event_path)
            visual_payload = load_pickle(visual_path)
            fps = float(visual_payload["fps"])
            subtitles = _load_raw_subtitles(video_id, config.raw_subtitle_dir, fps)
            shot_lookup = {
                int(shot["shot_id"]): shot
                for shot in visual_payload["shots"]
            }

            video_event_items: list[dict[str, Any]] = []
            video_shot_items: list[dict[str, Any]] = []

            for event in event_items:
                batch = _build_event_tensors(
                    event=event,
                    shot_lookup=shot_lookup,
                    fps=fps,
                    max_event_shots=int(train_config["max_event_shots"]),
                    max_keyframes_per_shot=int(train_config["max_keyframes_per_shot"]),
                    shot_duration_norm_sec=float(train_config["shot_metadata_duration_norm_sec"]),
                    keyframe_time_norm_sec=float(train_config["keyframe_metadata_time_norm_sec"]),
                    subtitles=subtitles,
                    subtitle_norm_sec=float(train_config["salient_subtitle_norm_sec"]),
                    device=runtime_device,
                )
                if batch is None:
                    continue
                encoded = _encode_event(model, batch, config.save_dtype)

                video_event_items.append(
                    {
                        "event_id": int(event["event_id"]),
                        "shot_ids": [int(shot_id) for shot_id in batch["shot_ids"]],
                        "start_time_sec": float(event["start_time_sec"]),
                        "end_time_sec": float(event["end_time_sec"]),
                        "embedding": encoded["event_embedding"],
                        "num_shots": len(batch["shot_ids"]),
                        "shot_attention_weights": encoded["shot_attention_weights"][: len(batch["shot_ids"])],
                    }
                )

                for local_index, shot_id in enumerate(batch["shot_ids"]):
                    shot_info = batch["shot_infos"][local_index]
                    num_keyframes = int(np.asarray(shot_info["keyframe_embeddings"]).shape[0])
                    video_shot_items.append(
                        {
                            "event_id": int(event["event_id"]),
                            "shot_id": int(shot_id),
                            "start_time_sec": float(shot_info["start_time_sec"]),
                            "end_time_sec": float(shot_info["end_time_sec"]),
                            "embedding": encoded["shot_embeddings"][local_index],
                            "clip_space_embedding": encoded["shot_clip_embeddings"][local_index],
                            "subtitle_overlap_score": float(batch["subtitle_overlap_scores"][local_index]),
                            "num_keyframes": num_keyframes,
                            "keyframe_attention_weights": encoded["keyframe_attention_weights"][local_index][
                                :num_keyframes
                            ],
                        }
                    )

            save_pickle(
                {
                    "video_id": video_id,
                    "events": video_event_items,
                },
                event_output_path,
            )
            save_pickle(
                {
                    "video_id": video_id,
                    "shots": video_shot_items,
                },
                shot_output_path,
            )
            summary["done"].append(
                {
                    "video_id": video_id,
                    "num_events": len(video_event_items),
                    "num_shots": len(video_shot_items),
                    "event_output_path": str(event_output_path),
                    "shot_output_path": str(shot_output_path),
                }
            )
        except Exception as exc:
            summary["failed"].append(
                {
                    "video_id": video_id,
                    "error": repr(exc),
                }
            )

    save_json(
        {
            "config": {
                "event_dir": str(config.event_dir),
                "visual_embedding_dir": str(config.visual_embedding_dir),
                "checkpoint_path": str(config.checkpoint_path),
                "event_output_dir": str(config.event_output_dir),
                "shot_output_dir": str(config.shot_output_dir),
                "raw_subtitle_dir": None if config.raw_subtitle_dir is None else str(config.raw_subtitle_dir),
                "clip_model_path_override": None
                if config.clip_model_path_override is None
                else str(config.clip_model_path_override),
                "start_index": config.start_index,
                "end_index": config.end_index,
                "video_ids": config.video_ids,
                "device": runtime_device,
                "save_dtype": config.save_dtype,
                "overwrite": config.overwrite,
            },
            "summary": summary,
        },
        config.event_output_dir.parent / "video_dataset_encode_summary.json",
    )
    return summary
