from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import EncodeConfig
from .dataset import HierarchicalTemporalEncodeDatasetBuilder, hierarchical_temporal_event_collate_fn
from .io_utils import ensure_dir, l2_normalize, save_json, save_pickle
from .train import build_model_from_batch


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32)


def encode(config: EncodeConfig):
    ensure_dir(config.output_dir)
    dataset = HierarchicalTemporalEncodeDatasetBuilder(config).build_dataset()
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=hierarchical_temporal_event_collate_fn,
    )
    first_batch = next(iter(loader))
    model = build_model_from_batch(type("Cfg", (), asdict(config))(), first_batch).to(config.device)

    if config.checkpoint_path is not None:
        checkpoint = torch.load(config.checkpoint_path, map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    events_by_video: dict[str, list[dict]] = defaultdict(list)
    shots_by_video: dict[str, list[dict]] = defaultdict(list)

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Encoding hierarchical temporal embeddings"):
            keyframe_vectors = batch["keyframe_vectors"].to(config.device)
            keyframe_metadata = batch["keyframe_metadata"].to(config.device)
            keyframe_mask = batch["keyframe_mask"].to(config.device)
            shot_metadata = batch["shot_metadata"].to(config.device)
            shot_mask = batch["shot_mask"].to(config.device)

            outputs = model.encode_visual(
                keyframe_vectors=keyframe_vectors,
                keyframe_metadata=keyframe_metadata,
                keyframe_mask=keyframe_mask,
                shot_metadata=shot_metadata,
                shot_mask=shot_mask,
            )
            event_embeddings = _to_numpy(outputs["event_embeddings"])
            shot_embeddings = _to_numpy(outputs["shot_embeddings"])
            shot_clip_embeddings = _to_numpy(outputs["shot_clip_embeddings"])
            shot_attention_weights = _to_numpy(outputs["shot_attention_weights"])
            keyframe_attention_weights = _to_numpy(outputs["keyframe_attention_weights"])

            for batch_index, video_id in enumerate(batch["video_ids"]):
                valid_len = int(batch["shot_mask"][batch_index].sum().item())
                event_record = {
                    "event_id": int(batch["event_ids"][batch_index]),
                    "start_time_sec": float(batch["start_time_sec"][batch_index]),
                    "end_time_sec": float(batch["end_time_sec"][batch_index]),
                    "caption": "",
                    "translated_caption": "",
                    "embedding": event_embeddings[batch_index].tolist(),
                    "num_shots": valid_len,
                    "shot_attention_weights": shot_attention_weights[batch_index, :valid_len].tolist(),
                }
                events_by_video[video_id].append(event_record)

                for local_shot_index in range(valid_len):
                    num_keyframes = int(batch["keyframe_mask"][batch_index, local_shot_index].sum().item())
                    shots_by_video[video_id].append(
                        {
                            "event_id": int(batch["event_ids"][batch_index]),
                            "shot_id": int(batch["shot_ids"][batch_index][local_shot_index]),
                            "start_time_sec": float(batch["shot_start_times"][batch_index][local_shot_index]),
                            "end_time_sec": float(batch["shot_end_times"][batch_index][local_shot_index]),
                            "embedding": shot_embeddings[batch_index, local_shot_index].tolist(),
                            "clip_space_embedding": shot_clip_embeddings[batch_index, local_shot_index].tolist(),
                            "subtitle_overlap_score": float(batch["subtitle_overlap_scores"][batch_index][local_shot_index]),
                            "num_keyframes": num_keyframes,
                            "keyframe_attention_weights": keyframe_attention_weights[
                                batch_index,
                                local_shot_index,
                                :num_keyframes,
                            ].tolist(),
                        }
                    )

    for video_id, event_items in events_by_video.items():
        video_dir = config.output_dir / video_id
        ensure_dir(video_dir)
        event_matrix = np.asarray([item["embedding"] for item in event_items], dtype=np.float32)
        np.save(video_dir / "event_temporal_embeddings.npy", l2_normalize(event_matrix))
        save_json(
            [
                {
                    "event_id": item["event_id"],
                    "start_time_sec": item["start_time_sec"],
                    "end_time_sec": item["end_time_sec"],
                    "caption": item["caption"],
                    "translated_caption": item["translated_caption"],
                    "num_shots": item["num_shots"],
                }
                for item in event_items
            ],
            video_dir / "events_temporal.json",
        )

    for video_id, shot_items in shots_by_video.items():
        save_pickle(
            {
                "video_id": video_id,
                "shots": shot_items,
            },
            config.output_dir / "shot_temporal" / f"{video_id}.pkl",
        )

    save_json(
        {
            "config": asdict(config),
            "num_videos": len(events_by_video),
            "num_events": sum(len(items) for items in events_by_video.values()),
            "num_shots": sum(len(items) for items in shots_by_video.values()),
        },
        config.output_dir / "encode_summary.json",
    )
    return config.output_dir
