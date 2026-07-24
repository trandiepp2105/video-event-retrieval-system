import math
from typing import Any

import numpy as np

from .config import SlowFastShotFeatureConfig


def build_subshot_ranges(
    shot: dict[str, Any],
    config: SlowFastShotFeatureConfig,
) -> list[tuple[float, float]]:
    start_sec = float(shot["start_time_sec"])
    end_sec = float(shot["end_time_sec"])
    if end_sec <= start_sec:
        end_sec = start_sec + 1e-3

    shot_duration = end_sec - start_sec
    if shot_duration <= config.clip_duration_sec:
        return [(start_sec, end_sec)]

    num_subshots = int(math.ceil(shot_duration / config.clip_duration_sec))
    boundaries = np.linspace(start_sec, end_sec, num_subshots + 1)
    ranges: list[tuple[float, float]] = []
    for idx in range(num_subshots):
        sub_start = float(boundaries[idx])
        sub_end = float(boundaries[idx + 1])
        if sub_end <= sub_start:
            sub_end = sub_start + 1e-3
        ranges.append((sub_start, sub_end))
    return ranges


def build_all_subshot_records(
    shots: list[dict[str, Any]],
    config: SlowFastShotFeatureConfig,
) -> list[dict[str, Any]]:
    subshot_records: list[dict[str, Any]] = []
    for shot in shots:
        for subshot_index, (clip_start, clip_end) in enumerate(build_subshot_ranges(shot, config)):
            subshot_records.append(
                {
                    "shot_id": int(shot["shot_id"]),
                    "subshot_index": int(subshot_index),
                    "clip_start": float(clip_start),
                    "clip_end": float(clip_end),
                }
            )
    subshot_records.sort(key=lambda record: (record["clip_start"], record["clip_end"], record["shot_id"]))
    return subshot_records
