from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .config import EventGroupingDatasetConfig


class DPSegmenter:
    def __init__(self, config: EventGroupingDatasetConfig):
        self.config = config

    def event_duration(self, shot_table: List[Dict[str, Any]], start: int, end_exclusive: int) -> float:
        return float(shot_table[end_exclusive - 1]["end_time_sec"] - shot_table[start]["start_time_sec"])

    def is_forced_singleton_long_shot(self, shot_table: List[Dict[str, Any]], shot_index: int) -> bool:
        return float(shot_table[shot_index]["duration_sec"]) > self.config.max_event_duration_sec

    def transition_valid(self, shot_table: List[Dict[str, Any]], start: int, end_exclusive: int) -> bool:
        num_shots = end_exclusive - start
        dur = self.event_duration(shot_table, start, end_exclusive)
        if num_shots == 1 and self.is_forced_singleton_long_shot(shot_table, start):
            return True
        if dur < self.config.min_event_duration_sec:
            return False
        if dur > self.config.max_event_duration_sec:
            return False
        return True

    def cut_reward_after_event(self, end_exclusive: int, n: int, boundary_rows: List[Dict[str, Any]]) -> float:
        if end_exclusive >= n:
            return 0.0
        b_idx = end_exclusive - 1
        row = boundary_rows[b_idx]
        reward = float(row["boundary_score"]) - self.config.cut_penalty
        if not row.get("is_candidate", False):
            reward -= self.config.non_candidate_penalty
        reward -= float(row.get("subtitle_bridge_penalty", 0.0))
        return float(reward)

    def _segment_chunk(self, shot_table: List[Dict[str, Any]], boundary_rows: List[Dict[str, Any]], start_offset: int = 0):
        n = len(shot_table)
        NEG_INF = -1e18
        dp = np.full(n + 1, NEG_INF, dtype=np.float64)
        backptr = [None] * (n + 1)
        dp[0] = 0.0

        for end in range(1, n + 1):
            best_score = NEG_INF
            best_start = None
            for start in range(0, end):
                if dp[start] <= NEG_INF / 2:
                    continue
                if not self.transition_valid(shot_table, start, end):
                    continue
                reward = self.cut_reward_after_event(end, n, boundary_rows)
                score = dp[start] + reward
                if score > best_score:
                    best_score = score
                    best_start = start
            dp[end] = best_score
            backptr[end] = best_start

        if backptr[n] is None:
            raise RuntimeError(
                f"DP không tìm được segmentation hợp lệ cho chunk bắt đầu tại shot index {start_offset}. "
                "Hãy giảm min_event_duration_sec hoặc tăng max_event_duration_sec."
            )

        ranges = []
        cur = n
        while cur > 0:
            prev = backptr[cur]
            if prev is None:
                raise RuntimeError("Backpointer bị lỗi trong lúc reconstruct.")
            ranges.append((prev + start_offset, cur + start_offset))
            cur = prev
        ranges.reverse()
        return ranges

    def segment(self, shot_table: List[Dict[str, Any]], boundary_rows: List[Dict[str, Any]]):
        n = len(shot_table)
        forced_indices = [idx for idx in range(n) if self.is_forced_singleton_long_shot(shot_table, idx)]
        if not forced_indices:
            ranges = self._segment_chunk(shot_table, boundary_rows, start_offset=0)
            return ranges, None, None

        ranges = []
        chunk_start = 0
        pending_prefix_start = None

        for forced_idx in forced_indices:
            if chunk_start < forced_idx:
                chunk_duration = self.event_duration(shot_table, chunk_start, forced_idx)
                if chunk_duration < self.config.min_event_duration_sec:
                    if ranges:
                        prev_start, _ = ranges[-1]
                        ranges[-1] = (prev_start, forced_idx)
                    else:
                        pending_prefix_start = chunk_start
                else:
                    chunk_ranges = self._segment_chunk(
                        shot_table[chunk_start:forced_idx],
                        boundary_rows[chunk_start:forced_idx - 1] if forced_idx - chunk_start >= 2 else [],
                        start_offset=chunk_start,
                    )
                    ranges.extend(chunk_ranges)

            event_start = pending_prefix_start if pending_prefix_start is not None else forced_idx
            ranges.append((event_start, forced_idx + 1))
            pending_prefix_start = None
            chunk_start = forced_idx + 1

        if chunk_start < n:
            chunk_duration = self.event_duration(shot_table, chunk_start, n)
            if chunk_duration < self.config.min_event_duration_sec and ranges:
                prev_start, _ = ranges[-1]
                ranges[-1] = (prev_start, n)
            else:
                chunk_ranges = self._segment_chunk(
                    shot_table[chunk_start:n],
                    boundary_rows[chunk_start:n - 1] if n - chunk_start >= 2 else [],
                    start_offset=chunk_start,
                )
                ranges.extend(chunk_ranges)

        return ranges, None, None
