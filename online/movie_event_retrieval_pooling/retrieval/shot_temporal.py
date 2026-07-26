from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..metadata import MetadataRepository
from ..schemas import ShotResult


@dataclass(frozen=True)
class TemporalStageShot:
    stage_index: int
    shot_id: str
    event_id: str
    video_id: str
    shot_order: int
    start_time_sec: float
    end_time_sec: float
    score: float
    evidence: dict[str, Any]


class ShotTemporalChainService:
    def __init__(self, metadata: MetadataRepository) -> None:
        self.metadata = metadata

    def search(
        self,
        *,
        stage_results: list[list[ShotResult]],
        top_k: int,
        window_size_shots: int,
        lambda_skip: float,
        min_stage_gap_shots: int,
        group_gap_shots: int,
    ) -> list[dict[str, Any]]:
        num_stages = len(stage_results)
        if num_stages == 0:
            return []

        candidate_map: dict[str, dict[str, Any]] = {}
        for stage_idx, results in enumerate(stage_results):
            for shot in results:
                payload = candidate_map.setdefault(
                    shot.shot_id,
                    {
                        "shot": shot,
                        "stage_scores": [0.0] * num_stages,
                        "evidence": {},
                    },
                )
                payload["stage_scores"][stage_idx] = max(payload["stage_scores"][stage_idx], float(shot.score))
                payload["evidence"][stage_idx] = shot.evidence

        if not candidate_map:
            return []

        temporal_candidates = []
        for shot_id, payload in candidate_map.items():
            shot = self.metadata.shots[shot_id]
            temporal_candidates.append(
                {
                    "shot_id": shot_id,
                    "video_id": shot.video_id,
                    "shot_order": int(shot.shot_order),
                    "start_time_sec": float(shot.start_time_sec),
                    "end_time_sec": float(shot.end_time_sec),
                    "event_id": shot.event_id,
                    "stage_scores": tuple(float(x) for x in payload["stage_scores"]),
                    "evidence": payload["evidence"],
                }
            )

        temporal_candidates.sort(key=lambda item: (item["video_id"], item["shot_order"]))
        temporal_groups = self._group_temporal_candidates(temporal_candidates, group_gap_shots=group_gap_shots)

        final_chains: list[dict[str, Any]] = []
        for temporal_group in temporal_groups:
            if not temporal_group:
                continue
            timeline = self._build_timeline(temporal_group)
            aggregated_timeline = self._aggregate_stage_scores_over_windows(
                timeline=timeline,
                num_stages=num_stages,
                window_size_shots=window_size_shots,
            )
            best_score, decision, prev_state = self._soft_temporal_dp(
                aggregated_timeline=aggregated_timeline,
                num_stages=num_stages,
                lambda_skip=lambda_skip,
                min_stage_gap_shots=min_stage_gap_shots,
            )
            best_final_score = 0.0
            best_final_stage = -1
            best_final_time_idx = -1
            for time_idx in range(len(aggregated_timeline)):
                for stage_idx in range(num_stages):
                    if best_score[time_idx][stage_idx] > best_final_score:
                        best_final_score = best_score[time_idx][stage_idx]
                        best_final_stage = stage_idx
                        best_final_time_idx = time_idx
            if best_final_stage == -1 or best_final_score <= 0:
                continue
            chain: list[dict[str, Any]] = []
            skipped_stages = 0
            state = (best_final_time_idx, best_final_stage)
            while state is not None:
                time_idx, stage_idx = state
                action = decision[time_idx][stage_idx]
                if action == "carry":
                    state = prev_state[time_idx][stage_idx]
                    continue
                if action == "skip":
                    skipped_stages += 1
                    state = prev_state[time_idx][stage_idx]
                    continue
                if action in {"start", "transition"}:
                    support = aggregated_timeline[time_idx]["stage_supports"][stage_idx]
                    if support["shot_id"] is not None and support["score"] > 0:
                        chain.append(
                            {
                                "stage_index": stage_idx,
                                "shot_id": support["shot_id"],
                                "score": float(support["score"]),
                            }
                        )
                    state = prev_state[time_idx][stage_idx]
                    continue
                break
            chain.reverse()
            if chain and self._is_valid_temporal_chain(chain, min_stage_gap_shots=min_stage_gap_shots):
                matched_stage_indices = [item["stage_index"] for item in chain]
                stage_payloads = []
                for item in chain:
                    shot = self.metadata.shots[item["shot_id"]]
                    stage_payloads.append(
                        {
                            "stage_index": item["stage_index"],
                            "shot_id": shot.shot_id,
                            "event_id": shot.event_id,
                            "video_id": shot.video_id,
                            "shot_order": int(shot.shot_order),
                            "start_time_sec": float(shot.start_time_sec),
                            "end_time_sec": float(shot.end_time_sec),
                            "score": float(item["score"]),
                        }
                    )
                final_chains.append(
                    {
                        "video_id": stage_payloads[0]["video_id"],
                        "score": float(best_final_score),
                        "num_stages_matched": len(chain),
                        "num_stages_skipped": skipped_stages,
                        "matched_stage_indices": matched_stage_indices,
                        "chain": stage_payloads,
                    }
                )
        final_chains.sort(
            key=lambda item: (
                item["num_stages_matched"],
                -item["num_stages_skipped"],
                item["score"],
            ),
            reverse=True,
        )
        return final_chains[: int(top_k)]

    @staticmethod
    def _group_temporal_candidates(
        temporal_candidates: list[dict[str, Any]],
        *,
        group_gap_shots: int,
    ) -> list[list[dict[str, Any]]]:
        groups: list[list[dict[str, Any]]] = []
        for item in temporal_candidates:
            if not groups:
                groups.append([item])
                continue
            previous = groups[-1][-1]
            if item["video_id"] != previous["video_id"] or item["shot_order"] - previous["shot_order"] > group_gap_shots:
                groups.append([item])
            else:
                groups[-1].append(item)
        return groups

    @staticmethod
    def _build_timeline(temporal_group: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return list(temporal_group)

    @staticmethod
    def _aggregate_stage_scores_over_windows(
        *,
        timeline: list[dict[str, Any]],
        num_stages: int,
        window_size_shots: int,
    ) -> list[dict[str, Any]]:
        aggregated = []
        for center_item in timeline:
            center_shot_order = center_item["shot_order"]
            aggregated_scores = []
            for stage_idx in range(num_stages):
                best_score = 0.0
                best_shot_id = None
                best_shot_order = None
                for neighbor in timeline:
                    if abs(neighbor["shot_order"] - center_shot_order) <= window_size_shots:
                        candidate_score = float(neighbor["stage_scores"][stage_idx])
                        if candidate_score > best_score:
                            best_score = candidate_score
                            best_shot_id = neighbor["shot_id"]
                            best_shot_order = int(neighbor["shot_order"])
                aggregated_scores.append(
                    {
                        "score": float(best_score),
                        "shot_id": best_shot_id,
                        "shot_order": best_shot_order,
                    }
                )
            aggregated.append(
                {
                    "video_id": center_item["video_id"],
                    "center_shot_order": center_shot_order,
                    "stage_supports": aggregated_scores,
                }
            )
        return aggregated

    @staticmethod
    def _soft_temporal_dp(
        *,
        aggregated_timeline: list[dict[str, Any]],
        num_stages: int,
        lambda_skip: float,
        min_stage_gap_shots: int,
    ) -> tuple[list[list[float]], list[list[str | None]], list[list[tuple[int, int] | None]]]:
        num_points = len(aggregated_timeline)
        best_score = [[0.0] * num_stages for _ in range(num_points)]
        decision: list[list[str | None]] = [[None] * num_stages for _ in range(num_points)]
        prev_state: list[list[tuple[int, int] | None]] = [[None] * num_stages for _ in range(num_points)]
        last_shot_order: list[list[int | None]] = [[None] * num_stages for _ in range(num_points)]

        for time_idx in range(num_points):
            for stage_idx in range(num_stages):
                support = aggregated_timeline[time_idx]["stage_supports"][stage_idx]
                current_score = float(support["score"])
                current_shot_order = support["shot_order"]

                if current_score > 0 and current_shot_order is not None:
                    best_score[time_idx][stage_idx] = current_score
                    decision[time_idx][stage_idx] = "start"
                    last_shot_order[time_idx][stage_idx] = int(current_shot_order)
                else:
                    decision[time_idx][stage_idx] = "empty"

                if time_idx > 0 and best_score[time_idx - 1][stage_idx] > best_score[time_idx][stage_idx]:
                    best_score[time_idx][stage_idx] = best_score[time_idx - 1][stage_idx]
                    decision[time_idx][stage_idx] = "carry"
                    prev_state[time_idx][stage_idx] = (time_idx - 1, stage_idx)
                    last_shot_order[time_idx][stage_idx] = last_shot_order[time_idx - 1][stage_idx]

                if time_idx > 0 and stage_idx > 0 and current_score > 0 and current_shot_order is not None:
                    previous_score = best_score[time_idx - 1][stage_idx - 1]
                    previous_shot_order = last_shot_order[time_idx - 1][stage_idx - 1]
                    if (
                        previous_score > 0
                        and previous_shot_order is not None
                        and int(current_shot_order) - int(previous_shot_order) >= min_stage_gap_shots
                    ):
                        transition_score = previous_score + current_score
                        if transition_score > best_score[time_idx][stage_idx]:
                            best_score[time_idx][stage_idx] = transition_score
                            decision[time_idx][stage_idx] = "transition"
                            prev_state[time_idx][stage_idx] = (time_idx - 1, stage_idx - 1)
                            last_shot_order[time_idx][stage_idx] = int(current_shot_order)

                if stage_idx > 0:
                    best_previous_stage_score = 0.0
                    best_previous_stage_state = None
                    for previous_time_idx in range(time_idx + 1):
                        if best_score[previous_time_idx][stage_idx - 1] > best_previous_stage_score:
                            best_previous_stage_score = best_score[previous_time_idx][stage_idx - 1]
                            best_previous_stage_state = (previous_time_idx, stage_idx - 1)
                    skip_score = lambda_skip * best_previous_stage_score
                    if skip_score > best_score[time_idx][stage_idx] and best_previous_stage_state is not None:
                        prev_t, prev_s = best_previous_stage_state
                        best_score[time_idx][stage_idx] = skip_score
                        decision[time_idx][stage_idx] = "skip"
                        prev_state[time_idx][stage_idx] = best_previous_stage_state
                        last_shot_order[time_idx][stage_idx] = last_shot_order[prev_t][prev_s]

        return best_score, decision, prev_state

    def _is_valid_temporal_chain(self, chain: list[dict[str, Any]], *, min_stage_gap_shots: int) -> bool:
        if len(chain) <= 1:
            return True
        seen_shots = set()
        previous_video_id = None
        previous_shot_order = None
        for item in chain:
            shot = self.metadata.shots[item["shot_id"]]
            if shot.shot_id in seen_shots:
                return False
            seen_shots.add(shot.shot_id)
            if previous_video_id is not None and shot.video_id != previous_video_id:
                return False
            if previous_shot_order is not None and int(shot.shot_order) - int(previous_shot_order) < min_stage_gap_shots:
                return False
            previous_video_id = shot.video_id
            previous_shot_order = int(shot.shot_order)
        return True
