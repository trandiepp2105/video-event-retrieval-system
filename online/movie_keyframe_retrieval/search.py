from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

from .encoders import E5TextEncoder, OpenClipTextEncoder
from .registry import IndexRegistry
from .schemas import SearchResult, StageQuery


def _normalize_score_map(score_map: dict[tuple[str, int], float]) -> dict[tuple[str, int], float]:
    if not score_map:
        return {}
    max_score = max(float(score) for score in score_map.values())
    if max_score <= 0:
        return {key: 0.0 for key in score_map}
    return {key: float(score) / max_score for key, score in score_map.items()}


def _range_hits_by_video(results: list[SearchResult]) -> dict[str, list[SearchResult]]:
    grouped: dict[str, list[SearchResult]] = {}
    for item in results:
        grouped.setdefault(item.video_id, []).append(item)
    return grouped


@dataclass
class StageCandidate:
    video_id: str
    frame_idx: int
    fused_score: float
    visual_score: float = 0.0
    ocr_score: float = 0.0
    subtitle_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SearchEngine:
    def __init__(
        self,
        *,
        registry: IndexRegistry,
        visual_encoder: Optional[OpenClipTextEncoder] = None,
        subtitle_encoder: Optional[E5TextEncoder] = None,
    ) -> None:
        self.registry = registry
        self.visual_encoder = visual_encoder
        self.subtitle_encoder = subtitle_encoder

    @staticmethod
    def _score_point_hits(results: list[SearchResult]) -> dict[tuple[str, int], float]:
        score_map: dict[tuple[str, int], float] = {}
        for item in results:
            frame_idx = int(item.frame_start)
            key = (item.video_id, frame_idx)
            score_map[key] = max(score_map.get(key, float("-inf")), float(item.score))
        return _normalize_score_map(score_map)

    @staticmethod
    def _subtitle_support(
        *,
        candidate_key: tuple[str, int],
        subtitle_hits_by_video: dict[str, list[SearchResult]],
    ) -> float:
        video_id, frame_idx = candidate_key
        best_score = 0.0
        for hit in subtitle_hits_by_video.get(video_id, []):
            if int(hit.frame_start) <= frame_idx <= int(hit.frame_end):
                best_score = max(best_score, float(hit.score))
        return best_score

    def stage_search(
        self,
        stage_query: StageQuery,
        *,
        top_k: int = 100,
        visual_top_k: int = 300,
        ocr_top_k: int = 300,
        subtitle_top_k: int = 300,
        visual_weight: float = 0.45,
        ocr_weight: float = 0.35,
        subtitle_weight: float = 0.20,
        allow_subtitle_only: bool = False,
    ) -> list[StageCandidate]:
        visual_hits: list[SearchResult] = []
        subtitle_hits: list[SearchResult] = []
        ocr_hits: list[SearchResult] = []

        if stage_query.visual:
            if self.visual_encoder is None:
                raise ValueError("visual_encoder is required when stage_query.visual is provided")
            visual_index = self.registry.get("visual")
            q_visual = self.visual_encoder.encode_texts([stage_query.visual])[0]
            visual_hits = visual_index.search(q_visual, top_k=visual_top_k)

        if stage_query.subtitle:
            if self.subtitle_encoder is None:
                raise ValueError("subtitle_encoder is required when stage_query.subtitle is provided")
            subtitle_index = self.registry.get("subtitle")
            q_subtitle = self.subtitle_encoder.encode_queries([stage_query.subtitle])[0]
            subtitle_hits = subtitle_index.search(q_subtitle, top_k=subtitle_top_k)

        if stage_query.ocr:
            ocr_index = self.registry.get("ocr")
            ocr_hits = ocr_index.search(stage_query.ocr, top_k=ocr_top_k)

        visual_scores = self._score_point_hits(visual_hits)
        ocr_scores = self._score_point_hits(ocr_hits)
        subtitle_hits_by_video = _range_hits_by_video(subtitle_hits)

        candidate_keys = set(visual_scores.keys()) | set(ocr_scores.keys())
        if allow_subtitle_only and not candidate_keys:
            for hit in subtitle_hits:
                candidate_keys.add((hit.video_id, (int(hit.frame_start) + int(hit.frame_end)) // 2))

        subtitle_support_raw = {
            key: self._subtitle_support(candidate_key=key, subtitle_hits_by_video=subtitle_hits_by_video)
            for key in candidate_keys
        }
        subtitle_scores = _normalize_score_map(subtitle_support_raw)

        candidates: list[StageCandidate] = []
        for video_id, frame_idx in candidate_keys:
            visual_score = float(visual_scores.get((video_id, frame_idx), 0.0))
            ocr_score = float(ocr_scores.get((video_id, frame_idx), 0.0))
            subtitle_score = float(subtitle_scores.get((video_id, frame_idx), 0.0))
            fused_score = (
                float(visual_weight) * visual_score
                + float(ocr_weight) * ocr_score
                + float(subtitle_weight) * subtitle_score
            )
            candidates.append(
                StageCandidate(
                    video_id=video_id,
                    frame_idx=int(frame_idx),
                    fused_score=float(fused_score),
                    visual_score=visual_score,
                    ocr_score=ocr_score,
                    subtitle_score=subtitle_score,
                )
            )

        candidates.sort(key=lambda item: item.fused_score, reverse=True)
        return candidates[: int(top_k)]

    def temporal_search(
        self,
        stage_queries: list[StageQuery],
        *,
        top_k: int = 20,
        per_stage_top_k: int = 100,
        visual_top_k: int = 300,
        ocr_top_k: int = 300,
        subtitle_top_k: int = 300,
        visual_weight: float = 0.45,
        ocr_weight: float = 0.35,
        subtitle_weight: float = 0.20,
    ) -> list[dict[str, Any]]:
        if not stage_queries:
            return []
        stage_candidates = [
            self.stage_search(
                stage_query=stage_query,
                top_k=per_stage_top_k,
                visual_top_k=visual_top_k,
                ocr_top_k=ocr_top_k,
                subtitle_top_k=subtitle_top_k,
                visual_weight=visual_weight,
                ocr_weight=ocr_weight,
                subtitle_weight=subtitle_weight,
            )
            for stage_query in stage_queries
        ]
        if any(not items for items in stage_candidates):
            return []

        all_video_ids = sorted({candidate.video_id for items in stage_candidates for candidate in items})
        chains: list[dict[str, Any]] = []

        for video_id in all_video_ids:
            per_stage_video_candidates = [
                [candidate for candidate in items if candidate.video_id == video_id]
                for items in stage_candidates
            ]
            if any(not items for items in per_stage_video_candidates):
                continue

            dp_scores: list[list[float]] = []
            dp_prev: list[list[Optional[int]]] = []

            first_stage = sorted(per_stage_video_candidates[0], key=lambda item: item.frame_idx)
            dp_scores.append([candidate.fused_score for candidate in first_stage])
            dp_prev.append([None for _ in first_stage])

            stage_lists = [first_stage]
            for stage_idx in range(1, len(per_stage_video_candidates)):
                current_items = sorted(per_stage_video_candidates[stage_idx], key=lambda item: item.frame_idx)
                prev_items = stage_lists[-1]
                prev_scores = dp_scores[-1]
                current_scores = [float("-inf")] * len(current_items)
                current_prev: list[Optional[int]] = [None] * len(current_items)
                for curr_idx, curr in enumerate(current_items):
                    best_score = float("-inf")
                    best_prev_idx: Optional[int] = None
                    for prev_idx, prev in enumerate(prev_items):
                        if prev.frame_idx >= curr.frame_idx:
                            continue
                        score = prev_scores[prev_idx] + curr.fused_score
                        if score > best_score:
                            best_score = score
                            best_prev_idx = prev_idx
                    current_scores[curr_idx] = best_score
                    current_prev[curr_idx] = best_prev_idx
                stage_lists.append(current_items)
                dp_scores.append(current_scores)
                dp_prev.append(current_prev)

            last_scores = dp_scores[-1]
            for last_idx, total_score in enumerate(last_scores):
                if total_score == float("-inf"):
                    continue
                chain_items: list[dict[str, Any]] = []
                cursor = last_idx
                valid = True
                for stage_idx in range(len(stage_lists) - 1, -1, -1):
                    item = stage_lists[stage_idx][cursor]
                    chain_items.append(
                        {
                            "stage_index": stage_idx,
                            **item.to_dict(),
                        }
                    )
                    prev_cursor = dp_prev[stage_idx][cursor]
                    if stage_idx > 0 and prev_cursor is None:
                        valid = False
                        break
                    if prev_cursor is not None:
                        cursor = prev_cursor
                if not valid:
                    continue
                chain_items.reverse()
                chains.append(
                    {
                        "video_id": video_id,
                        "total_score": float(total_score),
                        "stages": chain_items,
                    }
                )

        chains.sort(key=lambda item: item["total_score"], reverse=True)
        return chains[: int(top_k)]
