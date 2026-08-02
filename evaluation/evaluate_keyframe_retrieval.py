from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_K_VALUES = [1, 5, 10, 20, 50, 100]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_query_items(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected query file to contain a list: {path}")
    return payload


def load_result_items(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError(f"Expected result file to contain a dict with a results list: {path}")
    return payload["results"]


def load_fps_map(path: Path) -> dict[str, float]:
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return {str(item["video_id"]): float(item["fps"]) for item in payload["results"]}
    if isinstance(payload, dict):
        return {str(video_id): float(fps) for video_id, fps in payload.items()}
    raise ValueError(f"Unsupported FPS file format: {path}")


def reciprocal_rank(first_rank: int | None) -> float:
    return 0.0 if first_rank is None else 1.0 / first_rank


def segment_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def normalize_single_keyframe(result: Any) -> dict[str, Any] | None:
    try:
        key, score = result
        video_id, frame_idx = key
        return {"video_id": str(video_id), "frame_idx": int(frame_idx), "score": float(score)}
    except (TypeError, ValueError, IndexError):
        return None


def normalize_keyframe_chain(result: Any, fps_map: dict[str, float]) -> dict[str, Any] | None:
    if not isinstance(result, list) or not result:
        return None

    frames: list[dict[str, Any]] = []
    for step in result:
        try:
            key, payload = step
            video_id, frame_idx = key
            score = payload[1] if isinstance(payload, list) and len(payload) > 1 else None
            stage_index = payload[2] if isinstance(payload, list) and len(payload) > 2 else None
            frames.append(
                {
                    "video_id": str(video_id),
                    "frame_idx": int(frame_idx),
                    "score": score,
                    "stage_index": stage_index,
                }
            )
        except (TypeError, ValueError, IndexError):
            return None

    video_id = frames[0]["video_id"]
    if video_id not in fps_map:
        return None
    times = [frame["frame_idx"] / fps_map[video_id] for frame in frames]
    return {
        "video_id": video_id,
        "start_time_sec": min(times),
        "end_time_sec": max(times),
        "frames": frames,
    }


def evaluate_query(
    query: dict[str, Any],
    result_item: dict[str, Any],
    fps_map: dict[str, float],
    k_values: list[int],
) -> dict[str, Any]:
    gt_video_id = str(query["video_id"])
    gt_start = float(query["start_time_sec"])
    gt_end = float(query["end_time_sec"])
    raw_results = result_item.get("results") or []
    mode = result_item.get("mode", "single-stage")

    first_video_rank: int | None = None
    first_chain_rank: int | None = None
    hits = {k: {"video": 0, "chain": 0} for k in k_values}

    normalized: list[dict[str, Any]] = []
    for rank, raw_result in enumerate(raw_results, start=1):
        if mode == "single-stage":
            parsed = normalize_single_keyframe(raw_result)
            if parsed is None:
                continue
            video_id = parsed["video_id"]
            fps = fps_map.get(video_id)
            time_sec = None if fps is None else parsed["frame_idx"] / fps
            overlap = (
                segment_overlap(time_sec, time_sec, gt_start, gt_end)
                if time_sec is not None
                else 0.0
            )
            is_video_hit = video_id == gt_video_id
            is_chain_hit = bool(is_video_hit and time_sec is not None and gt_start <= time_sec <= gt_end)
            normalized.append(
                {
                    "rank": rank,
                    "video_id": video_id,
                    "frame_idx": parsed["frame_idx"],
                    "time_sec": time_sec,
                    "start_time_sec": time_sec,
                    "end_time_sec": time_sec,
                    "overlap_sec": overlap if is_video_hit else 0.0,
                    "score": parsed["score"],
                    "is_video_hit": is_video_hit,
                    "is_chain_hit": is_chain_hit,
                }
            )
        else:
            parsed_chain = normalize_keyframe_chain(raw_result, fps_map)
            if parsed_chain is None:
                continue
            video_id = parsed_chain["video_id"]
            overlap = segment_overlap(
                parsed_chain["start_time_sec"],
                parsed_chain["end_time_sec"],
                gt_start,
                gt_end,
            )
            is_video_hit = video_id == gt_video_id
            is_chain_hit = bool(is_video_hit and overlap > 0)
            normalized.append(
                {
                    "rank": rank,
                    "video_id": video_id,
                    "start_time_sec": parsed_chain["start_time_sec"],
                    "end_time_sec": parsed_chain["end_time_sec"],
                    "overlap_sec": overlap if is_video_hit else 0.0,
                    "is_video_hit": is_video_hit,
                    "is_chain_hit": is_chain_hit,
                    "frames": parsed_chain["frames"],
                }
            )

        if first_video_rank is None and is_video_hit:
            first_video_rank = rank
        if first_chain_rank is None and is_chain_hit:
            first_chain_rank = rank

    for k in k_values:
        top_k = [item for item in normalized if item["rank"] <= k]
        hits[k]["video"] = int(any(item["is_video_hit"] for item in top_k))
        hits[k]["chain"] = int(any(item["is_chain_hit"] for item in top_k))

    return {
        "query_id": int(query.get("query_id", result_item.get("query_id", -1))),
        "video_id": gt_video_id,
        "gt_start_time_sec": gt_start,
        "gt_end_time_sec": gt_end,
        "mode": mode,
        "first_video_rank": first_video_rank,
        "first_chain_rank": first_chain_rank,
        "video_rr": reciprocal_rank(first_video_rank),
        "chain_rr": reciprocal_rank(first_chain_rank),
        "hits": hits,
        "top_results": normalized,
    }


def summarize(per_query: list[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
    total = len(per_query)
    metrics: dict[str, Any] = {
        "num_queries": total,
        "video_mrr": sum(item["video_rr"] for item in per_query) / total if total else 0.0,
        "chain_mrr": sum(item["chain_rr"] for item in per_query) / total if total else 0.0,
        "hits_at_k": {},
    }
    for k in k_values:
        metrics["hits_at_k"][str(k)] = {
            "video_hit": sum(item["hits"][k]["video"] for item in per_query) / total if total else 0.0,
            "chain_hit": sum(item["hits"][k]["chain"] for item in per_query) / total if total else 0.0,
        }
    return metrics


def print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate keyframe-based retrieval results.")
    parser.add_argument("--keyframe_results_json", type=Path, required=True)
    parser.add_argument("--queries_json", type=Path, required=True)
    parser.add_argument("--video_fps_json", type=Path, required=True)
    parser.add_argument("--k_values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    parser.add_argument("--per_query_output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = load_query_items(args.queries_json)
    results = load_result_items(args.keyframe_results_json)
    fps_map = load_fps_map(args.video_fps_json)
    if len(queries) != len(results):
        raise ValueError(f"Query/result length mismatch: {len(queries)} != {len(results)}")

    k_values = sorted(set(args.k_values))
    per_query = [
        evaluate_query(query, result_item, fps_map, k_values)
        for query, result_item in zip(queries, results)
    ]
    summary = summarize(per_query, k_values)
    print_summary(summary)

    if args.per_query_output is not None:
        args.per_query_output.parent.mkdir(parents=True, exist_ok=True)
        with args.per_query_output.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_query": per_query}, f, ensure_ascii=False, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
