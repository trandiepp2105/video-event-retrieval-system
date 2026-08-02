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


def reciprocal_rank(first_rank: int | None) -> float:
    return 0.0 if first_rank is None else 1.0 / first_rank


def segment_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def normalize_event(result: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return {
            "video_id": str(result["video_id"]),
            "start_time_sec": float(result["start_time_sec"]),
            "end_time_sec": float(result["end_time_sec"]),
            "event_id": result.get("event_id"),
            "score": result.get("score"),
        }
    except (KeyError, TypeError, ValueError):
        return None


def normalize_shot_chain(result: dict[str, Any]) -> dict[str, Any] | None:
    chain = result.get("chain")
    if not isinstance(chain, list) or not chain:
        return None
    try:
        return {
            "video_id": str(result.get("video_id") or chain[0]["video_id"]),
            "start_time_sec": float(chain[0]["start_time_sec"]),
            "end_time_sec": float(chain[-1]["end_time_sec"]),
            "score": result.get("score"),
            "num_stages_matched": result.get("num_stages_matched"),
            "num_stages_skipped": result.get("num_stages_skipped"),
            "chain": chain,
        }
    except (KeyError, TypeError, ValueError):
        return None


def is_relevant_segment(
    result: dict[str, Any],
    gt_video_id: str,
    gt_start: float,
    gt_end: float,
) -> tuple[bool, float]:
    if result["video_id"] != gt_video_id:
        return False, 0.0
    overlap = segment_overlap(result["start_time_sec"], result["end_time_sec"], gt_start, gt_end)
    return overlap > 0, overlap


def first_relevant_rank(
    results: list[dict[str, Any]],
    gt_video_id: str,
    gt_start: float,
    gt_end: float,
) -> int | None:
    for rank, result in enumerate(results, start=1):
        is_relevant, _ = is_relevant_segment(result, gt_video_id, gt_start, gt_end)
        if is_relevant:
            return rank
    return None


def first_video_rank(results: list[dict[str, Any]], gt_video_id: str) -> int | None:
    for rank, result in enumerate(results, start=1):
        if result["video_id"] == gt_video_id:
            return rank
    return None


def video_hits_at_k(
    results: list[dict[str, Any]],
    gt_video_id: str,
    k_values: list[int],
) -> dict[int, int]:
    out: dict[int, int] = {}
    for k in k_values:
        top_k = results[:k]
        out[k] = int(any(item["video_id"] == gt_video_id for item in top_k))
    return out


def chain_hits_at_k(
    results: list[dict[str, Any]],
    gt_video_id: str,
    gt_start: float,
    gt_end: float,
    k_values: list[int],
) -> dict[int, int]:
    out: dict[int, dict[str, int]] = {}
    for k in k_values:
        top_k = results[:k]
        out[k] = int(any(is_relevant_segment(item, gt_video_id, gt_start, gt_end)[0] for item in top_k))
    return out


def event_recalls_at_k(
    results: list[dict[str, Any]],
    gt_video_id: str,
    gt_start: float,
    gt_end: float,
    k_values: list[int],
) -> dict[int, int]:
    return chain_hits_at_k(results, gt_video_id, gt_start, gt_end, k_values)


def evaluate_query(
    query: dict[str, Any],
    result_item: dict[str, Any],
    k_values: list[int],
) -> dict[str, Any]:
    gt_video_id = str(query["video_id"])
    gt_start = float(query["start_time_sec"])
    gt_end = float(query["end_time_sec"])

    event_candidates = [
        event
        for raw in result_item.get("top_event_candidates", [])
        if isinstance(raw, dict)
        for event in [normalize_event(raw)]
        if event is not None
    ]
    shot_chains = [
        chain
        for raw in result_item.get("top_shot_chains", [])
        if isinstance(raw, dict)
        for chain in [normalize_shot_chain(raw)]
        if chain is not None
    ]

    ranks = {
        "event_video": first_video_rank(event_candidates, gt_video_id),
        "event": first_relevant_rank(event_candidates, gt_video_id, gt_start, gt_end),
        "chain": first_relevant_rank(shot_chains, gt_video_id, gt_start, gt_end),
    }

    return {
        "query_id": int(query.get("query_id", result_item.get("query_id", -1))),
        "video_id": gt_video_id,
        "gt_start_time_sec": gt_start,
        "gt_end_time_sec": gt_end,
        "first_ranks": ranks,
        "rr": {name: reciprocal_rank(rank) for name, rank in ranks.items()},
        "hits": {
            "event_video": video_hits_at_k(event_candidates, gt_video_id, k_values),
            "event_recall": event_recalls_at_k(event_candidates, gt_video_id, gt_start, gt_end, k_values),
            "chain": chain_hits_at_k(shot_chains, gt_video_id, gt_start, gt_end, k_values),
        },
        "counts": {
            "event_candidates": len(event_candidates),
            "shot_chains": len(shot_chains),
        },
    }


def summarize(per_query: list[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
    total = len(per_query)
    summary: dict[str, Any] = {"num_queries": total, "mrr": {}, "hits_at_k": {}}

    rr_names = sorted(per_query[0]["rr"].keys()) if per_query else []
    for name in rr_names:
        summary["mrr"][name] = sum(item["rr"][name] for item in per_query) / total if total else 0.0

    for k in k_values:
        summary["hits_at_k"][str(k)] = {
            "event_video_hit": sum(item["hits"]["event_video"][k] for item in per_query) / total if total else 0.0,
            "event_recall": sum(item["hits"]["event_recall"][k] for item in per_query) / total if total else 0.0,
            "chain_hit": sum(item["hits"]["chain"][k] for item in per_query) / total if total else 0.0,
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate shot/event-based retrieval results.")
    parser.add_argument("--pooling_results_json", type=Path, required=True)
    parser.add_argument("--queries_json", type=Path, required=True)
    parser.add_argument("--video_fps_json", type=Path, default=None, help="Accepted for CLI symmetry; not used.")
    parser.add_argument("--k_values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    parser.add_argument("--per_query_output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = load_query_items(args.queries_json)
    results = load_result_items(args.pooling_results_json)
    if len(queries) != len(results):
        raise ValueError(f"Query/result length mismatch: {len(queries)} != {len(results)}")

    k_values = sorted(set(args.k_values))
    per_query = [
        evaluate_query(query, result_item, k_values)
        for query, result_item in zip(queries, results)
    ]
    summary = summarize(per_query, k_values)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.per_query_output is not None:
        args.per_query_output.parent.mkdir(parents=True, exist_ok=True)
        with args.per_query_output.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_query": per_query}, f, ensure_ascii=False, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
