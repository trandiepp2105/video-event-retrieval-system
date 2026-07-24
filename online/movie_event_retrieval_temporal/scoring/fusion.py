from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def reciprocal_rank_score(rank: int, rrf_k: int) -> float:
    return 1.0 / float(rrf_k + max(rank, 1))


@dataclass
class ScoreAccumulator:
    scores: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, item_id: str, score: float, *, source: str, payload: dict[str, Any]) -> None:
        self.scores[item_id] = self.scores.get(item_id, 0.0) + float(score)
        bucket = self.evidence.setdefault(item_id, {})
        bucket[source] = payload

    def sorted_items(self) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda item: item[1], reverse=True)
