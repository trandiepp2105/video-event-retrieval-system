from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from threading import Lock
from typing import Any

from .config import SearchConfig
from .metadata import StoredMetadataLoader
from .retrieval import PoolingMovieEventRetriever


@dataclass(frozen=True)
class SegmentResult:
    video_id: str
    start_time_sec: float
    end_time_sec: float

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


class PoolingRetrievalService:
    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self._retriever: PoolingMovieEventRetriever | None = None
        self._load_summary: dict[str, Any] | None = None
        self._load_lock = Lock()
        self._search_lock = Lock()

    @property
    def is_loaded(self) -> bool:
        return self._retriever is not None

    @property
    def load_summary(self) -> dict[str, Any] | None:
        if self._load_summary is None:
            return None
        return dict(self._load_summary)

    def load(self) -> dict[str, Any]:
        with self._load_lock:
            if self._retriever is not None and self._load_summary is not None:
                return dict(self._load_summary)

            metadata = StoredMetadataLoader().load(self.config.store_dir)
            retriever = PoolingMovieEventRetriever(metadata, self.config.store_dir)
            summary = retriever.load_resources(self.config)
            self._retriever = retriever
            self._load_summary = summary
            return dict(summary)

    def search(self, query: str, *, top_k: int = 10) -> list[dict[str, str | float]]:
        query = str(query).strip()
        if not query:
            raise ValueError("query must not be empty")
        if int(top_k) <= 0:
            raise ValueError("top_k must be greater than 0")
        if self._retriever is None:
            raise RuntimeError("PoolingRetrievalService.load() must be called before search()")

        request_config = replace(
            self.config,
            query_json=None,
            raw_query=query,
            translated_query="",
            subtitle_query="",
            ocr_query="",
            final_top_k=max(int(top_k), self.config.final_top_k),
            temporal_chain_top_k=max(int(top_k), self.config.temporal_chain_top_k),
            output_json=None,
        )
        with self._search_lock:
            payload = self._retriever.search(request_config)
        return [item.to_dict() for item in self._extract_segments(payload, top_k=int(top_k))]

    @staticmethod
    def _extract_segments(payload: dict[str, Any], *, top_k: int) -> list[SegmentResult]:
        segments: list[SegmentResult] = []
        chains = payload.get("shot_temporal", {}).get("top_chains", [])
        for chain_result in chains:
            chain = chain_result.get("chain", [])
            if not chain:
                continue
            segments.append(
                SegmentResult(
                    video_id=str(chain_result["video_id"]),
                    start_time_sec=float(chain[0]["start_time_sec"]),
                    end_time_sec=float(chain[-1]["end_time_sec"]),
                )
            )
            if len(segments) >= top_k:
                return segments

        if segments:
            return segments

        for event in payload.get("final_events", []):
            segments.append(
                SegmentResult(
                    video_id=str(event["video_id"]),
                    start_time_sec=float(event["start_time_sec"]),
                    end_time_sec=float(event["end_time_sec"]),
                )
            )
            if len(segments) >= top_k:
                break
        return segments
