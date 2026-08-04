from __future__ import annotations

from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import SearchConfig
from .service import PoolingRetrievalService


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=1000)


class SegmentResponse(BaseModel):
    video_id: str
    start_time_sec: float
    end_time_sec: float


def create_app(config: SearchConfig) -> FastAPI:
    service = PoolingRetrievalService(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print("[API] Loading pooling retrieval resources...", flush=True)
        app.state.retrieval_service = service
        app.state.load_summary = service.load()
        summary = app.state.load_summary
        print(
            "[API] Retrieval resources ready | "
            f"videos={summary['videos']} events={summary['events']} "
            f"shots={summary['shots']} event_vectors={summary['event_vectors']} "
            f"shot_vectors={summary['shot_vectors']}",
            flush=True,
        )
        yield

    app = FastAPI(
        title="Movie Event Retrieval Pooling API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        retrieval_service: PoolingRetrievalService = request.app.state.retrieval_service
        return {
            "status": "ready" if retrieval_service.is_loaded else "loading",
            "resources": retrieval_service.load_summary,
        }

    @app.post("/search", response_model=list[SegmentResponse])
    def search(payload: SearchRequest, request: Request) -> list[dict[str, str | float]]:
        retrieval_service: PoolingRetrievalService = request.app.state.retrieval_service
        started_at = perf_counter()
        status = "failed"
        result_count = 0
        query_analyzer_time_sec = 0.0
        try:
            results, metrics = retrieval_service.search_with_metrics(
                payload.query,
                top_k=payload.top_k,
            )
            result_count = len(results)
            query_analyzer_time_sec = metrics["query_analyzer_time_sec"]
            status = "completed"
            return results
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        finally:
            elapsed_sec = perf_counter() - started_at
            print(
                f"[API] Search {status} | results={result_count} | "
                f"server_search_time_sec={elapsed_sec:.3f} | "
                f"query_analyzer_time_sec={query_analyzer_time_sec:.3f}",
                flush=True,
            )

    return app
