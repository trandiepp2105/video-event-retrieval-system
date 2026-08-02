from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
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
        try:
            return retrieval_service.search(payload.query, top_k=payload.top_k)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    return app
