from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from config import settings
from pipeline import PipelineRunner
from pipeline.storage import close_client, ensure_posts_index, get_client, ping

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if ping():
            logger.info(
                "Elasticsearch reachable; posts index %r %s",
                settings.posts_index,
                "created" if ensure_posts_index() else "already exists",
            )
        else:
            logger.warning(
                "Elasticsearch not reachable at %s on startup; continuing without index bootstrap.",
                settings.host,
            )
    except Exception:
        logger.exception("Elasticsearch bootstrap failed")

    yield

    close_client()


app = FastAPI(
    title="Blue Shield - Processing Service",
    description="Scheduled ingestion, analysis, and storage pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "status": "Online",
        "message": "Welcome to the 'Blue Shield' Processing Service",
        "docs": "/docs",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/health/elastic")
def elastic_health() -> dict[str, Any]:
    try:
        info = get_client().cluster.health()
    except Exception as exc:
        logger.warning("Elasticsearch health check failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Elasticsearch unreachable: {exc}",
        ) from exc
    return {"host": settings.host, "index": settings.posts_index, "cluster": info.body}


class RunJobRequest(BaseModel):
    sources: list[str] = Field(
        default=["telegram"],
        description="Sources to ingest from. Defaults to ['telegram'].",
    )
    limit_per_source: int = Field(
        default=10,
        ge=1,
        le=500,
        description="Max posts to fetch per source. Defaults to 10.",
    )


# Run the full pipeline once: ingest → vectorize → store.
@app.post("/jobs/run", status_code=status.HTTP_200_OK)
def run_job(body: RunJobRequest | None = None) -> dict[str, Any]:
    req = body or RunJobRequest()
    sources, limit = req.sources, req.limit_per_source

    try:
        runner = PipelineRunner(sources=sources, limit_per_source=limit)
        result = runner.run()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Pipeline job failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Pipeline job failed: {exc}",
        ) from exc

    return result.as_dict()
