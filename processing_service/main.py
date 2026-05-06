"""
FastAPI entrypoint for the Blue Shield Processing Service.

The processing service is a worker that runs three steps end-to-end:
  1. Ingest posts from external sources (Telegram, Reddit).
  2. Filter, analyze, and vectorize them.
  3. Store the enriched posts in Elasticsearch.

The HTTP surface exposed here is intentionally narrow:
  - `/health*`        ops probes
  - `/jobs/run`       trigger one pipeline run on demand

The same `PipelineRunner` is used by the scheduled cron job, so cron and
HTTP trigger the exact same code path.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from config import settings
from pipeline import PipelineRunner
from pipeline.storage import close_client, ensure_posts_index, get_client, ping

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the Elasticsearch client and ensure the posts index exists."""
    try:
        if ping():
            created = ensure_posts_index()
            logger.info(
                "Elasticsearch reachable; posts index %r %s",
                settings.posts_index,
                "created" if created else "already exists",
            )
        else:
            logger.warning(
                "Elasticsearch not reachable at %s on startup; continuing without index bootstrap.",
                settings.host,
            )
    except Exception:  # pragma: no cover - startup must not crash the app
        logger.exception("Elasticsearch bootstrap failed")

    yield

    close_client()


app = FastAPI(
    title="Blue Shield - Processing Service",
    description="Scheduled ingestion, analysis, and storage pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Health ------------------------------------------------------------------


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
    """Return the Elasticsearch cluster health or a 503 if unreachable."""
    client = get_client()
    try:
        info = client.cluster.health()
    except Exception as exc:
        logger.warning("Elasticsearch health check failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Elasticsearch unreachable: {exc}",
        ) from exc
    return {"host": settings.host, "index": settings.posts_index, "cluster": dict(info)}


# --- Jobs --------------------------------------------------------------------


class RunJobRequest(BaseModel):
    """Body for `POST /jobs/run`.

    All fields are optional so the endpoint can be triggered with an empty
    body for a "run with defaults" behaviour.
    """

    sources: Optional[list[str]] = Field(
        default=None,
        description="Sources to ingest from. Defaults to ['telegram', 'reddit'].",
    )
    limit_per_source: Optional[int] = Field(
        default=None,
        ge=1,
        le=500,
        description="Max posts to fetch per source. Defaults to 10.",
    )


@app.post("/jobs/run", status_code=status.HTTP_200_OK)
def run_job(body: Optional[RunJobRequest] = None) -> dict[str, Any]:
    """Run the full pipeline once: ingest → vectorize → store.

    Returns per-stage counts plus any storage errors. Runs synchronously on
    the current worker — fine for small batches; for larger runs the service
    should queue the job instead (out of scope here).
    """
    req = body or RunJobRequest()
    sources = req.sources or ["telegram", "reddit"]
    limit = req.limit_per_source or 10

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
