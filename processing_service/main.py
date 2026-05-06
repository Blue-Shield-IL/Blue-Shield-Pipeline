"""
FastAPI entrypoint for the Blue Shield Processing Service.

The processing service is a scheduled worker that:
  1. Ingests posts from external sources (e.g. Telegram, Reddit).
  2. Filters, labels, and vectorizes them.
  3. Stores the enriched posts in Elasticsearch (see `elastic.py`).

The HTTP surface exposed here is intentionally minimal: health probes for
ops, and the OpenAPI docs. Data ingress into Elasticsearch happens via the
storage module (`elastic.store_post` / `elastic.store_posts`) called by the
pipeline worker — the Node.js API Server is the public-facing HTTP layer.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status

from config import settings
from elastic import close_client, ensure_posts_index, get_client, ping

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
