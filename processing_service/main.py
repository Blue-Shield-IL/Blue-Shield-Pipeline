"""
FastAPI entrypoint for the Blue Shield Processing Service.

Responsibilities:
- Expose health endpoints (app + Elasticsearch connectivity).
- Accept validated posts (single and batch) and index them into Elasticsearch.
- Ensure the posts index exists on startup.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status

from config import settings
from elastic import (
    bulk_index_posts,
    close_client,
    ensure_posts_index,
    get_client,
    index_post,
    ping,
)
from models import Post

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
    description="Backend infrastructure for data ingestion and analysis",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Health endpoints --------------------------------------------------------


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


# --- Admin endpoints ---------------------------------------------------------


@app.post("/admin/posts-index", status_code=status.HTTP_200_OK)
def create_posts_index() -> dict[str, Any]:
    """Create the posts index with the expected mapping if it does not exist."""
    try:
        created = ensure_posts_index()
    except Exception as exc:
        logger.exception("Failed to ensure posts index")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to ensure posts index: {exc}",
        ) from exc
    return {"index": settings.posts_index, "created": created}


# --- Posts ingestion ---------------------------------------------------------


@app.post("/posts", status_code=status.HTTP_201_CREATED)
def ingest_post(post: Post) -> dict[str, Any]:
    """Validate a single post and index it into Elasticsearch."""
    try:
        result = index_post(post)
    except Exception as exc:
        logger.exception("Failed to index post %s", post.post_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to index post: {exc}",
        ) from exc
    return {"post": post.model_dump(mode="json"), "elastic": result}


@app.post("/posts/batch", status_code=status.HTTP_201_CREATED)
def ingest_posts_batch(posts: list[Post]) -> dict[str, Any]:
    """Validate a batch of posts and index them via the Elasticsearch _bulk API."""
    try:
        success, errors = bulk_index_posts(posts)
    except Exception as exc:
        logger.exception("Bulk indexing failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Bulk indexing failed: {exc}",
        ) from exc

    return {
        "requested": len(posts),
        "indexed": success,
        "errors": errors,
        "posts": [p.model_dump(mode="json") for p in posts],
    }


@app.get("/posts/schema")
def get_post_schema() -> dict[str, Any]:
    """Return the JSON Schema representation of the Post model."""
    return Post.model_json_schema()
