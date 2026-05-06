"""
Step 1 of the pipeline: ingestion from external sources.

This module is a thin coordinator; the real work lives in per-source
fetchers. Each fetcher takes a `limit` and returns a list of raw post dicts
in a shape the `vectorize` stage can consume.

Current fetchers:
    - `fetch_telegram`  (stub — replace with real Telegram API integration)
    - `fetch_reddit`    (stub — replace with real PRAW integration)

Rules for fetcher output dicts:
    - Must contain enough fields to build a `models.Post` after vectorization.
    - Timestamps should be timezone-aware ISO 8601 strings or datetimes.
    - `post_id` should be unique and stable per source so re-runs are idempotent.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

RawPost = dict[str, Any]
Fetcher = Callable[[int], list[RawPost]]


def fetch_telegram(limit: int = 10) -> list[RawPost]:
    """Pull the most recent `limit` posts from Telegram.

    Placeholder implementation until the Telegram integration lands. Returns
    deterministic-enough fake data for local development and example runs.
    """
    logger.info("Telegram fetcher: returning %d stub posts", limit)
    now = datetime.now(timezone.utc)
    return [
        {
            "post_id": f"telegram-stub-{int(now.timestamp())}-{i}",
            "text_content": f"Stub Telegram message {i}",
            "author": "@stub_channel",
            "platform": "telegram",
            "created_at": now.isoformat(),
            "likes": random.randint(0, 100),
            "hashtags": ["stub"],
            "language": "en",
        }
        for i in range(1, limit + 1)
    ]


def fetch_reddit(limit: int = 10) -> list[RawPost]:
    """Pull the most recent `limit` posts from Reddit.

    Placeholder implementation until the PRAW integration lands.
    """
    logger.info("Reddit fetcher: returning %d stub posts", limit)
    now = datetime.now(timezone.utc)
    return [
        {
            "post_id": f"reddit-stub-{int(now.timestamp())}-{i}",
            "text_content": f"Stub Reddit post {i}",
            "author": "u/stub_user",
            "platform": "reddit",
            "created_at": now.isoformat(),
            "likes": random.randint(0, 500),
            "comments_count": random.randint(0, 50),
            "language": "en",
        }
        for i in range(1, limit + 1)
    ]


# Registry so callers can request by source name.
FETCHERS: dict[str, Fetcher] = {
    "telegram": fetch_telegram,
    "reddit": fetch_reddit,
}


def ingest(sources: list[str], limit_per_source: int = 10) -> list[RawPost]:
    """Run the configured fetchers and return all raw posts concatenated.

    Unknown source names are logged and skipped rather than raised, so a
    single broken fetcher does not stop an entire pipeline run.
    """
    posts: list[RawPost] = []
    for source in sources:
        fetcher = FETCHERS.get(source)
        if fetcher is None:
            logger.warning("Unknown ingestion source %r; skipping", source)
            continue
        try:
            posts.extend(fetcher(limit_per_source))
        except Exception:
            logger.exception("Fetcher %r failed; continuing with other sources", source)
    logger.info("Ingested %d raw posts across sources %s", len(posts), sources)
    return posts
