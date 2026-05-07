"""Step 1 — Ingestion (stub)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

RawPost = dict[str, Any]


def fetch_telegram(limit: int = 10) -> list[RawPost]:
    """Stub — returns fake Telegram posts until the real fetcher lands."""
    now = datetime.now(timezone.utc)
    return [
        {
            "post_id": f"telegram-stub-{int(now.timestamp())}-{i}",
            "text_content": f"Stub Telegram message {i}",
            "author": "@stub_channel",
            "platform": "telegram",
            "created_at": now.isoformat(),
        }
        for i in range(1, limit + 1)
    ]


FETCHERS: dict[str, Any] = {
    "telegram": fetch_telegram,
}


def ingest(sources: list[str], limit_per_source: int = 10) -> list[RawPost]:
    """Run fetchers for the requested sources. Unknown sources are skipped."""
    posts: list[RawPost] = []
    for source in sources:
        fetcher = FETCHERS.get(source)
        if fetcher is None:
            logger.warning("Unknown source %r; skipping", source)
            continue
        try:
            posts.extend(fetcher(limit_per_source))
        except Exception:
            logger.exception("Fetcher %r failed", source)
    return posts
