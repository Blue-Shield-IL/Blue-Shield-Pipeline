"""Step 1 - Ingestion."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .telegram_adapter import TelegramAdapter

logger = logging.getLogger(__name__)

RawPost = dict[str, Any]
RawPostHandler = Callable[[RawPost], None]


def fetch_telegram(limit: int = 10) -> list[RawPost]:
    """Fetch a bounded batch of recent Telegram messages."""
    adapter = TelegramAdapter()
    try:
        return adapter.fetch_recent(limit)
    finally:
        adapter.disconnect()


def listen_telegram(on_post: RawPostHandler) -> None:
    """Keep one Telegram connection open and stream normalized messages."""
    adapter = TelegramAdapter()
    try:
        adapter.listen_forever(on_post)
    finally:
        adapter.disconnect()


FETCHERS: dict[str, Any] = {
    "telegram": fetch_telegram,
}

LISTENERS: dict[str, Any] = {
    "telegram": listen_telegram,
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


def ingest_forever(sources: list[str], on_post: RawPostHandler) -> None:
    """Run source listeners forever. Unknown sources are skipped."""
    for source in sources:
        listener = LISTENERS.get(source)
        if listener is None:
            logger.warning("Unknown source %r; skipping", source)
            continue
        try:
            listener(on_post)
        except Exception:
            logger.exception("Listener %r failed", source)
