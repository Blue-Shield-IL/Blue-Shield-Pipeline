"""Step 2 — Vectorize (stub)."""

from __future__ import annotations

import logging
import random

from config import settings
from models import Post

from .ingest import RawPost

logger = logging.getLogger(__name__)


def vectorize(raw_posts: list[RawPost]) -> tuple[list[Post], int]:
    """Convert raw dicts into Post objects with a fake embedding.

    Returns (successful_posts, failed_count) so the caller can report
    how many posts failed validation.
    """
    dims = settings.embedding_dims
    results: list[Post] = []
    failed = 0

    for raw in raw_posts:
        rng = random.Random(raw.get("text_content", ""))
        raw["embedding"] = [rng.random() for _ in range(dims)]
        try:
            results.append(Post(**raw))
        except Exception as exc:
            failed += 1
            logger.warning(
                "Vectorize: failed to build Post from post_id=%r: %s",
                raw.get("post_id", "unknown"),
                exc,
            )

    if failed:
        logger.warning("Vectorize: %d/%d posts failed validation", failed, len(raw_posts))

    return results, failed
