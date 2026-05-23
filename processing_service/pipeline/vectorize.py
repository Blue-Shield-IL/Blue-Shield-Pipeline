from __future__ import annotations

import logging

from config import settings
from models import ProcessedPost
from services import analyze_content_batch, filter_posts_batch, vectorize_texts_batch

from .ingest import RawPost

logger = logging.getLogger(__name__)


def vectorize(
    raw_posts: list[RawPost],
    batch_size: int | None = None,
) -> tuple[list[ProcessedPost], int]:
    """Run the full ML pipeline over raw_posts in batches."""
    if batch_size is None:
        batch_size = settings.ml_batch_size

    all_passed: list[ProcessedPost] = []
    total_filtered_out = 0
    total_failed = 0

    for chunk_start in range(0, len(raw_posts), batch_size):
        chunk = raw_posts[chunk_start: chunk_start + batch_size]
        try:
            passed, filtered_out = filter_posts_batch(chunk, batch_size=batch_size)
            total_filtered_out += filtered_out
            all_passed.extend(passed)
        except Exception as exc:
            total_failed += len(chunk)
            logger.warning(
                "Filter stage failed for chunk %d–%d (%d posts skipped): %s",
                chunk_start,
                chunk_start + len(chunk),
                len(chunk),
                exc,
            )

    if not all_passed:
        logger.info(
            "Vectorize complete: all %d posts were filtered out or failed at filter stage.",
            len(raw_posts),
        )
        return [], total_failed

    try:
        all_passed = analyze_content_batch(all_passed, batch_size=batch_size)
    except Exception as exc:
        total_failed += len(all_passed)
        logger.warning(
            "analyze_content_batch failed for all %d posts: %s", len(all_passed), exc
        )
        return [], total_failed

    try:
        all_passed = vectorize_texts_batch(all_passed, batch_size=batch_size)
    except Exception as exc:
        total_failed += len(all_passed)
        logger.warning(
            "vectorize_texts_batch failed for all %d posts: %s", len(all_passed), exc
        )
        return [], total_failed

    logger.info(
        "Vectorize complete: processed=%d filtered_out=%d failed=%d total=%d",
        len(all_passed),
        total_filtered_out,
        total_failed,
        len(raw_posts),
    )

    return all_passed, total_failed
