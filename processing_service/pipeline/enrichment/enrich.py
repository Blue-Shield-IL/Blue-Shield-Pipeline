from __future__ import annotations

import logging

from config import settings
from models import ProcessedPost
from .processor import analyze_content_batch, filter_posts_batch, vectorize_texts_batch
from ..ingestion import RawPost

logger = logging.getLogger(__name__)


def enrich_posts(
    raw_posts: list[RawPost],
    batch_size: int | None = None,
) -> tuple[list[ProcessedPost], int]:
    """Run the full ML enrichment pipeline over raw_posts in batches."""
    if batch_size is None:
        batch_size = settings.ml_batch_size

    all_passed: list[ProcessedPost] = []
    total_filtered_out = 0
    total_failed = 0

    for chunk_start in range(0, len(raw_posts), batch_size):
        chunk = raw_posts[chunk_start: chunk_start + batch_size]

        try:
            analyzed_chunk, validation_failures = analyze_content_batch(chunk)
            total_failed += validation_failures
        except Exception as exc:
            logger.warning("Analyze stage failed for chunk",
                           extra={"payload": {"chunk_start": chunk_start, "chunk_end": chunk_start + len(chunk)},
                                  "error": str(exc)})
            raise

        if not analyzed_chunk:
            continue

        passed, filtered_out = filter_posts_batch(analyzed_chunk)
        total_filtered_out += filtered_out
        all_passed.extend(passed)

    if not all_passed:
        logger.info("Enrichment complete: all posts were filtered out or failed at filter stage",
                    extra={"payload": {"total_posts": len(raw_posts)}})
        return [], total_failed

    try:
        logger.info("Vectorize stage starting", extra={"payload": {"posts_count": len(all_passed)}})
        all_passed = vectorize_texts_batch(all_passed)
    except Exception as exc:
        logger.warning("Vectorize stage failed. Saving without vectors.",
                       extra={"payload": {"posts_count": len(all_passed)}, "error": str(exc)})

    logger.info("Enrichment complete", extra={"payload": {
        "processed": len(all_passed),
        "filtered_out": total_filtered_out,
        "failed": total_failed,
        "total": len(raw_posts)
    }})

    return all_passed, total_failed
