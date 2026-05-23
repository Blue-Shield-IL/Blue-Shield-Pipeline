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

    # We process in chunks to protect the LLM API from payload overload
    for chunk_start in range(0, len(raw_posts), batch_size):
        chunk = raw_posts[chunk_start: chunk_start + batch_size]
        
        # 1. Analyze (ML Inference)
        try:
            analyzed_chunk, validation_failures = analyze_content_batch(chunk)
            total_failed += validation_failures
        except Exception as exc:
            total_failed += len(chunk)
            logger.warning(
                "Analysis stage failed for chunk %d–%d (%d posts skipped): %s",
                chunk_start,
                chunk_start + len(chunk),
                len(chunk),
                exc,
            )
            continue

        if not analyzed_chunk:
            continue

        # 2. Filter (Business Logic)
        try:
            passed, filtered_out = filter_posts_batch(analyzed_chunk)
            total_filtered_out += filtered_out
            all_passed.extend(passed)
        except Exception as exc:
            total_failed += len(analyzed_chunk)
            logger.warning("Filter stage failed for chunk %d–%d: %s", chunk_start, chunk_start + len(chunk), exc)
            continue

    if not all_passed:
        logger.info(
            "Enrichment complete: all %d posts were filtered out or failed at filter stage.",
            len(raw_posts),
        )
        return [], total_failed

    # 3. Vectorize (Embeddings Inference)
    try:
        # vectorize_texts_batch natively chunks to size 100 inside the gemini service
        all_passed = vectorize_texts_batch(all_passed)
    except Exception as exc:
        total_failed += len(all_passed)
        logger.warning(
            "Vectorize stage failed for all %d posts: %s", len(all_passed), exc
        )
        return [], total_failed

    logger.info(
        "Enrichment complete: processed=%d filtered_out=%d failed=%d total=%d",
        len(all_passed),
        total_filtered_out,
        total_failed,
        len(raw_posts),
    )

    return all_passed, total_failed
