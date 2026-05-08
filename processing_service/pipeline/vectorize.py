from __future__ import annotations

import logging

from models import ProcessedPost
from services import analyze_content, filter_post, vectorize_text

from .ingest import RawPost

logger = logging.getLogger(__name__)


def vectorize(raw_posts: list[RawPost]) -> tuple[list[ProcessedPost], int]:
    results: list[ProcessedPost] = []
    failed = 0
    filtered_out = 0

    for raw_post_data in raw_posts:
        try:
            processed_post = filter_post(raw_post_data)
            if processed_post is None:
                filtered_out += 1
                logger.debug(
                    "Post filtered out due to low score: post_id=%r",
                    raw_post_data.get("post_id", "unknown"),
                )
            else:
                processed_post = analyze_content(processed_post)
                processed_post = vectorize_text(processed_post)
                results.append(processed_post)
        except Exception as exc:
            failed += 1
            logger.warning(
                "Pipeline failed for post_id=%r at filter/analyze/vectorize stage: %s",
                raw_post_data.get("post_id", "unknown"),
                exc,
            )

    logger.info(
        "Vectorize batch complete: processed=%d filtered_out=%d failed=%d total=%d",
        len(results),
        filtered_out,
        failed,
        len(raw_posts),
    )

    return results, failed
