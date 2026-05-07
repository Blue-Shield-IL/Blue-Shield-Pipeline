"""
Step 2 — Process and vectorize posts using ML models.

Orchestrates: filter → analyze → vectorize for each raw post.
"""

from __future__ import annotations

import logging
from typing import Any

from models import ProcessedPost

from .ml import analyze_content, filter_post, vectorize_text

LOGGER = logging.getLogger(__name__)

RawPost = dict[str, Any]


def process_post(raw_post_data: RawPost) -> ProcessedPost | None:
    """Run the full ML pipeline on a single raw post.

    Returns None if the post is filtered out (below antisemitism threshold).
    """
    processed_post = filter_post(raw_post_data)
    if processed_post is None:
        return None

    processed_post = analyze_content(processed_post)
    processed_post = vectorize_text(processed_post)
    return processed_post


def vectorize(raw_posts: list[RawPost]) -> tuple[list[ProcessedPost], int]:
    """Process a batch of raw posts through the ML pipeline.

    Returns (successful_posts, failed_count).
    Posts that are filtered out (low score) count as successful (not failures).
    """
    results: list[ProcessedPost] = []
    failed = 0
    filtered_out = 0

    for raw in raw_posts:
        try:
            result = process_post(raw)
            if result is not None:
                results.append(result)
            else:
                filtered_out += 1
        except Exception as exc:
            failed += 1
            LOGGER.warning(
                "ML pipeline failed for post_id=%r: %s",
                raw.get("post_id", "unknown"),
                exc,
            )

    LOGGER.info(
        "Vectorize: %d processed, %d filtered out, %d failed (of %d total)",
        len(results),
        filtered_out,
        failed,
        len(raw_posts),
    )
    return results, failed
