"""
Step 2 of the pipeline: filter + analyze + vectorize.

Eventually this will call a Hugging Face / OpenAI model to:
  - score "probability of antisemitism" and drop low-scoring posts
  - label sentiment (Hostile / Neutral / Supportive) and IHRA category
  - extract keywords, country, author influence
  - produce a dense embedding for semantic search

Current implementation is a stub that produces a random `embedding` so
downstream storage and kNN can be exercised end to end.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from config import settings
from models import Post

from .ingest import RawPost

logger = logging.getLogger(__name__)


def _embed(text: str, dims: int) -> list[float]:
    """Produce a dense vector for the given text.

    Placeholder — real implementation will use sentence-transformers or
    an equivalent embedding model whose output size matches `dims`.
    """
    # Seed with the text so the same post always produces the same vector.
    # This keeps re-runs idempotent during development.
    rng = random.Random(text)
    return [rng.random() for _ in range(dims)]


def _extract_keywords(text: str) -> list[str]:
    """Very naive keyword extractor: the 5 longest distinct words.

    Placeholder until the real keyword tagger lands.
    """
    seen: set[str] = set()
    tokens = [t.strip(".,!?\"'()[]{}").lower() for t in text.split()]
    tokens = [t for t in tokens if len(t) > 3]
    tokens.sort(key=len, reverse=True)
    result: list[str] = []
    for tok in tokens:
        if tok in seen:
            continue
        seen.add(tok)
        result.append(tok)
        if len(result) >= 5:
            break
    return result


def vectorize(raw_posts: list[RawPost]) -> list[Post]:
    """Enrich raw posts with keywords + embedding and coerce into `Post`s.

    Raw dicts that fail validation are logged and skipped so one bad record
    does not poison the whole batch.
    """
    dims = settings.embedding_dims
    enriched: list[Post] = []
    for raw in raw_posts:
        data: dict[str, Any] = dict(raw)  # copy so we don't mutate the caller's dict
        text = data.get("text_content", "")
        data.setdefault("keywords", _extract_keywords(text))
        data["embedding"] = _embed(text, dims)
        try:
            enriched.append(Post(**data))
        except Exception:
            logger.exception("Vectorization failed for post %r; skipping", data.get("post_id"))
    logger.info("Vectorized %d/%d posts (dims=%d)", len(enriched), len(raw_posts), dims)
    return enriched
