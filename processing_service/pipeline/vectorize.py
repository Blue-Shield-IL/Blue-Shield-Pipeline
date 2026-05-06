"""Step 2 — Vectorize (stub)."""

from __future__ import annotations

import random

from config import settings
from models import Post

from .ingest import RawPost


def vectorize(raw_posts: list[RawPost]) -> list[Post]:
    """Convert raw dicts into Post objects with a fake embedding."""
    dims = settings.embedding_dims
    results: list[Post] = []
    for raw in raw_posts:
        rng = random.Random(raw.get("text_content", ""))
        raw["embedding"] = [rng.random() for _ in range(dims)]
        try:
            results.append(Post(**raw))
        except Exception:
            continue

    return results
