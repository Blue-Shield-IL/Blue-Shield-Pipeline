"""
Example showing how steps 1 & 2 of the pipeline hand posts to step 3 (storage).

Run from inside `processing_service/` after activating the venv:

    venv/Scripts/python.exe examples/store_posts_example.py

This file fakes the Telegram ingestion + vectorization outputs so you can
exercise the Elasticsearch storage layer in isolation. Replace the fake
`ingest_from_telegram` and `vectorize` calls with the real implementations
once steps 1 and 2 land.
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make `config`, `elastic`, `models` importable when running from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from elastic import ensure_posts_index, ping, store_posts
from models import Post


def ingest_from_telegram() -> list[dict]:
    """Stand-in for the Telegram ingestion stage (step 1)."""
    now = datetime.now(timezone.utc)
    return [
        {
            "post_id": f"telegram-demo-{i}",
            "text_content": f"Sample Telegram message {i}",
            "author": "@channel_demo",
            "platform": "telegram",
            "created_at": now.isoformat(),
            "likes": random.randint(0, 100),
            "hashtags": ["demo"],
            "language": "en",
        }
        for i in range(1, 4)
    ]


def vectorize(raw_posts: list[dict]) -> list[Post]:
    """Stand-in for the vectorization stage (step 2).

    The real implementation will produce a dense embedding from a model such
    as sentence-transformers. Here we just attach a random vector of the
    configured dimension so the storage shape matches production.
    """
    dims = settings.embedding_dims
    result: list[Post] = []
    for raw in raw_posts:
        raw["keywords"] = ["demo", raw["platform"]]
        raw["embedding"] = [random.random() for _ in range(dims)]
        result.append(Post(**raw))
    return result


def main() -> None:
    if not ping():
        raise SystemExit(
            "Elasticsearch not reachable. Check VPN and .env credentials before running."
        )

    created = ensure_posts_index()
    print(f"[store] posts index {'created' if created else 'already existed'}")

    raw = ingest_from_telegram()
    print(f"[ingest] got {len(raw)} raw posts")

    enriched = vectorize(raw)
    print(f"[vectorize] produced {len(enriched)} embedded posts (dims={settings.embedding_dims})")

    success, errors = store_posts(enriched, refresh=True)
    print(f"[store] indexed={success} errors={len(errors)}")
    if errors:
        for err in errors:
            print(f"  error: {err}")


if __name__ == "__main__":
    main()
