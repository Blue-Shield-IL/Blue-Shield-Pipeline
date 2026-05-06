"""
Step 3 of the pipeline: Elasticsearch storage.

Given posts that have already been ingested and vectorized, persist them to
Elasticsearch so the API Server and dashboard can query them.

Public surface — callers should depend only on these:
    - `get_client()` / `close_client()`     lifecycle
    - `ping()`                               connectivity check
    - `ensure_posts_index()`                 create-if-missing + mapping
    - `store_post(post)`                     store one post
    - `store_posts(posts)`                   store many posts (bulk)

The index mapping includes a `dense_vector` field for the embedding produced
by the vectorization stage, enabling semantic search from the API Server.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from typing import Any

from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import TransportError

from config import ElasticsearchSettings, settings
from models import Post

logger = logging.getLogger(__name__)


def _build_mapping(embedding_dims: int) -> dict[str, Any]:
    """Build the posts-index mapping, parameterized by embedding dimension."""
    return {
        "mappings": {
            "properties": {
                "post_id": {"type": "keyword"},
                "text_content": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 8192}},
                },
                "author": {"type": "keyword"},
                "platform": {"type": "keyword"},
                "created_at": {"type": "date"},
                "likes": {"type": "long"},
                "shares": {"type": "long"},
                "comments_count": {"type": "long"},
                "views": {"type": "long"},
                "hashtags": {"type": "keyword"},
                "url": {"type": "keyword"},
                "language": {"type": "keyword"},
                "keywords": {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": embedding_dims,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        }
    }


# Pre-built default mapping using the dimensions from settings. Callers who need
# a different dimension (e.g. in tests) can pass their own via `ensure_posts_index`.
POSTS_INDEX_MAPPING: dict[str, Any] = _build_mapping(settings.embedding_dims)


_client: Elasticsearch | None = None


def _build_client(cfg: ElasticsearchSettings) -> Elasticsearch:
    """Create a new Elasticsearch client from the given settings."""
    if not cfg.password:
        # Surface misconfiguration early; ES will otherwise return 401.
        logger.warning("ELASTIC_PASSWORD is empty; authentication will likely fail.")

    client_kwargs: dict[str, Any] = {
        "hosts": [cfg.host],
        "basic_auth": (cfg.username, cfg.password),
        "request_timeout": cfg.request_timeout,
        "verify_certs": cfg.verify_certs,
    }
    if cfg.ca_certs:
        client_kwargs["ca_certs"] = cfg.ca_certs

    # When verify_certs=False, urllib3 emits an InsecureRequestWarning per call.
    # Silence it once, here, since the team knowingly runs against a self-signed cert.
    if not cfg.verify_certs:
        try:
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:  # pragma: no cover - best-effort
            warnings.filterwarnings("ignore", message="Unverified HTTPS request")

    return Elasticsearch(**client_kwargs)


def get_client() -> Elasticsearch:
    """Return a process-wide Elasticsearch client, building it on first use."""
    global _client
    if _client is None:
        _client = _build_client(settings)
    return _client


def close_client() -> None:
    """Close the cached client, if any. Safe to call multiple times."""
    global _client
    if _client is not None:
        try:
            _client.close()
        finally:
            _client = None


def ping() -> bool:
    """Return True if the cluster is reachable."""
    client = get_client()
    try:
        client.cluster.health(request_timeout=settings.request_timeout)
        return True
    except TransportError as exc:
        logger.warning("Elasticsearch ping failed: %s", exc)
        return False


def ensure_posts_index(
    index: str | None = None,
    embedding_dims: int | None = None,
) -> bool:
    """Create the posts index with the expected mapping if it doesn't exist.

    Returns True if the index was created during this call, False if it already
    existed. Does not modify the mapping of an existing index — changing
    `embedding_dims` on an existing index requires a reindex.
    """
    client = get_client()
    target = index or settings.posts_index
    if client.indices.exists(index=target):
        return False

    mapping = _build_mapping(embedding_dims or settings.embedding_dims)
    client.indices.create(index=target, **mapping)
    logger.info(
        "Created Elasticsearch index %r (embedding_dims=%d)",
        target,
        embedding_dims or settings.embedding_dims,
    )
    return True


def _post_to_doc(post: Post) -> dict[str, Any]:
    """Serialize a `Post` to a dict suitable for Elasticsearch indexing.

    `exclude_none=True` keeps the document compact by skipping fields the
    upstream stages haven't populated yet (e.g. a post without an embedding
    during a pre-vectorization dry run).
    """
    return post.model_dump(mode="json", exclude_none=True)


def store_post(
    post: Post,
    *,
    index: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Store a single post in Elasticsearch, keyed by `post_id` for idempotency.

    Re-storing a post with the same `post_id` updates the existing document.
    """
    client = get_client()
    target = index or settings.posts_index
    return dict(
        client.index(
            index=target,
            id=post.post_id,
            document=_post_to_doc(post),
            refresh="wait_for" if refresh else False,
        )
    )


def store_posts(
    posts: Iterable[Post],
    *,
    index: str | None = None,
    refresh: bool = False,
) -> tuple[int, list[dict[str, Any]]]:
    """Store many posts in a single bulk request.

    Returns (successful_count, errors). `errors` is a list of per-document
    failure payloads as returned by the _bulk API; it is empty on full success.
    """
    client = get_client()
    target = index or settings.posts_index

    actions = (
        {
            "_op_type": "index",
            "_index": target,
            "_id": post.post_id,
            "_source": _post_to_doc(post),
        }
        for post in posts
    )

    success, errors = helpers.bulk(
        client,
        actions,
        refresh="wait_for" if refresh else False,
        raise_on_error=False,
        raise_on_exception=False,
    )
    return success, list(errors)  # type: ignore[arg-type]
