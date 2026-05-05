"""
Elasticsearch client and indexing helpers.

This module encapsulates:
- Construction of the Elasticsearch client from settings.
- The index mapping for the `posts` index.
- Create-if-missing logic for the index.
- Insertion helpers for single and batch posts.

Callers should prefer `get_client()` so the client is lazily created and
reused for the life of the process.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Iterable, Optional

from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import TransportError

from config import ElasticsearchSettings, settings
from models import Post

logger = logging.getLogger(__name__)

# Index mapping for the `posts` index. Mirrors the fields on `Post`.
# `text` is analyzed for full-text search; keyword fields are for filtering/aggregations.
POSTS_INDEX_MAPPING: dict[str, Any] = {
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
            # `embedding` is reserved for the future vectorization stage.
            # Dimension will be finalized once the embedding model is chosen.
            # Uncomment and set `dims` to enable semantic search.
            # "embedding": {"type": "dense_vector", "dims": 384, "index": True, "similarity": "cosine"},
        }
    }
}


_client: Optional[Elasticsearch] = None


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
    """Return True if the cluster is reachable and responds to /_cluster/health."""
    client = get_client()
    try:
        # `ping()` swallows exceptions and returns False; use health for clearer errors.
        client.cluster.health(request_timeout=settings.request_timeout)
        return True
    except TransportError as exc:
        logger.warning("Elasticsearch ping failed: %s", exc)
        return False


def ensure_posts_index(index: Optional[str] = None) -> bool:
    """Create the posts index with the expected mapping if it doesn't exist.

    Returns True if the index was created during this call, False if it already existed.
    """
    client = get_client()
    target = index or settings.posts_index
    if client.indices.exists(index=target):
        return False
    client.indices.create(index=target, **POSTS_INDEX_MAPPING)
    logger.info("Created Elasticsearch index %r", target)
    return True


def _post_to_doc(post: Post) -> dict[str, Any]:
    """Serialize a `Post` to a dict suitable for Elasticsearch indexing."""
    # mode="json" converts datetime -> ISO 8601 string, which ES `date` type parses natively.
    return post.model_dump(mode="json")


def index_post(post: Post, index: Optional[str] = None, refresh: bool = False) -> dict[str, Any]:
    """Index a single post. Uses `post_id` as the document id for idempotency."""
    client = get_client()
    target = index or settings.posts_index
    doc = _post_to_doc(post)
    response = client.index(
        index=target,
        id=post.post_id,
        document=doc,
        refresh="wait_for" if refresh else False,
    )
    return dict(response)


def bulk_index_posts(
    posts: Iterable[Post],
    index: Optional[str] = None,
    refresh: bool = False,
) -> tuple[int, list[dict[str, Any]]]:
    """Bulk-index a collection of posts.

    Returns (successful_count, errors) where errors is a list of per-document
    failure payloads as returned by the _bulk API.
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

    # `raise_on_error=False` lets us report partial failures instead of aborting the batch.
    success, errors = helpers.bulk(
        client,
        actions,
        refresh="wait_for" if refresh else False,
        raise_on_error=False,
        raise_on_exception=False,
    )
    # `errors` from helpers.bulk with raise_on_error=False is a list[dict].
    return success, list(errors)  # type: ignore[arg-type]
