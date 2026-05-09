from __future__ import annotations

import logging
import types
import warnings
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Union, get_args, get_origin

from elasticsearch import Elasticsearch, helpers

from config import Settings, settings
from models import Post, ProcessedPost

logger = logging.getLogger(__name__)


# Fields that should be mapped as `keyword` (exact match) instead of `text` (analyzed).
_KEYWORD_FIELDS = {
    "post_id",
    "author",
    "platform",
    "language",
    "url",
    "sentiment",
    "country_of_origin",
}


def _unwrap_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(annotation) if a is not type(None)]
        return args[0] if len(args) == 1 else annotation

    return annotation


def _annotation_to_es(name: str, annotation: Any, embedding_dims: int) -> dict[str, Any]:
    inner = _unwrap_optional(annotation)
    origin = get_origin(inner)

    # list[float] → dense_vector
    if origin is list and get_args(inner) == (float,):
        return {
            "type": "dense_vector",
            "dims": embedding_dims,
            "index": True,
            "similarity": "cosine",
        }

    # list[str] → keyword (multi-value)
    if origin is list and get_args(inner) == (str,):
        return {"type": "keyword"}

    # Keyword override for specific fields
    if name in _KEYWORD_FIELDS:
        return {"type": "keyword"}

    # Scalar types
    if inner is str:
        return {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 8192}}}
    if inner is int:
        return {"type": "long"}
    if inner is float:
        return {"type": "double"}
    if inner is datetime:
        return {"type": "date"}

    # Fallback
    return {"type": "keyword"}


def _build_mapping(embedding_dims: int) -> dict[str, Any]:
    """Derive ES mapping from the ProcessedPost model fields."""
    properties: dict[str, Any] = {}
    for name, field_info in ProcessedPost.model_fields.items():
        properties[name] = _annotation_to_es(name, field_info.annotation, embedding_dims)
    return {"mappings": {"properties": properties}}


_client: Elasticsearch | None = None


def _build_client(cfg: Settings) -> Elasticsearch:
    client_kwargs: dict[str, Any] = {
        "hosts": [cfg.host],
        "basic_auth": (cfg.username, cfg.password),
        "request_timeout": cfg.request_timeout,
        "verify_certs": cfg.verify_certs,
    }
    if cfg.ca_certs:
        client_kwargs["ca_certs"] = cfg.ca_certs

    if not cfg.verify_certs:
        try:
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            warnings.filterwarnings("ignore", message="Unverified HTTPS request")

    return Elasticsearch(**client_kwargs)


def get_client() -> Elasticsearch:
    global _client
    if _client is None:
        _client = _build_client(settings)

    return _client


def close_client() -> None:
    global _client
    if _client is not None:
        try:
            _client.close()
        finally:
            _client = None


def ping() -> bool:
    try:
        get_client().cluster.health()
        return True
    except Exception as exc:
        logger.warning("Elasticsearch ping failed: %s", exc)
        return False


def _resolve_embedding_dims() -> int:
    """Get the embedding dimension to use for the ES mapping.

    Prefer the sentence model's actual dimension (so the mapping always
    matches what `vectorize_text` produces). Fall back to the configured
    setting if the model isn't importable (e.g. in lightweight test envs).
    """
    try:
        from services.ml_services import get_embedding_dims

        return get_embedding_dims()
    except Exception as exc:
        logger.warning(
            "Could not determine sentence model dims, using ELASTIC_EMBEDDING_DIMS=%d: %s",
            settings.embedding_dims,
            exc,
        )
        return settings.embedding_dims


def ensure_posts_index(index: str | None = None) -> bool:
    client = get_client()
    target = index or settings.posts_index
    if client.indices.exists(index=target):
        return False
    try:
        client.indices.create(index=target, **_build_mapping(_resolve_embedding_dims()))
    except Exception:
        logger.exception("Failed to create index %r", target)
        raise
    logger.info("Created index %r", target)
    return True


def store_post(post: Post, *, index: str | None = None, refresh: bool = False) -> dict[str, Any]:
    client = get_client()
    target = index or settings.posts_index
    doc = post.model_dump(mode="json", exclude_none=True)
    resp = client.index(
        index=target, id=post.post_id, document=doc, refresh="wait_for" if refresh else False
    )

    return resp.body


def store_posts(
    posts: Iterable[Post], *, index: str | None = None, refresh: bool = False
) -> tuple[int, list]:
    client = get_client()
    target = index or settings.posts_index

    actions = (
        {
            "_op_type": "index",
            "_index": target,
            "_id": post.post_id,
            "_source": post.model_dump(mode="json", exclude_none=True),
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

    return success, errors if isinstance(errors, list) else []
