from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from typing import Any

from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import TransportError

from config import Settings, settings
from models import Post

logger = logging.getLogger(__name__)


# Maps Python/Pydantic types to Elasticsearch field types.
# When the schema mission adds fields to Post, this handles the mapping automatically.
_TYPE_MAP: dict[str, dict[str, Any] | None] = {
    "str": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 8192}}},
    "int": {"type": "long"},
    "float": {"type": "double"},
    "datetime": {"type": "date"},
    "list[str]": {"type": "keyword"},
    "list[float]": None,  # handled specially as dense_vector
}

# Fields that should be mapped as `keyword` (exact match) instead of `text` (analyzed).
_KEYWORD_FIELDS = {"post_id", "author", "platform", "language", "url"}


def _build_mapping(embedding_dims: int) -> dict[str, Any]:
    properties: dict[str, Any] = {}

    for name, field_info in Post.model_fields.items():
        annotation = str(field_info.annotation).replace("typing.Optional[", "").rstrip("]")
        annotation = annotation.split(" | ")[0] if " | " in annotation else annotation

        # Dense vector gets special treatment
        if annotation == "list[float]":
            properties[name] = {
                "type": "dense_vector",
                "dims": embedding_dims,
                "index": True,
                "similarity": "cosine",
            }
        elif name in _KEYWORD_FIELDS:
            properties[name] = {"type": "keyword"}
        elif annotation in _TYPE_MAP and _TYPE_MAP[annotation] is not None:
            properties[name] = _TYPE_MAP[annotation]
        else:
            # Fallback: keyword for unknown types
            properties[name] = {"type": "keyword"}

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
    except TransportError as exc:
        logger.warning("Elasticsearch ping failed: %s", exc)
        return False


def ensure_posts_index(index: str | None = None) -> bool:
    client = get_client()
    target = index or settings.posts_index
    if client.indices.exists(index=target):
        return False
    client.indices.create(index=target, **_build_mapping(settings.embedding_dims))
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
