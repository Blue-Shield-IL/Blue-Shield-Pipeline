import logging
from collections.abc import Iterable
from typing import Any

from config import settings
from models import Post
from .adapters.elastic_adapter import ElasticsearchAdapter

logger = logging.getLogger(__name__)

# Global adapter instance
elastic_adapter: ElasticsearchAdapter | None = None


def get_adapter() -> ElasticsearchAdapter:
    global elastic_adapter
    if elastic_adapter is None:
        elastic_adapter = ElasticsearchAdapter(settings)
    return elastic_adapter


def get_client() -> Any:
    """Returns the underlying elasticsearch client from the adapter."""
    return get_adapter().client


def close_client() -> None:
    global elastic_adapter
    if elastic_adapter is not None:
        elastic_adapter.close()
        elastic_adapter = None


def ping() -> bool:
    return get_adapter().ping()


def ensure_posts_index(index: str | None = None) -> bool:
    adapter = get_adapter()
    target = index or settings.posts_index
    return adapter.ensure_index(target, settings.embedding_dims)


def store_post(post: Post, *, index: str | None = None, refresh: bool = False) -> dict[str, Any]:
    adapter = get_adapter()
    target = index or settings.posts_index
    return adapter.store_post(post, target, refresh)


def store_posts(
    posts: Iterable[Post], *, index: str | None = None, refresh: bool = False
) -> tuple[int, list]:
    adapter = get_adapter()
    target = index or settings.posts_index
    return adapter.store_posts(posts, target, refresh)
