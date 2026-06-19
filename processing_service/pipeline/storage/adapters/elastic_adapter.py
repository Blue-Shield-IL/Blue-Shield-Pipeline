import logging
import types
import warnings
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Union, get_args, get_origin

from elasticsearch import Elasticsearch, helpers

from config import Settings
from models import Post, ProcessedPost

logger = logging.getLogger(__name__)


class ElasticsearchAdapter:
    """Wrapper class for Elasticsearch interactions."""

    _KEYWORD_FIELDS = {
        "post_id",
        "author",
        "platform",
        "language",
        "url",
        "channel",
        "sentiment",
        "country_of_origin",
    }

    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.client = self._build_client(cfg)

    def _build_client(self, cfg: Settings) -> Elasticsearch:
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

    @staticmethod
    def _unwrap_optional(annotation: Any) -> Any:
        origin = get_origin(annotation)
        if origin is Union or origin is types.UnionType:
            args = [a for a in get_args(annotation) if a is not type(None)]
            return args[0] if len(args) == 1 else annotation
        return annotation

    @classmethod
    def _annotation_to_es(cls, name: str, annotation: Any, embedding_dims: int) -> dict[str, Any]:
        inner = cls._unwrap_optional(annotation)
        origin = get_origin(inner)

        if origin is list and get_args(inner) == (float,):
            return {
                "type": "dense_vector",
                "dims": embedding_dims,
                "index": True,
                "similarity": "cosine",
            }

        if origin is list and get_args(inner) == (str,):
            return {"type": "keyword"}

        if name in cls._KEYWORD_FIELDS:
            return {"type": "keyword"}

        if inner is str:
            return {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 8192}}}
        if inner is int:
            return {"type": "long"}
        if inner is float:
            return {"type": "double"}
        if inner is datetime:
            return {"type": "date"}

        return {"type": "keyword"}

    @classmethod
    def build_mapping(cls, embedding_dims: int) -> dict[str, Any]:
        """Derive ES mapping from the ProcessedPost model fields."""
        properties: dict[str, Any] = {}
        for name, field_info in ProcessedPost.model_fields.items():
            properties[name] = cls._annotation_to_es(name, field_info.annotation, embedding_dims)
        return {"mappings": {"properties": properties}}

    def close(self) -> None:
        if self.client:
            self.client.close()

    def ping(self) -> bool:
        try:
            self.client.cluster.health()
            return True
        except Exception as exc:
            logger.warning("Elasticsearch ping failed", extra={"error": str(exc)})
            return False

    def ensure_index(self, index_name: str, embedding_dims: int) -> bool:
        """Ensure the target index exists, creating it using the dynamically built mapping if not."""
        if self.client.indices.exists(index=index_name):
            return False

        mapping = self.build_mapping(embedding_dims)
        try:
            self.client.indices.create(index=index_name, **mapping)
        except Exception as exc:
            if "resource_already_exists_exception" in str(exc):
                return False
            logger.error("Failed to create index", extra={"payload": {"index_name": index_name}, "error": str(exc)})
            raise
        logger.info("Created index", extra={"payload": {"index_name": index_name}})
        return True

    def store_post(self, post: Post, index_name: str, refresh: bool = False) -> dict[str, Any]:
        doc = post.model_dump(mode="json", exclude_none=True)
        resp = self.client.index(
            index=index_name, id=post.post_id, document=doc, refresh="wait_for" if refresh else False
        )
        return resp.body

    def store_posts(
        self, posts: Iterable[Post], index_name: str, refresh: bool = False
    ) -> tuple[int, list]:
        actions = (
            {
                "_op_type": "index",
                "_index": index_name,
                "_id": post.post_id,
                "_source": post.model_dump(mode="json", exclude_none=True),
            }
            for post in posts
        )

        success, errors = helpers.bulk(
            self.client,
            actions,
            refresh="wait_for" if refresh else False,
            raise_on_error=False,
            raise_on_exception=False,
        )

        errors_list = errors if isinstance(errors, list) else []
        for err in errors_list:
            logger.error("Elasticsearch bulk index error", extra={"error": str(err)})

        return success, errors_list
