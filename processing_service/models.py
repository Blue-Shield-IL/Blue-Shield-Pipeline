"""
Pydantic data models for the Processing Service.

The `Post` model is the canonical schema for social-media posts flowing
through the Blue Shield pipeline. It is produced by the vectorization stage
and handed to the storage layer for indexing into Elasticsearch.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class Post(BaseModel):
    """A normalized social-media post ready for storage in Elasticsearch.

    Mirrors the schema defined in `.kiro/specs/posts-data-schema/design.md`.
    All timestamps are represented as ISO 8601 strings on the wire and as
    `datetime` objects in memory. `embedding` is populated by the
    vectorization stage before the post reaches the storage layer.
    """

    post_id: str = Field(..., description="Unique identifier of the post.")
    text_content: str = Field(..., description="Raw textual content of the post.")
    author: str = Field(..., description="Author or username of the post.")
    platform: str = Field(..., description="Source platform, e.g. telegram, reddit.")
    created_at: datetime = Field(..., description="When the post was created at source.")

    # Engagement metrics — optional, default to zero so partial-source data is tolerated.
    likes: int = 0
    shares: int = 0
    comments_count: int = 0
    views: int = 0

    # Free-form metadata.
    hashtags: list[str] = Field(default_factory=list)
    url: str | None = None
    language: str | None = None
    keywords: list[str] = Field(default_factory=list)

    # Vector produced by the upstream vectorization stage. Optional so the
    # schema can represent a post before it has been embedded (e.g. for debug
    # dumps), but the storage layer expects this to be populated in production.
    embedding: list[float] | None = Field(
        default=None,
        description="Dense vector embedding produced by the vectorization stage.",
    )

    @field_validator("post_id", "text_content")
    @classmethod
    def _non_empty_after_strip(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value
