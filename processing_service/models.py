"""
Pydantic data models for the Processing Service.

The `Post` model is the canonical schema for social-media posts flowing
through the Blue Shield pipeline. It is used for both API request validation
and for shaping documents stored in Elasticsearch.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Post(BaseModel):
    """A normalized social-media post.

    Mirrors the schema defined in `.kiro/specs/posts-data-schema/design.md`.
    All timestamps are represented as ISO 8601 strings on the wire and as
    `datetime` objects in memory.
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
    url: Optional[str] = None
    language: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)

    @field_validator("post_id", "text_content")
    @classmethod
    def _non_empty_after_strip(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value
