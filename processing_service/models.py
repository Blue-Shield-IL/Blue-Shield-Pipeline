"""
NOTE: This is a minimal stub.
Only the fields needed by the storage layer are declared here.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Post(BaseModel):
    post_id: str = Field(..., description="Unique identifier of the post.")
    text_content: str = Field(..., description="Raw textual content of the post.")
    author: str = Field(..., description="Author or username.")
    platform: str = Field(..., description="Source platform, e.g. telegram.")
    created_at: datetime = Field(..., description="When the post was created at source.")

    # Vector produced by the vectorization stage.
    embedding: list[float] | None = None
