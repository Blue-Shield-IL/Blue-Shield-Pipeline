from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

class Post(BaseModel):
    post_id: str
    text_content: str
    author: str
    platform: str
    created_at: datetime
    likes: int = 0
    shares: int = 0
    comments_count: int = 0
    views: int = 0
    hashtags: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    url: Optional[str] = None
    language: Optional[str] = None

    @field_validator("post_id", "text_content")
    @classmethod
    def field_must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field must be non-empty")
        return value


class ProcessedPost(Post):
    antisemitism_score: Optional[float] = None
    sentiment: Optional[str] = None
    ihra_labels: list[str] = Field(default_factory=list)
    country_of_origin: Optional[str] = None
    text_vector: list[float] = Field(default_factory=list)
