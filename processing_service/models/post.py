from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class Author(BaseModel):
    id: str | None = None
    name: str | None = None
    username: str | None = None
    phone: str | None = Field(default=None, exclude=True)


class Channel(BaseModel):
    id: str | None = None
    name: str | None = None
    username: str | None = None
    description: str | None = Field(default=None, exclude=True)


class Post(BaseModel):
    post_id: str
    text_content: str
    author: Author | None = None
    channel: Channel | None = None
    platform: str
    created_at: datetime
    likes: int = 0
    shares: int = 0
    comments_count: int = 0
    views: int = 0
    hashtags: list[str] = Field(default_factory=list)
    mentions: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    url: str | None = None
    language: str | None = None

    @field_validator("post_id", "text_content")
    @classmethod
    def field_must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field must be non-empty")
        return value


class ProcessedPost(Post):
    antisemitism_score: float | None = None
    sentiment: str | None = None
    ihra_labels: list[str] = Field(default_factory=list)
    country_of_origin: str | None = None
    text_vector: list[float] | None = None
