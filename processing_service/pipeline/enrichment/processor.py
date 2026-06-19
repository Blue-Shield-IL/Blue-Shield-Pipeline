import logging
from typing import Any

from config import settings
from consts.labels import IHRA_LABELS, KEYWORD_LABELS
from models import Post, ProcessedPost
from .adapters.gemini_adapter import GeminiAdapter

LOGGER = logging.getLogger(__name__)

gemini_adapter = GeminiAdapter()


def analyze_content_batch(
    raw_posts: list[dict[str, Any]]
) -> tuple[list[ProcessedPost], int]:
    """
    Validates RawPosts, calls Gemini API to extract antisemitism scores and labels,
    and returns a list of ProcessedPost objects.
    Also returns the count of items that failed validation.
    """
    posts: list[Post] = []
    validation_failures = 0

    for rp in raw_posts:
        try:
            posts.append(Post.model_validate(rp))
        except Exception as exc:
            LOGGER.warning("Post validation failed (skipping)", extra={"error": str(exc)})
            validation_failures += 1

    if not posts:
        return [], validation_failures

    contexts = [
        {
            "text": p.text_content,
            "author_phone": p.author_phone,
            "channel_description": p.channel_description
        }
        for p in posts
    ]

    results = gemini_adapter.analyze_posts(contexts, IHRA_LABELS, KEYWORD_LABELS)

    analyzed: list[ProcessedPost] = []
    for post, res in zip(posts, results):

        pp = ProcessedPost(
            **post.model_dump(),
            antisemitism_score=round(res.antisemitism_score, 6)
        )
        pp.ihra_labels = res.ihra_labels
        pp.keywords = sorted(list(set(post.keywords + res.keywords)))
        pp.sentiment = res.sentiment
        pp.country_of_origin = res.country_of_origin
        analyzed.append(pp)

    return analyzed, validation_failures


def filter_posts_batch(
    analyzed_posts: list[ProcessedPost],
    threshold: float | None = None
) -> tuple[list[ProcessedPost], int]:
    """
    Iterates through analyzed posts and drops any that fall below the antisemitism threshold.
    Returns (passed_posts, filtered_out_count).
    """
    if threshold is None:
        threshold = settings.filter_threshold

    passed: list[ProcessedPost] = []
    filtered_out = 0

    for post in analyzed_posts:
        LOGGER.info("Analyzing post", extra={"payload": {"post_id": post.post_id, "antisemitism_score": post.antisemitism_score}})

        if post.antisemitism_score < threshold:
            filtered_out += 1
        else:
            passed.append(post)

    return passed, filtered_out


def vectorize_texts_batch(posts: list[ProcessedPost]) -> list[ProcessedPost]:
    """Batch version of vectorize_text using Gemini."""
    texts = [p.text_content for p in posts]

    vectors = gemini_adapter.vectorize_texts(texts)

    for post, vector in zip(posts, vectors):
        post.text_vector = vector

    return posts


def count_tokens(text: str) -> int:
    return gemini_adapter.count_tokens(text)
