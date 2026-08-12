import logging
from typing import Any

from langdetect import detect, LangDetectException
from langfuse import observe

from config import settings
from consts.labels import IHRA_LABELS, KEYWORD_LABELS
from models import Post, ProcessedPost
from .adapters.gemini_adapter import GeminiAdapter

LOGGER = logging.getLogger(__name__)

gemini_adapter = GeminiAdapter()


def _detect_language(text: str) -> str | None:
    if not text or len(text.strip()) < 20:
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None


@observe(name="analyze_content_batch")
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

    contexts = []
    for p in posts:
        ctx = {"text": p.text_content}
        if p.author:
            if p.author.phone:
                ctx["author_phone"] = p.author.phone
            if p.author.name:
                ctx["author_name"] = p.author.name
            if p.author.username:
                ctx["author_username"] = p.author.username
        if p.channel:
            if p.channel.description:
                ctx["channel_description"] = p.channel.description
            if p.channel.name:
                ctx["channel_name"] = p.channel.name
            if p.channel.username:
                ctx["channel_username"] = p.channel.username
        contexts.append(ctx)

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
        pp.language = _detect_language(post.text_content)
        analyzed.append(pp)

    return analyzed, validation_failures


@observe(name="filter_posts_batch")
def filter_posts_batch(
    analyzed_posts: list[ProcessedPost],
    threshold: float | None = None
) -> tuple[list[ProcessedPost], int]:
    """
    Drops posts that fall below the antisemitism threshold. Posts are kept when their content
    contains hate-speech, antisemitism, or anti-Zionism — regardless of whether the author
    endorses or condemns it. General neutral mentions of Jews/Israel are filtered out.
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


@observe(name="vectorize_texts_batch")
def vectorize_texts_batch(posts: list[ProcessedPost]) -> list[ProcessedPost]:
    """Batch version of vectorize_text using Gemini."""
    texts = [p.text_content for p in posts]

    vectors = gemini_adapter.vectorize_texts(texts)

    for post, vector in zip(posts, vectors):
        post.text_vector = vector

    return posts


def count_tokens(text: str) -> int:
    return gemini_adapter.count_tokens(text)
