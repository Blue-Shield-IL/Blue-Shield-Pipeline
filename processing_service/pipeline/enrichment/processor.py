import logging
from typing import Any

from config import settings
from models import Post, ProcessedPost
from .adapters.gemini_adapter import GeminiAdapter

LOGGER = logging.getLogger(__name__)

# Instantiate the adapter once to be reused
gemini_adapter = GeminiAdapter()

IHRA_LABELS = [
    "Calling for, aiding, or justifying the killing or harming of Jews",
    "Mendacious, dehumanizing, demonizing, or stereotypical allegations about Jews",
    "Accusing Jews as a people of being responsible for real or imagined wrongdoing",
    "Denying the fact, scope, mechanisms, or intentionality of the Holocaust",
    "Accusing the Jews as a people, or Israel as a state, of inventing or exaggerating the Holocaust",
    "Accusing Jewish citizens of dual loyalty",
    "Applying double standards to Israel not expected of any other democratic nation",
    "Using symbols and images associated with classic antisemitism to characterize Israel or Israelis",
    "Drawing comparisons of contemporary Israeli policy to that of the Nazis",
    "Holding Jews collectively responsible for actions of the State of Israel",
]

KEYWORD_LABELS = [
    # Events & context
    "October 7th Hamas attack",
    "Holocaust denial or distortion",
    "pogrom or mob violence against Jews",
    # Classic tropes
    "Jewish control of media",
    "Jewish control of banks or financial system",
    "Jewish world domination conspiracy",
    "blood libel",
    "great replacement theory",
    "Protocols of the Elders of Zion",
    # Modern coded language
    "globalist conspiracy",
    "George Soros conspiracy",
    "Rothschild conspiracy",
    "New World Order Jewish conspiracy",
    # Israel context
    "Israel genocide accusation",
    "Zionist occupation",
    "Israeli apartheid",
]


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
            LOGGER.warning("Post validation failed (skipping): %s", exc)
            validation_failures += 1

    if not posts:
        return [], validation_failures

    texts = [p.text_content for p in posts]

    # Call Gemini API to extract everything
    results = gemini_adapter.analyze_posts(texts, IHRA_LABELS, KEYWORD_LABELS)

    analyzed: list[ProcessedPost] = []
    for post, res in zip(posts, results):
        # Construct ProcessedPost from base fields + extracted Gemini analysis
        pp = ProcessedPost(
            **post.model_dump(),
            antisemitism_score=round(res.antisemitism_score, 6)
        )
        pp.ihra_labels = res.ihra_labels
        pp.keywords = sorted(res.keywords)
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
        LOGGER.info(
            "post_id=%s antisemitism_score=%.4f",
            post.post_id,
            post.antisemitism_score,
        )

        if post.antisemitism_score < threshold:
            filtered_out += 1
        else:
            passed.append(post)

    return passed, filtered_out


def vectorize_texts_batch(posts: list[ProcessedPost]) -> list[ProcessedPost]:
    """Batch version of vectorize_text using Gemini."""
    if not posts:
        return posts

    texts = [p.text_content for p in posts]

    # Use Gemini embeddings
    vectors = gemini_adapter.vectorize_texts(texts)

    for post, vector in zip(posts, vectors):
        post.text_vector = vector

    return posts
