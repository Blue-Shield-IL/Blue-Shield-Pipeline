import logging
from typing import Any

import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from config import settings
from models import Post, ProcessedPost

TEXT_CLASSIFIER = None
SENTENCE_MODEL = None
ZERO_SHOT_CLASSIFIER = None
NER_PIPELINE = None
HF_DEVICE = 0 if torch.cuda.is_available() else -1

LOGGER = logging.getLogger(__name__)

TOXICITY_WEIGHTS: dict[str, float] = {
    "label_3": 1.0,  # ANTISEMITISM
    "label_7": 0.2,  # RACISM
    "label_2": 0.1,  # RELIGION
    "label_6": 0.3   # POLITICS
}

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

SENTIMENT_LABELS = ["Supportive", "Neutral", "Negative", "Hostile"]


def get_text_classifier() -> Any:
    global TEXT_CLASSIFIER
    if TEXT_CLASSIFIER is None:
        TEXT_CLASSIFIER = pipeline(
            "text-classification",
            model=settings.filter_model_name,
            device=HF_DEVICE,
            padding=True,
            truncation=True,
            top_k=None,
            function_to_apply="sigmoid",
            dtype=torch.float16 if HF_DEVICE == 0 else torch.float32
        )
    return TEXT_CLASSIFIER


def get_sentence_model() -> SentenceTransformer:
    global SENTENCE_MODEL
    if SENTENCE_MODEL is None:
        SENTENCE_MODEL = SentenceTransformer(settings.sentence_model_name)
    return SENTENCE_MODEL


def get_embedding_dims() -> int:
    """Return the embedding dimension produced by the sentence model."""
    model = get_sentence_model()
    getter = getattr(model, "get_embedding_dimension", None) or getattr(
        model, "get_sentence_embedding_dimension", None
    )
    dims = getter() if getter else None
    if dims is None:
        return settings.embedding_dims
    return int(dims)


def get_zero_shot_classifier() -> Any:
    global ZERO_SHOT_CLASSIFIER
    if ZERO_SHOT_CLASSIFIER is None:
        ZERO_SHOT_CLASSIFIER = pipeline(
            "zero-shot-classification", model=settings.zero_shot_model_name, device=HF_DEVICE
        )
    return ZERO_SHOT_CLASSIFIER


def get_ner_pipeline() -> Any:
    global NER_PIPELINE
    if NER_PIPELINE is None:
        NER_PIPELINE = pipeline(
            "ner",
            model=settings.ner_model_name,
            aggregation_strategy="simple",
            device=HF_DEVICE,
        )  # ty:ignore[no-matching-overload]
    return NER_PIPELINE


def filter_posts_batch(
    raw_posts: list[dict[str, Any]],
    threshold: float | None = None,
    batch_size: int | None = None,
) -> tuple[list[ProcessedPost], int]:
    """
    Runs unitary/unbiased-toxic-roberta over each post, computes a weighted
    compound toxicity score from all output dimensions, and keeps only posts
    that meet the threshold.

    Returns (passed_posts, filtered_out_count).
    Validation failures are counted as filtered-out.
    """
    if threshold is None:
        threshold = settings.filter_threshold
    if batch_size is None:
        batch_size = settings.ml_batch_size

    posts: list[Post] = []
    filtered_out = 0
    for rp in raw_posts:
        try:
            posts.append(Post.model_validate(rp))
        except Exception as exc:
            LOGGER.warning("Post validation failed (skipping): %s", exc)
            filtered_out += 1

    if not posts:
        return [], filtered_out

    texts = [p.text_content for p in posts]
    classifier = get_text_classifier()
    all_results: list[list[dict]] = classifier(texts, batch_size=batch_size)

    passed: list[ProcessedPost] = []
    for post, results in zip(posts, all_results):
        # Build a label → score mapping from raw model output
        dim_scores: dict[str, float] = {
            str(r.get("label", "")).lower(): float(r.get("score", 0.0))
            for r in results
        }

        compound = sum(
            dim_scores.get(dim, 0.0) * weight
            for dim, weight in TOXICITY_WEIGHTS.items()
        )

        LOGGER.info(
            "post_id=%s compound=%.4f dims=%s",
            post.post_id,
            compound,
            {d: f"{dim_scores.get(d, 0.0):.3f}" for d in TOXICITY_WEIGHTS},
        )

        # print(ProcessedPost(
        #             **post.model_dump(),
        #             antisemitism_score=round(compound, 6)
        #         ), compound,
        #     {d: f"{dim_scores.get(d, 0.0):.3f}" for d in TOXICITY_WEIGHTS})
        print(post.text_content, compound, results)
        if compound < threshold:
            filtered_out += 1
        else:
            passed.append(
                ProcessedPost(
                    **post.model_dump(),
                    antisemitism_score=round(compound, 6)
                )
            )

    return passed, filtered_out


def analyze_content_batch(
    posts: list[ProcessedPost],
    label_threshold: float | None = None,
    batch_size: int | None = None,
) -> list[ProcessedPost]:
    """Runs sentiment, zero-shot, and NER over a list."""
    if not posts:
        return posts
    if batch_size is None:
        batch_size = settings.ml_batch_size

    texts = [p.text_content for p in posts]

    try:
        zero_shot_classifier = get_zero_shot_classifier()
        candidates = IHRA_LABELS + KEYWORD_LABELS + SENTIMENT_LABELS
        zero_shot_results = zero_shot_classifier(
            texts, candidate_labels=candidates, multi_label=True, batch_size=batch_size
        )

        if label_threshold is None:
            label_threshold = settings.label_threshold

        for post, zs_result in zip(posts, zero_shot_results):
            labels = zs_result.get("labels", [])
            scores = zs_result.get("scores", [])
            label_scores = dict(zip(labels, scores, strict=False))

            post.ihra_labels = [
                label for label in IHRA_LABELS if float(label_scores.get(label, 0.0)) >= label_threshold
            ]

            merged_keywords = set(post.keywords)
            for keyword in KEYWORD_LABELS:
                if float(label_scores.get(keyword, 0.0)) >= label_threshold:
                    merged_keywords.add(keyword)
            post.keywords = sorted(merged_keywords)

            sentiment_scores = {s: float(label_scores.get(s, 0.0)) for s in SENTIMENT_LABELS}
            post.sentiment = max(sentiment_scores, key=sentiment_scores.get)

    except Exception as exc:
        LOGGER.exception("Batch zero-shot (IHRA/keywords/sentiment) failed: %s", exc)

    # --- NER ---
    try:
        ner = get_ner_pipeline()
        # HF NER pipeline returns list[list[dict]] for list input
        ner_results = ner(texts, batch_size=batch_size)
        for post, entities in zip(posts, ner_results):
            for entity in entities:
                if str(entity.get("entity_group", "")).upper() == "LOC":
                    word = str(entity.get("word", "")).strip()
                    if word:
                        post.country_of_origin = word
                        break
    except Exception as exc:
        LOGGER.exception("Batch NER country extraction failed: %s", exc)

    return posts


def vectorize_texts_batch(
    posts: list[ProcessedPost],
    batch_size: int | None = None,
) -> list[ProcessedPost]:
    """Batch version of vectorize_text — encodes all posts in a single encode() call."""
    if not posts:
        return posts
    if batch_size is None:
        batch_size = settings.ml_batch_size

    model = get_sentence_model()
    texts = [p.text_content for p in posts]
    # SentenceTransformer.encode() natively supports batch_size
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    for post, vector in zip(posts, vectors):
        post.text_vector = [float(v) for v in vector.tolist()]
    return posts


