import logging
from typing import Any

import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from config import settings
from models import Post, ProcessedPost

TEXT_CLASSIFIER = None
SENTENCE_MODEL = None
SENTIMENT_CLASSIFIER = None
ZERO_SHOT_CLASSIFIER = None
NER_PIPELINE = None
HF_DEVICE = 0 if torch.cuda.is_available() else -1

LOGGER = logging.getLogger(__name__)

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
    "Holocaust",
    "October 7th",
    "Zionist",
    "synagogue",
    "antisemitism",
    "Jewish conspiracy",
    "globalists",
    "media control",
    "bank control",
    "dual loyalty",
]

SENTIMENT_LABEL_MAP = {
    "LABEL_0": "Hostile",
    "LABEL_1": "Neutral",
    "LABEL_2": "Supportive",
}


def get_text_classifier() -> Any:
    global TEXT_CLASSIFIER
    if TEXT_CLASSIFIER is None:
        TEXT_CLASSIFIER = pipeline(
            "text-classification", model=settings.filter_model_name, device=HF_DEVICE
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


def get_sentiment_classifier() -> Any:
    global SENTIMENT_CLASSIFIER
    if SENTIMENT_CLASSIFIER is None:
        SENTIMENT_CLASSIFIER = pipeline(
            "text-classification", model=settings.sentiment_model_name, device=HF_DEVICE
        )
    return SENTIMENT_CLASSIFIER


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


def filter_post(raw_post_data: dict[str, Any], threshold: float = 0.6) -> ProcessedPost | None:
    """Filter a post by the score of a configured target label."""
    post = Post.model_validate(raw_post_data)
    classifier = get_text_classifier()
    results = classifier(post.text_content, truncation=True, top_k=None)

    target_label = settings.filter_target_label.upper()

    score = 0.0
    found = False
    for r in results:
        if str(r.get("label", "")).upper() == target_label:
            score = float(r.get("score", 0.0))
            found = True
            break

    if not found:
        LOGGER.warning(
            "FILTER_TARGET_LABEL=%r not found in classifier output labels %s. "
            "All posts will be filtered out. Check FILTER_MODEL_NAME and FILTER_TARGET_LABEL.",
            target_label,
            [str(r.get("label", "")) for r in results],
        )
        return None

    if score < threshold:
        return None

    return ProcessedPost(**post.model_dump(), antisemitism_score=score)


def analyze_content(processed_post: ProcessedPost) -> ProcessedPost:
    text = processed_post.text_content

    try:
        sentiment_classifier = get_sentiment_classifier()
        sentiment_result = sentiment_classifier(text, truncation=True)[0]
        raw_label = str(sentiment_result.get("label", "")).strip().upper()
        processed_post.sentiment = SENTIMENT_LABEL_MAP.get(raw_label, "Neutral")
    except Exception as exc:
        LOGGER.exception("Sentiment analysis failed: %s", exc)

    try:
        zero_shot_classifier = get_zero_shot_classifier()
        candidates = IHRA_LABELS + KEYWORD_LABELS
        zero_shot_result = zero_shot_classifier(text, candidate_labels=candidates, multi_label=True)

        labels = zero_shot_result.get("labels", [])
        scores = zero_shot_result.get("scores", [])
        label_scores = dict(zip(labels, scores, strict=False))

        processed_post.ihra_labels = [
            label for label in IHRA_LABELS if float(label_scores.get(label, 0.0)) >= 0.35
        ]

        merged_keywords = set(processed_post.keywords)
        for keyword in KEYWORD_LABELS:
            if float(label_scores.get(keyword, 0.0)) >= 0.35:
                merged_keywords.add(keyword)
        processed_post.keywords = sorted(merged_keywords)
    except Exception as exc:
        LOGGER.exception("Zero-shot IHRA/keyword extraction failed: %s", exc)

    try:
        ner = get_ner_pipeline()
        entities = ner(text)
        country = None
        for entity in entities:
            entity_group = str(entity.get("entity_group", "")).upper()
            if entity_group == "LOC":
                word = str(entity.get("word", "")).strip()
                if word:
                    country = word
                    break
        processed_post.country_of_origin = country
    except Exception as exc:
        LOGGER.exception("NER country extraction failed: %s", exc)

    return processed_post


def vectorize_text(processed_post: ProcessedPost) -> ProcessedPost:
    model = get_sentence_model()
    vector = model.encode(processed_post.text_content)
    processed_post.text_vector = [float(value) for value in vector.tolist()]
    return processed_post
