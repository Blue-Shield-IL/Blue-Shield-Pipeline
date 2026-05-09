import logging
import os
from typing import Any

import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline

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
        model_name = os.getenv(
            "FILTER_MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english"
        )
        TEXT_CLASSIFIER = pipeline("text-classification", model=model_name, device=HF_DEVICE)
    return TEXT_CLASSIFIER


def get_sentence_model() -> SentenceTransformer:
    global SENTENCE_MODEL
    if SENTENCE_MODEL is None:
        model_name = os.getenv("SENTENCE_MODEL_NAME", "all-MiniLM-L6-v2")
        SENTENCE_MODEL = SentenceTransformer(model_name)
    return SENTENCE_MODEL


def get_embedding_dims() -> int:
    """Return the embedding dimension produced by the sentence model.

    Used by the storage layer to size the ES dense_vector field so the
    mapping always matches what `vectorize_text` actually produces.
    Falls back to ELASTIC_EMBEDDING_DIMS if the model can't report its size.
    """
    model = get_sentence_model()
    # `get_embedding_dimension` is the new name; fall back to the old one
    # for older sentence-transformers versions.
    getter = getattr(model, "get_embedding_dimension", None) or getattr(
        model, "get_sentence_embedding_dimension", None
    )
    dims = getter() if getter else None
    if dims is None:
        return int(os.getenv("ELASTIC_EMBEDDING_DIMS", "384"))
    return int(dims)


def get_sentiment_classifier() -> Any:
    global SENTIMENT_CLASSIFIER
    if SENTIMENT_CLASSIFIER is None:
        model_name = os.getenv("SENTIMENT_MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment")
        SENTIMENT_CLASSIFIER = pipeline("text-classification", model=model_name, device=HF_DEVICE)
    return SENTIMENT_CLASSIFIER


def get_zero_shot_classifier() -> Any:
    global ZERO_SHOT_CLASSIFIER
    if ZERO_SHOT_CLASSIFIER is None:
        model_name = os.getenv("ZERO_SHOT_MODEL_NAME", "facebook/bart-large-mnli")
        ZERO_SHOT_CLASSIFIER = pipeline(
            "zero-shot-classification", model=model_name, device=HF_DEVICE
        )
    return ZERO_SHOT_CLASSIFIER


def get_ner_pipeline() -> Any:
    global NER_PIPELINE
    if NER_PIPELINE is None:
        model_name = os.getenv("NER_MODEL_NAME", "dbmdz/bert-large-cased-finetuned-conll03-english")
        NER_PIPELINE = pipeline(
            "ner",
            model=model_name,
            aggregation_strategy="simple",
            device=HF_DEVICE,
        )  # ty:ignore[no-matching-overload]
    return NER_PIPELINE


def filter_post(raw_post_data: dict[str, Any], threshold: float = 0.6) -> ProcessedPost | None:
    """Filter a post by the score of a configured target label.

    The classifier model and the label whose score is treated as the
    "antisemitism probability" are both env-configurable via
    FILTER_MODEL_NAME and FILTER_TARGET_LABEL. The default SST-2 model
    treats "POSITIVE" as the target — that's a stand-in for development;
    swap it for an actual antisemitism/abuse classifier in production.
    """
    post = Post.model_validate(raw_post_data)
    classifier = get_text_classifier()
    # Request scores for all labels so we can pick the target class by name,
    # not rely on whichever label the classifier happens to argmax to.
    results = classifier(post.text_content, truncation=True, top_k=None)

    target_label = os.getenv("FILTER_TARGET_LABEL", "NEGATIVE").upper()
    available_labels = [str(r.get("label", "")).upper() for r in results]

    if target_label not in available_labels:
        LOGGER.warning(
            "FILTER_TARGET_LABEL=%r not found in classifier output labels %s. "
            "All posts will be filtered out. Check FILTER_MODEL_NAME and FILTER_TARGET_LABEL.",
            target_label,
            available_labels,
        )
        return None

    score = 0.0
    for r in results:
        if str(r.get("label", "")).upper() == target_label:
            score = float(r.get("score", 0.0))
            break

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
    except Exception:
        LOGGER.exception("Sentiment analysis failed; using previous/default sentiment")

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
    except Exception:
        LOGGER.exception("Zero-shot IHRA/keyword extraction failed; using previous/default values")

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
    except Exception:
        LOGGER.exception("NER country extraction failed; using previous/default country")

    return processed_post


def vectorize_text(processed_post: ProcessedPost) -> ProcessedPost:
    model = get_sentence_model()
    vector = model.encode(processed_post.text_content)
    processed_post.text_vector = [float(value) for value in vector.tolist()]
    return processed_post
