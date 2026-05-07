"""
ML services for the processing pipeline (step 2).

All heavy imports (torch, transformers, sentence-transformers) are lazy
so that tests can mock the functions without needing the ML stack installed.
"""

import logging
import os
from typing import Any

from models import Post, ProcessedPost

LOGGER = logging.getLogger(__name__)

# Lazy-loaded model singletons
TEXT_CLASSIFIER = None
SENTENCE_MODEL = None
SENTIMENT_CLASSIFIER = None
ZERO_SHOT_CLASSIFIER = None
NER_PIPELINE = None
HF_DEVICE: int | None = None

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


def _get_device() -> int:
    global HF_DEVICE
    if HF_DEVICE is None:
        import torch

        HF_DEVICE = 0 if torch.cuda.is_available() else -1
    return HF_DEVICE


def get_text_classifier() -> Any:
    global TEXT_CLASSIFIER
    if TEXT_CLASSIFIER is None:
        from transformers import pipeline as hf_pipeline

        model_name = os.getenv(
            "FILTER_MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english"
        )
        TEXT_CLASSIFIER = hf_pipeline("text-classification", model=model_name)
    return TEXT_CLASSIFIER


def get_sentence_model() -> Any:
    global SENTENCE_MODEL
    if SENTENCE_MODEL is None:
        from sentence_transformers import SentenceTransformer

        model_name = os.getenv("SENTENCE_MODEL_NAME", "all-MiniLM-L6-v2")
        SENTENCE_MODEL = SentenceTransformer(model_name)
    return SENTENCE_MODEL


def get_sentiment_classifier() -> Any:
    global SENTIMENT_CLASSIFIER
    if SENTIMENT_CLASSIFIER is None:
        from transformers import pipeline as hf_pipeline

        model_name = os.getenv("SENTIMENT_MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment")
        SENTIMENT_CLASSIFIER = hf_pipeline(
            "text-classification", model=model_name, device=_get_device()
        )
    return SENTIMENT_CLASSIFIER


def get_zero_shot_classifier() -> Any:
    global ZERO_SHOT_CLASSIFIER
    if ZERO_SHOT_CLASSIFIER is None:
        from transformers import pipeline as hf_pipeline

        model_name = os.getenv("ZERO_SHOT_MODEL_NAME", "facebook/bart-large-mnli")
        ZERO_SHOT_CLASSIFIER = hf_pipeline(
            "zero-shot-classification", model=model_name, device=_get_device()
        )
    return ZERO_SHOT_CLASSIFIER


def get_ner_pipeline() -> Any:
    global NER_PIPELINE
    if NER_PIPELINE is None:
        from transformers import pipeline as hf_pipeline

        model_name = os.getenv("NER_MODEL_NAME", "dbmdz/bert-large-cased-finetuned-conll03-english")
        NER_PIPELINE = hf_pipeline(
            "ner", model=model_name, aggregation_strategy="simple", device=_get_device()
        )
    return NER_PIPELINE


def filter_post(raw_post_data: dict[str, Any], threshold: float = 0.6) -> ProcessedPost | None:
    """Binary filter: score post and discard if below threshold."""
    post = Post.model_validate(raw_post_data)
    classifier = get_text_classifier()
    result = classifier(post.text_content, truncation=True)[0]

    score = float(result.get("score", 0.0))
    if score < threshold:
        return None

    return ProcessedPost(**post.model_dump(), antisemitism_score=score)


def analyze_content(processed_post: ProcessedPost) -> ProcessedPost:
    """Run sentiment, IHRA labels, keywords, and NER on a post."""
    text = processed_post.text_content

    try:
        sentiment_classifier = get_sentiment_classifier()
        sentiment_result = sentiment_classifier(text, truncation=True)[0]
        raw_label = str(sentiment_result.get("label", "")).strip()
        processed_post.sentiment = SENTIMENT_LABEL_MAP.get(
            raw_label, SENTIMENT_LABEL_MAP.get(raw_label.lower(), "Neutral")
        )
    except Exception:
        LOGGER.exception("Sentiment analysis failed")

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
        LOGGER.exception("Zero-shot IHRA/keyword extraction failed")

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
        LOGGER.exception("NER country extraction failed")

    return processed_post


def vectorize_text(processed_post: ProcessedPost) -> ProcessedPost:
    """Produce a dense embedding for the post text."""
    model = get_sentence_model()
    vector = model.encode(processed_post.text_content)
    processed_post.text_vector = [float(value) for value in vector.tolist()]
    return processed_post
