import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_ML_INTEGRATION_TESTS", "0") != "1",
        reason="Requires heavy ML models. Set RUN_ML_INTEGRATION_TESTS=1 to run.",
    ),
]

from models import ProcessedPost
from pipeline.ml import analyze_content, filter_post, vectorize_text

HIGH_CONFIDENCE_POST = {
    "post_id": "real-001",
    "text_content": (
        "This is an absolutely wonderful and joyful event that everyone loves. "
        "The community came together in a fantastic celebration of unity and hope."
    ),
    "author": "reporter99",
    "platform": "twitter",
    "created_at": "2026-05-05T10:00:00Z",
}

AMBIGUOUS_POST = {
    "post_id": "real-002",
    "text_content": "ok",
    "author": "user_x",
    "platform": "facebook",
    "created_at": "2026-05-05T11:00:00Z",
}

ANALYSIS_POST = {
    "post_id": "real-003",
    "text_content": (
        "Jewish people control the banks and media in Germany. #conspiracy #antisemitism"
    ),
    "author": "badactor",
    "platform": "telegram",
    "created_at": "2026-05-05T09:00:00Z",
    "hashtags": ["#conspiracy", "#antisemitism"],
}


class TestRealFilterPost:
    def test_high_confidence_post_returns_valid_result(self):
        result = filter_post(HIGH_CONFIDENCE_POST)
        if result is not None:
            assert isinstance(result, ProcessedPost)
            assert isinstance(result.antisemitism_score, float)
            assert 0.0 <= result.antisemitism_score <= 1.0

    def test_score_is_always_float(self):
        result = filter_post(AMBIGUOUS_POST, threshold=0.0)
        assert result is not None
        assert isinstance(result.antisemitism_score, float)

    def test_invalid_post_raises_on_empty_text(self):
        with pytest.raises(Exception):
            filter_post({**HIGH_CONFIDENCE_POST, "text_content": "  "})


class TestRealVectorizeText:
    def test_vector_is_384_dims(self):
        post = ProcessedPost.model_validate(HIGH_CONFIDENCE_POST)
        result = vectorize_text(post)
        assert len(result.text_vector) == 384

    def test_vector_values_are_floats(self):
        post = ProcessedPost.model_validate(HIGH_CONFIDENCE_POST)
        result = vectorize_text(post)
        assert all(isinstance(v, float) for v in result.text_vector)

    def test_different_texts_produce_different_vectors(self):
        post_a = ProcessedPost.model_validate(HIGH_CONFIDENCE_POST)
        post_b = ProcessedPost.model_validate(AMBIGUOUS_POST)
        result_a = vectorize_text(post_a)
        result_b = vectorize_text(post_b)
        assert result_a.text_vector != result_b.text_vector

    def test_same_text_produces_deterministic_vector(self):
        post_a = ProcessedPost.model_validate(HIGH_CONFIDENCE_POST)
        post_b = ProcessedPost.model_validate(HIGH_CONFIDENCE_POST)
        result_a = vectorize_text(post_a)
        result_b = vectorize_text(post_b)
        assert result_a.text_vector == result_b.text_vector


@pytest.fixture(scope="session")
def analyzed_post():
    post = ProcessedPost.model_validate(ANALYSIS_POST)
    return analyze_content(post)


class TestRealAnalyzeContent:
    def test_sentiment_is_valid(self, analyzed_post):
        assert analyzed_post.sentiment is None or analyzed_post.sentiment in {
            "Hostile",
            "Neutral",
            "Supportive",
        }

    def test_ihra_labels_is_list(self, analyzed_post):
        assert isinstance(analyzed_post.ihra_labels, list)
        assert all(isinstance(label, str) for label in analyzed_post.ihra_labels)

    def test_country_of_origin(self, analyzed_post):
        assert analyzed_post.country_of_origin is None or isinstance(
            analyzed_post.country_of_origin, str
        )

    def test_antisemitic_post_gets_ihra_labels(self, analyzed_post):
        assert len(analyzed_post.ihra_labels) > 0
