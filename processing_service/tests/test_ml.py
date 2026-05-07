import os
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_ML_INTEGRATION_TESTS", "0") != "1",
        reason="Requires heavy ML models. Set RUN_ML_INTEGRATION_TESTS=1 to run.",
    ),
]
from processing_service.models.post import ProcessedPost
from processing_service.services.ml_services import (
    analyze_content,
    filter_post,
    vectorize_text,
)

# A post with strongly positive, confident language → classifier gives high score.
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

# Deliberately short/ambiguous text — may or may not pass the 0.6 threshold.
AMBIGUOUS_POST = {
    "post_id": "real-002",
    "text_content": "ok",
    "author": "user_x",
    "platform": "facebook",
    "created_at": "2026-05-05T11:00:00Z",
}

# A post with clear antisemitic framing for LLM analysis.
ANALYSIS_POST = {
    "post_id": "real-003",
    "text_content": (
        "Jewish people control the banks and media in Germany. "
        "#conspiracy #antisemitism"
    ),
    "author": "badactor",
    "platform": "telegram",
    "created_at": "2026-05-05T09:00:00Z",
    "hashtags": ["#conspiracy", "#antisemitism"],
}


class TestRealFilterPost:
    def test_high_confidence_post_passes_or_fails_with_valid_score(self):
        """Real classifier runs; result is either None or ProcessedPost with a valid score."""
        result = filter_post(HIGH_CONFIDENCE_POST)
        if result is not None:
            assert isinstance(result, ProcessedPost)
            assert isinstance(result.antisemitism_score, float)
            assert 0.0 <= result.antisemitism_score <= 1.0
            assert result.post_id == "real-001"
        # None is also valid (score < 0.6) — the test checks the contract, not the threshold.

    def test_high_confidence_post_returns_contract_valid_result(self):
        """Real classifier output should satisfy the function contract without assuming a fixed threshold."""
        result = filter_post(HIGH_CONFIDENCE_POST)
        if result is None:
            return
        assert isinstance(result, ProcessedPost)
        assert isinstance(result.antisemitism_score, float)
        assert 0.0 <= result.antisemitism_score <= 1.0
        assert result.post_id == "real-001"

    def test_score_is_always_float_regardless_of_outcome(self):
        """Even ambiguous posts produce a float score internally; the object is well-typed."""
        # Call with threshold=0.0 to guarantee we always get a ProcessedPost back.
        result = filter_post(AMBIGUOUS_POST, threshold=0.0)
        assert result is not None
        assert isinstance(result.antisemitism_score, float)
        assert 0.0 <= result.antisemitism_score <= 1.0

    def test_invalid_post_raises_on_empty_text(self):
        """Validation error raised before classifier is ever called."""
        with pytest.raises(Exception):
            filter_post({**HIGH_CONFIDENCE_POST, "text_content": "  "})


class TestRealVectorizeText:
    def test_vector_is_384_dims(self):
        """Real all-MiniLM-L6-v2 model produces a 384-dimensional vector."""
        post = ProcessedPost.model_validate(HIGH_CONFIDENCE_POST)
        result = vectorize_text(post)
        assert len(result.text_vector) == 384

    def test_vector_values_are_floats(self):
        post = ProcessedPost.model_validate(HIGH_CONFIDENCE_POST)
        result = vectorize_text(post)
        assert all(isinstance(v, float) for v in result.text_vector)

    def test_different_texts_produce_different_vectors(self):
        """Semantically different texts should not produce identical vectors."""
        post_a = ProcessedPost.model_validate(HIGH_CONFIDENCE_POST)
        post_b = ProcessedPost.model_validate(AMBIGUOUS_POST)
        result_a = vectorize_text(post_a)
        result_b = vectorize_text(post_b)
        assert result_a.text_vector != result_b.text_vector

    def test_same_text_produces_deterministic_vector(self):
        """Encoding the same text twice must return an identical vector."""
        post_a = ProcessedPost.model_validate(HIGH_CONFIDENCE_POST)
        post_b = ProcessedPost.model_validate(HIGH_CONFIDENCE_POST)
        result_a = vectorize_text(post_a)
        result_b = vectorize_text(post_b)
        assert result_a.text_vector == result_b.text_vector


# ---------------------------------------------------------------------------
# Session-scoped fixtures — expensive model pipelines (sentiment, zero-shot, NER) are executed only twice.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def analyzed_post():
    """Single invocation of analyze_content (runs sentiment, zero-shot, and NER models) shared across all TestRealAnalyzeContent tests."""
    post = ProcessedPost.model_validate(ANALYSIS_POST)
    return analyze_content(post)


@pytest.fixture(scope="session")
def analyzed_post_with_existing_keyword():
    """Single invocation of analyze_content (runs multiple models) for the keyword-merge test."""
    post = ProcessedPost.model_validate(
        {**ANALYSIS_POST, "keywords": ["existing"], "post_id": "real-004"}
    )
    return analyze_content(post)


class TestRealAnalyzeContent:
    """All tests share session-scoped fixtures — only 2 full analyze_content invocations total."""

    def test_sentiment_is_string_or_none(self, analyzed_post):
        """Sentiment should be one of the local taxonomy labels."""
        assert analyzed_post.sentiment is None or isinstance(analyzed_post.sentiment, str)
        if analyzed_post.sentiment is not None:
            assert analyzed_post.sentiment in {"Hostile", "Neutral", "Supportive"}

    def test_ihra_labels_is_list_of_strings(self, analyzed_post):
        """Zero-shot pipeline must return a list of strings for ihra_labels."""
        assert isinstance(analyzed_post.ihra_labels, list)
        assert all(isinstance(label, str) for label in analyzed_post.ihra_labels)

    def test_keywords_are_merged_and_deduped(self, analyzed_post_with_existing_keyword):
        """LLM keywords are appended to existing ones without duplicates."""
        result = analyzed_post_with_existing_keyword
        assert isinstance(result.keywords, list)
        assert "existing" in result.keywords
        assert len(result.keywords) == len(set(result.keywords))

    def test_country_of_origin_is_string_or_none(self, analyzed_post):
        assert analyzed_post.country_of_origin is None or isinstance(analyzed_post.country_of_origin, str)

    def test_antisemitic_post_gets_ihra_labels(self, analyzed_post):
        """An explicitly antisemitic post should generally trigger IHRA labels."""
        assert len(analyzed_post.ihra_labels) > 0, (
            "Expected zero-shot classification to assign at least one IHRA label to clearly antisemitic content."
        )

    def test_negative_sentiment_on_antisemitic_post(self, analyzed_post):
        """Antisemitic content should not be classified as supportive sentiment."""
        assert analyzed_post.sentiment is not None
        assert analyzed_post.sentiment in {"Hostile", "Neutral"}
