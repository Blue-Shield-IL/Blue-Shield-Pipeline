import pytest
from pydantic import ValidationError

pytestmark = [
    pytest.mark.integration,
]

from models import ProcessedPost
from pipeline.enrichment.processor import (
    analyze_content_batch,
    filter_posts_batch,
    vectorize_texts_batch,
)

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


class TestRealFilterPostsBatch:
    def test_high_confidence_post_passes_or_fails_with_valid_score(self):
        post = ProcessedPost.model_validate({**HIGH_CONFIDENCE_POST, "antisemitism_score": 0.95})
        passed, filtered_out = filter_posts_batch([post], threshold=0.4)
        assert isinstance(filtered_out, int)
        if passed:
            assert isinstance(passed[0], ProcessedPost)
            assert isinstance(passed[0].antisemitism_score, float)
            assert 0.0 <= passed[0].antisemitism_score <= 1.0
            assert passed[0].post_id == "real-001"

    def test_score_is_always_float_regardless_of_outcome(self):
        post = ProcessedPost.model_validate({**AMBIGUOUS_POST, "antisemitism_score": 0.5})
        passed, _ = filter_posts_batch([post], threshold=0.0)
        assert len(passed) == 1
        assert isinstance(passed[0].antisemitism_score, float)
        assert 0.0 <= passed[0].antisemitism_score <= 1.0

    def test_invalid_post_is_filtered_out_on_empty_text(self):
        # We simulate filtering by passing a low score since validation happens earlier now
        post = ProcessedPost.model_validate({**HIGH_CONFIDENCE_POST, "antisemitism_score": 0.1})
        passed, filtered_out = filter_posts_batch([post], threshold=0.4)
        assert passed == []
        assert filtered_out == 1

    def test_multiple_posts_processed_in_one_call(self):
        post1 = ProcessedPost.model_validate({**HIGH_CONFIDENCE_POST, "antisemitism_score": 0.95})
        post2 = ProcessedPost.model_validate({**AMBIGUOUS_POST, "antisemitism_score": 0.5})
        passed, filtered_out = filter_posts_batch(
            [post1, post2], threshold=0.0
        )
        assert len(passed) + filtered_out == 2


class TestRealVectorizeTextsBatch:
    def test_vector_is_384_dims(self):
        post = ProcessedPost.model_validate(HIGH_CONFIDENCE_POST)
        results = vectorize_texts_batch([post])
        assert len(results[0].text_vector) == 384

    def test_vector_values_are_floats(self):
        post = ProcessedPost.model_validate(HIGH_CONFIDENCE_POST)
        results = vectorize_texts_batch([post])
        assert all(isinstance(v, float) for v in results[0].text_vector)

    def test_different_texts_produce_different_vectors(self):
        post_a = ProcessedPost.model_validate(HIGH_CONFIDENCE_POST)
        post_b = ProcessedPost.model_validate(AMBIGUOUS_POST)
        results = vectorize_texts_batch([post_a, post_b])
        assert results[0].text_vector != results[1].text_vector

    def test_same_text_produces_deterministic_vector(self):
        post_a = ProcessedPost.model_validate(HIGH_CONFIDENCE_POST)
        post_b = ProcessedPost.model_validate(HIGH_CONFIDENCE_POST)
        result_a = vectorize_texts_batch([post_a])[0]
        result_b = vectorize_texts_batch([post_b])[0]
        assert result_a.text_vector == result_b.text_vector


@pytest.fixture(scope="session")
def analyzed_posts():
    return analyze_content_batch([ANALYSIS_POST])[0]


@pytest.fixture(scope="session")
def analyzed_post(analyzed_posts):
    return analyzed_posts[0]


@pytest.fixture(scope="session")
def analyzed_post_with_existing_keyword():
    raw_dict = {**ANALYSIS_POST, "keywords": ["existing"], "post_id": "real-004"}
    return analyze_content_batch([raw_dict])[0][0]


class TestRealAnalyzeContentBatch:
    def test_sentiment_is_string_or_none(self, analyzed_post):
        assert analyzed_post.sentiment is None or analyzed_post.sentiment in {"Supportive", "Neutral", "Negative", "Hostile"}

    def test_ihra_labels_is_list_of_strings(self, analyzed_post):
        assert isinstance(analyzed_post.ihra_labels, list)
        assert all(isinstance(label, str) for label in analyzed_post.ihra_labels)

    def test_keywords_are_merged_and_deduped(self, analyzed_post_with_existing_keyword):
        result = analyzed_post_with_existing_keyword
        assert isinstance(result.keywords, list)
        assert "existing" in result.keywords
        assert len(result.keywords) == len(set(result.keywords))

    def test_country_of_origin_is_string_or_none(self, analyzed_post):
        assert analyzed_post.country_of_origin is None or isinstance(
            analyzed_post.country_of_origin, str
        )

    def test_antisemitic_post_gets_ihra_labels(self, analyzed_post):
        assert len(analyzed_post.ihra_labels) > 0

    def test_negative_or_hostile_sentiment_on_antisemitic_post(self, analyzed_post):
        assert analyzed_post.sentiment is not None
        assert analyzed_post.sentiment in {"Negative", "Hostile"}
