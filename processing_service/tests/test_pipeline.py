from unittest.mock import MagicMock, patch

import pytest

from models import Post, ProcessedPost

SAMPLE_RAW_POST = {
    "post_id": "post-001",
    "text_content": "This is a test social media post about current events.",
    "author": "user123",
    "platform": "twitter",
    "created_at": "2026-05-05T12:00:00Z",
    "likes": 10,
    "shares": 2,
    "hashtags": ["#news"],
}


class TestPostModel:
    def test_valid_post_parses(self):
        post = Post.model_validate(SAMPLE_RAW_POST)
        assert post.post_id == "post-001"
        assert post.likes == 10

    def test_empty_post_id_rejected(self):
        with pytest.raises(Exception):
            Post.model_validate({**SAMPLE_RAW_POST, "post_id": "   "})

    def test_empty_text_content_rejected(self):
        with pytest.raises(Exception):
            Post.model_validate({**SAMPLE_RAW_POST, "text_content": ""})


class TestFilterPost:
    @patch("pipeline.ml.get_text_classifier")
    def test_high_score_returns_processed_post(self, mock_get_classifier):
        mock_classifier = MagicMock(return_value=[{"label": "POSITIVE", "score": 0.95}])
        mock_get_classifier.return_value = mock_classifier

        from pipeline.ml import filter_post

        result = filter_post(SAMPLE_RAW_POST)
        assert result is not None
        assert isinstance(result, ProcessedPost)
        assert result.antisemitism_score == pytest.approx(0.95)

    @patch("pipeline.ml.get_text_classifier")
    def test_low_score_returns_none(self, mock_get_classifier):
        mock_classifier = MagicMock(return_value=[{"label": "NEGATIVE", "score": 0.3}])
        mock_get_classifier.return_value = mock_classifier

        from pipeline.ml import filter_post

        result = filter_post(SAMPLE_RAW_POST)
        assert result is None


class TestVectorizeText:
    @patch("pipeline.ml.get_sentence_model")
    def test_vector_populated(self, mock_get_model):
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(384)
        mock_get_model.return_value = mock_model

        from pipeline.ml import vectorize_text

        post = ProcessedPost.model_validate(SAMPLE_RAW_POST)
        result = vectorize_text(post)
        assert len(result.text_vector) == 384
        assert all(isinstance(v, float) for v in result.text_vector)


class TestPipelineOrchestration:
    @patch("pipeline.vectorize.vectorize_text")
    @patch("pipeline.vectorize.analyze_content")
    @patch("pipeline.vectorize.filter_post")
    def test_full_pipeline_returns_processed_post(
        self,
        mock_filter,
        mock_analyze,
        mock_vectorize,
    ):
        from pipeline.vectorize import process_post

        fake_post = ProcessedPost.model_validate(SAMPLE_RAW_POST)
        fake_post.antisemitism_score = 0.85
        fake_post.sentiment = "Hostile"
        fake_post.text_vector = [0.0] * 384

        mock_filter.return_value = fake_post
        mock_analyze.return_value = fake_post
        mock_vectorize.return_value = fake_post

        result = process_post(SAMPLE_RAW_POST)
        assert result is not None
        assert result.post_id == "post-001"
        assert result.antisemitism_score == pytest.approx(0.85)

    @patch("pipeline.vectorize.filter_post")
    def test_filtered_post_returns_none(self, mock_filter):
        from pipeline.vectorize import process_post

        mock_filter.return_value = None
        result = process_post(SAMPLE_RAW_POST)
        assert result is None
