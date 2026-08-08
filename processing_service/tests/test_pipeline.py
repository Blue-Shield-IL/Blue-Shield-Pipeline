from unittest.mock import MagicMock, patch
import pytest
from pydantic import ValidationError

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
        with pytest.raises(ValidationError, match="post_id"):
            Post.model_validate({**SAMPLE_RAW_POST, "post_id": "   "})

    def test_empty_text_content_rejected(self):
        with pytest.raises(ValidationError, match="text_content"):
            Post.model_validate({**SAMPLE_RAW_POST, "text_content": ""})

class TestProcessor:
    @patch("pipeline.enrichment.processor.gemini_adapter")
    def test_analyze_content_batch(self, mock_gemini):
        from pipeline.enrichment.processor import analyze_content_batch
        from pipeline.enrichment.adapters.gemini_adapter import PostAnalysis
        
        mock_gemini.analyze_posts.return_value = [
            PostAnalysis(
                antisemitism_score=0.95,
                ihra_labels=[],
                keywords=[],
                sentiment="Negative",
                country_of_origin=None
            )
        ]
        
        analyzed, failed = analyze_content_batch([SAMPLE_RAW_POST])
        assert failed == 0
        assert len(analyzed) == 1
        assert analyzed[0].antisemitism_score == 0.95
        assert analyzed[0].sentiment == "Negative"

    def test_filter_posts_batch(self):
        from pipeline.enrichment.processor import filter_posts_batch
        
        post = ProcessedPost.model_validate({**SAMPLE_RAW_POST, "antisemitism_score": 0.95})
        passed, filtered = filter_posts_batch([post], threshold=0.5)
        assert len(passed) == 1
        assert filtered == 0
        
        post2 = ProcessedPost.model_validate({**SAMPLE_RAW_POST, "antisemitism_score": 0.2})
        passed, filtered = filter_posts_batch([post2], threshold=0.5)
        assert len(passed) == 0
        assert filtered == 1

    @patch("pipeline.enrichment.processor.gemini_adapter")
    def test_vectorize_texts_batch(self, mock_gemini):
        from pipeline.enrichment.processor import vectorize_texts_batch
        
        mock_gemini.vectorize_texts.return_value = [[0.1] * 384]
        
        post = ProcessedPost.model_validate({**SAMPLE_RAW_POST, "antisemitism_score": 0.95})
        result = vectorize_texts_batch([post])
        assert len(result) == 1
        assert len(result[0].text_vector) == 384

@pytest.mark.integration
class TestEndToEndWithElastic:
    """Integration tests that run the real pipeline and verify posts land in ES."""

    @pytest.fixture(autouse=True)
    def _require_es(self):
        from pipeline.storage import ping
        if not ping():
            pytest.skip("Elasticsearch not reachable")

    def test_full_pipeline_stores_posts_in_elastic(self):
        from config import settings
        from pipeline.storage import ensure_posts_index, get_client, store_posts
        from pipeline.enrichment import enrich_posts

        ensure_posts_index()

        test_posts = [
            {
                "post_id": "e2e-test-hostile-1",
                "text_content": "Jews control the world banks and all media. Wake up!",
                "author": "testuser",
                "platform": "telegram",
                "created_at": "2026-05-09T12:00:00Z",
            },
            {
                "post_id": "e2e-test-neutral-1",
                "text_content": "Beautiful sunny day at the beach with my family.",
                "author": "happyuser",
                "platform": "telegram",
                "created_at": "2026-05-09T12:02:00Z",
            },
        ]

        # Use new enrich_posts API
        results, failed = enrich_posts(test_posts)
        assert failed == 0
        assert len(results) > 0  # at least the hostile one should pass

        stored, errors = store_posts(results, refresh=True)
        assert stored == len(results)
        assert errors == []

        # Verify in ES
        c = get_client()
        for post in results:
            doc = c.get(index=settings.posts_index, id=post.post_id, source_includes=["*"])
            src = doc["_source"]
            assert src["post_id"] == post.post_id
            assert "antisemitism_score" in src
            assert len(src.get("text_vector", [])) == 384
