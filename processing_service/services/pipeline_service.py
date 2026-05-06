import logging
from typing import Any, Optional

from processing_service.models.post import ProcessedPost
from processing_service.services.ml_services import analyze_content, filter_post, vectorize_text

LOGGER = logging.getLogger(__name__)

def process_post(raw_post_data: dict[str, Any]) -> Optional[ProcessedPost]:
    try:
        processed_post = filter_post(raw_post_data)
        if processed_post is None:
            LOGGER.info("Post filtered out due to low score")
            return None

        processed_post = analyze_content(processed_post)
        processed_post = vectorize_text(processed_post)

        return processed_post
    except Exception as exc:
        LOGGER.exception("Pipeline processing failed")
        raise RuntimeError("Pipeline processing failed") from exc
