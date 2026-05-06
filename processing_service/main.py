import logging
import time
import requests
from typing import List, Dict, Any

from processing_service.services.pipeline_service import process_post

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

# TODO: Replace with actual source API URL.
SOURCE_API_URL = "http://placeholder-source-api.com/posts"

def fetch_raw_posts() -> List[Dict[str, Any]]:
    """Fetches a batch of raw posts from a Source API."""
    try:
        LOGGER.info(f"Fetching posts from {SOURCE_API_URL}...")
        response = requests.get(SOURCE_API_URL, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        LOGGER.error(f"Failed to fetch posts from Source API: {e}")
        # Return empty list on failure so loop continues
        return []
def run_service_loop():
    LOGGER.info("Starting Blue Shield Processing Service loop...")

    while True:
        raw_posts = fetch_raw_posts()
        
        if not raw_posts:
            LOGGER.info("No posts fetched in this batch.")
        else:
            LOGGER.info(f"Fetched {len(raw_posts)} posts to process.")
            for raw_post in raw_posts:
                try:
                    processed_post = process_post(raw_post)
                    if processed_post is not None:
                        LOGGER.info(f"Successfully processed post {raw_post.get('post_id', 'unknown')}.")
                        # TODO: Implement Elasticsearch Save.
                except Exception as e:
                    LOGGER.error(f"Error processing post {raw_post.get('post_id', 'unknown')}: {e}")

        LOGGER.info("Batch complete. Sleeping for 60 seconds...")
        time.sleep(60)

if __name__ == "__main__":
    run_service_loop()