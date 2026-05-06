"""Blue Shield data pipeline package.

Three steps, one runner:
    1. `ingest`   — fetch raw posts from external sources.
    2. `vectorize` — filter / analyze / embed each post.
    3. `storage`   — persist the enriched posts to Elasticsearch.

The `runner.PipelineRunner` orchestrates all three and is the single entry
point used by both the scheduled cron job and the `POST /jobs/run` endpoint.
"""

from .runner import JobResult, PipelineRunner

__all__ = ["JobResult", "PipelineRunner"]
