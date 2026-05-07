"""Blue Shield data pipeline package.

Three steps, one runner:
    1. `ingest`    — fetch raw posts from external sources.
    2. `vectorize` — filter / analyze / embed (ML models).
    3. `storage`   — persist to Elasticsearch.
"""

from .runner import JobResult, PipelineRunner

__all__ = ["JobResult", "PipelineRunner"]
