"""
Pipeline orchestrator.

Runs the three steps end-to-end and reports per-stage stats. This is the
single entry point used by:
  - the scheduled cron job
  - the `POST /jobs/run` HTTP endpoint

Designed so the caller does not need to know which fetchers or models are
wired behind each step — they just request a run.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .ingest import ingest
from .storage import ensure_posts_index, store_posts
from .vectorize import vectorize

logger = logging.getLogger(__name__)


@dataclass
class JobResult:
    """Result of one pipeline run, safe to serialize to JSON."""

    job_id: str
    started_at: str                 # ISO 8601 UTC
    finished_at: str                # ISO 8601 UTC
    duration_seconds: float
    sources: list[str]
    ingested: int = 0
    vectorized: int = 0
    stored: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class PipelineRunner:
    """Glue around the three pipeline steps.

    Instantiate once (or per-request) and call `run()` with the sources to
    hit and how many posts to request from each.
    """

    def __init__(self, sources: list[str], limit_per_source: int = 10) -> None:
        if not sources:
            raise ValueError("sources must be a non-empty list")
        if limit_per_source <= 0:
            raise ValueError("limit_per_source must be positive")
        self.sources = sources
        self.limit_per_source = limit_per_source

    def run(self) -> JobResult:
        job_id = str(uuid.uuid4())
        start_wall = datetime.now(timezone.utc)
        start_perf = time.perf_counter()

        logger.info(
            "Starting pipeline job %s sources=%s limit=%d",
            job_id,
            self.sources,
            self.limit_per_source,
        )

        # Ensure the target index exists before we try to store anything. Safe
        # to call every run — it's a no-op once the index is in place.
        ensure_posts_index()

        # Step 1
        raw = ingest(self.sources, self.limit_per_source)

        # Step 2
        enriched = vectorize(raw)

        # Step 3
        stored = 0
        errors: list[dict[str, Any]] = []
        if enriched:
            stored, errors = store_posts(enriched, refresh=False)

        duration = time.perf_counter() - start_perf
        result = JobResult(
            job_id=job_id,
            started_at=start_wall.isoformat(),
            finished_at=datetime.now(timezone.utc).isoformat(),
            duration_seconds=round(duration, 3),
            sources=self.sources,
            ingested=len(raw),
            vectorized=len(enriched),
            stored=stored,
            errors=errors,
        )
        logger.info(
            "Pipeline job %s done in %.2fs: ingested=%d vectorized=%d stored=%d errors=%d",
            job_id,
            duration,
            result.ingested,
            result.vectorized,
            result.stored,
            len(result.errors),
        )
        return result
