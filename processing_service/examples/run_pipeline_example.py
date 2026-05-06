"""
Run the full pipeline locally without going through the HTTP endpoint.

This is the same code path the `POST /jobs/run` endpoint and the scheduled
cron job use — it just drives `PipelineRunner` directly. Handy for sanity
checks before deploying changes to any of the three steps.

Usage from inside `processing_service/`:

    venv/Scripts/python.exe examples/run_pipeline_example.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Make the sibling modules importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import PipelineRunner
from pipeline.storage import ping

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    if not ping():
        raise SystemExit(
            "Elasticsearch not reachable. Check VPN and .env credentials before running."
        )

    runner = PipelineRunner(sources=["telegram", "reddit"], limit_per_source=3)
    result = runner.run()
    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()
