import asyncio
import logging
import json
import uuid
from typing import Any

from .enrichment import enrich_posts
from .enrichment.processor import count_tokens
from .storage import store_posts
from config import settings

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    def __init__(self, queue: asyncio.Queue, flush_interval_sec: float = 87.0):
        self.queue = queue
        self.flush_interval_sec = flush_interval_sec
        self.buffer: list[dict[str, Any]] = []

    async def run_forever(self):
        logger.info(f"Orchestrator started. Flushing every {self.flush_interval_sec}s.")
        while True:
            start_time = asyncio.get_running_loop().time()

            while not self.queue.empty():
                self.buffer.append(self.queue.get_nowait())

            if self.buffer:
                await self._flush()

            elapsed = asyncio.get_running_loop().time() - start_time
            sleep_time = max(0.0, self.flush_interval_sec - elapsed)
            await asyncio.sleep(sleep_time)

    async def _flush(self):
        job_id = str(uuid.uuid4())
        flush_start = asyncio.get_running_loop().time()

        logger.info("flush_start", extra={"payload": {"job_id": job_id, "posts": len(self.buffer)}})

        while self.buffer:
            combined_text = "\n".join([post.get("text_content", "") for post in self.buffer])

            if len(combined_text) < settings.safe_char_limit:
                break

            loop = asyncio.get_running_loop()
            tokens = await loop.run_in_executor(None, count_tokens, combined_text)

            if tokens <= settings.token_limit:
                break

            logger.warning("drop_batch", extra={"payload": {"job_id": job_id, "tokens": tokens, "limit": settings.token_limit}})

            token_per_char = tokens / max(len(combined_text), 1)

            while self.buffer:
                current_len = sum(len(p.get("text_content", "")) + 1 for p in self.buffer)
                estimated_tokens = current_len * token_per_char
                if estimated_tokens <= settings.token_limit:
                    break
                self.buffer.pop(0)

        if not self.buffer:
            return

        batch_to_process = self.buffer.copy()
        self.buffer.clear()

        loop = asyncio.get_running_loop()
        try:
            logger.info("enriching", extra={"payload": {"job_id": job_id, "posts": len(batch_to_process)}})
            enriched, enrich_failures = await loop.run_in_executor(None, enrich_posts, batch_to_process)

            if enriched:
                stored, errors = await loop.run_in_executor(None, store_posts, enriched)
                duration = asyncio.get_running_loop().time() - flush_start
                logger.info("flush_complete", extra={"payload": {
                    "job_id": job_id,
                    "stored": stored,
                    "enrich_failures": enrich_failures,
                    "storage_errors": len(errors),
                    "duration_sec": round(duration, 2)
                }})
            else:
                duration = asyncio.get_running_loop().time() - flush_start
                logger.info("no_posts_passed", extra={"payload": {
                    "job_id": job_id,
                    "duration_sec": round(duration, 2)
                }})
        except Exception as e:
            duration = asyncio.get_running_loop().time() - flush_start

            requeued_count = 0
            for post in batch_to_process:
                retry_count = post.get("_retry_count", 0) + 1
                if retry_count <= 2:
                    post["_retry_count"] = retry_count
                    self.buffer.append(post)
                    requeued_count += 1

            logger.error("flush_error", extra={"payload": {
                "job_id": job_id,
                "duration_sec": round(duration, 2),
                "requeued_for_retry": requeued_count,
                "dropped": len(batch_to_process) - requeued_count
            }, "error": str(e)})
