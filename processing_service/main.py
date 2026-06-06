import asyncio
import logging
import sys

from config import settings
from pipeline.ingestion.ingest import ingest_forever, ingest
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.storage import ensure_posts_index, ping, close_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


async def cron_fetchers_stub(queue: asyncio.Queue, fetchers: list[str], interval_sec: int):
    if not fetchers:
        return
    logger.info("Cron fetchers started for sources: %s", fetchers)
    loop = asyncio.get_running_loop()
    while True:
        try:
            posts = await loop.run_in_executor(None, ingest, fetchers, 50)
            for post in posts:
                await queue.put(post)
        except Exception:
            logger.exception("Cron fetcher failed")

        await asyncio.sleep(interval_sec)


def start_listeners(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue, listeners: list[str]):
    if not listeners:
        return

    def on_post(post):
        try:
            loop.call_soon_threadsafe(queue.put_nowait, post)
        except Exception:
            logger.exception("Failed to queue post")

    logger.info("Starting listeners for sources: %s", listeners)
    ingest_forever(sources=listeners, on_post=on_post)


async def main():
    logger.info("Initializing Blue Shield Processing Daemon...")

    try:
        if ping():
            ensure_posts_index()
            logger.info(f"Elasticsearch reachable; posts index {settings.posts_index!r} ensured.")
        else:
            logger.warning(f"Elasticsearch not reachable at {settings.host} on startup.")
    except Exception:
        logger.exception("Elasticsearch bootstrap failed")

    queue = asyncio.Queue()

    orchestrator = PipelineOrchestrator(queue=queue, flush_interval_sec=settings.flush_interval_sec)
    orchestrator_task = asyncio.create_task(orchestrator.run_forever())

    cron_task = asyncio.create_task(
        cron_fetchers_stub(queue, settings.ingestion_fetchers, settings.cron_fetch_interval_seconds)
    )

    loop = asyncio.get_running_loop()
    listener_future = loop.run_in_executor(None, start_listeners, loop, queue, settings.ingestion_listeners)

    logger.info("Daemon is now running.")

    try:
        await asyncio.gather(orchestrator_task, cron_task, listener_future)
    except asyncio.CancelledError:
        logger.info("Daemon shutdown requested.")
    finally:
        close_client()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Exiting on Ctrl+C")
