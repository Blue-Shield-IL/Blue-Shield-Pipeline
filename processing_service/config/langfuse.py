import logging

from langfuse import Langfuse

from config import settings

logger = logging.getLogger(__name__)

langfuse_client: Langfuse | None = None


def init_langfuse():
    """Initialize Langfuse client if enabled and keys are configured."""
    global langfuse_client

    if not settings.langfuse_enabled:
        logger.info("Langfuse disabled (LANGFUSE_ENABLED=false)")
        return

    if not settings.langfuse_secret_key or not settings.langfuse_public_key:
        logger.warning("Langfuse enabled but keys not set — skipping initialization")
        return

    langfuse_client = Langfuse(
        secret_key=settings.langfuse_secret_key,
        public_key=settings.langfuse_public_key,
        host=settings.langfuse_host,
    )
    logger.info("Langfuse initialized", extra={"payload": {"host": settings.langfuse_host}})


def get_langfuse_client() -> Langfuse | None:
    return langfuse_client
