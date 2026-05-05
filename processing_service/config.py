"""
Configuration for the Blue Shield Processing Service.

All runtime settings are loaded from environment variables, with sensible
defaults for local development. Secrets (Elasticsearch password, etc.) must
never be hard-coded; put them in a local .env file that is git-ignored.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Load .env from the processing_service directory (if present) into os.environ.
# This is a no-op in environments that already inject the variables (e.g. prod).
load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ElasticsearchSettings:
    """Elasticsearch connection and index settings."""

    # Full URL including scheme and port, e.g. "https://10.10.248.126:9200"
    host: str
    username: str
    password: str
    # Index used for storing processed social media posts.
    posts_index: str
    # TLS verification. Set to False only when using the self-signed dev cert
    # and no CA bundle is available. Prefer providing ca_certs in production.
    verify_certs: bool
    # Optional path to a CA certificate bundle used to verify the ES cert.
    ca_certs: Optional[str]
    # Client-side timeouts (seconds).
    request_timeout: float

    @classmethod
    def from_env(cls) -> "ElasticsearchSettings":
        host = os.getenv("ELASTIC_HOST", "https://localhost:9200")
        username = os.getenv("ELASTIC_USERNAME", "elastic")
        password = os.getenv("ELASTIC_PASSWORD", "")
        posts_index = os.getenv("ELASTIC_POSTS_INDEX", "posts")
        verify_certs = _get_bool("ELASTIC_VERIFY_CERTS", False)
        ca_certs = os.getenv("ELASTIC_CA_CERTS") or None
        request_timeout = float(os.getenv("ELASTIC_REQUEST_TIMEOUT", "30"))

        return cls(
            host=host,
            username=username,
            password=password,
            posts_index=posts_index,
            verify_certs=verify_certs,
            ca_certs=ca_certs,
            request_timeout=request_timeout,
        )


# Singleton-style accessor. Import `settings` where needed.
settings = ElasticsearchSettings.from_env()
