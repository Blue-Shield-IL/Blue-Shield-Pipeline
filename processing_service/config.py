import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() == "true"


def _int_env(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def _float_env(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _optional_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    return int(raw) if raw else None


@dataclass(frozen=True)
class Settings:
    # Elasticsearch
    host: str = field(default_factory=lambda: os.getenv("ELASTIC_HOST", "https://localhost:9200"))
    username: str = field(default_factory=lambda: os.getenv("ELASTIC_USERNAME", "elastic"))
    password: str = field(default_factory=lambda: os.getenv("ELASTIC_PASSWORD", ""))
    posts_index: str = field(default_factory=lambda: os.getenv("ELASTIC_POSTS_INDEX", "posts"))
    verify_certs: bool = field(default_factory=lambda: _bool_env("ELASTIC_VERIFY_CERTS", "false"))
    ca_certs: str | None = field(default_factory=lambda: os.getenv("ELASTIC_CA_CERTS") or None)
    request_timeout: float = field(
        default_factory=lambda: _float_env("ELASTIC_REQUEST_TIMEOUT", "30")
    )
    embedding_dims: int = field(default_factory=lambda: _int_env("ELASTIC_EMBEDDING_DIMS", "384"))
    telegram_enabled: bool = field(default_factory=lambda: _bool_env("TELEGRAM_ENABLED", "false"))
    telegram_api_id: int | None = field(
        default_factory=lambda: _optional_int_env("TELEGRAM_API_ID")
    )
    telegram_api_hash: str | None = field(
        default_factory=lambda: os.getenv("TELEGRAM_API_HASH") or None
    )
    telegram_session_file: str = field(
        default_factory=lambda: os.getenv("TELEGRAM_SESSION_FILE", "telegram.session")
    )
    telegram_supplier_channel: str | None = field(
        default_factory=lambda: os.getenv("TELEGRAM_SUPPLIER_CHANNEL") or None
    )
    telegram_fetch_limit: int = field(
        default_factory=lambda: _int_env("TELEGRAM_FETCH_LIMIT", "10")
    )
    telegram_batch_size: int = field(default_factory=lambda: _int_env("TELEGRAM_BATCH_SIZE", "50"))
    telegram_flush_seconds: float = field(
        default_factory=lambda: _float_env("TELEGRAM_FLUSH_SECONDS", "5")
    )
    telegram_startup_backfill_limit: int = field(
        default_factory=lambda: _int_env("TELEGRAM_STARTUP_BACKFILL_LIMIT", "100")
    )
    telegram_request_timeout: int = field(
        default_factory=lambda: _int_env("TELEGRAM_REQUEST_TIMEOUT", "30")
    )

    ml_batch_size: int = field(default_factory=lambda: _int_env("ML_BATCH_SIZE", "50"))
    # ML Models (Gemini)
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    gemini_model_name: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
    )
    gemini_embedding_model_name: str = field(
        default_factory=lambda: os.getenv("GEMINI_EMBEDDING_MODEL_NAME", "gemini-embedding-2")
    )
    filter_threshold: float = field(
        default_factory=lambda: float(os.getenv("FILTER_THRESHOLD", "0.45"))
    )

settings = Settings()
