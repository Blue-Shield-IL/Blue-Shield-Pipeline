import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() == "true"


@dataclass(frozen=True)
class Settings:
    host: str = field(default_factory=lambda: os.getenv("ELASTIC_HOST", "https://localhost:9200"))
    username: str = field(default_factory=lambda: os.getenv("ELASTIC_USERNAME", "elastic"))
    password: str = field(default_factory=lambda: os.getenv("ELASTIC_PASSWORD", ""))
    posts_index: str = field(default_factory=lambda: os.getenv("ELASTIC_POSTS_INDEX", "posts"))
    verify_certs: bool = field(default_factory=lambda: _bool_env("ELASTIC_VERIFY_CERTS", "false"))
    ca_certs: str | None = field(default_factory=lambda: os.getenv("ELASTIC_CA_CERTS") or None)
    request_timeout: float = field(
        default_factory=lambda: float(os.getenv("ELASTIC_REQUEST_TIMEOUT", "30"))
    )
    embedding_dims: int = field(
        default_factory=lambda: int(os.getenv("ELASTIC_EMBEDDING_DIMS", "384"))
    )


settings = Settings()
