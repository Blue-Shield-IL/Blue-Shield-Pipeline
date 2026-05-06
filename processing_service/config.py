import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    host = os.getenv("ELASTIC_HOST", "https://localhost:9200")
    username = os.getenv("ELASTIC_USERNAME", "elastic")
    password = os.getenv("ELASTIC_PASSWORD", "")
    posts_index = os.getenv("ELASTIC_POSTS_INDEX", "posts")
    verify_certs = os.getenv("ELASTIC_VERIFY_CERTS", "false").lower() == "true"
    ca_certs = os.getenv("ELASTIC_CA_CERTS") or None
    request_timeout = float(os.getenv("ELASTIC_REQUEST_TIMEOUT", "30"))
    embedding_dims = int(os.getenv("ELASTIC_EMBEDDING_DIMS", "384"))


settings = Settings()
