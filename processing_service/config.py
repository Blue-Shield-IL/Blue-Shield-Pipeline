import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() == "true"


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
        default_factory=lambda: float(os.getenv("ELASTIC_REQUEST_TIMEOUT", "30"))
    )
    embedding_dims: int = field(
        default_factory=lambda: int(os.getenv("ELASTIC_EMBEDDING_DIMS", "384"))
    )

    # ML Models
    filter_model_name: str = field(
        default_factory=lambda: os.getenv(
            "FILTER_MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english"
        )
    )
    filter_target_label: str = field(
        default_factory=lambda: os.getenv("FILTER_TARGET_LABEL", "NEGATIVE")
    )
    sentence_model_name: str = field(
        default_factory=lambda: os.getenv("SENTENCE_MODEL_NAME", "all-MiniLM-L6-v2")
    )
    sentiment_model_name: str = field(
        default_factory=lambda: os.getenv(
            "SENTIMENT_MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment"
        )
    )
    zero_shot_model_name: str = field(
        default_factory=lambda: os.getenv("ZERO_SHOT_MODEL_NAME", "facebook/bart-large-mnli")
    )
    ner_model_name: str = field(
        default_factory=lambda: os.getenv(
            "NER_MODEL_NAME", "dbmdz/bert-large-cased-finetuned-conll03-english"
        )
    )


settings = Settings()
