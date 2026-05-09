# Blue Shield - Processing Service

Python worker that runs the Blue Shield data pipeline:

1. **Ingest** — fetch posts from Telegram (stub, real integration coming).
2. **Process** — filter by antisemitism score, analyze sentiment/IHRA labels/keywords/country, vectorize with sentence-transformers.
3. **Store** — persist enriched posts to Elasticsearch with dense-vector embeddings.

Triggered via `POST /jobs/run` or a scheduled cron job — same code path.

---

## Tech Stack

- Python 3.10+
- FastAPI (health + job trigger)
- Elasticsearch 9.x (posts + dense-vector embeddings + kNN search)
- HuggingFace Transformers (text classification, sentiment, zero-shot, NER)
- Sentence-Transformers (all-MiniLM-L6-v2 for embeddings)
- Ruff (formatter + linter)

---

## Getting Started

```bash
cd processing_service
python -m venv venv
.\venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Copy `.env.example` → `.env` and fill in credentials.
The team ES is at `https://10.10.248.126:9200` (VPN required).

---

## Running

```bash
uvicorn main:app --reload
```

On startup the service pings ES and creates the `posts` index if missing.

Swagger UI: http://127.0.0.1:8000/docs

---

## Endpoints

| Method | Path              | Description                                          |
|--------|-------------------|------------------------------------------------------|
| GET    | `/health`         | App health.                                          |
| GET    | `/health/elastic` | ES cluster health (503 if unreachable).              |
| POST   | `/jobs/run`       | Run the full pipeline (ingest → process → store).    |

### `POST /jobs/run`

```json
// Request (all optional)
{ "sources": ["telegram"], "limit_per_source": 10 }

// Response
{
  "job_id": "uuid",
  "duration_seconds": 13.08,
  "ingested": 10,
  "vectorized": 8,
  "vectorize_failures": 0,
  "stored": 8,
  "errors": []
}
```

---

## Project Structure

```
processing_service/
├── main.py              FastAPI app + /jobs/run endpoint
├── config.py            Env-driven ES settings
├── models/
│   ├── __init__.py
│   └── post.py            Post + ProcessedPost schemas
├── services/
│   ├── __init__.py
│   └── ml_services.py    ML models (filter, sentiment, IHRA, NER, vectorize)
├── pipeline/
│   ├── __init__.py
│   ├── ingest.py          Step 1 — source fetchers (stub)
│   ├── vectorize.py       Step 2 — orchestrates services/ml_services
│   ├── storage.py         Step 3 — Elasticsearch persistence
│   └── runner.py          PipelineRunner (used by cron + /jobs/run)
├── tests/
│   ├── conftest.py
│   ├── test_pipeline.py   Unit tests (mocked, 8 tests)
│   └── test_ml.py         Integration tests (real models, gated)
├── .env.example
├── pyproject.toml       Ruff + pytest + ty config
└── requirements.txt
```

---

## Environment Variables

| Variable                 | Default                                          | Purpose                                    |
|--------------------------|--------------------------------------------------|--------------------------------------------|
| `ELASTIC_HOST`           | `https://localhost:9200`                         | ES node URL.                               |
| `ELASTIC_USERNAME`       | `elastic`                                        | ES username.                               |
| `ELASTIC_PASSWORD`       | —                                                | ES password. **Do not commit.**            |
| `ELASTIC_POSTS_INDEX`    | `posts`                                          | Target index.                              |
| `ELASTIC_VERIFY_CERTS`   | `false`                                          | TLS verification.                          |
| `ELASTIC_CA_CERTS`       | —                                                | Path to CA bundle (optional).              |
| `ELASTIC_REQUEST_TIMEOUT`| `30`                                             | Timeout in seconds.                        |
| `ELASTIC_EMBEDDING_DIMS` | `384`                                            | Fallback vector dimension.                 |
| `FILTER_MODEL_NAME`      | `distilbert-base-uncased-finetuned-sst-2-english`| Binary classifier model.                   |
| `FILTER_TARGET_LABEL`    | `POSITIVE`                                       | Label treated as "antisemitic" class.      |
| `SENTENCE_MODEL_NAME`    | `all-MiniLM-L6-v2`                              | Embedding model (384-dim).                 |
| `SENTIMENT_MODEL_NAME`   | `cardiffnlp/twitter-roberta-base-sentiment`      | Sentiment classifier.                      |
| `ZERO_SHOT_MODEL_NAME`   | `facebook/bart-large-mnli`                       | Zero-shot for IHRA + keywords.             |
| `NER_MODEL_NAME`         | `dbmdz/bert-large-cased-finetuned-conll03-english`| NER for country extraction.               |

---

## Testing

```bash
# Unit tests (mocked, no ML models needed)
pytest tests/test_pipeline.py -v

# Integration tests (downloads ~5GB of models on first run)
RUN_ML_INTEGRATION_TESTS=1 pytest tests/test_ml.py -v
```
