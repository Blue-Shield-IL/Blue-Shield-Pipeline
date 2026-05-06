# Blue Shield - Processing Service

Python worker that runs the Blue Shield data pipeline:

1. **Ingest** — fetch posts from Telegram.
2. **Vectorize** — filter, label, and embed each post.
3. **Store** — persist enriched posts to Elasticsearch.

Triggered via `POST /jobs/run` or a scheduled cron job — same code path.

---

## Tech Stack

- Python 3.10+
- FastAPI (health + job trigger)
- Elasticsearch 9.x (posts + dense-vector embeddings)
- Ruff (formatter + linter)

---

## Getting Started

```bash
cd processing_service
python -m venv venv
.\venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Copy `.env.example` → `.env` and fill in the Elasticsearch credentials.


## Running

```bash
uvicorn main:app --reload
```

On startup the service pings ES and creates the `posts` index if missing.

Swagger UI: http://127.0.0.1:8000/docs

---

## Endpoints

| Method | Path              | Description                                      |
|--------|-------------------|--------------------------------------------------|
| GET    | `/health`         | App health.                                      |
| GET    | `/health/elastic` | ES cluster health (503 if unreachable).          |
| POST   | `/jobs/run`       | Run the full pipeline (ingest → vectorize → store). |

### `POST /jobs/run`

```json
// Request (all optional — defaults to all sources, limit 10)
{ "sources": ["telegram"], "limit_per_source": 10 }

// Response
{
  "job_id": "uuid",
  "duration_seconds": 1.18,
  "ingested": 20,
  "vectorized": 20,
  "stored": 20,
  "errors": []
}
```

---

## Project Structure

```
processing_service/
├── main.py              FastAPI app + /jobs/run endpoint
├── config.py            Env-driven settings
├── models.py            Post schema (stub — full schema is a separate mission)
├── pipeline/
│   ├── __init__.py
│   ├── ingest.py          Step 1 — source fetchers (stubs until integrations land)
│   ├── vectorize.py       Step 2 — embed posts
│   ├── storage.py         Step 3 — Elasticsearch persistence
│   └── runner.py          Orchestrator
├── .env.example
├── pyproject.toml       Ruff + ty config
└── requirements.txt
```

---

## Environment Variables

| Variable                 | Default                        | Purpose                          |
|--------------------------|--------------------------------|----------------------------------|
| `ELASTIC_HOST`           | `https://localhost:9200`       | ES node URL.                     |
| `ELASTIC_USERNAME`       | `elastic`                      | ES username.                     |
| `ELASTIC_PASSWORD`       | —                              | ES password. **Do not commit.**  |
| `ELASTIC_POSTS_INDEX`    | `posts`                        | Target index.                    |
| `ELASTIC_VERIFY_CERTS`   | `false`                        | TLS verification.                |
| `ELASTIC_CA_CERTS`       | —                              | Path to CA bundle (optional).    |
| `ELASTIC_REQUEST_TIMEOUT`| `30`                           | Timeout in seconds.              |
| `ELASTIC_EMBEDDING_DIMS` | `384`                          | Vector dimension.                |
