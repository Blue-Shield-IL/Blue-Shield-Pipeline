# Blue Shield - Processing Service

FastAPI-based Python worker that runs the Blue Shield data pipeline end-to-end:

1. **Ingest** posts from external sources (Telegram, Reddit, ...)
2. **Analyze + vectorize** each post (filtering, labeling, embedding)
3. **Store** the enriched posts in Elasticsearch for the API Server to query

The Node.js API Server is the public-facing HTTP layer. This service's HTTP surface is intentionally minimal — health probes plus a single job-trigger endpoint.

---

## Tech Stack

- Framework: FastAPI (health endpoints + job trigger + lifespan)
- Storage: Elasticsearch 9.x (posts with dense-vector embeddings)
- Language: Python 3.10+
- Environment Management: venv

---

## Getting Started

### Navigate to Directory
```bash
cd processing_service
```

### Setup Virtual Environment
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Configure Environment Variables
Copy `.env.example` to `.env` and fill in the Elasticsearch credentials. The team-hosted Elasticsearch is reachable via the project VPN at `https://10.10.248.126:9200`.

| Variable                   | Purpose                                                                 |
|----------------------------|-------------------------------------------------------------------------|
| `ELASTIC_HOST`             | Full URL of the ES node.                                                |
| `ELASTIC_USERNAME`         | ES username (default `elastic`).                                        |
| `ELASTIC_PASSWORD`         | ES password. **Do not commit.**                                         |
| `ELASTIC_POSTS_INDEX`      | Target index for processed posts (default `posts`).                     |
| `ELASTIC_VERIFY_CERTS`     | `false` for the self-signed dev cert, `true` when a CA bundle is set.   |
| `ELASTIC_CA_CERTS`         | Optional path to the Elasticsearch CA bundle (`http_ca.crt`).           |
| `ELASTIC_REQUEST_TIMEOUT`  | Client request timeout in seconds (default `30`).                       |
| `ELASTIC_EMBEDDING_DIMS`   | Dimension of the vector embedding (default `384`).                      |

---

## Running the Service
```bash
uvicorn main:app --reload
```

On startup the service pings Elasticsearch and creates the `posts` index if it does not yet exist.

### Endpoints

| Method | Path                | Description                                                                 |
|--------|---------------------|-----------------------------------------------------------------------------|
| GET    | `/`                 | Service root.                                                               |
| GET    | `/health`           | Basic app health.                                                           |
| GET    | `/health/elastic`   | Elasticsearch cluster health (503 if unreachable).                          |
| POST   | `/jobs/run`         | Run the full pipeline once (ingest → vectorize → store).                    |

Swagger UI: http://127.0.0.1:8000/docs

### `POST /jobs/run`

Request body (all fields optional — defaults run all known sources with a small limit):

```json
{
  "sources": ["telegram", "reddit"],
  "limit_per_source": 10
}
```

Response:

```json
{
  "job_id": "…uuid…",
  "started_at": "2026-05-05T20:30:00+00:00",
  "finished_at": "2026-05-05T20:30:01.182000+00:00",
  "duration_seconds": 1.18,
  "sources": ["telegram", "reddit"],
  "ingested": 20,
  "vectorized": 20,
  "stored": 20,
  "errors": []
}
```

The same `PipelineRunner` is used by the scheduled cron job, so cron and HTTP trigger the exact same code path.

---

## Project Structure

```
processing_service/
├── main.py                 FastAPI app: health + /jobs/run + lifespan
├── config.py               Environment-driven settings
├── models.py               Shared Pydantic schemas (`Post`)
├── pipeline/               The three pipeline steps + orchestrator
│   ├── __init__.py
│   ├── ingest.py             Step 1 — source fetchers
│   ├── vectorize.py          Step 2 — filter / label / embed
│   ├── storage.py            Step 3 — Elasticsearch persistence
│   └── runner.py             Orchestrator used by cron and /jobs/run
├── examples/
│   └── run_pipeline_example.py
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

The pipeline surface a caller needs is just:

```python
from pipeline import PipelineRunner

result = PipelineRunner(sources=["telegram"], limit_per_source=50).run()
```

And for direct storage use (e.g. from an ad-hoc script):

```python
from pipeline.storage import ensure_posts_index, store_posts
```

---

## Data Model

The `Post` Pydantic model in `models.py` defines what goes into Elasticsearch:

| Field            | Type              | Required | Notes                                  |
|------------------|-------------------|----------|----------------------------------------|
| `post_id`        | `str`             | yes      | Non-empty; used as the ES document id. |
| `text_content`   | `str`             | yes      | Non-empty.                             |
| `author`         | `str`             | yes      |                                        |
| `platform`       | `str`             | yes      | e.g. `telegram`, `reddit`.             |
| `created_at`     | `datetime`        | yes      | ISO 8601 on the wire.                  |
| `likes` / `shares` / `comments_count` / `views` | `int` | no | default `0`.                           |
| `hashtags`       | `list[str]`       | no       |                                        |
| `url`            | `str`             | no       |                                        |
| `language`       | `str`             | no       |                                        |
| `keywords`       | `list[str]`       | no       |                                        |
| `embedding`      | `list[float]`     | no       | Dense vector from the vectorization stage. |
