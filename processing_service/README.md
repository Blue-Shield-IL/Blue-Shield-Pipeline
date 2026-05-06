# Blue Shield - Processing Service

FastAPI-based Python worker that runs the Blue Shield data pipeline end-to-end:

1. **Ingest** posts from external sources (Telegram, Reddit, etc.)
2. **Analyze + vectorize** the posts (filtering, labeling, embedding)
3. **Store** the enriched posts in Elasticsearch for the API Server to query

The Node.js API Server is the public-facing HTTP layer. This service's HTTP surface is intentionally minimal — just health probes and OpenAPI docs.

---

## Tech Stack
* Framework: FastAPI (for health endpoints + lifespan)
* Storage: Elasticsearch 9.x (posts with dense-vector embeddings)
* Language: Python 3.10+
* Environment Management: venv

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

## Storage Module API

The storage layer lives in `elastic.py` and is the public interface for step 3 of the pipeline. Typical usage from the ingestion/vectorization worker:

```python
from elastic import ensure_posts_index, store_post, store_posts
from models import Post

ensure_posts_index()                # idempotent, run once at startup

store_post(post)                    # single-post write
success, errors = store_posts(posts)  # bulk write
```

- **`ensure_posts_index(index=None, embedding_dims=None)`** — creates the index with the expected mapping (including `dense_vector` for `embedding`) if it doesn't exist. No-op otherwise.
- **`store_post(post, *, index=None, refresh=False)`** — indexes one `Post`, keyed by `post_id` for idempotent re-runs.
- **`store_posts(posts, *, index=None, refresh=False) -> (success_count, errors)`** — bulk-indexes an iterable of posts via the `_bulk` API.
- **`ping()`**, **`get_client()`**, **`close_client()`** — lifecycle helpers.

See `examples/store_posts_example.py` for a runnable end-to-end demo (with fake ingestion + vectorization) you can use as a template.

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

---

## Running the Health Server (optional)
```bash
uvicorn main:app --reload
```

On startup, the service pings Elasticsearch and creates the `posts` index with the expected mapping if it does not yet exist.

| Method | Path                | Description                                           |
|--------|---------------------|-------------------------------------------------------|
| GET    | `/`                 | Service root.                                         |
| GET    | `/health`           | Basic app health.                                     |
| GET    | `/health/elastic`   | Elasticsearch cluster health (503 if unreachable).    |

Swagger UI: http://127.0.0.1:8000/docs

---

## Project Structure
* `main.py` — FastAPI app with health endpoints and ES bootstrap on startup.
* `config.py` — Environment-driven settings.
* `elastic.py` — Elasticsearch storage module (step 3 of the pipeline).
* `models.py` — `Post` Pydantic schema.
* `examples/store_posts_example.py` — End-to-end pipeline demo with faked ingestion/vectorization.
* `requirements.txt` — Project dependencies.
* `.env.example` — Template for local environment configuration.
* `tests/` — Pytest + Hypothesis test suite.
