# Blue Shield - Processing Service

This repository contains the **Processing Service** for the Blue Shield project. It is a FastAPI-based Python backend responsible for the end-to-end data pipeline, including ingestion, filtering, and analysis.

---

## Tech Stack
* Framework: FastAPI
* Server: Uvicorn
* Language: Python 3.10+
* Storage: Elasticsearch (posts + future embeddings)
* Environment Management: venv

---

## Getting Started

### Navigate to Directory
```bash
cd processing_service
```

### Setup Virtual Environment
```bash
# Create the environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Configure Environment Variables
Copy `.env.example` to `.env` and fill in the Elasticsearch credentials. The team-hosted Elasticsearch is reachable via the project VPN at `https://10.10.248.126:9200`.

```bash
cp .env.example .env   # or manually create a .env file on Windows
```

Key variables:

| Variable                   | Purpose                                                                 |
|----------------------------|-------------------------------------------------------------------------|
| `ELASTIC_HOST`             | Full URL of the ES node, e.g. `https://10.10.248.126:9200`.             |
| `ELASTIC_USERNAME`         | ES username (default `elastic`).                                        |
| `ELASTIC_PASSWORD`         | ES password. **Do not commit.**                                         |
| `ELASTIC_POSTS_INDEX`      | Target index for processed posts (default `posts`).                     |
| `ELASTIC_VERIFY_CERTS`     | `false` for the self-signed dev cert, `true` when a CA bundle is set.   |
| `ELASTIC_CA_CERTS`         | Optional path to the Elasticsearch CA bundle (`http_ca.crt`).           |
| `ELASTIC_REQUEST_TIMEOUT`  | Client request timeout in seconds (default `30`).                       |

---

## Running the Server
```bash
uvicorn main:app --reload
```

On startup, the service pings Elasticsearch and creates the `posts` index with the expected mapping if it does not yet exist.

---

## API Documentation
Once the server is running:
* Swagger UI: http://127.0.0.1:8000/docs
* ReDoc: http://127.0.0.1:8000/redoc

### Endpoints at a glance

| Method | Path                   | Description                                               |
|--------|------------------------|-----------------------------------------------------------|
| GET    | `/`                    | Service root.                                             |
| GET    | `/health`              | Basic app health.                                         |
| GET    | `/health/elastic`      | Elasticsearch cluster health (503 if unreachable).        |
| POST   | `/admin/posts-index`   | Create the posts index if it doesn't exist.               |
| POST   | `/posts`               | Validate and index a single post.                         |
| POST   | `/posts/batch`         | Validate and bulk-index a list of posts.                  |
| GET    | `/posts/schema`        | JSON Schema of the `Post` model.                          |

---

## Project Structure
* `main.py` — FastAPI app, lifespan, route definitions.
* `config.py` — Environment-driven settings.
* `elastic.py` — Elasticsearch client, index mapping, insert helpers.
* `models.py` — Pydantic `Post` schema.
* `requirements.txt` — Project dependencies.
* `.env.example` — Template for local environment configuration.
* `tests/` — Pytest + Hypothesis test suite.
