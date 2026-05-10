# Blue Shield - Processing Service

Python worker that runs the Blue Shield data pipeline:

1. **Ingest** — fetch posts from Telegram (stub, real integration coming).
2. **Process** — filter by antisemitism score, analyze (sentiment/IHRA labels/keywords/country), vectorize with sentence-transformers. Posts below the score threshold are discarded.
3. **Store** — persist all posts that passed the filter to Elasticsearch with dense-vector embeddings.

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
The team ES host is shared internally (VPN required).

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
│   ├── test_pipeline.py   Unit tests (mocked) + E2E test (real ML + ES)
│   └── test_ml.py         ML integration tests (real models, gated)
├── .env.example
├── pyproject.toml       Ruff + pytest + ty config
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
| `TELEGRAM_ENABLED`       | `false`                        | Enables Telegram integration.    |
| `TELEGRAM_API_ID`        |                                | Telegram app API ID.             |
| `TELEGRAM_API_HASH`      |                                | Telegram app API hash.           |
| `TELEGRAM_SESSION_FILE`  | `telegram.session`             | Local Telethon session file.     |
| `TELEGRAM_SUPPLIER_CHANNEL` |                           | Telegram channel handle or link. |
| `TELEGRAM_FETCH_LIMIT`   | `10`                           | Default batch fetch size.        |
| `TELEGRAM_BATCH_SIZE`    | `50`                           | Reserved for listener batching.  |
| `TELEGRAM_FLUSH_SECONDS` | `5`                            | Reserved listener flush window.  |
| `TELEGRAM_STARTUP_BACKFILL_LIMIT` | `100`                 | Reserved startup backfill size.  |
| `TELEGRAM_REQUEST_TIMEOUT` | `30`                         | Telegram request timeout.        |
| `FILTER_MODEL_NAME`      | `distilbert-base-uncased-finetuned-sst-2-english` | Binary classifier model. |
| `FILTER_TARGET_LABEL`    | `NEGATIVE`                     | Label treated as flagged class (matches SST-2). |
| `FILTER_THRESHOLD`       | `0.75`                         | Minimum score to pass the filter (0.0–1.0). |
| `SENTENCE_MODEL_NAME`    | `all-MiniLM-L6-v2`             | Embedding model (384-dim).       |
| `SENTIMENT_MODEL_NAME`   | `cardiffnlp/twitter-roberta-base-sentiment` | Sentiment classifier. |
| `ZERO_SHOT_MODEL_NAME`   | `facebook/bart-large-mnli`     | Zero-shot for IHRA + keywords.   |
| `NER_MODEL_NAME`         | `dbmdz/bert-large-cased-finetuned-conll03-english` | NER for country extraction. |

## Telegram Setup

The Telegram integration uses a Telethon user session. You must register a
Telegram app once, configure the Telegram environment variables, and create a
local authorized session file before fetching or listening to messages.

### 1. Register a Telegram app

1. Go to `https://my.telegram.org`.
2. Sign in with the phone number of the Telegram account that will read the
   supplier channel.
3. Open `API development tools`.
4. Create an app if you do not already have one.
5. Copy the generated `api_id` and `api_hash`.

### 2. Configure `.env`

Set the Telegram fields in `processing_service/.env`:

```env
TELEGRAM_ENABLED=true
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_SESSION_FILE=telegram.session
TELEGRAM_SUPPLIER_CHANNEL=@your_channel_or_t_me_link
TELEGRAM_FETCH_LIMIT=10
TELEGRAM_REQUEST_TIMEOUT=30
```

Notes:

- `TELEGRAM_API_ID` should be numeric and should not be quoted.
- `TELEGRAM_SESSION_FILE` is the local Telethon session file.
- `TELEGRAM_SUPPLIER_CHANNEL` should be a resolvable public handle such as
  `@channel_name`, a `https://t.me/...` link, or a private channel the
  authenticated account already has access to.

### 3. Create the Telegram session

Run this once from `processing_service`:

```bash
python -c "from telethon.sync import TelegramClient; from config import settings; client = TelegramClient(settings.telegram_session_file, settings.telegram_api_id, settings.telegram_api_hash); client.start(); print('authorized=', client.is_user_authorized()); print(client.get_me()); client.disconnect()"
```

What to expect:

- Telethon may ask for your phone number.
- Telegram will send a login code.
- If two-factor authentication is enabled, Telethon may also ask for your
  password.
- On success, the session file configured by `TELEGRAM_SESSION_FILE` is created
  or updated locally.

### 4. Test the batch fetch

To fetch a bounded batch of recent messages:

```bash
python -c "from pipeline.ingest import fetch_telegram; import json; print(json.dumps(fetch_telegram(5), indent=2, ensure_ascii=False))"
```

This connects, fetches up to `5` recent messages from the configured supplier
channel, prints them, and disconnects.

### 5. Run the always-on listener

To keep one Telegram connection open and print each new normalized message:

```bash
python listen_telegram.py
```

Stop it with `Ctrl+C`.
