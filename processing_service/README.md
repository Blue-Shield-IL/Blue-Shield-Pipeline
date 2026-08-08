# Blue Shield - Processing Service

Python daemon that runs the Blue Shield data pipeline using `asyncio`:

1. **Ingest** — fetch and listen for posts from external sources (e.g., Telegram) using dedicated fetchers and listeners.
2. **Process** — analyze content via Gemini LLM (sentiment/IHRA labels/keywords/country), filter by antisemitism score, and vectorize using Gemini embeddings.
3. **Store** — persist all posts that passed the filter to Elasticsearch with dense-vector embeddings.

The pipeline runs continuously in the background, orchestrating the ingestion flow via asynchronous queues and scheduled cron jobs.

---

## Tech Stack

- Python 3.10+
- `asyncio` for background daemon and orchestration
- Elasticsearch 9.x (posts + dense-vector embeddings + kNN search)
- Gemini API (LLM for text analysis and embeddings)
- Telethon (for Telegram ingestion)
- Ruff (formatter + linter)
- Pytest

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Access to the internal Elasticsearch host (VPN required)
- A valid Gemini API Key
- Telegram API credentials (if using Telegram ingestion)

### Installation

```bash
cd processing_service
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

Copy `.env.example` → `.env` and fill in your credentials.

---

## Configuration (`.env`)

Key environment variables to configure:

| Variable                 | Purpose                          |
|--------------------------|----------------------------------|
| `ELASTIC_HOST`           | ES node URL.                     |
| `ELASTIC_USERNAME`       | ES username.                     |
| `ELASTIC_PASSWORD`       | ES password.                     |
| `GEMINI_API_KEY`         | Your Gemini API key.             |
| `INGESTION_LISTENERS`    | Comma-separated list of active listeners (e.g., `telegram`). |
| `INGESTION_FETCHERS`     | Comma-separated list of active fetchers (e.g., `telegram`). |
| `CRON_FETCH_INTERVAL_SECONDS`| How often fetchers run (default `60`).|

See `config.py` for all available settings.

---

## Telegram Setup

The Telegram integration uses a Telethon user session. You must register a Telegram app once, configure the environment variables, and create a local authorized session file before the daemon can fetch or listen to messages.

### 1. Register a Telegram app

1. Go to `https://my.telegram.org`.
2. Sign in with the phone number of the Telegram account that will read the supplier channel.
3. Open **API development tools**.
4. Create an app if you do not already have one.
5. Copy the generated `api_id` and `api_hash`.

### 2. Configure `.env`

Set the Telegram fields in `processing_service/.env`:

```env
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_SESSION_FILE=telegram.session
TELEGRAM_SUPPLIER_CHANNELS=@your_channel_or_t_me_link,@another_channel
```

Notes:
- `TELEGRAM_API_ID` should be numeric and should not be quoted.
- `TELEGRAM_SESSION_FILE` is the local Telethon session file.
- `TELEGRAM_SUPPLIER_CHANNELS` should be a comma-separated list of resolvable public handles (e.g., `@channel_name`), an `https://t.me/...` link, or private channels the account has access to.

### 3. Create the Telegram session

Run this once from `processing_service` directory to initialize the local session:

```bash
python -c "from telethon.sync import TelegramClient; from config import settings; client = TelegramClient(settings.telegram_session_file, settings.telegram_api_id, settings.telegram_api_hash); client.start(); print('authorized=', client.is_user_authorized()); print(client.get_me()); client.disconnect()"
```

- Telethon will ask for your phone number and login code.
- If 2FA is enabled, it may ask for your password.
- On success, the session file (`TELEGRAM_SESSION_FILE`) is created locally.

---

## Running the Daemon

```bash
python main.py
```

On startup, the daemon:
1. Pings Elasticsearch and ensures the target `posts` index exists.
2. Starts the Orchestrator to monitor the internal `asyncio` queue.
3. Starts the specified **Listeners** (real-time stream).
4. Starts the specified **Fetchers** on a scheduled cron interval.

---

## Project Structure & Flow

```
processing_service/
├── main.py              Main daemon entrypoint, manages asyncio loop, orchestrator, and ingestion
├── config.py            Env-driven configuration for ES, Gemini, Telegram, etc.
├── models/
│   └── post.py          Pydantic schemas for Post and ProcessedPost
├── pipeline/
│   ├── orchestrator.py  Batches and routes queued posts to enrichment & storage
│   ├── ingestion/       Step 1 — Source fetchers and listeners
│   │   ├── ingest.py    Maps ingestion names (e.g. 'telegram') to adapters
│   │   └── adapters/    Specific ingestion implementations (e.g. telegram_adapter.py)
│   ├── enrichment/      Step 2 — Gemini API analysis and vectorization
│   └── storage/         Step 3 — Elasticsearch persistence
├── tests/               Unit tests and E2E integration tests
├── .env.example
├── pyproject.toml
└── requirements.txt
```

### The Pipeline Flow

1. **Ingestion**: Listeners (`listen_telegram`) and Fetchers (`fetch_telegram`) stream raw posts into an `asyncio.Queue` in `main.py`.
2. **Orchestration**: `PipelineOrchestrator` groups items from the queue into batches based on `ML_BATCH_SIZE` or elapsed time (`FLUSH_INTERVAL_SEC`).
3. **Enrichment**: The orchestrator sends batches to `enrich_posts` in the enrichment module, which uses the `GeminiAdapter` to analyze content, filter by score, and vectorize the passed posts.
4. **Storage**: The finalized `ProcessedPost` objects are inserted into the `posts` index in Elasticsearch using the storage module.

---

## Testing

Run tests with `pytest`:

```bash
pytest tests/
```
The test suite includes unit tests (`test_pipeline.py`) with mocked adapters, and end-to-end/integration tests (`test_ml.py` and `TestEndToEndWithElastic`). 
**Note**: Integration tests require a valid `GEMINI_API_KEY` and a reachable Elasticsearch instance.
