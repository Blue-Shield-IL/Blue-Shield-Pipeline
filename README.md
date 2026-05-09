# Blue Shield Pipeline

Data processing pipeline for the Blue Shield project — monitors, analyzes, and stores social media posts related to antisemitism and public diplomacy.

## Architecture

```
[Telegram / Reddit] → Processing Service → Elasticsearch → [API Server → Dashboard]
                      (this repo)
```

The Processing Service runs a 3-step pipeline:
1. **Ingest** posts from external sources
2. **Process** with ML models (filter, sentiment, IHRA labels, keywords, NER, embeddings)
3. **Store** in Elasticsearch with dense vectors for semantic search

## Repositories

| Repo | Purpose |
|------|---------|
| [Blue-Shield-Pipeline](https://github.com/eden0501/Blue-Shield-Pipeline) | Data ingestion + ML processing + ES storage (this repo) |
| [Blue-Shield-Back](https://github.com/eden0501/Blue-Shield-Back) | Node.js API Server (auth, queries, aggregations) |
| [Blue-Shield-Client](https://github.com/eden0501/Blue-Shield-Client) | React dashboard |

## Quick Start

See [`processing_service/README.md`](processing_service/README.md) for setup instructions.
