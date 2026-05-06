# Blue Shield - Processing Service

A Producer-Consumer worker script for the Blue Shield data pipeline. It continuously retrieves batches of raw social posts from a Source API and runs them through local Hugging Face models to filter and extract insights.

**Note: This service runs entirely locally on your hardware. No external APIs (like OpenAI or Google AI) are used.**

## Folder Structure

- `main.py`: The worker script that runs the infinite fetch-process loop.
- `models/`: Pydantic schemas (`Post` and `ProcessedPost`) for data validation.
- `services/ml_services.py`: The local NLP pipeline (Binary Filtering, Sentiment, Zero-shot IHRA classification, NER, and Dense Vectorization).
- `services/pipeline_service.py`: End-to-end orchestration of the ML steps.
- `tests/`: Pytest suite verifying the pipeline and ML functions.
- `requirements.txt`: Python dependencies.

## Getting Started

1. Work from the project root (one level above `processing_service/`):

```bash
cd Blue-Shield-Pipeline
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate        # Windows
source .venv/bin/activate       # Mac / Linux
```

3. Install dependencies:

```bash
pip install -r processing_service/requirements.txt
```

4. Configure your `.env` file (Optional):

Key variables:
- `FILTER_MODEL_NAME` — default `distilbert-base-uncased-finetuned-sst-2-english`

5. Run the Worker **from the project root** (so Python resolves the `processing_service.*` package):

```bash
python processing_service/main.py
```
*Note: Ensure the `SOURCE_API_URL` inside `main.py` is pointed to your actual data source.*

6. Run Tests:

```bash
python -m pytest processing_service/tests/ -v
```

## Local Models Used
The pipeline downloads and executes the following models locally via the `transformers` and `sentence-transformers` libraries:
- **Filtering:** `distilbert-base-uncased-finetuned-sst-2-english`
- **Sentiment:** `cardiffnlp/twitter-roberta-base-sentiment`
- **Analysis:** `facebook/bart-large-mnli`
- **NER (Country extraction):** `dbmdz/bert-large-cased-finetuned-conll03-english`
- **Vectorization:** `all-MiniLM-L6-v2`
