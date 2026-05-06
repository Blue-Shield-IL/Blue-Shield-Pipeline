# Blue Shield Pipeline

Blue Shield is a data processing pipeline that ingests raw social media posts and runs them through a series of local, privacy-preserving AI models to detect and analyze antisemitism.

The core of the pipeline is the `processing_service`, which functions as a producer-consumer worker. It retrieves posts, passes them through a local Hugging Face NLP pipeline (for filtering, sentiment analysis, zero-shot IHRA categorization, and entity extraction), and generates dense vectors for semantic search.

See `processing_service/README.md` for setup and execution details.
