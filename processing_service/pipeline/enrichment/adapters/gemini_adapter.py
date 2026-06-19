import json
import logging

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from config import settings
from models.post_analysis import PostAnalysis

logger = logging.getlogger(__name__)


class GeminiAdapter:
    """Wrapper class for Google Gemini API."""

    def __init__(self):
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model_name
        self.embedding_model = settings.gemini_embedding_model_name
        self.embedding_dims = settings.embedding_dims

    def analyze_posts(
        self, contexts: list[dict], ihra_labels: list[str], keyword_labels: list[str]
    ) -> list[PostAnalysis]:
        if not contexts:
            return []

        prompt = f"""
You are an expert hate-speech and geopolitical analyst.
Analyze the following batch of social media posts.

For each post, output a JSON object containing:
- antisemitism_score: A float from 0.0 to 1.0 indicating if the text is antisemitic or justifies violence against Jews. IMPORTANT: Base your score strictly on the actual semantic topic and meaning of the text. Do not over-flag or give high scores merely because of the presence of specific punctuation, formatting (e.g., echo brackets like '(((' or ')))'), or specific author names if the text itself has no connection to Jews or antisemitism.
- ihra_labels: A list of strings matching ONLY the following allowed IHRA labels:
{json.dumps(ihra_labels, indent=2)}
- keywords: A list of strings matching ONLY the following allowed keyword labels:
{json.dumps(keyword_labels, indent=2)}
- sentiment: "Supportive", "Neutral", "Negative", or "Hostile".
- country_of_origin: Infer the geographic origin of the post using the provided metadata.
  1. Phone number (if available) - use the country code.
  2. Channel description (if available) - extract the location or primary country of focus.
  3. Language - if the text is in a country-specific language (e.g. German -> Germany, Farsi -> Iran), use it.
  4. Context - use slang, local events, or politicians mentioned to guess the country.
  Return null ONLY if it is completely impossible to infer the country.

Here are the posts to analyze:
"""
        for i, ctx in enumerate(contexts):
            prompt += f"\n[POST {i}]\nText: {ctx['text']}\n"
            if ctx.get('author_phone'):
                prompt += f"Author Phone: {ctx['author_phone']}\n"
            if ctx.get('channel_description'):
                prompt += f"Channel Description: {ctx['channel_description']}\n"

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=list[PostAnalysis],
                    temperature=0.0,
                ),
            )

            raw_results = json.loads(response.text)
            results = [PostAnalysis(**res) for res in raw_results]

            if len(results) != len(contexts):
                raise RuntimeError(f"Expected {len(contexts)} results from Gemini, got {len(results)}.")

            return results

        except Exception as e:
            logger.error("Gemini API analysis failed", extra={"error": str(e)})
            raise

    def vectorize_texts(self, texts: list[str]) -> list[list[float] | None]:
        """Get embeddings using Gemini model sequentially."""
        if not texts:
            return []

        results: list[list[float] | None] = [None for _ in texts]

        for i, text in enumerate(texts):
            if text.strip():
                try:
                    response = self.client.models.embed_content(
                        model=self.embedding_model,
                        contents=text,
                        config=types.EmbedContentConfig(
                            output_dimensionality=self.embedding_dims,
                        )
                    )
                    if response.embeddings:
                        results[i] = response.embeddings[0].values
                except Exception as e:
                    logger.warning("Embedding failed for a text", extra={"error": str(e)})

        return results

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text using the Gemini model."""
        try:
            response = self.client.models.count_tokens(model=self.model, contents=text)
            return response.total_tokens
        except Exception as e:
            logger.error("Gemini API count_tokens failed", extra={"error": str(e)})
            return len(text) // 4
