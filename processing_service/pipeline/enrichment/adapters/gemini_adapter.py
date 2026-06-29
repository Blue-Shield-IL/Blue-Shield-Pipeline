import json
import logging

from google import genai
from google.genai import types
from langfuse import observe

from config import get_langfuse_client, settings
from models.post_analysis import PostAnalysis

logger = logging.getLogger(__name__)


class GeminiAdapter:
    """Wrapper class for Google Gemini API."""

    def __init__(self):
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
        self.client = genai.Client(api_key=settings.gemini_api_key, http_options={'timeout': 300000})
        self.model = settings.gemini_model_name
        self.embedding_model = settings.gemini_embedding_model_name
        self.embedding_dims = settings.embedding_dims

    @observe(
        as_type="generation", name="gemini-analyze-posts", capture_input=False, capture_output=False
    )
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
  2. Channel/Author Description or Name/Username (if available) - extract the location or primary country of focus.
  3. Language - if the text is in a country-specific language (e.g. German -> Germany, Farsi -> Iran), use it.
  4. Context - use slang, local events, or politicians mentioned to guess the country.
  Return null ONLY if it is completely impossible to infer the country.

Here are the posts to analyze:
"""
        for i, ctx in enumerate(contexts):
            prompt += f"\n[POST {i}]\nText: {ctx['text']}\n"
            if ctx.get("author_phone"):
                prompt += f"Author Phone: {ctx['author_phone']}\n"
            if ctx.get("author_name"):
                prompt += f"Author Name: {ctx['author_name']}\n"
            if ctx.get("author_username"):
                prompt += f"Author Username: {ctx['author_username']}\n"
            if ctx.get("channel_name"):
                prompt += f"Channel Name: {ctx['channel_name']}\n"
            if ctx.get("channel_username"):
                prompt += f"Channel Username: {ctx['channel_username']}\n"
            if ctx.get("channel_description"):
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

            usage = getattr(response, "usage_metadata", None)
            client = get_langfuse_client()
            if client:
                client.update_current_generation(
                    model=self.model,
                    input={"posts_count": len(contexts)},
                    output=response.text[:500],
                    usage_details={
                        "input": getattr(usage, "prompt_token_count", 0),
                        "output": getattr(usage, "candidates_token_count", 0),
                    },
                )

            raw_results = json.loads(response.text)
            results = [PostAnalysis(**res) for res in raw_results]

            if len(results) != len(contexts):
                logger.warning(
                    f"Expected {len(contexts)} results from Gemini, got {len(results)}. Proceeding with partial results."
                )

            return results

        except Exception as e:
            logger.error("Gemini API analysis failed", extra={"error": str(e)})
            raise

    @observe(
        as_type="generation",
        name="gemini-vectorize-texts",
        capture_input=False,
        capture_output=False,
    )
    def vectorize_texts(self, texts: list[str]) -> list[list[float] | None]:
        """Get embeddings using Gemini model sequentially."""
        if not texts:
            return []

        import concurrent.futures

        results: list[list[float] | None] = [None for _ in texts]

        def _embed_single(i: int, text: str):
            if not text.strip():
                return
            try:
                response = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=text,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.embedding_dims,
                    ),
                )
                if response.embeddings:
                    results[i] = response.embeddings[0].values
            except Exception as e:
                logger.warning("Embedding failed for a text", extra={"error": str(e)})

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_embed_single, i, text) for i, text in enumerate(texts)]
            concurrent.futures.wait(futures)

        client = get_langfuse_client()
        if client:
            client.update_current_generation(
                model=self.embedding_model,
                input={"texts_count": len(texts)},
            )
        return results

    @observe(name="gemini-count-tokens", capture_input=False, capture_output=False)
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text using the Gemini model."""
        try:
            response = self.client.models.count_tokens(model=self.model, contents=text)
            return response.total_tokens
        except Exception as e:
            logger.error("Gemini API count_tokens failed", extra={"error": str(e)})
            return len(text) // 4
