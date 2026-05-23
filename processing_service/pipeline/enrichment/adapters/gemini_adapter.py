import json
import logging

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from config import settings

LOGGER = logging.getLogger(__name__)


class PostAnalysis(BaseModel):
    antisemitism_score: float = Field(
        description="A gradient score between 0.0 and 1.0 representing the confidence that the post contains "
                    "antisemitism or justifies violence against Jews."
    )
    ihra_labels: list[str] = Field(
        description="List of applicable IHRA antisemitism labels."
    )
    keywords: list[str] = Field(
        description="List of applicable keyword labels (e.g., 'October 7th Hamas attack', 'blood libel')."
    )
    sentiment: str = Field(
        description="The overall sentiment of the text. One of: Supportive, Neutral, Negative, Hostile."
    )
    country_of_origin: str | None = Field(
        default=None,
        description="The country or location mentioned as the origin/target if applicable. Null if none."
    )


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
        self, texts: list[str], ihra_labels: list[str], keyword_labels: list[str]
    ) -> list[PostAnalysis]:
        if not texts:
            return []

        prompt = f"""
You are an expert hate-speech and geopolitical analyst.
Analyze the following batch of social media posts.

For each post, output a JSON object containing:
- antisemitism_score: A float from 0.0 to 1.0 indicating if the text is antisemitic or justifies violence against Jews.
- ihra_labels: A list of strings matching ONLY the following allowed IHRA labels:
{json.dumps(ihra_labels, indent=2)}
- keywords: A list of strings matching ONLY the following allowed keyword labels:
{json.dumps(keyword_labels, indent=2)}
- sentiment: "Supportive", "Neutral", "Negative", or "Hostile".
- country_of_origin: Any extracted country/location from the text. Null if none.

Here are the posts to analyze:
"""
        for i, text in enumerate(texts):
            prompt += f"\n[POST {i}]\n{text}\n"

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

            if len(results) != len(texts):
                LOGGER.error(f"Expected {len(texts)} results from Gemini, got {len(results)}. Padding/truncating.")
                while len(results) < len(texts):
                    results.append(
                        PostAnalysis(
                            antisemitism_score=0.0,
                            ihra_labels=[],
                            keywords=[],
                            sentiment="Neutral",
                            country_of_origin=None,
                        )
                    )
                results = results[: len(texts)]

            return results

        except Exception as e:
            LOGGER.exception(f"Gemini API analysis failed: {e}")
            return [
                PostAnalysis(
                    antisemitism_score=0.0,
                    ihra_labels=[],
                    keywords=[],
                    sentiment="Neutral",
                    country_of_origin=None,
                )
                for _ in texts
            ]

    def vectorize_texts(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings using Gemini model."""
        if not texts:
            return []

        import concurrent.futures

        def embed_one(text: str) -> list[float]:
            if not text.strip():
                return [0.0] * self.embedding_dims
            try:
                response = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=text,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.embedding_dims,
                    )
                )
                return response.embeddings[0].values
            except Exception as e:
                LOGGER.exception(f"Gemini API embedding failed: {e}")
                return [0.0] * self.embedding_dims

        # Using a thread pool since the google-genai SDK embed_content
        # doesn't natively batch multiple documents in a single request
        # (passing a list of strings concatenates them into one embedding).

        all_embeddings = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            all_embeddings = list(executor.map(embed_one, texts))

        return all_embeddings
