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
    def _build_system_instruction(self, ihra_labels: list[str], keyword_labels: list[str]) -> str:
        return f"""You are an antisemitism monitoring analyst. Analyze posts for antisemitism relevance.

RULES (follow strictly):

R1-SCORE: antisemitism_score = relevance to antisemitism/anti-Zionism (0.0–1.0), based on CONTENT, not author intent.
  0.0–0.2: No relevance. Tech, sports, weather, cultural content with no Israel/conflict framing.
  0.3–0.4: Low. Factual conflict reporting WITHOUT editorial framing or loaded terms.
  0.5–0.6: Relevant. Anti-Zionist framing, loaded language ("occupation", "massacre", "genocide"), one-sided war reporting designed to generate anti-Israel sentiment, reporting on antisemitic incidents.
  0.7–0.8: High. Tropes, conspiracy theories, dehumanization, glorifying armed resistance against Israel.
  0.9–1.0: Extreme. Calls for violence, Holocaust denial, explicit hate-speech.

R2-SENTIMENT: the author's stance toward Jews/Israel. Choose one:
  "Supportive" = defends Jews/Israel, condemns antisemitism.
  "Neutral" = balanced factual reporting with no editorial framing. Use ONLY for truly objective reporting.
  "Negative" = any anti-Israel framing, loaded language, one-sided reporting, delegitimization, tropes, celebrating anti-Israel violence.
  "Hostile" = ONLY explicit calls for violence/genocide. Rare.

R3-DECISION RULES (override general scoring when they apply):
  - Celebratory tone toward anti-Israel armed groups (Hezbollah, Hamas, etc.) or "resistance" = Negative sentiment, score >= 0.7. NEVER Supportive.
  - Celebrating violence against Israel-linked or US-Israeli targets ("beautiful strike", "direct hit") = score >= 0.7, Negative.
  - "Zionist enemy", "Zionist entity", "Zionist project", "occupation forces", "Israeli enemy" = delegitimization, score >= 0.5, Negative.
  - War reporting with loaded terms ("massacre", "martyrs", "genocide", "terrorist Israel") = score >= 0.5, Negative. These are NOT neutral.
  - Showing civilian casualties with anti-Israel framing (dead children, destroyed homes) = score >= 0.5, Negative.
  - Coded references to Jews ("them", "those people", "the tribe", "goyim", "goy") with negative context = score >= 0.6.
  - Comparing Israeli leaders to Nazis/terrorists or vice versa = score >= 0.5, Negative.
  - Reporting on antisemitic incidents (attacks, vandalism, hate crimes) = score 0.5–0.6, Neutral sentiment.
  - Conspiracy framing ("Israel played the X card", "Jewish lobby controls") = score >= 0.6.
  - Unverified or exaggerated claims against Israel (inflated casualty numbers, fabricated atrocities, unsubstantiated accusations) = score >= 0.6, Negative. Disinformation designed to demonize Israel is antisemitic content.
  - ONLY score 0.0–0.2 if the post is about tech, culture, sports, or mentions Israel/Jews without ANY conflict, criticism, or loaded framing.
  - Do NOT lower score just because tone is calm, academic, or celebratory.
  - Do NOT raise score based on punctuation, formatting, or author names alone.

R4-COUNTRY: where the antisemitism/anti-Zionism originates from. NOT the country being discussed or criticized.
  1. Author nationality/location explicitly stated in text → that country.
  2. Anti-Zionist or antisemitic activity happens in a named place (e.g. attack in Iraq, strike in Kurdistan, antisemitic incident in France) → that country.
  3. Metadata (phone code, channel location) as weak signal only.
  null in ALL other cases. Do not confuse the country being discussed with the country of origin. Countries merely mentioned or criticized in the post are NOT automatically the origin.
  NEVER use "Palestine", "Gaza", or "West Bank" as country_of_origin — use "Israel" or null instead.

OUTPUT: ihra_labels from ONLY: {json.dumps(ihra_labels)}
keywords from ONLY: {json.dumps(keyword_labels)}"""

    def _build_few_shot_examples(self) -> list[types.Content]:
        examples = [
            ("Jewish community center in Toronto hosts annual Hanukkah celebration with record attendance",
             '{"antisemitism_score": 0.1, "sentiment": "Supportive", "ihra_labels": [], "keywords": [], "country_of_origin": null}'),
            ("I'm starting to notice who really controls Hollywood and the banks. Just look at the names. Do your own research.",
             '{"antisemitism_score": 0.45, "sentiment": "Negative", "ihra_labels": ["Stereotypes about Jewish power"], "keywords": ["conspiracy"], "country_of_origin": "United States"}'),
            ("A rabbi was attacked outside a synagogue in Reykjavik. Police are investigating it as a hate crime.",
             '{"antisemitism_score": 0.55, "sentiment": "Neutral", "ihra_labels": [], "keywords": ["antisemitic attacks"], "country_of_origin": "Iceland"}'),
            ("We congratulate the heroic resistance on its glorious victory against the Zionist enemy entity. The Zionist project will be defeated.",
             '{"antisemitism_score": 0.7, "sentiment": "Negative", "ihra_labels": ["Denying Israel\'s right to exist"], "keywords": ["anti-Zionism"], "country_of_origin": null}'),
            ("The Holocaust is the biggest lie in history. 6 million is a fabricated number used to guilt the West.",
             '{"antisemitism_score": 0.9, "sentiment": "Negative", "ihra_labels": ["Holocaust denial"], "keywords": ["Holocaust denial"], "country_of_origin": null}'),
        ]
        contents = []
        for user_text, model_response in examples:
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=f"[POST 0]\nText: {user_text}")]))
            contents.append(types.Content(role="model", parts=[types.Part.from_text(text=f"[{model_response}]")]))
        return contents

    def analyze_posts(
        self, contexts: list[dict], ihra_labels: list[str], keyword_labels: list[str]
    ) -> list[PostAnalysis]:
        if not contexts:
            return []

        prompt = "Analyze these posts:\n"
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

        few_shot = self._build_few_shot_examples()
        few_shot.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=few_shot,
                config=types.GenerateContentConfig(
                    system_instruction=self._build_system_instruction(ihra_labels, keyword_labels),
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

            if not response.text:
                raise ValueError(f"Gemini returned empty response (finish_reason={getattr(response.candidates[0], 'finish_reason', 'unknown') if response.candidates else 'no_candidates'})")

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
