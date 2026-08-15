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
        return f"""You are an expert hate-speech and geopolitical analyst specializing in antisemitism monitoring.

antisemitism_score and sentiment measure DIFFERENT things:
- antisemitism_score: relevance to hate-speech, antisemitism, or anti-Zionism (0.0–1.0). Score based on content, not author stance.
- sentiment: the AUTHOR'S OWN stance toward Jews/Israel.

Scoring guide:
  0.0–0.2: No antisemitism. General mentions of Jews/Israel, cultural content, neutral news.
  0.3–0.4: Mild — dog-whistles, coded language, ambiguous hostility.
  0.5–0.6: Anti-Zionist or critical — one-sided framing against Israel, critical war reporting, delegitimization, reporting on antisemitic incidents.
  0.7–0.8: Clear antisemitic content — tropes, conspiracy theories, dehumanization, anti-Zionism with hateful rhetoric.
  0.9–1.0: Extreme — calls for violence, Holocaust denial, explicit hate-speech targeting Jews.

Output fields:
- antisemitism_score: float 0.0–1.0. Do not over-flag based on punctuation, formatting, or author names alone.
- ihra_labels: list from ONLY: {json.dumps(ihra_labels)}
- keywords: list from ONLY: {json.dumps(keyword_labels)}
- sentiment: one of:
  "Supportive" — pro-Jewish/pro-Israel, condemning antisemitism.
  "Neutral" — no stance, factual reporting.
  "Negative" — critical, conspiracy theories, tropes, delegitimization, inflammatory rhetoric.
  "Hostile" — ONLY explicit calls for violence, direct threats, endorsing genocide. Should be rare.
- country_of_origin: where the antisemitism/anti-Zionism originates from.
  1. If author nationality/location is stated in text, use that country.
  2. If antisemitic activity happens in a specific place, use that country.
  3. Use metadata (phone code, channel location) as supporting signals.
  Return null if no clear geographic link — do NOT guess from topic or channel name."""

    def _build_few_shot_examples(self) -> list[types.Content]:
        examples = [
            ("Jewish community center in Toronto hosts annual Hanukkah celebration with record attendance",
             '{"antisemitism_score": 0.1, "sentiment": "Supportive", "ihra_labels": [], "keywords": [], "country_of_origin": null}'),
            ("Israeli tech startup raises $50M in Series B funding for cybersecurity platform",
             '{"antisemitism_score": 0.0, "sentiment": "Neutral", "ihra_labels": [], "keywords": [], "country_of_origin": null}'),
            ("I'm starting to notice who really controls Hollywood and the banks. Just look at the names. Do your own research.",
             '{"antisemitism_score": 0.45, "sentiment": "Negative", "ihra_labels": ["Stereotypes about Jewish power"], "keywords": ["conspiracy"], "country_of_origin": "United States"}'),
            ("Israel's settlement expansion violates international law and amounts to apartheid, according to Human Rights Watch report",
             '{"antisemitism_score": 0.5, "sentiment": "Negative", "ihra_labels": ["Double standards applied to Israel"], "keywords": ["apartheid"], "country_of_origin": "United States"}'),
            ("The Zionist lobby has purchased every member of Congress. America is ZOG — a Zionist Occupied Government serving its Jewish masters.",
             '{"antisemitism_score": 0.75, "sentiment": "Negative", "ihra_labels": ["Stereotypes about Jewish power", "Accusations of dual loyalty"], "keywords": ["conspiracy", "ZOG"], "country_of_origin": null}'),
            ("The Holocaust is the biggest lie in history. 6 million is a fabricated number used to guilt the West into supporting the illegitimate state of Israel.",
             '{"antisemitism_score": 0.9, "sentiment": "Negative", "ihra_labels": ["Holocaust denial", "Denying Israel\'s right to exist"], "keywords": ["Holocaust denial"], "country_of_origin": null}'),
            ("Day of the rope is coming for every last one of them. The ovens need to be fired up again.",
             '{"antisemitism_score": 1.0, "sentiment": "Hostile", "ihra_labels": ["Calls for violence against Jews"], "keywords": ["violence", "genocide"], "country_of_origin": null}'),
            ("Antisemitic attacks in France rose 300% last year according to government statistics. Jewish schools now require armed guards.",
             '{"antisemitism_score": 0.55, "sentiment": "Neutral", "ihra_labels": [], "keywords": ["antisemitic attacks"], "country_of_origin": "France"}'),
            ("French journalist recalls seeing enormous territories destroyed in Gaza during a humanitarian aid mission, the result of intensive Israeli bombardments.",
             '{"antisemitism_score": 0.45, "sentiment": "Negative", "ihra_labels": [], "keywords": [], "country_of_origin": "France"}'),
        ]
        contents = []
        for user_text, model_response in examples:
            contents.append(types.Content(role="user", parts=[types.Part.from_text(f"[POST 0]\nText: {user_text}")]))
            contents.append(types.Content(role="model", parts=[types.Part.from_text(f"[{model_response}]")]))
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
        few_shot.append(types.Content(role="user", parts=[types.Part.from_text(prompt)]))

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
