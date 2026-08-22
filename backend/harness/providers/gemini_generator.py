"""
Generation provider using Gemini. The prompt explicitly instructs the model
to answer only from the provided context and to say so plainly when the
context doesn't contain the answer -- this is the generation-side half of
groundedness; the guardrails module (separate) does the post-hoc check that
the answer actually stayed grounded.
"""

import os
from ..errors import GenerationError
from ..interfaces import GeneratorProvider

REFUSAL_PHRASE = "I don't have enough information in the provided context to answer that."

_PROMPT_TEMPLATE = """You are a retrieval-augmented assistant. Answer the question using ONLY the context passages below. Do not use outside knowledge.

If the context does not contain enough information to answer, respond with exactly this sentence and nothing else:
"{refusal}"

Context passages:
{context}

Question: {query}

Answer:"""


class GeminiGenerator(GeneratorProvider):
    name = "gemini"

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")
        self._client = None

    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model_name)
        return self._client

    def generate(self, query: str, context_chunks: list[str]) -> str:
        if not context_chunks:
            return REFUSAL_PHRASE

        context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))
        prompt = _PROMPT_TEMPLATE.format(refusal=REFUSAL_PHRASE, context=context, query=query)

        try:
            model = self._get_client()
            response = model.generate_content(prompt)
            text = (response.text or "").strip()
            if not text:
                raise GenerationError("Gemini returned an empty response")
            return text
        except GenerationError:
            raise
        except Exception as e:
            raise GenerationError(f"Gemini generation failed: {e}") from e
