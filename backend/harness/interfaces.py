"""
Abstract interfaces for the two swappable external calls in the pipeline:
speech-to-text and answer generation. The harness only ever talks to these
interfaces, never to a specific vendor SDK directly -- so switching from
Whisper to ElevenLabs, or from Gemini to another model, means writing one
new provider class, not touching pipeline.py.
"""

from abc import ABC, abstractmethod
from .schemas import GuardrailResult


class STTProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """Returns transcribed text. Raises TranscriptionError on failure."""
        raise NotImplementedError

    name: str = "base_stt"


class GeneratorProvider(ABC):
    @abstractmethod
    def generate(self, query: str, context_chunks: list[str]) -> str:
        """Returns generated answer text. Raises GenerationError on failure."""
        raise NotImplementedError

    name: str = "base_generator"


class GuardrailChecker(ABC):
    """Injected into the harness so the full guardrails module (off-topic
    detection, groundedness scoring, unsafe-input handling) can be swapped
    in later without touching pipeline.py. See guardrails/basic_guardrails.py
    for a minimal implementation used until the full module lands."""

    @abstractmethod
    def check_pre_generation(
        self, query: str, retrieval_scores: list[float], top_chunk_text: str | None = None,
    ) -> GuardrailResult:
        """Runs after retrieval, before generation. Used to catch off-topic
        queries where nothing relevant was retrieved. top_chunk_text is
        optional context (the text of the highest-scoring retrieved chunk)
        that richer implementations can use for a lexical sanity check;
        simple implementations can ignore it."""
        raise NotImplementedError

    @abstractmethod
    def check_post_generation(self, query: str, answer: str, context_chunks: list[str]) -> GuardrailResult:
        """Runs after generation. Used to catch ungrounded/hallucinated answers."""
        raise NotImplementedError
