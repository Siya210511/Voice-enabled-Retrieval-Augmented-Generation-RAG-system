"""
Fake providers with no external dependencies -- used to unit-test the
harness's orchestration logic (retries, error recovery, structured I/O)
in isolation from real APIs and API keys. Also handy for CI.
"""

import random
from ..errors import TranscriptionError, GenerationError
from ..interfaces import STTProvider, GeneratorProvider


class FakeSTT(STTProvider):
    name = "fake_stt"

    def __init__(self, fixed_text: str = "who built the taj mahal", fail_rate: float = 0.0):
        self.fixed_text = fixed_text
        self.fail_rate = fail_rate

    def transcribe(self, audio_path: str) -> str:
        if random.random() < self.fail_rate:
            raise TranscriptionError("simulated STT failure")
        return self.fixed_text


class FakeGenerator(GeneratorProvider):
    name = "fake_generator"

    def __init__(self, fail_rate: float = 0.0):
        self.fail_rate = fail_rate

    def generate(self, query: str, context_chunks: list[str]) -> str:
        if random.random() < self.fail_rate:
            raise GenerationError("simulated generation failure")
        if not context_chunks:
            return "I don't have enough information in the provided context to answer that."
        return f"Based on the context, here is an answer to '{query}': {context_chunks[0][:80]}"
