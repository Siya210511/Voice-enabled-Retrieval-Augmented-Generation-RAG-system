"""
STT providers.

ElevenLabsSTT is the spec-compliant option (task requires Sarvam or
ElevenLabs). WhisperSTT wraps your partner's existing transcribe.py so you
can keep developing/demoing locally without an ElevenLabs API key, and swap
it out for the final submission by changing one line where the harness is
constructed (see pipeline.py's `build_default_harness`).

Confirm with the organizers whether Whisper is acceptable before relying on
it for your actual submission -- the spec explicitly names only two options.
"""

import os
from ..errors import TranscriptionError
from ..interfaces import STTProvider


class ElevenLabsSTT(STTProvider):
    name = "elevenlabs"

    def __init__(self, api_key: str | None = None, model_id: str = "scribe_v1"):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        self.model_id = model_id
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not set")

    def transcribe(self, audio_path: str) -> str:
        try:
            from elevenlabs.client import ElevenLabs

            client = ElevenLabs(api_key=self.api_key)
            with open(audio_path, "rb") as f:
                result = client.speech_to_text.convert(
                    file=f,
                    model_id=self.model_id,
                )
            text = getattr(result, "text", None)
            if not text:
                raise TranscriptionError("ElevenLabs returned empty transcript")
            return text
        except TranscriptionError:
            raise
        except Exception as e:
            raise TranscriptionError(f"ElevenLabs STT failed: {e}") from e


class WhisperSTT(STTProvider):
    """Wraps the existing src/stt/transcribe.py logic. Dev/demo use only --
    not one of the two spec-approved STT options."""

    name = "whisper_local"

    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            import whisper
            self._model = whisper.load_model(self.model_size)
        return self._model

    def transcribe(self, audio_path: str) -> str:
        try:
            model = self._get_model()
            result = model.transcribe(audio_path)
            text = result.get("text", "").strip()
            if not text:
                raise TranscriptionError("Whisper returned empty transcript")
            return text
        except TranscriptionError:
            raise
        except Exception as e:
            raise TranscriptionError(f"Whisper STT failed: {e}") from e
