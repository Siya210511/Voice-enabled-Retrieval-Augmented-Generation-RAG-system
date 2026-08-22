"""
The harness: orchestrates STT -> retrieval -> pre-generation guardrail ->
generation -> post-generation guardrail, with retries and structured
input/output at every stage. This is the "proper harness" the spec asks
for, as opposed to a single raw prompt-in/text-out call.

Design choices worth knowing about:
  - Every stage wraps its provider call in retry_with_backoff and converts
    failures into a structured *Result with status=FAILED rather than
    raising -- so the caller (API layer, tests) always gets a well-formed
    PipelineResponse, never a bare traceback.
  - Guardrails run as a real stage, not an afterthought bolted onto
    generation: a failed pre-generation check skips the (expensive, slow)
    generation call entirely.
  - text_query bypasses STT entirely -- useful for testing the
    retrieval/generation stages independently of audio, and for a
    text-input fallback in the UI if the mic fails.
"""

import time

from .schemas import (
    PipelineRequest, PipelineResponse, StageStatus,
    TranscriptionResult, RetrievalResult, RetrievedChunkView,
    GenerationResult, GuardrailResult,
)
from .errors import (
    TranscriptionError, RetrievalError, GenerationError,
    retry_with_backoff,
)
from .interfaces import STTProvider, GeneratorProvider, GuardrailChecker


class RAGHarness:
    def __init__(
        self,
        stt: STTProvider | None,
        retriever,                       # embedding.retriever.Retriever instance
        generator: GeneratorProvider,
        guardrails: GuardrailChecker,
        top_k_default: int = 5,
        max_attempts: int = 3,
    ):
        self.stt = stt
        self.retriever = retriever
        self.generator = generator
        self.guardrails = guardrails
        self.top_k_default = top_k_default
        self.max_attempts = max_attempts

    def run(self, request: PipelineRequest) -> PipelineResponse:
        overall_start = time.perf_counter()

        # ---- Stage 1: transcription (skipped if text_query provided) ----
        if request.text_query:
            transcription = TranscriptionResult(
                status=StageStatus.SKIPPED, text=request.text_query, provider="text_input",
            )
            query_text = request.text_query
        else:
            transcription = self._run_transcription(request.audio_path)
            if transcription.status == StageStatus.FAILED:
                return self._early_exit(
                    request, transcription=transcription,
                    blocked_reason=f"Transcription failed: {transcription.error}",
                    overall_start=overall_start,
                )
            query_text = transcription.text

        # ---- Stage 2: retrieval ----
        retrieval = self._run_retrieval(request, query_text)
        if retrieval.status == StageStatus.FAILED:
            return self._early_exit(
                request, transcription=transcription, retrieval=retrieval,
                blocked_reason=f"Retrieval failed: {retrieval.error}",
                overall_start=overall_start,
            )

        # ---- Stage 3: pre-generation guardrail ----
        scores = [c.score for c in retrieval.chunks]
        top_chunk_text = retrieval.chunks[0].text if retrieval.chunks else None
        guardrail_pre = self.guardrails.check_pre_generation(query_text, scores, top_chunk_text=top_chunk_text)
        if not guardrail_pre.passed:
            return self._early_exit(
                request, transcription=transcription, retrieval=retrieval,
                guardrail_pre=guardrail_pre,
                blocked_reason=guardrail_pre.reason,
                overall_start=overall_start,
                final_answer="I don't have relevant information to answer that question.",
            )

        # ---- Stage 4: generation ----
        context_texts = [c.text for c in retrieval.chunks]
        generation = self._run_generation(query_text, context_texts)
        if generation.status == StageStatus.FAILED:
            return self._early_exit(
                request, transcription=transcription, retrieval=retrieval,
                guardrail_pre=guardrail_pre, generation=generation,
                blocked_reason=f"Generation failed: {generation.error}",
                overall_start=overall_start,
            )

        # ---- Stage 5: post-generation guardrail ----
        guardrail_post = self.guardrails.check_post_generation(
            query_text, generation.answer, context_texts,
        )
        final_answer = generation.answer
        blocked_reason = None
        if not guardrail_post.passed:
            final_answer = "I'm not confident enough in this answer to share it -- it may not be fully supported by the retrieved context."
            blocked_reason = guardrail_post.reason

        total_latency = time.perf_counter() - overall_start
        return PipelineResponse(
            query_text=query_text,
            transcription=transcription,
            retrieval=retrieval,
            generation=generation,
            guardrail_pre=guardrail_pre,
            guardrail_post=guardrail_post,
            final_answer=final_answer,
            total_latency_sec=total_latency,
            status=StageStatus.OK if guardrail_post.passed else StageStatus.FAILED,
            blocked_reason=blocked_reason,
        )

    # ---------------- internal stage runners ----------------

    def _run_transcription(self, audio_path: str) -> TranscriptionResult:
        try:
            @retry_with_backoff(max_attempts=self.max_attempts, exceptions=(TranscriptionError,))
            def _call():
                return self.stt.transcribe(audio_path)

            (text, attempts) = _call()
            return TranscriptionResult(
                status=StageStatus.OK, text=text, provider=self.stt.name, attempts=attempts,
            )
        except TranscriptionError as e:
            return TranscriptionResult(status=StageStatus.FAILED, error=str(e), provider=getattr(self.stt, "name", None))

    def _run_retrieval(self, request: PipelineRequest, query_text: str) -> RetrievalResult:
        try:
            @retry_with_backoff(max_attempts=self.max_attempts, exceptions=(Exception,))
            def _call():
                results, latency = self.retriever.retrieve(
                    query_text,
                    k=request.top_k or self.top_k_default,
                    language=request.language_filter,
                    only_selected=request.only_selected,
                )
                return results, latency

            (result_tuple, attempts) = _call()
            results, latency = result_tuple
            chunk_views = [
                RetrievedChunkView(chunk_id=r.chunk_id, text=r.text, score=r.score, doc_id=r.doc_id)
                for r in results
            ]
            return RetrievalResult(
                status=StageStatus.OK, chunks=chunk_views, latency_sec=latency,
                strategy=self.retriever.strategy, attempts=attempts,
            )
        except Exception as e:
            return RetrievalResult(status=StageStatus.FAILED, error=str(e), strategy=getattr(self.retriever, "strategy", None))

    def _run_generation(self, query_text: str, context_texts: list[str]) -> GenerationResult:
        try:
            @retry_with_backoff(max_attempts=self.max_attempts, exceptions=(GenerationError,))
            def _call():
                start = time.perf_counter()
                answer = self.generator.generate(query_text, context_texts)
                return answer, time.perf_counter() - start

            ((answer, latency), attempts) = _call()
            return GenerationResult(
                status=StageStatus.OK, answer=answer, model=self.generator.name,
                latency_sec=latency, attempts=attempts,
            )
        except GenerationError as e:
            return GenerationResult(status=StageStatus.FAILED, error=str(e), model=getattr(self.generator, "name", None))

    def _early_exit(
        self, request, transcription=None, retrieval=None,
        guardrail_pre=None, generation=None, guardrail_post=None,
        blocked_reason=None, overall_start=None, final_answer=None,
    ) -> PipelineResponse:
        return PipelineResponse(
            query_text=request.text_query,
            transcription=transcription,
            retrieval=retrieval,
            generation=generation,
            guardrail_pre=guardrail_pre,
            guardrail_post=guardrail_post,
            final_answer=final_answer or "Sorry, I couldn't process that request.",
            total_latency_sec=time.perf_counter() - overall_start if overall_start else 0.0,
            status=StageStatus.FAILED,
            blocked_reason=blocked_reason,
        )


def build_default_harness(strategy: str = "metadata_aware", use_real_providers: bool = False, use_full_guardrails: bool = True):
    """Convenience constructor. use_real_providers=False wires up fake STT/
    generator so you can smoke-test the harness without API keys; flip it
    to True once ELEVENLABS_API_KEY / GEMINI_API_KEY are set.
    use_full_guardrails=True uses guardrails.full_guardrails.FullGuardrails
    (recommended); set False to fall back to the basic placeholder."""
    from embedding.retriever import Retriever

    retriever = Retriever(strategy=strategy)

    if use_full_guardrails:
        from guardrails.full_guardrails import FullGuardrails
        guardrails = FullGuardrails()
    else:
        from .basic_guardrails import BasicGuardrails
        guardrails = BasicGuardrails()

    if use_real_providers:
        from .providers.stt_providers import ElevenLabsSTT
        from .providers.gemini_generator import GeminiGenerator
        stt = ElevenLabsSTT()
        generator = GeminiGenerator()
    else:
        from .providers.fake_providers import FakeSTT, FakeGenerator
        stt = FakeSTT()
        generator = FakeGenerator()

    return RAGHarness(stt=stt, retriever=retriever, generator=generator, guardrails=guardrails)
