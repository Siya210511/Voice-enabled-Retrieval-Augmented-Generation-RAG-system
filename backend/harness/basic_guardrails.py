"""
Minimal guardrail implementation so the harness has real pre/post checks
to call today. This is intentionally basic (threshold + keyword overlap) --
the full guardrails module (semantic off-topic detection, unsafe-input
classification, proper hallucination scoring) should replace this class.
Because it implements GuardrailChecker, swapping it in the harness is a
one-line change: `harness = build_default_harness(guardrails=FullGuardrails())`.
"""

import re
from .interfaces import GuardrailChecker
from .schemas import GuardrailResult


class BasicGuardrails(GuardrailChecker):
    def __init__(self, min_retrieval_score: float = 0.15, min_overlap_ratio: float = 0.1):
        self.min_retrieval_score = min_retrieval_score
        self.min_overlap_ratio = min_overlap_ratio

    def check_pre_generation(self, query: str, retrieval_scores: list[float]) -> GuardrailResult:
        if not retrieval_scores or max(retrieval_scores) < self.min_retrieval_score:
            return GuardrailResult(
                passed=False,
                reason=(
                    f"Top retrieval score {max(retrieval_scores) if retrieval_scores else 0:.3f} "
                    f"is below threshold {self.min_retrieval_score} -- likely off-topic query."
                ),
                checked_stage="pre_generation",
            )
        return GuardrailResult(passed=True, checked_stage="pre_generation")

    def check_post_generation(self, query: str, answer: str, context_chunks: list[str]) -> GuardrailResult:
        # Refusals are trivially "grounded" -- the model correctly declined.
        if "don't have enough information" in answer.lower():
            return GuardrailResult(passed=True, checked_stage="post_generation")

        answer_words = set(re.findall(r"\w+", answer.lower()))
        context_words = set(re.findall(r"\w+", " ".join(context_chunks).lower()))
        if not answer_words:
            return GuardrailResult(passed=False, reason="Empty answer", checked_stage="post_generation")

        overlap_ratio = len(answer_words & context_words) / len(answer_words)
        if overlap_ratio < self.min_overlap_ratio:
            return GuardrailResult(
                passed=False,
                reason=(
                    f"Answer/context word overlap ({overlap_ratio:.2f}) below threshold "
                    f"({self.min_overlap_ratio}) -- answer may not be grounded in retrieved context."
                ),
                checked_stage="post_generation",
            )
        return GuardrailResult(passed=True, checked_stage="post_generation")
