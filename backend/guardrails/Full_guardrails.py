"""
FullGuardrails: implements harness.interfaces.GuardrailChecker, combining
the three checks in this module. Drop-in replacement for
harness.basic_guardrails.BasicGuardrails -- swap it in the harness
constructor and nothing else needs to change:

    from guardrails.full_guardrails import FullGuardrails
    harness = RAGHarness(..., guardrails=FullGuardrails())

Pre-generation: unsafe-input filter runs first (cheapest, most important to
catch early), then off-topic detection (score threshold + lexical sanity
check against the top retrieved chunk).
Post-generation: groundedness check (stopword-filtered overlap + per-
sentence support check).
"""

from harness.interfaces import GuardrailChecker
from harness.schemas import GuardrailResult

from .unsafe_input import UnsafeInputFilter
from .off_topic import OffTopicDetector
from .groundedness import GroundednessChecker


class FullGuardrails(GuardrailChecker):
    def __init__(
        self,
        min_retrieval_score: float = 0.15,
        min_overall_overlap: float = 0.25,
        min_sentence_overlap: float = 0.2,
    ):
        self.unsafe_filter = UnsafeInputFilter()
        self.off_topic_detector = OffTopicDetector(min_retrieval_score=min_retrieval_score)
        self.groundedness_checker = GroundednessChecker(
            min_overall_overlap=min_overall_overlap,
            min_sentence_overlap=min_sentence_overlap,
        )

    def check_pre_generation(
        self, query: str, retrieval_scores: list[float], top_chunk_text: str | None = None,
    ) -> GuardrailResult:
        safety = self.unsafe_filter.check(query)
        if not safety.passed:
            return GuardrailResult(passed=False, reason=safety.reason, checked_stage="pre_generation")

        topic = self.off_topic_detector.check(query, retrieval_scores, top_chunk_text)
        if not topic.passed:
            return GuardrailResult(passed=False, reason=topic.reason, checked_stage="pre_generation")

        return GuardrailResult(passed=True, checked_stage="pre_generation")

    def check_post_generation(self, query: str, answer: str, context_chunks: list[str]) -> GuardrailResult:
        score = self.groundedness_checker.check(answer, context_chunks)
        return GuardrailResult(passed=score.passed, reason=score.reason, checked_stage="post_generation")
