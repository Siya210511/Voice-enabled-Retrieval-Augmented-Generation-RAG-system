"""
Groundedness checking: does the generated answer actually come from the
retrieved context, or is the model padding/hallucinating?

Two checks, both needed:
  1. Document-level overlap: content-word overlap between the whole answer
     and the whole context, with stopwords stripped (fixes the bug found
     in BasicGuardrails, where "the"/"is" inflated the score).
  2. Per-sentence support: split the answer into sentences and require each
     *substantive* sentence to have meaningful overlap with at least one
     context chunk individually. This catches the case where the answer
     as a whole scores fine on aggregate overlap (because one sentence is
     well-grounded) but includes an additional fabricated sentence riding
     along -- aggregate scoring alone would miss that.

This is still lexical (word-overlap based), not a semantic entailment model.
It will not catch a hallucination that happens to reuse the right words in
a false claim (e.g. swapping two entities that both appear in context). If
you have budget left, replacing step 2 with an NLI model or asking the
generator to self-cite which chunk supports each sentence would close that
gap -- noted here rather than silently pretending this is airtight.
"""

import re
from dataclasses import dataclass
from .stopwords import content_words

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+")


@dataclass
class GroundednessScore:
    passed: bool
    overall_overlap_ratio: float
    unsupported_sentences: list[str]
    reason: str | None = None


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]


class GroundednessChecker:
    def __init__(
        self,
        min_overall_overlap: float = 0.25,
        min_sentence_overlap: float = 0.2,
        refusal_markers: tuple[str, ...] = ("don't have enough information", "cannot answer", "no relevant information"),
    ):
        self.min_overall_overlap = min_overall_overlap
        self.min_sentence_overlap = min_sentence_overlap
        self.refusal_markers = refusal_markers

    def check(self, answer: str, context_chunks: list[str]) -> GroundednessScore:
        answer_lower = answer.lower()
        if any(marker in answer_lower for marker in self.refusal_markers):
            # A correct refusal is trivially grounded -- it makes no claims.
            return GroundednessScore(passed=True, overall_overlap_ratio=1.0, unsupported_sentences=[])

        context_words = content_words(" ".join(context_chunks))
        answer_words = content_words(answer)

        if not answer_words:
            return GroundednessScore(
                passed=False, overall_overlap_ratio=0.0, unsupported_sentences=[],
                reason="Answer contained no substantive content words to check.",
            )

        overall_overlap = len(answer_words & context_words) / len(answer_words)

        # per-sentence check
        per_chunk_words = [content_words(c) for c in context_chunks]
        unsupported = []
        for sentence in split_sentences(answer):
            s_words = content_words(sentence)
            if not s_words:
                continue  # e.g. a bare "Yes." or "No." -- nothing to check
            best_ratio = 0.0
            for chunk_words in per_chunk_words:
                if not chunk_words:
                    continue
                ratio = len(s_words & chunk_words) / len(s_words)
                best_ratio = max(best_ratio, ratio)
            if best_ratio < self.min_sentence_overlap:
                unsupported.append(sentence)

        passed = overall_overlap >= self.min_overall_overlap and not unsupported
        reason = None
        if not passed:
            reasons = []
            if overall_overlap < self.min_overall_overlap:
                reasons.append(
                    f"overall content-word overlap {overall_overlap:.2f} below "
                    f"threshold {self.min_overall_overlap}"
                )
            if unsupported:
                reasons.append(f"{len(unsupported)} sentence(s) not supported by any single context chunk")
            reason = "; ".join(reasons)

        return GroundednessScore(
            passed=passed,
            overall_overlap_ratio=overall_overlap,
            unsupported_sentences=unsupported,
            reason=reason,
        )
