"""
Off-topic detection: decides whether a query is even answerable from this
corpus, before spending a generation call on it.

Uses two signals together, not just one:
  1. Retrieval score threshold -- the vector-DB signal. Fast, but can be
     fooled by an embedding model that assigns spuriously high similarity
     to unrelated text (we saw exactly this with the hash-based fallback
     embedder during testing).
  2. Lexical overlap sanity check -- does the query share ANY content word
     with its own top-retrieved chunk? A genuinely on-topic query almost
     always will; a query that scored well by embedding-space coincidence
     but shares zero vocabulary is a red flag worth catching independently.

Requiring both signals to agree makes this more robust than either alone,
and is cheap enough to run on every request (no extra model calls).
"""

from dataclasses import dataclass
from .stopwords import content_words


@dataclass
class TopicCheckResult:
    passed: bool
    reason: str | None = None


class OffTopicDetector:
    def __init__(self, min_retrieval_score: float = 0.15, require_lexical_overlap: bool = True):
        self.min_retrieval_score = min_retrieval_score
        self.require_lexical_overlap = require_lexical_overlap

    def check(self, query: str, retrieval_scores: list[float], top_chunk_text: str | None) -> TopicCheckResult:
        if not retrieval_scores:
            return TopicCheckResult(passed=False, reason="No chunks retrieved at all.")

        top_score = max(retrieval_scores)
        if top_score < self.min_retrieval_score:
            return TopicCheckResult(
                passed=False,
                reason=f"Top retrieval score {top_score:.3f} below threshold {self.min_retrieval_score}.",
            )

        if self.require_lexical_overlap and top_chunk_text:
            query_words = content_words(query)
            chunk_words = content_words(top_chunk_text)
            if query_words and not (query_words & chunk_words):
                return TopicCheckResult(
                    passed=False,
                    reason=(
                        "Query shares no vocabulary with its own top-retrieved chunk -- "
                        "likely an embedding-space false positive rather than a real topic match."
                    ),
                )

        return TopicCheckResult(passed=True)
