"""
Strategy 2: Semantic chunking.

Instead of cutting text every N words, we split sentences into groups and
measure the embedding similarity between consecutive sentences. A big drop
in similarity signals a topic shift -- that's where we cut. This keeps each
chunk about one coherent idea, which improves retrieval precision on
passages that ramble across sub-topics (common in MSMARCO-style passages).

Falls back to a simple sentence-count grouping if no embedding model is
available, so the pipeline never hard-fails in an offline/CI environment.
"""

import re
from .base import BaseChunker, Chunk

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+")  # includes Hindi danda '।' for Indic text


def split_sentences(text: str) -> list[str]:
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


class SemanticChunker(BaseChunker):
    name = "semantic"

    def __init__(
        self,
        embed_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        similarity_threshold: float = 0.55,
        min_sentences_per_chunk: int = 2,
        max_sentences_per_chunk: int = 8,
    ):
        self.similarity_threshold = similarity_threshold
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.max_sentences_per_chunk = max_sentences_per_chunk
        self._model = None
        self._embed_model_name = embed_model_name

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._embed_model_name)
            except Exception:
                self._model = False  # sentinel: embeddings unavailable, use fallback
        return self._model

    def chunk_document(self, doc_id: str, text: str, **kwargs) -> list[Chunk]:
        sentences = split_sentences(text)
        if not sentences:
            return []

        model = self._get_model()

        if model:
            groups = self._group_by_similarity(sentences, model)
        else:
            groups = self._group_fallback(sentences)

        chunks: list[Chunk] = []
        cursor = 0
        for i, group in enumerate(groups):
            chunk_text = " ".join(group)
            start_char = text.find(group[0], cursor)
            if start_char == -1:
                start_char = cursor
            end_char = start_char + len(chunk_text)
            cursor = end_char

            chunks.append(
                Chunk(
                    text=chunk_text,
                    doc_id=doc_id,
                    chunk_index=i,
                    strategy=self.name,
                    start_char=start_char,
                    end_char=end_char,
                    extra={"num_sentences": len(group), "used_embeddings": bool(model)},
                )
            )
        return chunks

    def _group_by_similarity(self, sentences: list[str], model) -> list[list[str]]:
        import numpy as np

        embeddings = model.encode(sentences, normalize_embeddings=True)
        groups: list[list[str]] = [[sentences[0]]]

        for i in range(1, len(sentences)):
            sim = float(np.dot(embeddings[i - 1], embeddings[i]))
            same_topic = sim >= self.similarity_threshold
            current_group = groups[-1]

            if same_topic and len(current_group) < self.max_sentences_per_chunk:
                current_group.append(sentences[i])
            else:
                # avoid tiny orphan chunks by merging back if under the minimum
                if len(current_group) < self.min_sentences_per_chunk and groups:
                    current_group.append(sentences[i])
                else:
                    groups.append([sentences[i]])

        return groups

    def _group_fallback(self, sentences: list[str]) -> list[list[str]]:
        """No embedding model available: group by a fixed sentence count instead
        of failing. Keeps the pipeline runnable end-to-end in any environment."""
        size = self.max_sentences_per_chunk
        return [sentences[i:i + size] for i in range(0, len(sentences), size)]
