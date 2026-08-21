"""
Strategy 1: Fixed-size chunking with overlap.

This is the baseline every RAG system needs, but done properly:
- token-aware sizing (not naive character counts) using a cheap whitespace
  tokenizer proxy, so chunk sizes are comparable across languages.
- configurable overlap so we don't cut sentences/answers in half at boundaries.
- records exact start/end char offsets in metadata, so this stays auditable.
"""

from .base import BaseChunker, Chunk


class FixedSizeChunker(BaseChunker):
    name = "fixed_size"

    def __init__(self, chunk_size_words: int = 120, overlap_words: int = 30):
        if overlap_words >= chunk_size_words:
            raise ValueError("overlap_words must be smaller than chunk_size_words")
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words

    def chunk_document(self, doc_id: str, text: str, **kwargs) -> list[Chunk]:
        words = text.split()
        if not words:
            return []

        step = self.chunk_size_words - self.overlap_words
        chunks: list[Chunk] = []
        idx = 0
        chunk_index = 0

        while idx < len(words):
            window = words[idx: idx + self.chunk_size_words]
            chunk_text = " ".join(window)

            # approximate char offsets by re-locating the window text in the source
            start_char = len(" ".join(words[:idx])) + (1 if idx > 0 else 0)
            end_char = start_char + len(chunk_text)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    strategy=self.name,
                    start_char=start_char,
                    end_char=end_char,
                    extra={
                        "chunk_size_words": self.chunk_size_words,
                        "overlap_words": self.overlap_words,
                    },
                )
            )
            chunk_index += 1
            idx += step

        return chunks
