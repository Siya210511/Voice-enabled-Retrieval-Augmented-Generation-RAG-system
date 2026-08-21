"""
Base types shared by every chunking strategy.

A Chunk carries not just text, but metadata about where it came from and
how it was produced. This metadata is what lets retrieval be smarter than
"nearest text blob" -- we can filter/boost by source doc, strategy, position,
language, etc. at query time.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import hashlib


@dataclass
class Chunk:
    text: str
    doc_id: str                      # id of the source document/passage in MSMARCO-XI
    chunk_index: int                 # position of this chunk within its document
    strategy: str                    # "fixed_size" | "semantic" | "metadata_aware"
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    language: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Deterministic id so the same chunk always maps to the same vector-db key."""
        raw = f"{self.doc_id}:{self.strategy}:{self.chunk_index}:{self.text[:50]}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "strategy": self.strategy,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "language": self.language,
            **self.extra,
        }


class BaseChunker:
    """All chunkers implement chunk_document(doc_id, text) -> list[Chunk]."""

    name = "base"

    def chunk_document(self, doc_id: str, text: str, **kwargs) -> list[Chunk]:
        raise NotImplementedError

    def chunk_corpus(self, documents: dict[str, str], **kwargs) -> list[Chunk]:
        """documents: {doc_id: text}. Runs chunk_document over the whole corpus."""
        all_chunks: list[Chunk] = []
        for doc_id, text in documents.items():
            if not text or not text.strip():
                continue
            all_chunks.extend(self.chunk_document(doc_id, text, **kwargs))
        return all_chunks
