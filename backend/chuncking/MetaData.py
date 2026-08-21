"""
Strategy 3: Metadata-aware chunking.

Wraps another chunker (fixed-size or semantic) and enriches every resulting
chunk with metadata pulled from the source record itself -- not just derived
from the text. This is what lets retrieval do metadata filtering later
(e.g. "only search Hindi passages", "boost chunks from short documents",
"exclude chunks that are mostly numbers/tables").

For MSMARCO-XI specifically, each HF dataset row typically carries a query,
a set of passages, and per-passage `is_selected` / language info. We surface
that here so it travels with the chunk into the vector DB payload.
"""

from .base import BaseChunker, Chunk


class MetadataAwareChunker(BaseChunker):
    name = "metadata_aware"

    def __init__(self, base_chunker: BaseChunker):
        self.base_chunker = base_chunker

    def chunk_document(self, doc_id: str, text: str, record: dict | None = None, **kwargs) -> list[Chunk]:
        base_chunks = self.base_chunker.chunk_document(doc_id, text, **kwargs)
        record = record or {}

        doc_length_words = len(text.split())
        language = self._guess_language(text, record)

        for chunk in base_chunks:
            chunk.strategy = self.name
            chunk.language = language
            chunk.extra.update({
                "wrapped_strategy": self.base_chunker.name,
                "doc_length_words": doc_length_words,
                "is_selected": record.get("is_selected"),
                "query_id": record.get("query_id"),
                "source_query": record.get("query"),
                "is_short_doc": doc_length_words < 40,
            })
        return base_chunks

    @staticmethod
    def _guess_language(text: str, record: dict) -> str:
        if record.get("language"):
            return record["language"]
        # crude Devanagari check as a fallback signal for MSMARCO-XI's Indic subset
        devanagari_chars = sum(1 for ch in text if "\u0900" <= ch <= "\u097F")
        return "hi" if devanagari_chars > len(text) * 0.2 else "en"

    def chunk_corpus(self, documents: dict[str, str], records: dict[str, dict] | None = None, **kwargs) -> list[Chunk]:
        records = records or {}
        all_chunks: list[Chunk] = []
        for doc_id, text in documents.items():
            if not text or not text.strip():
                continue
            all_chunks.extend(
                self.chunk_document(doc_id, text, record=records.get(doc_id), **kwargs)
            )
        return all_chunks
