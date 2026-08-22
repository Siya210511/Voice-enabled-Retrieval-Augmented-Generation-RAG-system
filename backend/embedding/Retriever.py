"""
Loads a pre-built index + metadata store for one strategy and exposes
top-k retrieval, with:
  - optional metadata filtering (language, is_selected) applied post-search
    by over-fetching then filtering, so filtering doesn't require a
    different index type.
  - per-call latency measurement, since retrieval latency is exactly the
    number the submission's P50/P70/P100 analytics needs.

This is intentionally the *only* file the harness/RAG pipeline should need
to import from the embedding module.
"""

import os
import time
from dataclasses import dataclass

from .embed import Embedder
from .build_indexes import INDEX_DIR
from .vector_index import VectorIndex

import json


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    doc_id: str
    metadata: dict


class Retriever:
    def __init__(self, strategy: str = "metadata_aware", embedder: Embedder | None = None):
        self.strategy = strategy
        self.embedder = embedder or Embedder()

        index_prefix = os.path.join(INDEX_DIR, strategy)
        self.index = VectorIndex.load(index_prefix, dim=self.embedder.dim)
        self.metadata_by_id = self._load_metadata(strategy)

    @staticmethod
    def _load_metadata(strategy: str) -> dict[str, dict]:
        path = os.path.join(INDEX_DIR, f"{strategy}.metadata.jsonl")
        metadata = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                metadata[row["chunk_id"]] = row
        return metadata

    def retrieve(
        self,
        query: str,
        k: int = 5,
        language: str | None = None,
        only_selected: bool = False,
    ) -> tuple[list[RetrievedChunk], float]:
        """Returns (results, latency_seconds). Latency covers embedding the
        query + vector search only -- not generation -- matching the spec's
        "chunking + vector DB retrieval" latency target."""

        start = time.perf_counter()

        query_vec = self.embedder.encode([query])[0]

        # over-fetch when filtering so post-filter results still fill k
        fetch_k = k * 4 if (language or only_selected) else k
        raw_results = self.index.search(query_vec, k=fetch_k)

        results = []
        for chunk_id, score in raw_results:
            meta = self.metadata_by_id.get(chunk_id)
            if meta is None:
                continue
            if language and meta.get("language") != language:
                continue
            if only_selected and not meta.get("is_selected"):
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=meta["text"],
                    score=score,
                    doc_id=meta["doc_id"],
                    metadata=meta,
                )
            )
            if len(results) >= k:
                break

        latency = time.perf_counter() - start
        return results, latency
