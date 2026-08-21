"""
Runs every chunking strategy over the corpus and writes each strategy's
output to its own file under data/chunks/. This lets you A/B the strategies
downstream (embed + index each separately, compare retrieval quality)
instead of committing to one chunking approach blind.

Usage:
    python -m chunking.pipeline
"""

import json
import os
import time

from .base import Chunk
from .fixed_size import FixedSizeChunker
from .semantic import SemanticChunker
from .metadata_aware import MetadataAwareChunker
from .load_data import load_from_jsonl, OUTPUT_PATH

CHUNKS_DIR = os.path.join("data", "chunks")


def build_strategies() -> dict[str, object]:
    fixed = FixedSizeChunker(chunk_size_words=120, overlap_words=30)
    semantic = SemanticChunker(similarity_threshold=0.55)
    metadata_aware = MetadataAwareChunker(base_chunker=fixed)

    return {
        "fixed_size": fixed,
        "semantic": semantic,
        "metadata_aware": metadata_aware,
    }


def save_chunks(chunks: list[Chunk], strategy_name: str):
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    path = os.path.join(CHUNKS_DIR, f"{strategy_name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    return path


def run(raw_documents_path: str = OUTPUT_PATH):
    if not os.path.exists(raw_documents_path):
        raise FileNotFoundError(
            f"{raw_documents_path} not found. Run `python -m chunking.load_data` first."
        )

    documents, records = load_from_jsonl(raw_documents_path)
    print(f"Loaded {len(documents)} source documents.")

    strategies = build_strategies()
    summary = {}

    for name, chunker in strategies.items():
        start = time.perf_counter()
        if name == "metadata_aware":
            chunks = chunker.chunk_corpus(documents, records=records)
        else:
            chunks = chunker.chunk_corpus(documents)
        elapsed = time.perf_counter() - start

        path = save_chunks(chunks, name)
        summary[name] = {
            "num_chunks": len(chunks),
            "avg_chunk_words": (
                sum(len(c.text.split()) for c in chunks) / len(chunks) if chunks else 0
            ),
            "build_time_sec": round(elapsed, 3),
            "output_path": path,
        }
        print(f"[{name}] {len(chunks)} chunks -> {path} ({elapsed:.2f}s)")

    summary_path = os.path.join(CHUNKS_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary written to {summary_path}")

    return summary


if __name__ == "__main__":
    run()
