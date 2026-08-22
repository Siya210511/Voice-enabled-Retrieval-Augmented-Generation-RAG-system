"""
Reads each chunking strategy's output from data/chunks/{strategy}.jsonl,
embeds the chunk text, and builds+persists a VectorIndex per strategy.

Also writes a metadata store (chunk_id -> full chunk dict) alongside each
index, since the index itself only knows vectors + chunk_ids -- the actual
text and metadata (doc_id, language, is_selected, etc.) needed for grounding
generation and guardrails lives in this side file.

Usage:
    python -m embedding.build_indexes
    python -m embedding.build_indexes --strategy fixed_size
"""

import argparse
import json
import os
import time

from .embed import Embedder
from .vector_index import VectorIndex

CHUNKS_DIR = os.path.join("data", "chunks")
INDEX_DIR = os.path.join("data", "indexes")

STRATEGIES = ["fixed_size", "semantic", "metadata_aware"]


def load_chunks(strategy: str) -> list[dict]:
    path = os.path.join(CHUNKS_DIR, f"{strategy}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python -m chunking.pipeline` first."
        )
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def save_metadata_store(chunks: list[dict], strategy: str):
    os.makedirs(INDEX_DIR, exist_ok=True)
    path = os.path.join(INDEX_DIR, f"{strategy}.metadata.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return path


def build_index_for_strategy(strategy: str, embedder: Embedder) -> dict:
    chunks = load_chunks(strategy)
    if not chunks:
        print(f"[{strategy}] no chunks found, skipping.")
        return {}

    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    start = time.perf_counter()
    vectors = embedder.encode(texts)
    embed_time = time.perf_counter() - start

    index = VectorIndex(dim=embedder.dim)
    index.build(vectors, chunk_ids)

    index_path_prefix = os.path.join(INDEX_DIR, strategy)
    index.save(index_path_prefix)
    metadata_path = save_metadata_store(chunks, strategy)

    stats = {
        "strategy": strategy,
        "num_chunks": len(chunks),
        "embed_time_sec": round(embed_time, 3),
        "using_real_model": embedder.using_real_model,
        "index_path_prefix": index_path_prefix,
        "metadata_path": metadata_path,
    }
    print(f"[{strategy}] indexed {len(chunks)} chunks in {embed_time:.2f}s "
          f"(real_model={embedder.using_real_model})")
    return stats


def run(strategies: list[str] | None = None):
    strategies = strategies or STRATEGIES
    embedder = Embedder()  # shared across strategies -- model load is the expensive part

    summary = {}
    for strategy in strategies:
        try:
            summary[strategy] = build_index_for_strategy(strategy, embedder)
        except FileNotFoundError as e:
            print(f"[{strategy}] skipped: {e}")

    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(os.path.join(INDEX_DIR, "build_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=STRATEGIES, default=None,
                         help="Build only this strategy's index. Default: all.")
    args = parser.parse_args()

    run(strategies=[args.strategy] if args.strategy else None)
