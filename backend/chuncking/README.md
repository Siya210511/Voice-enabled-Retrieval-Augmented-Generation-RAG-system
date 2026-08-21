# Chunking Module

Implements three chunking strategies over the `ai4bharat/MSMARCO-XI` corpus,
each producing its own chunk set so you can compare retrieval quality
across strategies rather than committing to one blind.

## Strategies

1. **Fixed-size (`fixed_size.py`)** — word-count windows with configurable
   overlap. Baseline; fast; deterministic offsets.
2. **Semantic (`semantic.py`)** — splits into sentences, groups consecutive
   sentences by embedding similarity, cuts at topic shifts. Falls back to
   fixed sentence-count grouping if `sentence-transformers` isn't installed,
   so the pipeline never hard-fails.
3. **Metadata-aware (`metadata_aware.py`)** — wraps another chunker (fixed-size
   by default) and attaches source metadata: query_id, original query text,
   `is_selected` flag from MS MARCO, guessed language (Hindi/English via
   Devanagari ratio), and doc-length flags. This is what enables filtered
   retrieval later (e.g. "only search Hindi passages").

Every chunk is a `Chunk` dataclass with a deterministic `chunk_id` (hash of
doc_id + strategy + position + text prefix), so the same chunk always maps
to the same vector-DB key across reruns.

## How to run

```bash
# 1. Install deps
pip install datasets sentence-transformers numpy

# 2. Pull and flatten the dataset (does this once, writes to disk)
python -m chunking.load_data

# 3. Run all three chunking strategies, write outputs to data/chunks/*.jsonl
python -m chunking.pipeline
```

Output: `data/chunks/fixed_size.jsonl`, `data/chunks/semantic.jsonl`,
`data/chunks/metadata_aware.jsonl`, plus `data/chunks/summary.json` with
chunk counts / avg size / build time per strategy — useful evidence for
your submission write-up on chunking design decisions.

## Notes on MSMARCO-XI shape

The loader (`load_data.py`) assumes the standard MS MARCO passage-ranking
shape: one row per query, with a nested `passages` field containing
`passage_text` (list) and `is_selected` (list of 0/1). If the actual
`ai4bharat/MSMARCO-XI` schema differs once you inspect it with
`ds.features`, adjust the field names at the top of `load_and_flatten()` —
the rest of the pipeline doesn't care about the source schema.

## Next steps (not in this module)

- Embed each strategy's chunks (e.g. multilingual sentence-transformers
  model) and load into FAISS/Chroma — one index per strategy, or one index
  with `strategy` as a metadata filter.
- Latency instrumentation should wrap retrieval only, separate from
  chunking (chunking is a one-time offline cost, not part of per-query
  latency).

