"""
Loads ai4bharat/MSMARCO-XI and normalizes it into the shape every chunker
expects: documents = {doc_id: text}, records = {doc_id: metadata dict}.

The HF dataset stores one row per query, with a *list* of passages attached
(MS MARCO's standard passage-ranking shape). We flatten that so each
passage becomes its own addressable document with a stable doc_id.

Run this once, offline, to produce data/raw_documents.jsonl. Everything
downstream (chunking, embedding, indexing) reads from that file so you are
not re-hitting the HF hub on every experiment.
"""

import json
import os

OUTPUT_PATH = os.path.join("data", "raw_documents.jsonl")


def load_and_flatten(split: str = "train", limit: int | None = None):
    """Returns (documents, records) ready for chunk_corpus()."""
    from datasets import load_dataset

    ds = load_dataset("ai4bharat/MSMARCO-XI", split=split)

    documents: dict[str, str] = {}
    records: dict[str, dict] = {}

    count = 0
    for row_idx, row in enumerate(ds):
        query = row.get("query") or row.get("query_text")
        query_id = row.get("query_id", row_idx)

        passages_field = row.get("passages") or row.get("passage_text") or []
        # HF MS-MARCO-style datasets often nest as {"passage_text": [...], "is_selected": [...]}
        if isinstance(passages_field, dict):
            texts = passages_field.get("passage_text", [])
            selected_flags = passages_field.get("is_selected", [0] * len(texts))
        elif isinstance(passages_field, list):
            texts = passages_field
            selected_flags = [None] * len(texts)
        else:
            texts = [str(passages_field)]
            selected_flags = [None]

        for p_idx, passage_text in enumerate(texts):
            if not passage_text:
                continue
            doc_id = f"{query_id}_{p_idx}"
            documents[doc_id] = passage_text
            records[doc_id] = {
                "query_id": query_id,
                "query": query,
                "is_selected": selected_flags[p_idx] if p_idx < len(selected_flags) else None,
            }
            count += 1
            if limit and count >= limit:
                return documents, records

    return documents, records


def save_to_jsonl(documents: dict[str, str], records: dict[str, dict], path: str = OUTPUT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for doc_id, text in documents.items():
            row = {"doc_id": doc_id, "text": text, **records.get(doc_id, {})}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_from_jsonl(path: str = OUTPUT_PATH):
    documents, records = {}, {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc_id = row.pop("doc_id")
            text = row.pop("text")
            documents[doc_id] = text
            records[doc_id] = row
    return documents, records


if __name__ == "__main__":
    # Example: pull a manageable slice first, not the whole dataset, while iterating.
    docs, recs = load_and_flatten(split="train", limit=2000)
    save_to_jsonl(docs, recs)
    print(f"Saved {len(docs)} passages to {OUTPUT_PATH}")
