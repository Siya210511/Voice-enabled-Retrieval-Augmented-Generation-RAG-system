"""
Stream MSMARCO-XI from Hugging Face WITHOUT loading entire dataset.
Process in batches, write directly to disk, never hold >1000 rows in memory.
"""

import json
import os
from pathlib import Path

OUTPUT_PATH = os.path.join("data", "raw_documents.jsonl")

def stream_and_save_dataset(
    split: str = "train",
    limit: int = 10000,  # ← CHANGE THIS: 10k-50k for demo, not 1.2M
    batch_size: int = 100
):
    """
    Stream dataset from HuggingFace and save incrementally to disk.
    Memory usage: ~50MB (only batch_size rows at a time)
    """
    from datasets import load_dataset
    
    print(f"🔄 Streaming {limit} passages from MSMARCO-XI...")
    
    # Load with streaming=True → doesn't download entire dataset
    ds = load_dataset(
        "ai4bharat/MSMARCO-XI",
        split=split,
        streaming=True  # ← KEY: Stream mode
    )
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    count = 0
    doc_id_counter = 0
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row_idx, row in enumerate(ds):
            if count >= limit:
                print(f"✅ Reached limit of {limit} passages")
                break
            
            query = row.get("query") or row.get("query_text", "")
            query_id = row.get("query_id", row_idx)
            
            # Handle passages (nested structure varies)
            passages_field = row.get("passages") or row.get("passage_text") or []
            
            if isinstance(passages_field, dict):
                texts = passages_field.get("passage_text", [])
                selected = passages_field.get("is_selected", [0] * len(texts))
            elif isinstance(passages_field, list):
                texts = passages_field
                selected = [None] * len(texts)
            else:
                texts = [str(passages_field)]
                selected = [None]
            
            # Save each passage
            for p_idx, text in enumerate(texts):
                if not text or not text.strip():
                    continue
                
                doc_id = f"doc_{doc_id_counter}"
                doc_id_counter += 1
                
                row_to_save = {
                    "doc_id": doc_id,
                    "text": text.strip(),
                    "query_id": query_id,
                    "query": query,
                    "is_selected": selected[p_idx] if p_idx < len(selected) else None,
                    "passage_idx": p_idx
                }
                
                # Write immediately (don't accumulate in memory)
                f.write(json.dumps(row_to_save, ensure_ascii=False) + "\n")
                
                count += 1
                
                # Progress every 1000
                if count % 1000 == 0:
                    print(f"  → Saved {count} passages...")
                
                if count >= limit:
                    break
    
    print(f"✅ Saved {count} passages to {OUTPUT_PATH}")
    return count


def load_from_jsonl(path: str = OUTPUT_PATH) -> tuple[dict, dict]:
    """Load from disk (never loads full dataset into memory)"""
    documents = {}
    records = {}
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc_id = row.pop("doc_id")
            text = row.pop("text")
            documents[doc_id] = text
            records[doc_id] = row
    
    return documents, records


if __name__ == "__main__":
    # For demo: use only 10,000 passages (can run on laptop)
    # For production: increase to 100,000
    stream_and_save_dataset(split="train", limit=10000)
    
    print("\n✅ Now run:")
    print("   python -m chunking.pipeline")
