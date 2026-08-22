"""
Thin wrapper around a FAISS index, with a numpy brute-force fallback when
faiss isn't installed. Both backends expose the same .search(query_vec, k)
interface, so the retriever never needs to know which one is active.

Uses IndexFlatIP (inner product) over normalized vectors == cosine similarity.
Flat/exact search is intentional: MSMARCO-XI-derived chunk counts are small
enough (tens of thousands, not tens of millions) that exact search comfortably
fits the <200ms retrieval budget without needing an approximate index (IVF/HNSW).
Revisit this if the corpus grows past ~500k vectors.
"""

import json
import os
import numpy as np


class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.chunk_ids: list[str] = []
        self._faiss_index = None
        self._flat_vectors = None  # numpy fallback storage
        self._using_faiss = False
        self._init_backend()

    def _init_backend(self):
        try:
            import faiss  # noqa: F401
            self._using_faiss = True
        except Exception:
            self._using_faiss = False
            print("[VectorIndex] faiss not installed -- using numpy brute-force fallback.")

    def build(self, vectors: np.ndarray, chunk_ids: list[str]):
        assert vectors.shape[0] == len(chunk_ids)
        self.chunk_ids = chunk_ids

        if self._using_faiss:
            import faiss
            index = faiss.IndexFlatIP(self.dim)
            index.add(vectors.astype("float32"))
            self._faiss_index = index
        else:
            self._flat_vectors = vectors.astype("float32")

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        query_vec = query_vec.reshape(1, -1).astype("float32")

        if self._using_faiss:
            scores, idxs = self._faiss_index.search(query_vec, k)
            results = [
                (self.chunk_ids[idx], float(score))
                for idx, score in zip(idxs[0], scores[0])
                if idx != -1
            ]
        else:
            sims = self._flat_vectors @ query_vec[0]
            top_k_idx = np.argsort(-sims)[:k]
            results = [(self.chunk_ids[i], float(sims[i])) for i in top_k_idx]

        return results

    def save(self, path_prefix: str):
        os.makedirs(os.path.dirname(path_prefix) or ".", exist_ok=True)

        with open(f"{path_prefix}.ids.json", "w", encoding="utf-8") as f:
            json.dump(self.chunk_ids, f)

        if self._using_faiss:
            import faiss
            faiss.write_index(self._faiss_index, f"{path_prefix}.faiss")
        else:
            np.save(f"{path_prefix}.vectors.npy", self._flat_vectors)

    @classmethod
    def load(cls, path_prefix: str, dim: int):
        instance = cls(dim=dim)

        with open(f"{path_prefix}.ids.json", "r", encoding="utf-8") as f:
            instance.chunk_ids = json.load(f)

        if instance._using_faiss and os.path.exists(f"{path_prefix}.faiss"):
            import faiss
            instance._faiss_index = faiss.read_index(f"{path_prefix}.faiss")
        else:
            instance._using_faiss = False
            instance._flat_vectors = np.load(f"{path_prefix}.vectors.npy")

        return instance
