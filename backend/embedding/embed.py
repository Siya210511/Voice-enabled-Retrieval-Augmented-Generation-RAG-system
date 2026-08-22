"""
Wraps a multilingual sentence-embedding model (needed since MSMARCO-XI mixes
Hindi/English). Falls back to a deterministic hash-based pseudo-embedding if
sentence-transformers isn't installed, so this module -- and everything that
imports it -- stays importable and testable in any environment, including
CI or a teammate's machine mid-setup.

DO NOT ship the fallback to production retrieval quality -- it exists so
the pipeline never hard-crashes and so you can unit-test indexing/retrieval
logic without downloading a model. Always check `embedder.using_real_model`
before trusting retrieval results.
"""

import hashlib
import numpy as np

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
FALLBACK_DIM = 384  # matches MiniLM-L12-v2's real output dim, so index code doesn't branch


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self.using_real_model = False
        self._load()

    def _load(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self.using_real_model = True
        except Exception as e:
            print(f"[Embedder] Falling back to pseudo-embeddings (real model unavailable: {e})")
            self._model = None
            self.using_real_model = False

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, FALLBACK_DIM), dtype="float32")

        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )
            return np.asarray(embeddings, dtype="float32")

        return self._fallback_encode(texts, normalize)

    def _fallback_encode(self, texts: list[str], normalize: bool) -> np.ndarray:
        """Deterministic pseudo-embedding: hash word n-grams into a fixed-size
        vector. Not semantically meaningful -- purely for keeping the pipeline
        runnable/testable without the real model installed."""
        vectors = np.zeros((len(texts), FALLBACK_DIM), dtype="float32")
        for i, text in enumerate(texts):
            for word in text.lower().split():
                h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
                vectors[i, h % FALLBACK_DIM] += 1.0
        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
        return vectors

    @property
    def dim(self) -> int:
        if self.using_real_model:
            return self._model.get_sentence_embedding_dimension()
        return FALLBACK_DIM
