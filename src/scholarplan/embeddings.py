"""Sentence embeddings for relevance ranking, de-duplication and the vector index.

The design chose *local* embeddings (Sentence-BERT; Reimers and Gurevych,
2019) over an embedding API: they cost nothing, do not send the corpus to a
third party, and are deterministic for a given model, which keeps the
relevance threshold meaningful from run to run. all-MiniLM-L6-v2 (384 dims)
was picked because it runs comfortably on a laptop CPU, as no GPU is
available on the development machine.

A hashing embedder is provided for the offline test suite so that tests never
download a model. It captures only word overlap, which is enough to exercise
the thresholds and ranking logic deterministically.
"""

from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache
from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) array of L2-normalised vectors."""
        ...


class HashingEmbedder:
    """Bag-of-words feature hashing: deterministic, dependency-free, offline."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]{3,}", (t or "").lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        # Imported lazily: torch takes several seconds to import and the mock
        # test-suite should not pay that cost.
        # Windows without Developer Mode cannot create symlinks; the HF cache
        # still works (it copies files), so the warning is noise in demo logs.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device="cpu")
        # sentence-transformers 6 renamed this method; support both so the
        # pinned minimum version (3.x) and current releases both work.
        get_dim = (getattr(self.model, "get_embedding_dimension", None)
                   or self.model.get_sentence_embedding_dimension)
        self.dim = int(get_dim())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


@lru_cache(maxsize=2)
def get_embedder(backend: str, model_name: str) -> Embedder:
    """Cached so the ~90 MB model is loaded once per process, not per agent step."""
    if backend == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(model_name)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity; inputs are already normalised, so a dot product suffices."""
    return a @ b.T
