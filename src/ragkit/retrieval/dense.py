"""Dense retrieval with a pluggable embedder.

* ``hashing``  - offline, deterministic, zero-download: signed feature hashing of word uni/bi-grams
  and character 4-grams. Captures sub-word / morphological similarity ("refundable" ~ "refund")
  that exact-match BM25 misses. It is NOT a semantic model.
* ``sentence-transformers`` - real semantic embeddings (config ``retrieval.embedder``).
  Needs ``pip install -e .[dense]`` and a one-off model download.
"""
from __future__ import annotations

import hashlib
import math
from typing import Protocol

import numpy as np

from .text import tokenize


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _h(self, feat: str) -> tuple[int, float]:
        d = hashlib.blake2b(feat.encode(), digest_size=8).digest()
        v = int.from_bytes(d, "big")
        return v % self.dim, 1.0 if (v >> 63) & 1 else -1.0

    def encode(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            toks = tokenize(t)
            feats = list(toks) + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
            for w in toks:
                padded = f"#{w}#"
                feats += [padded[i:i + 4] for i in range(max(1, len(padded) - 3))]
            counts: dict[str, int] = {}
            for f in feats:
                counts[f] = counts.get(f, 0) + 1
            for f, c in counts.items():
                i, sgn = self._h(f)
                m[r, i] += sgn * (1 + math.log(c))
            n = np.linalg.norm(m[r])
            if n:
                m[r] /= n
        return m


class STEmbedder:
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import
        self.model = SentenceTransformer(model)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def make_embedder(kind: str, st_model: str) -> Embedder:
    if kind == "sentence-transformers":
        return STEmbedder(st_model)
    return HashingEmbedder()


class DenseIndex:
    def __init__(self, embedder: Embedder, docs: list[str]):
        self.embedder = embedder
        self.matrix = embedder.encode(docs)

    def scores(self, query: str) -> np.ndarray:
        return self.matrix @ self.embedder.encode([query])[0]

    def top_k(self, query: str, k: int, sc: np.ndarray | None = None) -> list[tuple[int, float]]:
        sc = self.scores(query) if sc is None else sc
        order = np.argsort(-sc)[:k]
        return [(int(i), float(sc[i])) for i in order if sc[i] > 0]
