"""Okapi BM25 (dependency-free, ~40 lines) over the chunk corpus."""
from __future__ import annotations

import math
from collections import Counter

from .text import tokenize


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.dl = [sum(c.values()) for c in self.tf]
        self.avgdl = (sum(self.dl) / len(self.dl)) if docs else 0.0
        n = len(docs)
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, dl in zip(self.tf, self.dl, strict=True):
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        sc = self.scores(query)
        order = sorted(range(len(sc)), key=lambda i: -sc[i])[:k]
        return [(i, sc[i]) for i in order if sc[i] > 0]
