"""Second-stage re-ranking over the fused candidate list.

Why re-rank: BM25 and dense scores are not comparable and RRF only uses ranks. A re-ranker looks
at (query, chunk) *jointly* and produces a calibrated 0..1 score, which we need for the
"is there enough evidence to answer at all?" threshold in the context policy.

* ``heuristic``      - idf-weighted query-term coverage + bigram/phrase match + heading match +
                       a mild multiplicative prior from the fused rank. Offline and explainable.
* ``cross-encoder``  - sentence-transformers CrossEncoder (sigmoid-squashed). Drop-in upgrade.
* ``none``           - passthrough (min-max normalised fused score), used as an eval baseline.
"""
from __future__ import annotations

import math

from ..types import Hit
from .text import tokenize


class HeuristicReranker:
    def __init__(self, idf: dict[str, float]):
        self.idf = idf
        self.default_idf = max(idf.values()) if idf else 1.0

    def _w(self, t: str) -> float:
        return self.idf.get(t, self.default_idf)

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        q = tokenize(query)
        if not q or not hits:
            return hits
        qset = set(q)
        total = sum(self._w(t) for t in qset)
        qbi = set(zip(q, q[1:], strict=False))
        max_prior = max(h.score for h in hits) or 1.0
        for h in hits:
            body = tokenize(h.chunk.text)
            head = set(tokenize(h.chunk.section))
            bset = set(body)
            cov = sum(self._w(t) for t in qset if t in bset or t in head) / total
            bi = (len(qbi & set(zip(body, body[1:], strict=False))) / len(qbi)) if qbi else 0.0
            hd = sum(self._w(t) for t in qset if t in head) / total
            prior = h.score / max_prior
            # multiplicative prior: zero lexical evidence can never be rescued by rank alone
            s = (0.6 * cov + 0.2 * bi + 0.2 * hd) * (0.8 + 0.2 * prior)
            h.sources["rerank"] = round(s, 4)
            h.score = s
        return sorted(hits, key=lambda x: -x.score)


class CrossEncoderReranker:
    def __init__(self, model: str):
        from sentence_transformers import CrossEncoder
        self.ce = CrossEncoder(model)

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        raw = self.ce.predict([(query, f"{h.chunk.section}. {h.chunk.text}") for h in hits])
        for h, r in zip(hits, raw, strict=True):
            h.score = 1 / (1 + math.exp(-float(r)))
            h.sources["rerank"] = round(h.score, 4)
        return sorted(hits, key=lambda x: -x.score)


class NoReranker:
    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        lo, hi = min(h.score for h in hits), max(h.score for h in hits)
        for h in hits:
            h.score = (h.score - lo) / (hi - lo) if hi > lo else 1.0
        return hits


def make_reranker(kind: str, idf: dict[str, float], ce_model: str):
    if kind == "cross-encoder":
        return CrossEncoderReranker(ce_model)
    if kind == "none":
        return NoReranker()
    return HeuristicReranker(idf)
