"""Hybrid retriever: BM25 + dense -> Reciprocal Rank Fusion -> re-rank.

``mode`` lets the eval harness ablate each stage: bm25 | dense | hybrid | hybrid+rerank.
"""
from __future__ import annotations

import time

from ..config import Settings
from ..types import Chunk, Hit
from .bm25 import BM25
from .dense import DenseIndex, make_embedder
from .rerank import make_reranker


def index_text(c: Chunk) -> str:
    return f"{c.section}. {c.text}"   # heading path is part of the indexed text


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], settings: Settings):
        self.chunks = chunks
        self.cfg = settings.retrieval
        docs = [index_text(c) for c in chunks]
        self.bm25 = BM25(docs)
        self.dense = DenseIndex(make_embedder(self.cfg.embedder, self.cfg.st_model), docs)
        self.reranker = make_reranker(self.cfg.reranker, self.bm25.idf, self.cfg.ce_model)

    def _rrf(self, *rankings: list[tuple[int, float]]) -> dict[int, float]:
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, (idx, _) in enumerate(ranking, start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.cfg.rrf_k + rank)
        return fused

    def search(self, query: str, k: int = 5, mode: str = "hybrid+rerank",
               timings: dict | None = None) -> list[Hit]:
        """``timings`` (optional out-param) receives per-stage milliseconds for observability."""
        t0 = time.perf_counter()
        b = self.bm25.top_k(query, self.cfg.top_k_bm25)
        t1 = time.perf_counter()
        cos = self.dense.scores(query)
        d = self.dense.top_k(query, self.cfg.top_k_dense, cos)
        t2 = time.perf_counter()
        if timings is not None:
            timings.update(bm25=(t1 - t0) * 1e3, dense=(t2 - t1) * 1e3)
        if mode == "bm25":
            return [Hit(self.chunks[i], s, {"bm25": r}) for r, (i, s) in enumerate(b[:k], 1)]
        if mode == "dense":
            return [Hit(self.chunks[i], s, {"dense": r}) for r, (i, s) in enumerate(d[:k], 1)]

        fused = self._rrf(b, d)
        brank = {i: r for r, (i, _) in enumerate(b, 1)}
        drank = {i: r for r, (i, _) in enumerate(d, 1)}
        order = sorted(fused, key=lambda i: -fused[i])[: self.cfg.fused_candidates]
        hits = [Hit(self.chunks[i], fused[i],
                    {"bm25": brank.get(i), "dense": drank.get(i), "rrf": round(fused[i], 5),
                     "cos": float(cos[i])})
                for i in order]
        if mode == "hybrid":
            return hits[:k]
        t3 = time.perf_counter()
        out = self.reranker.rerank(query, hits)[:k]
        if timings is not None:
            timings["rerank"] = (time.perf_counter() - t3) * 1e3
        return out
