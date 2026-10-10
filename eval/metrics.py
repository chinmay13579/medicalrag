"""Pure metric functions (no I/O) so they are unit-testable.

Relevance is span-based: a chunk is relevant to a question if it contains one of the question's
`gold_spans` (normalised substring match). Labels therefore survive re-chunking experiments.
"""
from __future__ import annotations

import math
import re


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def is_relevant(chunk_text: str, gold_spans: list[str]) -> bool:
    t = norm(chunk_text)
    return any(norm(g) in t for g in gold_spans)


def rel_vector(chunk_texts: list[str], gold_spans: list[str]) -> list[int]:
    return [int(is_relevant(t, gold_spans)) for t in chunk_texts]


def hit_at_k(rel: list[int], k: int) -> float:
    return float(any(rel[:k]))


def span_recall_at_k(chunk_texts: list[str], gold_spans: list[str], k: int) -> float:
    """Fraction of gold spans covered by the top-k chunks."""
    if not gold_spans:
        return 1.0
    joined = " ".join(norm(t) for t in chunk_texts[:k])
    return sum(norm(g) in joined for g in gold_spans) / len(gold_spans)


def mrr(rel: list[int]) -> float:
    for i, r in enumerate(rel, 1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rel: list[int], k: int, n_relevant_total: int) -> float:
    dcg = sum(r / math.log2(i + 1) for i, r in enumerate(rel[:k], 1))
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(n_relevant_total, k) + 1))
    return dcg / ideal if ideal else 0.0


def keyword_coverage(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    a = norm(answer)
    return sum(norm(k) in a for k in keywords) / len(keywords)


def contains_any(text: str, needles: list[str]) -> list[str]:
    """Whole-token match (so 'Tom' does not fire on 'cusTOMer')."""
    return [n for n in needles
            if re.search(rf"(?<!\w){re.escape(n)}(?!\w)", text, re.I)]


def mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else float("nan")
