from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    section: str
    text: str            # pseudonymised text: what is indexed and what may reach the LLM
    n_words: int = 0
    anonymised: bool = False   # True for de-identified clinical reports (cannot be linked to a person)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    sources: dict = field(default_factory=dict)  # e.g. {"bm25": rank, "dense": rank, "rerank": s}
