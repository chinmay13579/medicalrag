"""Backends for the Streamlit app: call the pipeline in-process (demo) or the HTTP API (deployment).

In a real deployment the UI container only talks HTTP, never imports the pipeline, so the UI holds no
index, no pseudonymisation key and no documents.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Protocol


class Backend(Protocol):
    label: str

    def ask(self, query: str, debug: bool = True) -> dict: ...
    def traces(self, limit: int = 100) -> list[dict]: ...
    def metrics_text(self) -> str: ...
    def feedback(self, trace_id: str, rating: str, reason: str = "none") -> bool: ...
    def corpus(self) -> list[dict]: ...


class InProcessBackend:
    label = "in-process pipeline"

    def __init__(self, pipeline: Any):
        self.p = pipeline

    def ask(self, query: str, debug: bool = True) -> dict:
        return asdict(self.p.answer(query, debug=debug))

    def traces(self, limit: int = 100) -> list[dict]:
        return self.p.telemetry.recent(limit)

    def metrics_text(self) -> str:
        return self.p.telemetry.metrics.render().decode()

    def feedback(self, trace_id: str, rating: str, reason: str = "none") -> bool:
        return self.p.telemetry.add_feedback(trace_id, rating, reason)

    def corpus(self) -> list[dict]:
        return [{"chunk_id": c.chunk_id, "doc_id": c.doc_id, "section": c.section,
                 "n_words": c.n_words, "anonymised": c.anonymised, "text": c.text}
                for c in self.p.chunks]


class HttpBackend:
    def __init__(self, base_url: str = "", token: str | None = None, client: Any = None):
        import httpx
        self.c = client or httpx.Client(base_url=base_url, timeout=30)
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.label = f"HTTP API ({base_url or 'test client'})"

    def ask(self, query: str, debug: bool = True) -> dict:
        r = self.c.post("/v1/ask", json={"query": query, "debug": debug})
        r.raise_for_status()
        return r.json()

    def traces(self, limit: int = 100) -> list[dict]:
        r = self.c.get("/v1/traces", params={"limit": limit}, headers=self.headers)
        r.raise_for_status()
        return r.json()["traces"]

    def metrics_text(self) -> str:
        r = self.c.get("/metrics", headers=self.headers)
        r.raise_for_status()
        return r.text

    def feedback(self, trace_id: str, rating: str, reason: str = "none") -> bool:
        r = self.c.post("/v1/feedback", json={"trace_id": trace_id, "rating": rating, "reason": reason})
        r.raise_for_status()
        return r.json()["recorded"]

    def corpus(self) -> list[dict]:
        r = self.c.get("/v1/corpus", headers=self.headers)
        r.raise_for_status()
        return r.json()["chunks"]
