"""Telemetry facade used by the pipeline and the API: metrics + trace store + structured log.

What is (not) recorded, per request
  recorded   status, stage timings, counts (chunks/tokens/masked entities), retrieval scores,
             chunk ids, guardrail reason codes, keyed query hash + length, LLM token usage
  recorded   pseudonymised query text ONLY if observability.log_queries is true (dev)
  never      raw query, answer text, document text, vault contents, free-text feedback
"""
from __future__ import annotations

import hashlib
import hmac
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

from .. import __version__
from ..config import Settings
from ..logging import get_logger
from .metrics import Metrics
from .tracing import Trace, make_tracer

log = get_logger("ragkit.request")

FEEDBACK_RATINGS = {"up", "down"}
FEEDBACK_REASONS = {"none", "wrong_answer", "missing_info", "privacy_concern", "other"}


@dataclass
class RequestRecord:
    trace_id: str
    ts: float
    status: str
    total_ms: float
    stages: dict[str, float]
    query_hash: str
    query_len: int
    query_safe: str | None = None
    masked: dict[str, int] = field(default_factory=dict)
    n_context: int = 0
    context_tokens: int = 0
    top_score: float | None = None
    retrieved: list[dict] = field(default_factory=list)      # [{"chunk_id", "score"}]
    dropped: list[dict] = field(default_factory=list)        # [{"chunk_id", "reason"}]
    guard: dict[str, list[str]] = field(default_factory=dict)  # {"input": [...], "output": [...]}
    groundedness: float | None = None
    llm_usage: dict[str, int] | None = None
    feedback: str | None = None


class Telemetry:
    def __init__(self, settings: Settings, metrics: Metrics | None = None, tracer: Any = None):
        self.cfg = settings.observability
        self._key = settings.pseudo_key.encode()
        self.metrics = metrics or Metrics()
        self.tracer = tracer if tracer is not None else make_tracer(self.cfg.otel)
        self._records: deque[RequestRecord] = deque(maxlen=self.cfg.trace_buffer)
        self._lock = threading.Lock()
        self.metrics.build.labels(
            __version__, settings.retrieval.embedder, settings.retrieval.reranker,
            settings.llm.provider).set(1)

    # ---- helpers ------------------------------------------------------------------------------
    def new_trace(self, trace_id: str) -> Trace:
        return Trace(trace_id, self.tracer)

    def query_hash(self, query: str) -> str:
        """Keyed hash: stable for grouping repeat questions, not reversible by guessing the text."""
        return hmac.new(self._key, f"q|{query}".encode(), hashlib.sha256).hexdigest()[:10]

    def set_corpus(self, chunks: list, anon_results: dict) -> None:
        n_rep = sum(1 for c in chunks if c.anonymised)
        self.metrics.index_chunks.labels("document").set(len(chunks) - n_rep)
        self.metrics.index_chunks.labels("report").set(n_rep)
        for status in ("deidentified", "needs_review"):
            self.metrics.reports.labels(status).set(
                sum(1 for r in anon_results.values() if r.status == status))

    # ---- recording ----------------------------------------------------------------------------
    def record(self, rec: RequestRecord) -> None:
        m = self.metrics
        m.requests.labels(rec.status).inc()
        m.duration.observe(rec.total_ms / 1e3)
        for stage, ms in rec.stages.items():
            m.stage.labels(stage).observe(ms / 1e3)
        for guard, reasons in rec.guard.items():
            for reason in reasons:
                m.guard.labels(guard, reason).inc()
        for d in rec.dropped:
            m.dropped.labels(d["reason"]).inc()
        for kind, n in rec.masked.items():
            m.masked.labels(kind).inc(n)
        if rec.status in ("ok", "no_evidence", "blocked_output"):
            m.context_tokens.observe(rec.context_tokens)
            m.context_chunks.observe(rec.n_context)
        if rec.top_score is not None:
            m.top_score.observe(max(0.0, min(1.0, rec.top_score)))
        if rec.groundedness is not None:
            m.groundedness.observe(rec.groundedness)
        if rec.llm_usage:
            for direction, n in rec.llm_usage.items():
                m.llm_tokens.labels(direction).inc(n)
        with self._lock:
            self._records.append(rec)
        payload = asdict(rec)
        if not self.cfg.log_queries:
            payload.pop("query_safe", None)
        log.info("request", extra={"ctx": payload})

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            items = list(self._records)[-limit:]
        return [asdict(r) for r in reversed(items)]

    def add_feedback(self, trace_id: str, rating: str, reason: str = "none") -> bool:
        if rating not in FEEDBACK_RATINGS or reason not in FEEDBACK_REASONS:
            raise ValueError("invalid feedback code")
        with self._lock:
            for r in self._records:
                if r.trace_id == trace_id:
                    r.feedback = f"{rating}:{reason}"
                    break
            else:
                return False
        self.metrics.feedback.labels(rating, reason).inc()
        return True

    @staticmethod
    def now() -> float:
        return time.time()
