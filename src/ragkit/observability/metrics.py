"""Prometheus metrics.

Cardinality and privacy rules (enforced by tests/test_observability.py):
* labels are drawn from small closed sets (status, stage, guard, reason, PII *kind*); never from
  user input, query text, names, ids or document content;
* every Metrics instance owns its CollectorRegistry, so several pipelines (tests, workers) can
  coexist without "duplicated timeseries" errors.
"""
from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

_LAT = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
_STAGE = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
_UNIT = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

CONTENT_TYPE = CONTENT_TYPE_LATEST


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.requests = Counter(
            "ragkit_requests_total", "Questions processed, by outcome status.", ["status"], registry=r)
        self.duration = Histogram(
            "ragkit_request_duration_seconds", "End-to-end question latency.", buckets=_LAT, registry=r)
        self.stage = Histogram(
            "ragkit_stage_duration_seconds", "Latency per pipeline stage.", ["stage"],
            buckets=_STAGE, registry=r)
        self.guard = Counter(
            "ragkit_guardrail_events_total", "Guardrail interventions.", ["guard", "reason"], registry=r)
        self.dropped = Counter(
            "ragkit_context_chunks_dropped_total", "Retrieved chunks excluded by the context policy.",
            ["reason"], registry=r)
        self.masked = Counter(
            "ragkit_pii_masked_total", "Identifiers pseudonymised in user queries, by kind.",
            ["kind"], registry=r)
        self.context_tokens = Histogram(
            "ragkit_context_tokens", "Estimated tokens of retrieved context sent to the LLM.",
            buckets=(0, 25, 50, 100, 150, 250, 350, 450, 600), registry=r)
        self.context_chunks = Histogram(
            "ragkit_context_chunks", "Chunks sent to the LLM per question.",
            buckets=(0, 1, 2, 3, 4, 5), registry=r)
        self.top_score = Histogram(
            "ragkit_retrieval_top_score", "Best re-ranker score for a question (retrieval health).",
            buckets=_UNIT, registry=r)
        self.groundedness = Histogram(
            "ragkit_answer_groundedness", "Lexical groundedness of answers.", buckets=_UNIT, registry=r)
        self.llm_tokens = Counter(
            "ragkit_llm_tokens_total", "Tokens billed by the LLM provider.", ["direction"], registry=r)
        self.feedback = Counter(
            "ragkit_feedback_total", "User feedback on answers.", ["rating", "reason"], registry=r)
        self.http = Counter(
            "ragkit_http_requests_total", "HTTP requests by route template and status code.",
            ["route", "method", "code"], registry=r)
        self.index_chunks = Gauge(
            "ragkit_index_chunks", "Chunks in the index.", ["kind"], registry=r)
        self.reports = Gauge(
            "ragkit_reports_deidentified", "Clinical reports by de-identification status.",
            ["status"], registry=r)
        self.build = Gauge(
            "ragkit_build_info", "Build and configuration info (value is always 1).",
            ["version", "retrieval", "reranker", "llm"], registry=r)

    def render(self) -> bytes:
        return generate_latest(self.registry)
