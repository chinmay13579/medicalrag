"""Per-request stage tracing.

Always on and dependency-free: ``Trace.span(name)`` records wall-clock timings that feed the
Prometheus stage histogram, the in-memory trace store and the JSON log line.

Optionally each span is mirrored to OpenTelemetry (``observability.otel: true`` plus
``pip install .[otel]``). Span attributes are restricted to non-identifying scalars: counts, status
codes and chunk ids; never text from the query, the documents or the answer.
"""
from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

_SAFE = (int, float, bool)


@dataclass
class SpanRecord:
    name: str
    start_ms: float          # offset from trace start
    duration_ms: float
    attrs: dict[str, Any] = field(default_factory=dict)


class Trace:
    def __init__(self, trace_id: str, tracer: Any = None):
        self.trace_id = trace_id
        self.spans: list[SpanRecord] = []
        self._t0 = time.perf_counter()
        self._tracer = tracer

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[dict]:
        """Time a stage. The yielded dict may be filled with extra *non-identifying* attributes."""
        extra: dict[str, Any] = dict(attrs)
        start = time.perf_counter()
        otel_cm = self._tracer.start_as_current_span(f"ragkit.{name}") if self._tracer else None
        otel_span = otel_cm.__enter__() if otel_cm else None
        try:
            yield extra
        finally:
            dur = (time.perf_counter() - start) * 1e3
            self.spans.append(SpanRecord(name, round((start - self._t0) * 1e3, 3), round(dur, 3), extra))
            if otel_span is not None:
                otel_span.set_attribute("ragkit.trace_id", self.trace_id)
                for k, v in extra.items():
                    if isinstance(v, _SAFE) or (isinstance(v, str) and len(v) <= 40):
                        otel_span.set_attribute(f"ragkit.{k}", v)
                otel_cm.__exit__(None, None, None)

    def add(self, name: str, duration_ms: float, **attrs: Any) -> None:
        """Record a sub-stage that was timed elsewhere (e.g. inside the retriever)."""
        self.spans.append(SpanRecord(name, round((time.perf_counter() - self._t0) * 1e3, 3),
                                     round(duration_ms, 3), dict(attrs)))

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1e3, 3)

    def stage_ms(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for s in self.spans:
            out[s.name] = round(out.get(s.name, 0.0) + s.duration_ms, 3)
        return out


def make_tracer(enabled: bool) -> Any:
    """Return an OpenTelemetry tracer, or None when disabled / not installed."""
    if not enabled:
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except ImportError:
        return None
    provider = TracerProvider(resource=Resource.create({"service.name": "ragkit"}))
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    return trace.get_tracer("ragkit")
