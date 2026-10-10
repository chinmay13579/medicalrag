"""Read Prometheus text exposition into small, UI-friendly aggregates (no Streamlit dependency)."""
from __future__ import annotations

from dataclasses import dataclass

from prometheus_client.parser import text_string_to_metric_families


@dataclass(frozen=True)
class Sample:
    name: str
    labels: dict
    value: float


class Snapshot:
    def __init__(self, text: str):
        self.samples = [Sample(s.name, dict(s.labels), s.value)
                        for fam in text_string_to_metric_families(text) for s in fam.samples]

    def _match(self, name: str, labels: dict):
        return [s for s in self.samples
                if s.name == name and all(s.labels.get(k) == v for k, v in labels.items())]

    def total(self, name: str, **labels) -> float:
        return sum(s.value for s in self._match(name, labels))

    def by_label(self, name: str, label: str, **labels) -> dict[str, float]:
        out: dict[str, float] = {}
        for s in self._match(name, labels):
            if label in s.labels:
                out[s.labels[label]] = out.get(s.labels[label], 0.0) + s.value
        return out

    def mean(self, name: str, **labels) -> float | None:
        n = self.total(f"{name}_count", **labels)
        return self.total(f"{name}_sum", **labels) / n if n else None

    def quantile(self, name: str, q: float, **labels) -> float | None:
        """Prometheus-style histogram_quantile over cumulative `_bucket` samples."""
        buckets: dict[float, float] = {}
        for s in self._match(f"{name}_bucket", labels):
            le = float("inf") if s.labels["le"] == "+Inf" else float(s.labels["le"])
            buckets[le] = buckets.get(le, 0.0) + s.value
        if not buckets:
            return None
        edges = sorted(buckets)
        total = buckets[edges[-1]]
        if total == 0:
            return None
        rank, prev_edge, prev_cnt = q * total, 0.0, 0.0
        for edge in edges:
            cnt = buckets[edge]
            if cnt >= rank:
                if edge == float("inf"):
                    return prev_edge
                span = cnt - prev_cnt
                return prev_edge + (edge - prev_edge) * ((rank - prev_cnt) / span if span else 0.0)
            prev_edge, prev_cnt = edge, cnt
        return prev_edge
