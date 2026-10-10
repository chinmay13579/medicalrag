"""Static checks for deploy/ files (promtool and docker are not required).

Guards against the classic failure: a dashboard or alert referencing a metric or label that the code
no longer exports, so it silently shows nothing.
"""
import json
import re
from pathlib import Path

import yaml

from ragkit.observability import Metrics

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"

METRICS = Metrics()
# sample/series name -> label names the collector really has
SERIES: dict[str, set[str]] = {}
for name, coll in METRICS.registry._names_to_collectors.items():       # noqa: SLF001
    SERIES[name] = set(getattr(coll, "_labelnames", ()))
for fam in list(SERIES):                                                   # histogram helper series
    for suf in ("_bucket", "_sum", "_count"):
        if fam.endswith(("duration_seconds", "stage_duration_seconds", "tokens", "chunks", "score",
                         "groundedness")):
            SERIES.setdefault(fam + suf, SERIES[fam] | ({"le"} if suf == "_bucket" else set()))


def exprs_from_alerts() -> list[str]:
    doc = yaml.safe_load((DEPLOY / "observability/alerts.yml").read_text(encoding="utf-8"))
    return [r["expr"] for g in doc["groups"] for r in g["rules"]]


def exprs_from_dashboard() -> list[str]:
    dash = json.loads((DEPLOY / "grafana/dashboards/ragkit-overview.json").read_text(encoding="utf-8"))
    return [t["expr"] for p in dash["panels"] for t in p["targets"]]


def check(expr: str) -> list[str]:
    problems = []
    assert expr.count("(") == expr.count(")") and expr.count("{") == expr.count("}"), expr
    for m in re.finditer(r"\b(ragkit_[a-z0-9_]+)(\{([^}]*)\})?", expr):
        name, sel = m.group(1), m.group(3) or ""
        if name not in SERIES:
            problems.append(f"unknown metric {name}")
            continue
        for lab in re.findall(r"([a-z_]+)\s*(?:=~|!~|!=|=)", sel):
            if lab not in SERIES[name]:
                problems.append(f"{name} has no label {lab}")
    return problems          # grouping labels are checked separately, see the grouping-labels test


def test_alert_rules_are_well_formed_and_reference_real_metrics():
    doc = yaml.safe_load((DEPLOY / "observability/alerts.yml").read_text(encoding="utf-8"))
    for g in doc["groups"]:
        for r in g["rules"]:
            assert {"alert", "expr", "labels", "annotations"} <= set(r), r.get("alert")
            assert r["labels"]["severity"] in {"critical", "warning", "info"}
    for e in exprs_from_alerts():
        problems = [p for p in check(e) if "up" not in p]
        assert not problems, (e, problems)


def test_dashboard_is_valid_json_and_references_real_metrics():
    dash = json.loads((DEPLOY / "grafana/dashboards/ragkit-overview.json").read_text(encoding="utf-8"))
    assert dash["uid"] and len(dash["panels"]) >= 10
    ids = [p["id"] for p in dash["panels"]]
    assert len(ids) == len(set(ids))
    for e in exprs_from_dashboard():
        assert not check(e), (e, check(e))


def test_grouping_labels_exist_on_the_grouped_metric():
    for e in exprs_from_alerts() + exprs_from_dashboard():
        names = re.findall(r"\b(ragkit_[a-z0-9_]+)", e)
        for grp in re.findall(r"by \(([^)]*)\)", e):
            for lab in (x.strip() for x in grp.split(",")):
                if lab == "le":
                    continue
                assert any(lab in SERIES[n] for n in names if n in SERIES), (e, lab)


def test_prometheus_and_compose_files_parse_and_wire_up():
    prom = yaml.safe_load((DEPLOY / "observability/prometheus.yml").read_text(encoding="utf-8"))
    assert prom["scrape_configs"][0]["metrics_path"] == "/metrics"
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert {"api", "ui", "prometheus", "grafana"} <= set(compose["services"])
    assert compose["services"]["ui"]["environment"]["RAGKIT_API_URL"] == "http://api:8000"
    assert compose["services"]["api"]["environment"]["RAGKIT_ENV"] == "prod"
    for rel in ("deploy/observability/prometheus.yml", "deploy/observability/alerts.yml",
                "deploy/grafana/provisioning", "deploy/grafana/dashboards"):
        assert (ROOT / rel).exists()
    ds = yaml.safe_load((DEPLOY / "grafana/provisioning/datasources/prometheus.yml").read_text(encoding="utf-8"))
    assert ds["datasources"][0]["uid"] == "prometheus"


def test_every_dashboard_datasource_matches_provisioned_uid():
    dash = json.loads((DEPLOY / "grafana/dashboards/ragkit-overview.json").read_text(encoding="utf-8"))
    assert {p["datasource"]["uid"] for p in dash["panels"]} == {"prometheus"}
