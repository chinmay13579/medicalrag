from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from fastapi.testclient import TestClient  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from ragkit.api.app import create_app  # noqa: E402
from ragkit.pipeline import RagPipeline  # noqa: E402
from ragkit.ui.backend import HttpBackend, InProcessBackend  # noqa: E402
from ragkit.ui.stats import Snapshot  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "src/ragkit/ui/streamlit_app.py")

EXPO = """\
# TYPE h histogram
h_bucket{le="0.1",stage="a"} 2
h_bucket{le="0.5",stage="a"} 6
h_bucket{le="1.0",stage="a"} 10
h_bucket{le="+Inf",stage="a"} 10
h_sum{stage="a"} 4.0
h_count{stage="a"} 10
# TYPE c counter
c_total{status="ok"} 7
c_total{status="bad"} 3
"""


def test_snapshot_quantile_mean_and_labels():
    s = Snapshot(EXPO)
    assert s.by_label("c_total", "status") == {"ok": 7, "bad": 3}
    assert s.mean("h", stage="a") == pytest.approx(0.4)
    assert s.quantile("h", 0.5) == pytest.approx(0.1 + 0.4 * (5 - 2) / (6 - 2))   # interpolated in (0.1, 0.5]
    assert s.quantile("h", 0.99) == pytest.approx(0.5 + 0.5 * (9.9 - 6) / (10 - 6))
    assert s.quantile("missing", 0.5) is None


def test_http_backend_round_trip_against_real_api():
    be = HttpBackend(client=TestClient(create_app(RagPipeline())))
    a = be.ask("How long are audit logs retained?", True)
    assert a["status"] == "ok" and a["prompt"] is not None
    assert be.feedback(a["trace_id"], "up") is True
    tr = be.traces(5)
    assert tr[0]["trace_id"] == a["trace_id"] and tr[0]["feedback"] == "up:none"
    assert Snapshot(be.metrics_text()).total("ragkit_requests_total", status="ok") >= 1
    assert any(c["anonymised"] for c in be.corpus())


def test_inprocess_backend_matches_http_backend_shape():
    be = InProcessBackend(RagPipeline())
    a = be.ask("How long are audit logs retained?")
    assert set(a) >= {"trace_id", "answer", "status", "citations", "guardrails", "prompt", "timings_ms"}
    assert be.traces(1)[0]["trace_id"] == a["trace_id"]


# ---- the Streamlit app itself -------------------------------------------------------------------
@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    return at


def _button(at, label=None, key=None):
    return next(b for b in at.button if (label and b.label == label) or (key and b.key == key))


def test_app_renders_all_tabs(app):
    assert [t.label for t in app.tabs] == ["Ask", "Anonymise", "Observability", "Evaluation", "Corpus"]


def test_ask_flow_shows_answer_citations_and_prompt():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.text_input(key="question_input").set_value("How long are audit logs retained?")
    _button(at, "Ask").click().run()
    assert not at.exception, at.exception
    assert any("10 years" in m.value for m in at.markdown)
    assert any("answered" in s.value.lower() or "knowledge base" in s.value.lower() for s in at.success)
    codes = " ".join(c.value for c in at.code)
    assert "<context>" in codes and "CANARY-" in codes          # the exact pseudonymised prompt is inspectable


def test_pii_in_question_never_reaches_displayed_prompt():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.text_input(key="question_input").set_value(
        "Why did Jana Novakova (785214/4321) have missing lab results?")
    _button(at, "Ask").click().run()
    assert not at.exception, at.exception
    prompt_blocks = " ".join(c.value for c in at.code if "<question>" in c.value or "Rules:" in c.value)
    assert "<question>" in prompt_blocks
    for raw in ("Novakova", "785214"):
        assert raw not in prompt_blocks


def test_injection_example_is_refused_in_ui():
    at = AppTest.from_file(APP, default_timeout=120).run()
    _button(at, key="ex-5").click().run()
    assert not at.exception, at.exception
    assert any("refused_input" in w.value for w in at.warning)


def test_anonymise_tab_deidentifies_sample_report():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.selectbox(key="anon-sample").set_value("lab_report_hematology").run()
    _button(at, key="anon-run").click().run()
    assert not at.exception, at.exception
    text = " ".join(c.value for c in at.code)
    assert "[PATIENT]" in text and "postal area 130**" in text and "Date of birth: 1978" in text
    assert any("passed" in s.value for s in at.success)


def test_observability_tab_shows_metrics_after_traffic():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.text_input(key="question_input").set_value("How long are audit logs retained?")
    _button(at, "Ask").click().run()
    labels = {m.label for m in at.metric}
    assert {"Requests", "Latency p50", "Latency p95", "Mean top retrieval score"} <= labels
    assert not at.exception, at.exception
