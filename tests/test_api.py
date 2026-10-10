from fastapi.testclient import TestClient

from ragkit.api.app import create_app

client = TestClient(create_app())


def test_health():
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["chunks"] > 0


def test_ask_ok_and_security_headers():
    r = client.post("/v1/ask", json={"query": "How long are audit logs retained?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and "10 years" in body["answer"]
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "content-security-policy" in r.headers
    assert body["prompt"] is None  # debug off by default


def test_validation_rejects_empty_and_oversized():
    assert client.post("/v1/ask", json={"query": ""}).status_code == 422
    assert client.post("/v1/ask", json={"query": "x" * 3000}).status_code == 422


def test_ui_served():
    assert "Platform Policy Assistant" in client.get("/").text
