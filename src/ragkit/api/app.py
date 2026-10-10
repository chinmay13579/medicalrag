"""FastAPI service + minimal web UI.

GET  /               demo UI
GET  /healthz        liveness
GET  /readyz         readiness (index loaded)
GET  /metrics        Prometheus exposition (no PII; restrict at the network level or set RAGKIT_ADMIN_TOKEN)
POST /v1/ask         {"query": "...", "debug": false}                      -> Answer
POST /v1/feedback    {"trace_id", "rating": up|down, "reason": <code>}     -> {"recorded": bool}
GET  /v1/traces      recent request records (admin)       no query text unless observability.log_queries
GET  /v1/corpus      indexed chunk metadata + protected text (admin)

"Admin" endpoints require ``Authorization: Bearer $RAGKIT_ADMIN_TOKEN``. If the token is unset they
are open in dev and disabled (404) when RAGKIT_ENV=prod.
"""
from __future__ import annotations

import hmac
import os
import time
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from ..observability import CONTENT_TYPE
from ..pipeline import RagPipeline

STATIC = Path(__file__).parent / "static"
RATE_LIMIT, WINDOW = 30, 60  # requests / seconds / client (use an API gateway in real prod)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    debug: bool = False


class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., pattern=r"^[a-f0-9]{12}$")
    rating: Literal["up", "down"]
    # closed set of reason codes: free-text comments could contain patient data
    reason: Literal["none", "wrong_answer", "missing_info", "privacy_concern", "other"] = "none"


def _is_prod() -> bool:
    return os.environ.get("RAGKIT_ENV", "dev") == "prod"


def _check_admin(request: Request, *, open_when_unset: bool) -> None:
    token = os.environ.get("RAGKIT_ADMIN_TOKEN")
    if not token:
        if open_when_unset:
            return
        raise HTTPException(404, "not found")
    supplied = request.headers.get("authorization", "")
    if not hmac.compare_digest(supplied, f"Bearer {token}"):
        raise HTTPException(401, "unauthorised", headers={"WWW-Authenticate": "Bearer"})


def create_app(pipeline: RagPipeline | None = None) -> FastAPI:
    app = FastAPI(title="ragkit", version="0.1.0", docs_url="/docs")
    app.state.pipe = pipeline or RagPipeline()
    telemetry = app.state.pipe.telemetry
    hits: dict[str, deque] = defaultdict(deque)

    @app.middleware("http")
    async def security_headers_and_metrics(request: Request, call_next):
        resp = await call_next(request)
        route = request.scope.get("route")
        telemetry.metrics.http.labels(
            route.path if route is not None else "unmatched", request.method, str(resp.status_code)).inc()
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'")
        return resp

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "chunks": len(app.state.pipe.chunks)}

    @app.get("/readyz")
    def readyz():
        n = len(app.state.pipe.chunks)
        if n == 0:
            raise HTTPException(503, "index empty")
        return {"status": "ready", "chunks": n}

    @app.get("/metrics")
    def metrics(request: Request):
        _check_admin(request, open_when_unset=True)
        return Response(telemetry.metrics.render(), media_type=CONTENT_TYPE)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.post("/v1/ask")
    def ask(req: AskRequest, request: Request):
        ip = request.client.host if request.client else "unknown"
        now, q = time.time(), hits[ip]
        while q and now - q[0] > WINDOW:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            raise HTTPException(429, "rate limit exceeded")
        q.append(now)
        ans = app.state.pipe.answer(req.query, debug=req.debug and not _is_prod())
        return JSONResponse(asdict(ans))

    @app.post("/v1/feedback")
    def feedback(req: FeedbackRequest):
        return {"recorded": telemetry.add_feedback(req.trace_id, req.rating, req.reason)}

    @app.get("/v1/traces")
    def traces(request: Request, limit: int = 50):
        _check_admin(request, open_when_unset=not _is_prod())
        return {"traces": telemetry.recent(max(1, min(limit, 500)))}

    @app.get("/v1/corpus")
    def corpus(request: Request):
        _check_admin(request, open_when_unset=not _is_prod())
        return {"chunks": [
            {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "section": c.section, "n_words": c.n_words,
             "anonymised": c.anonymised, "text": c.text} for c in app.state.pipe.chunks]}

    return app


def get_app() -> FastAPI:  # uvicorn --factory ragkit.api.app:get_app
    return create_app()
