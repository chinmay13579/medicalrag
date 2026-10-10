"""End-to-end orchestration.

raw query
  -> [1] input guard (raw)             block injection / oversize / clinical-advice requests
  -> [2] pseudonymise query            per-request Vault (token -> original)
  -> [3] hybrid retrieve + re-rank     over a PII-free index
  -> [4] minimum-context prompt build  evidence floor, dedupe, caps, report/identity rule
  -> [5] residual-PII gate on prompt   fail closed
  -> [6] LLM
  -> [7] output guard (pre-rehydrate)  canary, PII, citations, groundedness
  -> [8] re-hydrate (query-origin tokens only)

Every stage runs inside a trace span; exactly one RequestRecord per question is handed to
``Telemetry`` (metrics + trace store + one structured log line), including refusals and errors.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from .config import Settings, get_settings
from .guardrails.input import check_input
from .guardrails.output import NO_ANSWER, check_output
from .ingestion.loader import load_corpus
from .llm.client import LLM, make_llm
from .logging import get_logger
from .observability import RequestRecord, Telemetry, Trace
from .prompting.builder import build_prompt
from .retrieval.hybrid import HybridRetriever

log = get_logger()
SCOPE_MESSAGE = ("This assistant explains platform policies and procedures. It can't provide medical "
                 "advice, diagnoses or dosing - please consult a clinician.")
# a query containing one of these refers to a specific person
_IDENTIFYING = ("[PERSON_", "[NATID_", "[ABHA_", "[MRN_", "[AADHAAR_", "[EMAIL_", "[PHONE_", "[SSN_", "[DOB_")


@dataclass
class Answer:
    trace_id: str
    answer: str
    status: str                       # ok | refused_input | no_evidence | blocked_output | blocked_prompt
    citations: list[dict] = field(default_factory=list)
    guardrails: dict = field(default_factory=dict)
    prompt: dict | None = None        # pseudonymised prompt, only populated when debug=True
    timings_ms: dict = field(default_factory=dict)


class RagPipeline:
    def __init__(self, settings: Settings | None = None, llm: LLM | None = None,
                 telemetry: Telemetry | None = None):
        self.s = settings or get_settings()
        corpus = load_corpus(self.s)
        self.chunks, self.pseudo, self.anon_results = corpus.chunks, corpus.pseudo, corpus.anon
        self.retriever = HybridRetriever(self.chunks, self.s)
        self.llm = llm or make_llm(self.s.llm)
        self.telemetry = telemetry or Telemetry(self.s)
        self.telemetry.set_corpus(self.chunks, self.anon_results)

    # -- retrieval only (used by eval) --------------------------------------------------------
    def retrieve(self, query: str, k: int = 5, mode: str = "hybrid+rerank"):
        safe_q, _ = self.pseudo.pseudonymise(query)
        return self.retriever.search(safe_q, k=k, mode=mode)

    def _build(self, query: str, mode: str, trace: Trace):
        with trace.span("pseudonymise") as sp:
            safe_q, vault = self.pseudo.pseudonymise(query)
            sp["entities"] = len(vault.mapping)
        timings: dict = {}
        with trace.span("retrieve") as sp:
            hits = self.retriever.search(safe_q, k=self.s.retrieval.fused_candidates, mode=mode,
                                         timings=timings)
            sp["candidates"] = len(hits)
        for name, ms in timings.items():                    # sub-stages (bm25 / dense / rerank)
            trace.add(f"retrieve.{name}", ms)
        identified = any(tok.startswith(_IDENTIFYING) for tok in vault.mapping)
        with trace.span("context_build") as sp:
            bundle = build_prompt(safe_q, hits, self.s.context, NO_ANSWER, identified_query=identified)
            sp["chunks"] = len(bundle.items)
        return safe_q, vault, bundle, hits

    def build(self, query: str, mode: str = "hybrid+rerank"):
        """Pseudonymise + retrieve + build prompt. Returns (safe_q, vault, bundle)."""
        safe_q, vault, bundle, _hits = self._build(query, mode, Trace("-"))
        return safe_q, vault, bundle

    # -- full answer --------------------------------------------------------------------------
    def answer(self, query: str, debug: bool = False) -> Answer:
        trace_id = uuid.uuid4().hex[:12]
        trace = self.telemetry.new_trace(trace_id)
        ctx: dict = {"guard": {}, "masked": {}, "retrieved": [], "dropped": []}
        ans: Answer | None = None
        try:
            with trace.span("request"):
                ans = self._answer(query, debug, trace, trace_id, ctx)
            return ans
        finally:
            stages = {k: v for k, v in trace.stage_ms().items() if k != "request"}
            total = trace.stage_ms().get("request", trace.total_ms)
            if ans is not None:
                ans.timings_ms = {**stages, "total": total}
            self.telemetry.record(RequestRecord(
                trace_id=trace_id, ts=self.telemetry.now(), status=ans.status if ans else "error",
                total_ms=total, stages=stages, query_hash=self.telemetry.query_hash(query),
                query_len=len(query), query_safe=ctx.get("query_safe"), masked=ctx["masked"],
                n_context=ctx.get("n_context", 0), context_tokens=ctx.get("context_tokens", 0),
                top_score=ctx.get("top_score"), retrieved=ctx["retrieved"], dropped=ctx["dropped"],
                guard=ctx["guard"], groundedness=ctx.get("groundedness"),
                llm_usage=ctx.get("llm_usage")))

    def _answer(self, query: str, debug: bool, trace: Trace, trace_id: str, ctx: dict) -> Answer:
        with trace.span("input_guard") as sp:
            iv = check_input(query)
            sp["allowed"] = iv.allowed
        if not iv.allowed:
            ctx["guard"]["input"] = iv.reasons
            msg = SCOPE_MESSAGE if "clinical_advice" in iv.reasons else "Your request can't be processed."
            return Answer(trace_id, msg, "refused_input", guardrails={"input": iv.reasons})

        safe_q, vault, bundle, hits = self._build(iv.query, "hybrid+rerank", trace)
        for tok in vault.mapping:
            kind = tok[1:].split("_", 1)[0]
            ctx["masked"][kind] = ctx["masked"].get(kind, 0) + 1
        if self.s.observability.log_queries:
            ctx["query_safe"] = safe_q
        ctx.update(n_context=len(bundle.items), context_tokens=bundle.context_tokens,
                   top_score=hits[0].score if hits else None,
                   retrieved=[{"chunk_id": h.chunk.chunk_id, "score": round(h.score, 4)} for h in hits[:5]],
                   dropped=bundle.dropped)

        prompt_view = ({"system": bundle.system, "user": bundle.user, "dropped": bundle.dropped,
                        "context_tokens": bundle.context_tokens} if debug else None)
        cites = [{"id": i.cid, "chunk_id": i.chunk_id, "doc": i.doc_id, "section": i.section,
                  "score": i.score} for i in bundle.items]

        if not bundle.items:
            return Answer(trace_id, NO_ANSWER, "no_evidence", [], {}, prompt_view)

        with trace.span("residual_pii_gate") as sp:
            residual = (self.s.privacy.fail_on_residual_pii
                        and bool(self.pseudo.residual(bundle.system + bundle.user)))
            sp["residual"] = residual
        if residual:
            ctx["guard"]["prompt"] = ["residual_pii"]
            log.error("residual_pii_in_prompt", extra={"ctx": {"trace_id": trace_id}})
            return Answer(trace_id, "Your request can't be processed.", "blocked_prompt",
                          guardrails={"prompt": ["residual_pii"]})

        with trace.span("llm", provider=self.s.llm.provider):
            raw = self.llm.generate(bundle, safe_q)
        ctx["llm_usage"] = bundle.usage

        with trace.span("output_guard") as sp:
            ov = check_output(raw, bundle.context_text, {i.cid for i in bundle.items},
                              bundle.canary, bundle.system, self.pseudo)
            sp["allowed"] = ov.allowed
        ctx["groundedness"] = ov.groundedness
        if not ov.allowed:
            ctx["guard"]["output"] = ov.reasons
            log.warning("output_blocked", extra={"ctx": {"trace_id": trace_id, "reasons": ov.reasons}})
            return Answer(trace_id, NO_ANSWER, "blocked_output", cites,
                          {"output": ov.reasons, "groundedness": ov.groundedness}, prompt_view)

        with trace.span("rehydrate"):
            final = vault.rehydrate(raw) if self.s.privacy.rehydrate == "query_only" else raw
        status = "no_evidence" if raw.strip() == NO_ANSWER else "ok"
        return Answer(trace_id, final, status, cites if status == "ok" else [],
                      {"groundedness": ov.groundedness}, prompt_view)
