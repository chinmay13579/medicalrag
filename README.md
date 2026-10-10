# ragkit: privacy-preserving RAG for health-data interoperability

A retrieval-augmented question-answering service for a healthcare data-interoperability platform (patients,
clinicians and research; Slovakia, Czechia, India; heading toward the European Health Data Space). Staff ask
policy and procedure questions and get cited answers; identifiers are masked before anything is indexed or
sent to a model, clinical reports are de-identified, and the system is observable without creating a new leak.

This README is also the project's architecture documentation, structured after the
[arc42](https://arc42.org) template. Diagrams are Mermaid and render on GitHub.

## Contents

| # | Section | Answers |
|---|---|---|
| - | [Quick start](#quick-start) | Install, run, test, and where things are |
| - | [Sample output](#sample-output) | Screenshots from a real run |
| 1 | [Introduction and goals](#1-introduction-and-goals) | What is it for, who uses it, which qualities matter most? |
| 2 | [Constraints](#2-constraints) | What may we not decide freely? |
| 3 | [Context and scope](#3-context-and-scope) | What is inside, what is outside, which interfaces? |
| 4 | [Solution strategy](#4-solution-strategy) | The key approaches, mapped to the quality goals |
| 5 | [Building block view](#5-building-block-view) | Static structure: modules and responsibilities |
| 6 | [Runtime view](#6-runtime-view) | How the blocks collaborate in the important scenarios |
| 7 | [Deployment view](#7-deployment-view) | Where it runs; configuration; CI |
| 8 | [Crosscutting concepts](#8-crosscutting-concepts) | Privacy model, security, observability, configuration, testing |
| 9 | [Architecture decisions](#9-architecture-decisions) | ADR-001 to ADR-013 |
| 10 | [Quality requirements](#10-quality-requirements) | Quality tree, scenarios, and the measured evidence |
| 11 | [Risks and technical debt](#11-risks-and-technical-debt) | What can go wrong; what we knowingly left undone |
| 12 | [Glossary](#12-glossary) | Terms |

## Status of the evidence

This documentation distinguishes three levels, because a privacy-critical system should not overclaim:

* **Verified** - exercised by an automated test or measured in this repository (82 unit tests, a
  59-case evaluation set with CI gates, a 38-test promptfoo suite run against the live service; the CI
  workflow's lint, test, evaluation-gate and promptfoo jobs also passed on GitHub-hosted Linux runners).
* **Written, not executed** - artefacts that could not be run in the authoring environment: the Docker
  image and `docker-compose.yml`, Prometheus alert rules (syntax-checked and cross-checked against the
  exported metric names, but not loaded into Prometheus), the Grafana dashboard, and the generative
  red-team. They are flagged wherever they appear.
* **Synthetic data** - all documents, reports, people, identifiers and policy values are invented. The
  policy numbers (24 h, cell size 10, ...) are placeholders, not the company's real policies.

## Related files

* Evaluation harness and gates: [`eval/`](eval), thresholds in `eval/thresholds.<profile>.json`
* Security tests: [`promptfoo/`](promptfoo)
* Ops artefacts: [`deploy/`](deploy), [`docker-compose.yml`](docker-compose.yml)
* Slide deck: [`docs/ragkit-overview.pptx`](docs/ragkit-overview.pptx)

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,ui,otel]"                        # API + Streamlit app + OpenTelemetry bridge

pytest -q                                              # 82 tests
python eval/run_eval.py                                # offline evaluation + gates (no downloads, no API key)

uvicorn --factory ragkit.api.app:get_app --port 8000   # API, /metrics, demo page at http://localhost:8000
streamlit run src/ragkit/ui/streamlit_app.py           # Streamlit app at http://localhost:8501 (in-process)
RAGKIT_API_URL=http://localhost:8000 streamlit run src/ragkit/ui/streamlit_app.py   # same, via the HTTP API

python scripts/show_prompt.py "Why did Jana Novakova's lab results not appear?"   # the exact masked prompt
python scripts/anonymise_report.py lab_report_hematology                          # before/after of a report

npm ci && npx promptfoo eval -c promptfoo/promptfooconfig.yaml   # security suite (needs Node >= 22)

pip install -e ".[dense]"                              # real embeddings + cross-encoder
RAGKIT_CONFIG=configs/semantic.yaml python eval/run_eval.py

docker compose up --build                              # API + UI + Prometheus + Grafana (needs .env; not executed)
```

`llm.provider: anthropic` (with `pip install -e ".[llm]"` and `ANTHROPIC_API_KEY`) swaps the offline extractive
answerer for Claude. Copy `.env.example` to `.env` for the settings that matter in production (`PSEUDO_KEY`,
`RAGKIT_ADMIN_TOKEN`).

Repository layout:

| Path | Contents |
|------|----------|
| `src/ragkit/` | `api`, `ui`, `pipeline`, `ingestion`, `privacy`, `retrieval`, `prompting`, `guardrails`, `llm`, `observability` |
| `data/` | `raw/` policy documents, `raw/reports/` clinical reports, `eval/` evaluation set and annotations (all synthetic) |
| `eval/` | metrics, harness, per-profile gates |
| `promptfoo/` | security and privacy regression suite, red-team configuration |
| `deploy/`, `docker-compose.yml`, `Dockerfile` | Prometheus, alert rules, Grafana, container build |
| `configs/` | `default.yaml` (offline) and `semantic.yaml` (real models) profiles |
| `tests/` | unit, component, UI (Streamlit `AppTest`) and ops-config tests |
| `docs/` | `screenshots/` and the slide deck `ragkit-overview.pptx` |

## Sample output

Captured from a real run (API and Streamlit app in HTTP mode, Edge driven by Playwright). The full set is in
[`docs/screenshots/`](docs/screenshots); a slide deck about the project is
[`docs/ragkit-overview.pptx`](docs/ragkit-overview.pptx).

| | |
|---|---|
| **Ask**: a question naming a patient. The prompt shows `[PERSON_ce1f64]`, not the name; the answer restores only what the user typed. | ![Ask](docs/screenshots/01-ask-answer-and-masked-prompt.png) |
| **Refused**: a clinical-advice request stops at the input guard (0.1 ms, no retrieval). | ![Refused](docs/screenshots/02-ask-clinical-advice-refused.png) |
| **De-identify** a lab report, side by side. | ![Anonymise](docs/screenshots/04-anonymise-report.png) |
| **Observability** after live traffic. | ![Observability](docs/screenshots/05-observability.png) |
| **promptfoo** results viewer: 38 / 38 cases, 432 / 432 assertions. | ![promptfoo](docs/screenshots/10-promptfoo-results.png) |
| **Tests and semantic-profile evaluation** (rendered from the captured command output, not a live terminal grab). | ![Eval](docs/screenshots/11-terminal-tests-and-eval.png) |

The API's `/docs` page could not be captured: Swagger UI loads its assets from a CDN that was unreachable from
the capture environment.

---

## 1. Introduction and goals

### 1.1 Requirements overview

ragkit lets platform staff ask questions in natural language about the company's **policies and
procedures** (consent and legal basis, FHIR interoperability, data residency, retention, incident
response, access control, research data access) and about **de-identified clinical reports**, and returns
short answers with citations.

| ID | Requirement | Realised by |
|----|-------------|-------------|
| F1 | Answer policy questions from an internal knowledge base, with source citations | hybrid retrieval, prompt builder, LLM adapter, output guard |
| F2 | Decline when the knowledge base has no supporting evidence | evidence floor in the context policy |
| F3 | Never give medical advice, diagnosis or dosing | input scope guard + system prompt |
| F4 | Make clinical reports (lab, radiology, discharge) searchable **without** exposing patients | irreversible de-identification at ingestion |
| F5 | Accept questions that mention people or identifiers without exposing them to the model | query pseudonymisation, per-request vault |
| F6 | Resist prompt injection, including injection hidden inside indexed documents | input guard, context sanitiser, output guard |
| F7 | Make the system's behaviour observable without creating a new privacy exposure | metrics, traces, structured logs, privacy-safe telemetry |
| F8 | Let staff inspect what the model saw, de-identify a report interactively, and watch the system | Streamlit app |
| F9 | Measure quality and safety and block regressions | evaluation harness, CI gates, promptfoo |

**Not a goal:** diagnosing, treating or advising on individual patients; acting as a system of record;
replacing the care provider's controller responsibilities.

### 1.2 Quality goals

Ordered by priority. Where goals conflict, the higher one wins (for example privacy over answer
completeness).

| Prio | Quality goal | Meaning here |
|------|--------------|--------------|
| 1 | **Privacy** | No patient or staff identifier reaches the index, the LLM prompt, logs, metrics, traces or feedback storage. Fail closed. |
| 2 | **Safety** | No medical advice; no report attributed to the wrong person; no obeying instructions found in data. |
| 3 | **Correctness / groundedness** | Answers come from, and cite, retrieved evidence; unsupported questions are declined, not guessed. |
| 4 | **Observability** | Every request leaves a trace; operators can see quality, latency, guardrail activity and failures. |
| 5 | **Testability** | Every protective behaviour has an automated check that can fail. |

Latency and cost are *constraints to be monitored*, not primary goals (see chapter 10).

### 1.3 Stakeholders

| Role | Expectation |
|------|-------------|
| Support / operations staff (users) | Fast, cited answers about policy; clear messages when the assistant declines |
| Data stewards, privacy team, DPO | Evidence that identifiers cannot leak; control over de-identification rules; the DPO signs off before anything is called anonymised |
| Clinicians (indirect) | Reports are never attached to the wrong patient; no advice masquerading as policy |
| Security engineering | Defence in depth; adversarial test suite; auditable telemetry |
| Platform / SRE | Metrics, alerts, dashboards, health endpoints; reproducible deployment |
| Researchers (indirect) | De-identified report content that retains clinical value |
| LLM provider (external) | Receives only pseudonymised text and minimal context |

---

## 2. Constraints

### 2.1 Technical constraints

| Constraint | Background |
|------------|-----------|
| Python >= 3.10 | `requires-python` in `pyproject.toml`. CI and the container use 3.11. A local interpreter at exactly 3.10.0 needs `pydantic < 2.10` (handled by an environment marker). |
| Must run **offline** with no API key | The default profile uses a hashing embedder, a heuristic re-ranker and an extractive answerer, so CI, evaluation and demos need no network and no model download. |
| Real models are optional extras | `pip install .[dense]` (sentence-transformers, cross-encoder), `.[llm]` (Anthropic), `.[ui]` (Streamlit), `.[otel]` (OpenTelemetry). |
| In-memory index | The corpus is small; the index is rebuilt at start-up. No vector database. |
| Identifiers must be handled before indexing | Required by the privacy goal; see ADR-001. |
| Secrets only via environment | `PSEUDO_KEY`, `RAGKIT_ADMIN_TOKEN`, `ANTHROPIC_API_KEY`, `GRAFANA_PASSWORD`. `PSEUDO_KEY` is mandatory when `RAGKIT_ENV=prod`. |
| Node >= 22 for promptfoo | The development machine has Node 18; promptfoo was run under a temporary Node 22. |

### 2.2 Organisational and regulatory constraints

| Constraint | Consequence |
|------------|-------------|
| GDPR Art. 9 (health data = special category) | Pseudonymised data is still personal data; a DPIA and a processor agreement with any model provider are required before real use (open item, chapter 11). |
| Regional data residency (EU data stays in the EU; Indian data stays in India) | Stated in the knowledge base. The service itself is region-agnostic: deploy one instance per region. Not implemented as a technical control. |
| Anonymisation requires DPO sign-off | Report processing is labelled *de-identified*, never *anonymous*; status `needs_review` blocks the claim. |
| European Health Data Space direction | Secondary-use opt-outs and small-numbers rules are reflected in the knowledge base content, not enforced by this service. |
| No real patient data in this repository | All data is synthetic. The Streamlit app warns against pasting real data into a shared deployment. |

### 2.3 Conventions

| Topic | Convention |
|-------|-----------|
| Configuration | YAML profile in `configs/`, overridable by environment variables. `configs/default.yaml` is offline; `configs/semantic.yaml` uses real models. |
| Language | Documents, queries, UI and code are English only. Names are matched accent-insensitively as a convenience, not as multilingual support. |
| Telemetry | Labels from closed sets only; no text from queries, documents or answers (see 8.3). |
| Tests | Behaviour that protects privacy or safety must have a test that fails when the behaviour is removed. |
| Documentation | arc42; Mermaid diagrams. |

---

## 3. Context and scope

### 3.1 Business context

```mermaid
flowchart LR
    staff["Platform staff<br/>(support, data stewards, privacy team)"]
    ops["Operators / SRE"]
    dpo["DPO / privacy team"]
    kb[("Knowledge base<br/>policy documents +<br/>clinical reports (files)")]
    llm["LLM provider<br/>(optional, external)"]
    sys["ragkit<br/>privacy-preserving Q&A"]

    staff -- "questions, feedback" --> sys
    sys -- "cited answers, refusals" --> staff
    kb -- "ingested at start-up" --> sys
    sys -- "pseudonymised prompt,<br/>minimum context" --> llm
    llm -- "answer text" --> sys
    sys -- "metrics, traces, logs" --> ops
    dpo -- "reviews de-identification status,<br/>approves rules" --> sys
```

| Partner | Inputs to ragkit | Outputs from ragkit |
|---------|------------------|---------------------|
| Platform staff | Natural-language questions (may contain names or identifiers); feedback codes | Answer + citations, or a refusal / "no evidence" message; optional view of the pseudonymised prompt |
| Knowledge base (files) | Markdown policy documents and clinical reports | - |
| LLM provider | Answer text | Pseudonymised system + user prompt (never raw identifiers) |
| Operators | - | Prometheus metrics, JSON logs, recent request records, health endpoints |
| DPO / privacy team | Rule changes, sign-off | De-identification log (categories and counts) and `needs_review` status |

### 3.2 Technical context

```mermaid
flowchart LR
    browser["Browser"]
    st["Streamlit UI :8501"]
    api["ragkit API :8000<br/>FastAPI"]
    prom["Prometheus :9090"]
    graf["Grafana :3000"]
    anth["Anthropic API<br/>(optional)"]
    hf["Model hub<br/>(build/first run only)"]
    files[("data/raw/*.md<br/>data/raw/reports/*.md")]

    browser --> st
    browser -- "legacy demo page" --> api
    st -- "HTTP: /v1/ask, /v1/traces,<br/>/v1/feedback, /v1/corpus, /metrics" --> api
    prom -- "scrape /metrics" --> api
    graf --> prom
    api --> anth
    api -.-> hf
    files --> api
```

| Interface | Protocol / format | Notes |
|-----------|-------------------|-------|
| `POST /v1/ask` | JSON in/out | Rate-limited (30 / 60 s / client, in-process). `debug` honoured only outside `prod`. |
| `POST /v1/feedback` | JSON; `rating` up/down, `reason` from a closed set | Free text is rejected (422): it could contain patient data. |
| `GET /v1/traces`, `GET /v1/corpus` | JSON | Admin: bearer `RAGKIT_ADMIN_TOKEN`; open in dev if unset; **404 in prod if unset**. |
| `GET /metrics` | Prometheus text | Contains no identifiers. Protected by the admin token if one is set. |
| `GET /healthz`, `GET /readyz` | JSON | Liveness; readiness (index non-empty). |
| Files | Markdown | Header fields (`Patient name:`, `Address:` ...) in reports drive de-identification rules. |
| Anthropic Messages API | HTTPS | Only when `llm.provider: anthropic`; token usage is recorded as a metric. |

### 3.3 Scope

**In scope:** ingestion and protection of documents and reports; retrieval and re-ranking; prompt
construction; guardrails; the LLM adapter; telemetry; the API and Streamlit UI; evaluation and security
test harnesses; deployment artefacts.

**Out of scope:** the source clinical systems and any HL7/FHIR integration (reports are provided as
files); identity and access management of end users (see chapter 11); a persistent vector store; model
training; the care provider's controller duties.

---

## 4. Solution strategy

| Quality goal | Strategy | Where | ADR |
|--------------|----------|-------|-----|
| **Privacy** | Protect identifiers **at the ingestion boundary** so the index is identifier-free; protect queries with the same keyed function so retrieval still works on masked text; re-scan the final prompt and **fail closed**. | `privacy/`, `ingestion/loader.py`, `pipeline.py` | 001, 013 |
| **Privacy (reports)** | A separate, **irreversible** path for clinical reports: generic unlinkable markers, generalised quasi-identifiers, residual re-scan, status `deidentified` / `needs_review`. | `privacy/anonymise.py` | 002 |
| **Privacy (minimisation)** | Send the model the least text that can answer: evidence floor, relative floor, de-duplication, chunk and token caps. No ids, paths or scores in the prompt. | `prompting/builder.py` | 005 |
| **Safety** | Refuse clinical-advice requests up front; never present a de-identified report as belonging to a named person; treat retrieved text as untrusted data. | `guardrails/input.py`, `prompting/builder.py` | 007, 009 |
| **Security** | Layered, independent controls (input, context, prompt gate, output) plus structural isolation; adversarial regression suite in promptfoo. | `guardrails/`, `promptfoo/` | 006 |
| **Correctness** | Hybrid retrieval (lexical + dense) fused by rank, then re-ranked; abstain when evidence is weak; check citations and groundedness on output. | `retrieval/`, `guardrails/output.py` | 003, 004 |
| **Observability** | One request record per question feeding Prometheus metrics, a bounded in-memory trace store and a structured log line; optional OpenTelemetry spans; telemetry itself is privacy-safe. | `observability/` | 010 |
| **Testability / portability** | Offline-first default profile so CI needs no network; real-model profile behind the same interfaces; per-profile evaluation gates. | `configs/`, `eval/` | 008, 012 |
| **Usability** | Streamlit app with swappable backend; shows the exact pseudonymised prompt and guardrail verdicts. | `ui/` | 011 |

### Guiding principles

1. **Protect early, once.** Identifiers are removed or tokenised before indexing, not at answer time.
2. **Two privacy tools, used deliberately.** Keyed pseudonymisation (linkable, reversible by the key
   holder) for queries and ordinary documents; irreversible de-identification for reports.
3. **Fail closed.** A residual identifier in a prompt, a bad citation, a leaked canary: the answer is
   withheld, not repaired.
4. **Defence in depth.** No single guard is trusted; each has an independent test, and the tests assert
   *outcomes* (nothing leaked), not that a particular regex fired.
5. **Measure, then claim.** Every number in these documents comes from a test or a measurement in this
   repository; limitations are listed in chapter 11.

---

## 5. Building block view

### 5.1 Level 1: whitebox overview

```mermaid
flowchart TB
    subgraph clients["Clients"]
        ui["ui<br/>Streamlit app"]
        web["api/static<br/>demo page"]
    end
    api["api<br/>FastAPI service"]
    subgraph core["Core"]
        pipe["pipeline<br/>orchestration"]
        ing["ingestion<br/>loader + chunking"]
        priv["privacy<br/>pseudonymise + anonymise"]
        ret["retrieval<br/>BM25, dense, RRF, re-rank"]
        prm["prompting<br/>minimum-context builder"]
        grd["guardrails<br/>input, injection, output"]
        llm["llm<br/>extractive / Anthropic"]
    end
    obs["observability<br/>metrics, tracing, telemetry"]
    cfg["config + logging"]
    subgraph quality["Quality tooling"]
        ev["eval<br/>metrics, run_eval, gates"]
        pf["promptfoo<br/>security suite"]
    end

    ui -- HTTP or in-process --> api
    web --> api
    api --> pipe
    pipe --> ing --> priv
    pipe --> priv
    pipe --> ret
    pipe --> prm
    pipe --> grd
    pipe --> llm
    pipe --> obs
    api --> obs
    ing --> ret
    cfg -.-> core
    ev --> pipe
    pf --> api
```

| Block | Responsibility | Key files |
|-------|----------------|-----------|
| **api** | HTTP surface, rate limit, security headers, admin gating, HTTP metrics | `api/app.py` |
| **ui** | Streamlit front-end and its backends (in-process / HTTP); Prometheus-text statistics | `ui/streamlit_app.py`, `ui/backend.py`, `ui/stats.py` |
| **pipeline** | The one place where stages are sequenced; emits exactly one `RequestRecord` per question | `pipeline.py` |
| **ingestion** | Read files, route documents vs reports, chunk | `ingestion/loader.py`, `ingestion/chunking.py` |
| **privacy** | Keyed pseudonymisation (queries, documents); irreversible report de-identification | `privacy/pseudonymise.py`, `privacy/anonymise.py` |
| **retrieval** | Tokeniser, BM25, dense index, RRF fusion, re-rankers | `retrieval/*.py` |
| **prompting** | Context selection policy and prompt construction | `prompting/builder.py` |
| **guardrails** | Injection / scope checks on input; output verification | `guardrails/*.py` |
| **llm** | `generate(bundle, question)` behind a protocol; extractive and Anthropic adapters | `llm/client.py` |
| **observability** | Prometheus metrics, per-request trace/spans, telemetry facade and trace store | `observability/*.py` |
| **config / logging** | Typed settings (YAML + env), JSON logger | `config.py`, `logging.py` |
| **eval** | Pure metric functions, harness, anonymisation metrics, per-profile gates | `eval/` |
| **promptfoo** | Outcome-based adversarial and privacy regression tests; generative red-team config | `promptfoo/` |
| **deploy** | Prometheus config, alert rules, Grafana dashboard, compose, Dockerfile | `deploy/`, `docker-compose.yml`, `Dockerfile` |

#### Dependency rules

* `privacy`, `retrieval`, `guardrails`, `prompting` do not import `pipeline`, `api` or `ui`.
* Only `pipeline` and `api` know about `observability`; lower layers return data, they do not emit telemetry.
* `ui` talks to the pipeline only through `Backend`; in HTTP mode it imports only `config` and the
  stateless `Anonymiser` / `Pseudonymiser` (for the demonstration tab), never the pipeline.
* The LLM adapter sees a `PromptBundle` that is already protected; it has no access to the vault.

### 5.2 Level 2: privacy

```mermaid
flowchart LR
    subgraph keyed["Pseudonymiser (linkable, keyed)"]
        det["detectors<br/>email, phone, NATID, ABHA,<br/>Aadhaar, MRN, DOB, card,<br/>persons (honorific + gazetteer)"]
        tok["token = HMAC(PSEUDO_KEY, kind|value)"]
        vault["Vault<br/>token to original<br/>(per request)"]
        det --> tok --> vault
    end
    subgraph irr["Anonymiser (irreversible)"]
        fld["labelled-field rules"]
        nar["narrative detectors<br/>+ per-report gazetteer<br/>+ kinship cues"]
        gen["generalise<br/>DOB to year, dates to year-month,<br/>age bands, postcode to 3 digits"]
        res["residual re-scan<br/>deidentified | needs_review"]
        fld --> gen
        nar --> gen --> res
    end
```

| Aspect | Pseudonymiser | Anonymiser |
|--------|---------------|------------|
| Used for | user queries; ordinary policy/case documents | clinical reports |
| Output | `[PERSON_1be848]` (same input, same token) | `[PATIENT]`, `[CLINICIAN]` (no hash, no linkage) |
| Reversible | by the key holder, via the per-request vault, for values the *user typed* | no |
| Names found here feed | the query gazetteer | the query gazetteer (so a typed name is masked) |
| Failure signal | residual scan of the prompt: `blocked_prompt` | `needs_review`, metric `ragkit_reports_deidentified` |

### 5.3 Level 2: retrieval and context policy

```mermaid
flowchart LR
    q["pseudonymised query"] --> bm["BM25"]
    q --> de["dense cosine"]
    bm --> rrf["RRF fusion<br/>(rank-based)"]
    de --> rrf
    rrf --> rr["re-rank<br/>heuristic | cross-encoder"]
    rr --> pol["context policy"]
    subgraph pol_inner["policy filters, in order"]
        a["evidence floor"] --> b["relative floor"] --> c["identity rule<br/>(anonymised report vs named query)"] --> d["injection filter"] --> e["de-duplicate"] --> f["max chunks / token budget"]
    end
    pol --- pol_inner
    pol --> out["PromptBundle"]
```

### 5.4 Level 2: observability

| Component | Role |
|-----------|------|
| `Metrics` | Owns a `CollectorRegistry`; defines all series with closed label sets |
| `Trace` / `SpanRecord` | Per-request stage timer; optional OpenTelemetry mirroring; attribute whitelist (scalars and short strings) |
| `Telemetry` | `record(RequestRecord)` updates metrics, appends to a bounded ring buffer, emits one JSON log line; keyed query hash; feedback with reason codes |
| `RequestRecord` | The single data contract for a request: status, stage timings, counts, scores, chunk ids, guard reasons, hash |

### 5.5 Core data structures

| Type | Purpose |
|------|---------|
| `Chunk` | `chunk_id`, `doc_id`, `section`, protected `text`, `n_words`, `anonymised` flag |
| `Hit` | chunk + score + per-stage ranks/scores (`bm25`, `dense`, `rrf`, `cos`, `rerank`) |
| `PromptBundle` | system, user, canary, selected `ContextItem`s, dropped chunks with reasons, token estimate, LLM usage |
| `Vault` | per-request token-to-original mapping; never persisted or logged |
| `Answer` | what the API returns: answer, status, citations, guardrail verdicts, optional prompt view, timings |
| `AnonResult` | de-identified text, category counts, residual flags, status, caveat note |
| `RequestRecord` | telemetry contract (5.4) |

---

## 6. Runtime view

### 6.1 Scenario: a question is answered (happy path)

```mermaid
sequenceDiagram
    autonumber
    actor U as Staff member
    participant A as API
    participant P as Pipeline
    participant G as Input guard
    participant S as Pseudonymiser
    participant R as Retriever
    participant B as Prompt builder
    participant L as LLM adapter
    participant O as Output guard
    participant T as Telemetry

    U->>A: POST /v1/ask {"query": "Why did Jana Novakova's lab results not appear?"}
    A->>P: answer(query)
    P->>G: check_input(raw query)
    G-->>P: allowed
    P->>S: pseudonymise(query)
    S-->>P: "Why did [PERSON_2804a3]'s ..." + Vault
    P->>R: search(masked query)
    R-->>P: ranked hits (bm25, dense, RRF, re-rank)
    P->>B: build_prompt(hits, identified_query=true)
    B-->>P: PromptBundle (minimum context, anonymised reports dropped)
    P->>P: residual-PII gate on system+user text
    P->>L: generate(bundle)
    L-->>P: raw answer with [c1] citations
    P->>O: check_output(raw, context, canary)
    O-->>P: allowed, groundedness
    P->>P: re-hydrate tokens that came from the query
    P->>T: record(RequestRecord)
    P-->>A: Answer
    A-->>U: answer + citations
```

Points to note: the guard sees the **raw** query (it must, to catch obfuscation), everything after step 4
sees only the masked text, and the output guard runs **before** re-hydration so that a model-invented
identifier cannot be laundered through the vault. Exactly one `RequestRecord` is produced per request,
including refusals and errors (status `error`).

### 6.2 Scenario: request refused at the input guard

Triggers: injection / exfiltration patterns, letter-spaced obfuscation, delimiter break-out, oversize
input, or a **clinical-advice** request (`prescribe`, `should I give ...`, `my patient has ...`,
`what is the diagnosis`).

```mermaid
sequenceDiagram
    participant P as Pipeline
    participant G as Input guard
    participant T as Telemetry
    P->>G: check_input(raw)
    G-->>P: blocked, reasons=[clinical_advice]
    P->>T: record(status=refused_input, guard.input=[clinical_advice])
    Note over P: No pseudonymisation, retrieval or LLM call happens.
```

The user sees a scope message for `clinical_advice` ("consult a clinician") and a generic message otherwise.

### 6.3 Scenario: ingestion and report de-identification (start-up)

```mermaid
sequenceDiagram
    participant L as Loader
    participant AN as Anonymiser
    participant PS as Pseudonymiser
    participant CH as Chunker
    participant IX as Indexes (BM25, dense)
    participant TM as Telemetry

    L->>L: read data/raw/*.md and data/raw/reports/*.md
    loop each report
        L->>AN: anonymise(report text)
        AN-->>L: de-identified text, log (counts), status, person names
    end
    L->>PS: build gazetteer from report names + cue-harvested document names
    loop each document
        L->>PS: pseudonymise(text) (vault discarded)
        L->>CH: chunk protected text
    end
    loop each report
        L->>CH: chunk de-identified text (anonymised=true)
    end
    L->>IX: build indexes over protected chunks only
    L->>TM: set_corpus(chunk counts, reports by status)
```

The original text of a report exists only inside `Anonymiser.anonymise`; the result object keeps
category counts and flags, not values. Person names go to the **query-side** gazetteer so a typed name
is masked, but the indexed text no longer contains them.

### 6.4 Scenario: a named person is asked about a de-identified report

Question: "What was Jana Novakova's hemoglobin?"

1. The query is pseudonymised; the vault contains a `[PERSON_...]` token, so the query is **identified**.
2. Retrieval finds the hematology report chunk (it matches "hemoglobin").
3. The context policy drops it with reason `anonymised_report_vs_identified_query`.
4. The prompt contains no report content; the answer is built from whatever else remains (here, an
   unrelated case note) or is "no evidence". The value 9.8 is never attached to the name.

Rationale: an anonymous record cannot be tied to a person, and presenting it as theirs would be a
patient-safety error rather than merely a privacy one. Verified by `tests/test_anonymise.py` and the
`rp02` eval case.

### 6.5 Scenario: observing the system

```mermaid
sequenceDiagram
    participant P as Pipeline
    participant T as Telemetry
    participant M as Metrics registry
    participant ST as Trace store (ring buffer)
    participant LG as JSON logger
    participant PR as Prometheus
    participant UI as Streamlit

    P->>T: record(RequestRecord)
    T->>M: counters, histograms (closed label sets)
    T->>ST: append (bounded)
    T->>LG: one line, no query text unless log_queries
    PR->>M: GET /metrics every 15 s
    UI->>M: GET /metrics (via API or in-process)
    UI->>ST: GET /v1/traces (admin)
```

### 6.6 Scenario: feedback

The UI shows "Helpful" / "Not helpful" plus a reason from a **closed set** (`wrong_answer`,
`missing_info`, `privacy_concern`, `other`). `POST /v1/feedback` validates the trace id format and the
codes, increments `ragkit_feedback_total{rating,reason}` and annotates the stored record. Free text is
rejected with 422 so patient data cannot arrive through the feedback door.

### 6.7 Failure behaviour

| Failure | Behaviour |
|---------|-----------|
| Residual identifier in the assembled prompt | `blocked_prompt`; LLM not called; metric `guard=prompt,reason=residual_pii`; critical alert |
| Model output contains an identifier, canary, bad citation or is ungrounded | `blocked_output`; user receives the "no information" message; alert if repeated |
| No chunk passes the evidence floor | `no_evidence`; LLM not called (cheaper, nothing to hallucinate from) |
| Report fails the post-anonymisation re-scan | status `needs_review`; gauge > 0; critical alert; (the report remains indexed until an operator acts: see chapter 11) |
| Exception inside a stage | the `RequestRecord` is still written with status `error`; the exception propagates (HTTP 500) |
| Optional dependency missing (OTel, Anthropic) | OTel silently disabled; Anthropic adapter raises a clear error at start-up |

---

## 7. Deployment view

> **Status:** the Docker image, `docker-compose.yml`, Prometheus rules and Grafana dashboard were
> **written but not executed** (no Docker or Prometheus in the authoring environment). They are
> statically validated by `tests/test_ops_config.py` (YAML/JSON parse; every metric and label used in
> alerts and dashboard exists in the real exposition). The API and the Streamlit app were run for real,
> including Streamlit-over-HTTP against a live API. The CI workflow itself ran on GitHub Actions (its
> `redteam` job needs an API key and is not part of push runs).

### 7.1 Topology (docker compose)

```mermaid
flowchart LR
    user["Browser"]
    subgraph host["Docker host / one region"]
        ui["ui<br/>Streamlit :8501<br/>image: EXTRAS=ui"]
        api["api<br/>uvicorn :8000<br/>RAGKIT_ENV=prod"]
        prom["prometheus :9090<br/>alerts.yml"]
        graf["grafana :3000<br/>dashboard ragkit overview"]
    end
    ext["Anthropic API (optional)"]

    user --> ui
    user --> graf
    ui -- "HTTP + admin bearer" --> api
    prom -- "scrape /metrics" --> api
    graf --> prom
    api -.-> ext
```

| Node | Image / process | Config | Holds |
|------|-----------------|--------|-------|
| api | `Dockerfile` (python:3.11-slim, non-root uid 10001), `uvicorn --factory ragkit.api.app:get_app` | env: `RAGKIT_ENV=prod`, `PSEUDO_KEY`, `RAGKIT_ADMIN_TOKEN`, optional `ANTHROPIC_API_KEY`; `RAGKIT_HOME=/app` | the index (in memory), the pseudonymisation key, the documents |
| ui | same image built with `EXTRAS=ui`, `streamlit run ...` | `RAGKIT_API_URL=http://api:8000`, `RAGKIT_ADMIN_TOKEN` | **nothing sensitive**: no index, no key |
| prometheus | `prom/prometheus` | `deploy/observability/prometheus.yml`, `alerts.yml` | time series (numbers only) |
| grafana | `grafana/grafana` | provisioning + dashboard JSON from `deploy/grafana/` | dashboards |

Residency: run **one stack per region** (EU, India). Nothing in the service moves data between them.

### 7.2 Environments

| Environment | How | Notable settings |
|-------------|-----|------------------|
| Developer | `pip install -e .[dev,ui,otel]`; `make serve`; `make ui` | dev key (insecure default), debug prompts on, admin endpoints open |
| CI | GitHub Actions: lint, pytest (incl. Streamlit AppTest), offline eval gate, promptfoo | Python 3.11; promptfoo on Node 22 |
| Staging / production | docker compose or any container platform | `RAGKIT_ENV=prod` (key mandatory, debug off, admin endpoints need a token) |

### 7.3 Configuration

| Variable | Purpose | Required |
|----------|---------|----------|
| `PSEUDO_KEY` | HMAC key for pseudonyms and the query hash | prod: yes |
| `RAGKIT_ENV` | `dev` / `prod` | no |
| `RAGKIT_CONFIG` | profile YAML (`configs/default.yaml`, `configs/semantic.yaml`) | no |
| `RAGKIT_HOME` | project root (data, configs); auto-detected from CWD otherwise | in containers |
| `RAGKIT_ADMIN_TOKEN` | bearer for `/v1/traces`, `/v1/corpus`, `/metrics` | prod: for those endpoints to exist |
| `RAGKIT_API_URL` | Streamlit HTTP mode | UI container |
| `RAGKIT_LOG_QUERIES` | include the **pseudonymised** query in telemetry (dev only) | no |
| `ANTHROPIC_API_KEY` | real LLM | if `llm.provider: anthropic` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP/HTTP collector for spans (`observability.otel: true`) | no |
| `LOG_LEVEL` | logger level | no |

### 7.4 Pipeline (CI)

1. `ruff check`; `pytest -q` (unit, observability, ops-config, Streamlit AppTest).
2. `python eval/run_eval.py` (offline profile): retrieval, end-to-end, safety and de-identification
   gates; non-zero exit on breach.
3. Start the API; `promptfoo eval` (38 deterministic tests); results uploaded as an artefact.
4. Weekly or by hand (`workflow_dispatch`): generative red-team. It needs an `ANTHROPIC_API_KEY` repository
   secret and skips cleanly without one, so it is **never executed so far**; push and PR runs do not include it.

The `semantic` profile gate is run manually (`make eval-semantic`) because it downloads two models.

### 7.5 Scaling notes

The index lives in each process's memory and is rebuilt at start-up (seconds at this size), the pseudonym
key is shared by configuration, and the telemetry store is per process. Horizontal scaling therefore
needs: a shared rate limiter (the current one is per process), aggregation of `/v1/traces` across
replicas (or a trace backend via OpenTelemetry), and a persistent vector store beyond roughly
tens of thousands of chunks.

---

## 8. Crosscutting concepts

### 8.1 Privacy model

| Concern | Concept |
|---------|---------|
| What is sensitive | patient identifiers (name, national ID / rodne cislo, ABHA, Aadhaar, MRN, phone, e-mail, address, dates of birth, relatives) and staff identifiers (names, e-mail, phone). Role mailboxes such as `privacy@` are public and left intact. |
| Where it is handled | at **ingestion** (documents, reports) and at **query time** (the query). Nothing downstream - index, prompt, logs, metrics, traces, feedback - receives raw identifiers. |
| Pseudonymisation | `[KIND_<6 hex of HMAC-SHA256(PSEUDO_KEY, kind|normalised value)>]`. Deterministic (retrieval works on masked text), keyed (not reversible by guessing), per-request vault for re-hydration. |
| Re-hydration policy | `query_only`: only tokens that came from the user's own question are restored. Identifiers that exist only in documents are never re-identified. |
| De-identification of reports | irreversible markers + generalisation (DOB to year or removed above age 89; other dates to year-month; 5-year age bands, `90+`; postcode to 3 digits). Output status `deidentified` / `needs_review`; always described as *de-identified, not anonymous*. |
| Identity rule | a de-identified report is never offered in answer to a query that names a person or identifier. |
| Defence in depth | residual scan of every prompt; output guard rejects any identifier-shaped text before re-hydration. |
| Legal framing | pseudonymised data remains personal data (GDPR); anonymisation needs DPO sign-off; a DPIA and a processor agreement are prerequisites for production. |

Known detector coverage is cue-based and format-based (chapter 11); the system is designed so that a
detector miss is caught by a later layer rather than trusted.

### 8.2 Security

| Layer | Control |
|-------|---------|
| Input | NFKC normalisation; length cap (800 chars); regex families for override, prompt extraction, role hijack, mass exfiltration (two phrasings), unmask requests, delimiter break-out, encoded payloads; de-spacing pass against `i g n o r e` obfuscation; clinical-advice scope patterns |
| Context | retrieved chunks containing injection patterns are dropped (`injection_in_context`) before prompting; documents are fenced as `<doc>` data and declared untrusted in the system prompt |
| Prompt | residual-PII gate (fail closed); per-request canary in the system prompt |
| Output | canary and system-prompt leak, residual identifiers, citation ids must exist, lexical groundedness >= 0.55 |
| Transport / API | security headers (CSP, nosniff, frame deny, no-store), pydantic request validation, per-client rate limit, admin bearer token with constant-time comparison, `debug` disabled in prod |
| Supply chain | pinned dependency ranges; secrets only via environment; container runs as non-root |
| Testing | promptfoo deterministic suite (38 tests, outcome-based) + generative red-team config (not yet run) |

Heuristic guards are a **cost-saving first filter**, not the defence: the structural properties
(identifier-free index and prompt) hold even if every regex is bypassed. A mutation check, with the input
guard disabled, made 6 of 20 earlier promptfoo tests fail while the other 14 still passed.

### 8.3 Observability concept

**Signals**

| Signal | Content | Sink |
|--------|---------|------|
| Metrics | request outcomes, latency, per-stage latency, guardrail events by guard and reason, chunks dropped by reason, identifiers masked by *kind*, context size, retrieval top score, groundedness, LLM tokens, feedback, HTTP by route template, index and de-identification gauges, build info | `/metrics` (Prometheus) |
| Traces | per-request ordered stage spans with durations; optional OpenTelemetry spans | in-memory ring buffer (`/v1/traces`); OTLP exporter |
| Logs | one JSON line per request with the `RequestRecord`; warnings for blocked output / residual PII | stderr |

**Privacy rules for telemetry (enforced by tests)**

* labels come from closed sets (status, stage, guard, reason, PII *kind*, route template); user input is never a label;
* the query appears only as a **keyed hash + length**; `observability.log_queries: true` (dev) adds the
  *pseudonymised* query; answers and document text are never recorded;
* span attributes are limited to scalars and short strings; chunk ids but no text;
* feedback uses reason codes only;
* `/v1/traces` and `/v1/corpus` are admin-only and disabled in prod without a token.

**Use**

* Dashboards: the Grafana board (16 panels) and the Streamlit Observability tab (KPIs, latency
  quantiles computed from histogram buckets, health checks, per-stage times, guardrail table, recent
  traces, trace inspector).
* Alerts: residual PII in a prompt, report failing de-identification (both critical), blocked outputs,
  input-guard spikes, high no-evidence rate, falling retrieval score, negative feedback, availability,
  5xx, p95 latency.
* Debugging: stage sub-timings (`retrieve.bm25`, `.dense`, `.rerank`) located the cross-encoder as the
  latency driver in the semantic profile (about 240 of 263 ms retrieval on CPU).

### 8.4 Configuration

Typed `Settings` (pydantic) loaded from a YAML profile, then overridden by environment variables.
Profiles differ only in embedder, re-ranker and the evidence-floor thresholds, which are calibrated per
profile (ADR-004). No behavioural flag is hidden in code.

### 8.5 Error handling and resilience

Fail closed on anything privacy-related; fail soft on optional features (OTel disabled if absent).
Stages are time-boxed implicitly (in-process, CPU-bound); the LLM call is the only network dependency and
is skipped when there is no evidence. No retries are implemented (single synchronous call).

### 8.6 Testing strategy

| Level | What | Count / evidence |
|-------|------|------------------|
| Unit | detectors, anonymiser, chunker, guards, metrics maths, root resolution | 82 tests in `tests/` |
| Component | API incl. admin gating, metrics, feedback, privacy of telemetry; OTel bridge | `test_api.py`, `test_observability.py` |
| UI | Streamlit app through `AppTest`, HTTP backend against the real API | `test_ui.py` |
| Ops config | alerts and dashboard reference real metrics and labels; compose wiring | `test_ops_config.py` |
| Quality | retrieval ablation, end-to-end, safety, de-identification metrics with gates | `eval/run_eval.py`, 59 cases + 4 annotated reports |
| Security | adversarial and privacy regression through the live HTTP interface | promptfoo, 38 tests |
| Mutation | disable a guard, confirm the suite notices | done once (input guard) |

### 8.7 Language and localisation

English only. Person-name matching folds diacritics, so a name typed without accents still matches the
accented form in a document; this is a typing convenience, not Slovak or Czech language support.

---

## 9. Architecture decisions

Format: context, decision, consequences. Status is *accepted* unless stated.

| ADR | Title |
|-----|-------|
| 001 | Protect identifiers before indexing; keyed deterministic pseudonyms |
| 002 | Irreversible de-identification path for clinical reports |
| 003 | Hybrid retrieval (BM25 + dense) fused by RRF, then re-ranked |
| 004 | Evidence floor from embedding cosine, not from cross-encoder probability |
| 005 | Minimum-context prompt policy; sentence pruning off |
| 006 | Layered fail-closed guardrails; heuristics first, adversarial tests as the proof |
| 007 | Never present a de-identified report as belonging to a named person |
| 008 | Offline-first default profile with a stand-in answerer |
| 009 | Clinical-advice scope guard at the input |
| 010 | Privacy-by-design observability (Prometheus + structured logs + optional OpenTelemetry) |
| 011 | Streamlit UI behind a swappable backend |
| 012 | Span-labelled evaluation with per-profile CI gates |
| 013 | Single-process, in-memory index |

---

#### ADR-001 Protect identifiers before indexing; keyed deterministic pseudonyms
**Context.** Staff questions and source documents contain patient and staff identifiers. The index, the
LLM prompt, logs and metrics must not.
**Decision.** Mask at the ingestion boundary and mask the query with the *same* function. Token =
`[KIND_<HMAC-SHA256(PSEUDO_KEY, kind|value)[:6]>]`. A per-request vault restores only values the user
typed (`rehydrate: query_only`).
**Consequences.** + No mapping table to guard at index time; retrieval still matches a typed name to its
document mentions. + A compromised index leaks tokens, not identifiers. - Security rests on `PSEUDO_KEY`;
rotating it changes every token and forces a re-index. - 6 hex characters per kind is a collision risk at
very large scale (acceptable here). - Detector misses become index content (mitigated by the residual scan).
**Rejected.** Masking only at prompt time (leaves identifiers in the index and logs); random tokens
(breaks cross-document linkage needed for retrieval).

#### ADR-002 Irreversible de-identification path for clinical reports
**Context.** Reports are headed for search and possibly research; linkable tokens are the wrong tool.
**Decision.** Rule-based transformation: labelled fields and narrative detectors replaced by generic
markers (no hash), DOB to year (removed above 89), other dates to year-month, 5-year age bands (`90+`),
postcode to 3 digits; residual re-scan sets `needs_review`; output is *de-identified, not anonymous*.
**Consequences.** + Aligns with the knowledge-base policy (small-numbers rule, DPO sign-off). + Result
object holds counts, not values. - Year-month dates lose ordering of events. - Rules, not a model:
indirect identifiers in free text (rare conditions, occupations, small places) are not addressed.
**Rejected.** Pseudonymising reports (keeps linkability); a consistent per-patient date shift (needs a
stable patient key that de-identification removes).

#### ADR-003 Hybrid retrieval fused by RRF, then re-ranked
**Context.** Exact terms (LOINC, "72 hours") and paraphrases ("find out who looked at their file") both occur.
**Decision.** BM25 and dense search, Reciprocal Rank Fusion (k = 60), then a re-ranker over <= 8 candidates.
**Consequences.** + Rank fusion needs no score calibration. + Measured: BM25 span-recall@3 0.913 -> hybrid
1.000; MRR 0.848 -> 0.942 with the cross-encoder. - The cross-encoder cost dominates latency (about 240 ms
on CPU) and cost one query's recall@3 versus plain hybrid (1.000 -> 0.978).

#### ADR-004 Evidence floor from embedding cosine, not cross-encoder probability
**Context.** Cross-encoder probabilities order well but were uncalibrated for abstention: a correct
passage for a paraphrased question scored 0.003 while an on-topic wrong one scored 0.3.
**Decision.** In the semantic profile abstain when the best embedding cosine is below `min_dense_cosine`
(0.25); the cross-encoder only orders; a relative floor (55 % of best) trims the tail. The offline profile
uses its heuristic re-ranker score with an absolute floor.
**Consequences.** + Off-topic questions are declined (cosine <= 0.13 off-topic vs >= 0.33 answerable). -
On-topic but unanswerable questions pass ("Which hospitals in Poland use the platform?"); needs an
LLM/NLI answerability check. Gate: `refusal_accuracy >= 0.75`.

#### ADR-005 Minimum-context prompt policy; sentence pruning off
**Decision.** Filters in order: evidence floor, relative floor, identity rule, injection filter,
de-duplication (Jaccard > 0.8), `max_chunks = 3`, token budget 450. No LLM call when nothing survives. Only
text and a citation label (`c1`) are sent. Sentence pruning exists but is off.
**Consequences.** + 64-81 mean context tokens vs about 2,700 for the corpus; cheaper, less to leak. -
Cross-section answers need several chunks. Pruning deleted answers on paraphrased queries, so it is opt-in.

#### ADR-006 Layered fail-closed guardrails; heuristics first, adversarial tests as the proof
**Decision.** Input regex guard (with normalisation and de-spacing) -> context sanitiser -> prompt
residual-PII gate -> output guard; structural isolation (data fenced and declared untrusted; canary).
Tests assert outcomes ("nothing leaked") rather than which guard fired.
**Consequences.** + Swapping a regex for a classifier does not invalidate the suite. - Regexes miss novel or
non-English phrasing; the French case is only checked for a safe outcome. A letter-spacing bypass was found
by promptfoo and fixed by a normaliser.

#### ADR-007 Never present a de-identified report as belonging to a named person
**Context.** "What was Jana Novakova's hemoglobin?" matches the hematology report on analyte names alone.
**Decision.** If the query contains a person, national ID, MRN, e-mail, phone, ABHA, Aadhaar, SSN or DOB
token, drop `anonymised` chunks (`anonymised_report_vs_identified_query`).
**Consequences.** + Prevents wrong-patient attribution. - Legitimate cohort questions that mention a
person are answered without report content.

#### ADR-008 Offline-first default profile with a stand-in answerer
**Decision.** `configs/default.yaml` uses a hashing embedder, heuristic re-ranker and an extractive
"LLM". `configs/semantic.yaml` and `llm.provider: anthropic` plug in the real components behind the same
interfaces.
**Consequences.** + CI needs no network or key; fast tests. - Answer-text metrics under-state real
quality; semantic quality needs the second profile; the real-LLM path has not been run.

#### ADR-009 Clinical-advice scope guard at the input
**Context.** Once diabetes lab reports were indexed, "What dose of metformin should I prescribe?" became
topically close and was answered. Relying on "nothing relevant found" stopped working.
**Decision.** Refuse advice-seeking phrasing at the input (`clinical_advice`), allow policy questions that
merely mention medication or diagnoses; add the rule to the system prompt too.
**Consequences.** + Deterministic and testable both ways. - Phrase-based; novel phrasings fall back to the
system prompt and retrieval. An intent classifier is the next step.

#### ADR-010 Privacy-by-design observability
**Context.** Telemetry for a health system can itself become the leak.
**Decision.** One `RequestRecord` per request -> Prometheus (closed label sets), bounded in-memory trace
store, one JSON log line; optional OpenTelemetry mirroring with an attribute whitelist; keyed query hash
instead of text; feedback reason codes; admin-gated trace and corpus endpoints; route templates for HTTP
metrics.
**Consequences.** + Tests prove the absence of identifiers in metrics, traces and logs. + Dashboards and
alerts are checked against the real metric names. - Debugging a single bad answer needs `log_queries` in dev
or the trace id plus the evaluation set. - Per-process trace store.
**Rejected.** Logging queries "for debugging"; free-text feedback; a third-party APM that would receive text.

#### ADR-011 Streamlit UI behind a swappable backend
**Decision.** `Backend` protocol with an in-process implementation (demo) and an HTTP implementation
(deployment). In HTTP mode the UI holds no index and no key.
**Consequences.** + The same app demos locally and deploys safely; tested through `AppTest` and against the
real API. - Anonymise/pseudonymise demo tabs run the stateless libraries locally in either mode; the visual
layout was not inspected in a real browser.

#### ADR-012 Span-labelled evaluation with per-profile CI gates
**Decision.** Relevance = chunk contains a gold *text span* (survives re-chunking). Metrics: retrieval
ablation (bm25, dense, hybrid, hybrid+rerank), end-to-end, safety, de-identification (planted identifiers
and must-survive strings). Hard zero gates for leaks; regression floors just below current values for quality.
**Consequences.** + A weak first eval (everything 1.0) was fixed by adding paraphrased cases. - Small,
author-written data: signals, not benchmarks.

#### ADR-013 Single-process, in-memory index
**Decision.** Rebuild BM25/dense/trace store in memory at start-up.
**Consequences.** + Simple, fast at this size. - No horizontal sharing of rate limits or traces; start-up
time grows with corpus size; see 7.5.

---

## 10. Quality requirements

### 10.1 Quality tree

```mermaid
mindmap
  root((ragkit quality))
    Privacy
      No identifiers in index, prompt, telemetry
      Report de-identification recall
      Fail closed
    Safety
      No medical advice
      No wrong-patient attribution
      Injection resistance
    Correctness
      Retrieval accuracy
      Groundedness and citations
      Abstain without evidence
    Observability
      Every request traced
      Alerts on privacy failures
      Telemetry is itself private
    Testability
      Gated evaluation
      Adversarial suite
    Efficiency
      Latency
      Minimal context
```

### 10.2 Quality scenarios and evidence

Measured on the development machine (CPU only, synthetic corpus of 8 documents + 4 reports = 48 chunks).
"Offline" is `configs/default.yaml`; "semantic" is `configs/semantic.yaml` (MiniLM embedder + MS-MARCO
cross-encoder).

| ID | Quality | Scenario (stimulus -> response) | Measure / target | Result | Evidence |
|----|---------|---------------------------------|------------------|--------|----------|
| Q1 | Privacy | Query contains name, national ID, e-mail, phone -> none appears in prompt, metrics, traces, logs | occurrences = 0 | **0** | `test_no_pii_in_metrics_traces_or_logs_by_default`; eval `prompt_pii_leaks = 0` |
| Q2 | Privacy | Index inspected for any planted report identifier | removal recall = 1.0 | **1.00** (74 planted, 4 reports) | `eval anon.phi_recall`, `test_anonymise.py`. *Planted by the author in expected formats: a regression guard, not field recall.* |
| Q3 | Privacy | Clinical content must survive de-identification | retention >= 0.95 | **1.00** (24 strings) | `anon.utility_retention` |
| Q4 | Privacy | A residual identifier in an assembled prompt | blocked, LLM not called | blocked (`blocked_prompt`) | `pipeline._answer`; alert `RagkitResidualPiiInPrompt` |
| Q5 | Safety | Dosing / diagnosis request | refused, no retrieval | refused (3 phrasings; 3 legitimate policy questions allowed) | `test_medical_advice_is_refused_up_front`, `test_input_guard_allows_legitimate_...` |
| Q6 | Safety | Named patient asked about a de-identified report | report content not shown | dropped (`anonymised_report_vs_identified_query`) | `test_named_person_query_never_gets_an_anonymised_report`, eval `rp02` |
| Q7 | Security | 38 adversarial / privacy / utility cases against the live API | 100 % pass | **38 / 38** | promptfoo `results.json` |
| Q8 | Security | Indirect injection planted in a case note | never reaches prompt | dropped (`injection_in_context`) | `test_poisoned_chunk_never_reaches_prompt` |
| Q9 | Correctness | Retrieval, semantic profile, 46 labelled queries (13 hard paraphrases) | span-recall@3 >= 0.97, MRR >= 0.93 | **0.978 / 0.942** | `eval` (gate) |
| Q10 | Correctness | Retrieval, offline profile | recall@3 >= 0.88, MRR >= 0.80 | 0.935 / 0.844 | `eval` (gate); paraphrases mostly fail by design |
| Q11 | Correctness | End-to-end status accuracy | >= 0.95 (semantic), >= 0.75 (offline) | 0.983 / 0.780 | `eval` |
| Q12 | Correctness | Answers are grounded and cite valid ids | groundedness >= 0.9 | 1.00 | `eval`, output guard |
| Q13 | Correctness | Off-topic / unanswerable questions | refusal accuracy | 0.75 (3 of 4); the miss is an on-topic hard negative | `eval` (gate = 0.75), ADR-004 |
| Q14 | Efficiency | Latency per question, excluding a network LLM | p95 < 2 s (alert threshold) | offline **p50 2.2 ms / p95 5.5 ms**; semantic **p50 241 ms / p95 418 ms** (re-rank about 240 ms) | per-stage spans |
| Q15 | Efficiency | Context sent to the model | <= 450 tokens | mean 64 (offline) / 81 (semantic) vs about 2,700 corpus | `eval context_tokens` |
| Q16 | Observability | Every outcome (incl. refusals, errors) produces a record, a metric and a log line | 1 per request | yes | `test_metrics_endpoint_counts_outcomes_and_stages` |
| Q17 | Observability | Dashboards and alerts reference only existing metrics and labels | 0 unknown | 0 | `test_ops_config.py` (checker proven to fail on bad names) |
| Q18 | Testability | A guard is removed | suite fails | 6 / 20 earlier promptfoo tests failed with the input guard off | mutation check (not repeated after corpus changes) |

### 10.3 Known gaps against the targets

* Q2/Q3 are measured on data written together with the detectors; field performance is unknown.
* Q13 is below an ideal 1.0 by design of the stand-in answerer and embedding-only abstention.
* Q14 excludes a hosted LLM, which has not been run; expect the network call to dominate.
* No availability, throughput or capacity targets were set or measured.

---

## 11. Risks and technical debt

### 11.1 Risks

Likelihood (L) and impact (I): H / M / L.

| ID | Risk | L | I | Mitigation in place | Residual action |
|----|------|---|---|---------------------|-----------------|
| R1 | A **name or identifier format is not detected** (names are cue- and gazetteer-based; unseen formats, typos, transliterations) and reaches the index or prompt | M | H | residual scan of every prompt (fails closed); output guard; promptfoo privacy cases; alert on `residual_pii` | NER (Presidio / clinical NER), sampled human audit, format inventory per country |
| R2 | **Indirect identifiers in report free text** (rare conditions, occupations, small places, dates in prose) re-identify a patient | M | H | labelled as *de-identified, not anonymous*; DPO sign-off required; dates generalised | k-anonymity / uniqueness analysis across a real corpus; clinical NER; manual review workflow |
| R3 | Rule-based de-identification **tested only on author-written reports** gives false confidence | H | M | disclosed (10.3); eval gate is a regression guard | evaluate on consented, de-identified real samples with independent annotation |
| R4 | **Pseudonymisation key compromise or loss** | L | H | key only via environment; prod refuses to start without it; tokens are keyed | secret manager, rotation procedure with re-index, separate keys per region |
| R5 | A report that fails the residual check **stays indexed** until an operator acts | M | H | gauge + critical alert | quarantine automatically: exclude `needs_review` documents from the index |
| R6 | **Prompt-injection bypass** of regex guards (novel, multilingual, encoded) | M | M | layered controls; structural isolation; no raw identifiers to steal; adversarial suite | classifier-based guard (Llama Guard / Prompt Guard); run the generative red-team regularly |
| R7 | **Medical advice slips through** the phrase-based scope guard | M | H | system-prompt rule; evidence floor | intent classifier; red-team plugin for specialised advice |
| R8 | **LLM hallucination** in production is unmeasured (the real LLM path has not been run) | H | M | groundedness and citation checks; abstain without evidence | run the evaluation and red-team with `llm.provider: anthropic`; human-rated sample |
| R9 | Evaluation set is **small and author-written** (59 cases) | H | M | disclosed; span labels; hard slice | grow with real, consented questions; add confidence intervals |
| R10 | **Telemetry exposure**: admin endpoints or metrics reachable by unauthorised parties | L | M | token, 404 in prod without token, no text in telemetry | network policy; mTLS or gateway auth in front of `/metrics` and `/v1/*` |
| R11 | **Unverified deployment artefacts** (Docker image, compose, Prometheus rules, Grafana JSON) fail on first use | M | M | static validation (`test_ops_config.py`); same code run live outside Docker | build and run the stack in CI; `promtool check rules` |
| R12 | **Runtime drift**: Python 3.10.0 / pydantic incompatibility, model libraries pinned in a scratch environment | M | L | environment marker; clean venv; CI on 3.11 | lock files; test the `dense` extra in CI |
| R13 | Cross-encoder **latency and recall regression** as the corpus grows | M | M | per-stage timings, alert on p95; hybrid fallback available | cache, smaller candidate set, GPU or distilled model |
| R14 | **No end-user authentication / authorisation**; re-hydration assumes the caller may see the named person's data | H | H | none in this service | put behind an IdP / gateway; enforce per-user access to the records referenced |

### 11.2 Technical debt

| Item | Why it exists | Cost of leaving it |
|------|---------------|--------------------|
| In-memory index and trace store, per-process rate limiter | simplicity at this scale | cannot scale out or survive restarts with history |
| Regex name/identifier detectors and guards | no model dependency, deterministic, testable | recall ceiling (R1, R6) |
| Extractive stand-in answerer | offline CI | answer-quality metrics are a floor; real path unverified |
| Chunker adds a full stop to list lines | simplest way to split result tables | affects exact text of some chunks (cosmetic) |
| `observability.otel` uses a console exporter when no OTLP endpoint is set | convenience | noisy if enabled accidentally in prod |
| Year-month date generalisation | no stable patient key after de-identification | event ordering lost inside a month |
| Dashboards reference Grafana datasource uid `prometheus` | provisioning convenience | breaks if the uid is changed |
| `tests/test_ops_config.py` reads `_names_to_collectors` (private API of `prometheus_client`) | no public equivalent | may need adjusting on upgrade |
| Streamlit layout not inspected in a real browser | no browser tooling in the authoring environment | cosmetic defects possible; functionality is covered by `AppTest` and a live HTTP-mode run |

### 11.3 Open decisions

1. Which NER stack for names (and is a clinical model permitted by the data-protection impact assessment)?
2. Who owns the DPO sign-off workflow that moves a report from *de-identified* to *anonymous*?
3. Do researchers need stable intra-patient date intervals (date shifting) badly enough to keep a
   per-patient secret inside the de-identification boundary?
4. Is a hosted LLM acceptable for pseudonymised health text under the processor agreement, or must the
   model run in-region?

---

## 12. Glossary

| Term | Definition |
|------|------------|
| **ABHA** | Ayushman Bharat Health Account number (India, format `##-####-####-####`); treated as a direct identifier. |
| **ABDM** | Ayushman Bharat Digital Mission; consent manager referenced in the knowledge base for Indian patients. |
| **Anonymisation** | Irreversible removal of the possibility to identify a person. In this project it requires DPO sign-off of a re-identification risk assessment. |
| **Anonymised chunk** | `Chunk.anonymised == True`: text from a de-identified clinical report. |
| **BM25** | Lexical ranking function used for keyword retrieval. |
| **Break-glass** | Emergency access to a record outside the usual assignment (policy topic in the knowledge base). |
| **Canary** | A random per-request marker placed in the system prompt; its appearance in an output signals prompt leakage. |
| **Context policy** | The ordered set of filters that decides which retrieved chunks reach the LLM. |
| **Cross-encoder** | Model scoring a (query, passage) pair jointly; used as the re-ranker in the semantic profile. |
| **De-identification** | Removal and generalisation of identifiers so a record is no longer directly attributable; not necessarily anonymous. |
| **DPIA / DPO** | Data protection impact assessment / Data Protection Officer. |
| **EHDS** | European Health Data Space. |
| **Evidence floor** | Minimum relevance (re-rank score or embedding cosine) below which the system answers "no information". |
| **FHIR / HL7 / LOINC / ICD-10 / SNOMED CT / ATC** | Healthcare interoperability standard and code systems named in the knowledge base. |
| **Gazetteer** | Set of person names harvested from the corpus and used to find names in free text. |
| **Groundedness** | Share of content words of an answer that occur in the supplied context (lexical proxy). |
| **Hard slice** | Evaluation questions phrased with little lexical overlap with the documents (paraphrases). |
| **MRN** | Medical record number; an identifier masked in queries and reports. |
| **MRR / nDCG / span recall** | Retrieval metrics: mean reciprocal rank; normalised discounted cumulative gain; share of gold text spans present in the top-k chunks. |
| **needs_review** | De-identification status meaning the residual re-scan found something; must be reviewed before the report is trusted. |
| **Pseudonymisation** | Replacing identifiers by tokens that can be reversed by someone holding the key (here an HMAC key plus a per-request vault). |
| **Query-origin re-hydration** | Restoring only those tokens that were produced from the user's own question. |
| **Residual-PII gate** | Re-scan of the assembled prompt (and of model output) for identifier patterns; failing closed. |
| **RRF** | Reciprocal Rank Fusion: combines ranked lists using only ranks. |
| **Request record** | The single telemetry data contract produced for each question. |
| **Rodne cislo (national ID)** | Slovak / Czech birth number, format `######/####`; treated as a direct identifier. |
| **Stand-in answerer** | The extractive "LLM" that quotes retrieved sentences so the system runs offline. |
| **Trace / span** | Per-request record of timed stages. |
| **Vault** | Per-request mapping from token to original value; never persisted or logged. |
