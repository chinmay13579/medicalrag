"""Typed settings: YAML defaults, overridable by env (RAGKIT_CONFIG, PSEUDO_KEY, RAGKIT_ENV)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel


def resolve_root() -> Path:
    """Project home: $RAGKIT_HOME, else the CWD if it looks like a project checkout / container
    workdir, else the repo layout relative to this file (editable installs)."""
    if os.environ.get("RAGKIT_HOME"):
        return Path(os.environ["RAGKIT_HOME"]).resolve()
    if (Path.cwd() / "configs" / "default.yaml").exists():
        return Path.cwd()
    return Path(__file__).resolve().parents[2]


ROOT = resolve_root()


class ChunkingCfg(BaseModel):
    target_words: int = 90
    overlap_sentences: int = 1
    min_words: int = 12


class RetrievalCfg(BaseModel):
    top_k_bm25: int = 10
    top_k_dense: int = 10
    rrf_k: int = 60
    fused_candidates: int = 8
    embedder: str = "hashing"
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker: str = "heuristic"
    ce_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class ContextCfg(BaseModel):
    max_chunks: int = 3
    max_context_tokens: int = 450
    min_rerank_score: float = 0.25
    relative_floor: float = 0.55
    min_dense_cosine: float = 0.0   # >0 enables a cosine evidence floor (semantic embedders)
    prune_sentences: bool = False


class PrivacyCfg(BaseModel):
    rehydrate: str = "query_only"
    fail_on_residual_pii: bool = True


class LLMCfg(BaseModel):
    provider: str = "extractive"
    model: str = "claude-sonnet-5-5"
    max_tokens: int = 400
    temperature: float = 0.0


class ObservabilityCfg(BaseModel):
    log_queries: bool = False      # False: log a keyed hash + length only, never query text
    trace_buffer: int = 500        # recent request records kept in memory for /v1/traces and the UI
    otel: bool = False             # export spans via OpenTelemetry (needs `pip install .[otel]`)


class Settings(BaseModel):
    data_dir: str = "data/raw"
    chunking: ChunkingCfg = ChunkingCfg()
    retrieval: RetrievalCfg = RetrievalCfg()
    context: ContextCfg = ContextCfg()
    privacy: PrivacyCfg = PrivacyCfg()
    llm: LLMCfg = LLMCfg()
    observability: ObservabilityCfg = ObservabilityCfg()
    env: str = "dev"
    pseudo_key: str = "dev-only-insecure-key"

    def path(self, p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else ROOT / q


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    cfg_path = Path(os.environ.get("RAGKIT_CONFIG", ROOT / "configs" / "default.yaml"))
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    s = Settings(**raw)
    s.env = os.environ.get("RAGKIT_ENV", "dev")
    if os.environ.get("RAGKIT_LOG_QUERIES") is not None:
        s.observability.log_queries = os.environ["RAGKIT_LOG_QUERIES"].lower() in ("1", "true", "yes")
    key = os.environ.get("PSEUDO_KEY")
    if key:
        s.pseudo_key = key
    elif s.env == "prod":
        raise RuntimeError("PSEUDO_KEY must be set when RAGKIT_ENV=prod")
    return s
