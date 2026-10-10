"""LLM adapters. The pipeline only depends on ``generate(bundle) -> str``.

* ExtractiveLLM  - offline, deterministic: returns the best-matching context sentences with citations.
                   Lets CI, evals and the demo run with no API key and zero cost.
* AnthropicLLM   - real model call; the prompt it receives is already pseudonymised + minimised.
"""
from __future__ import annotations

import os
from typing import Protocol

from ..config import LLMCfg
from ..guardrails.output import NO_ANSWER
from ..prompting.builder import _SENT, PromptBundle
from ..retrieval.text import tokenize


class LLM(Protocol):
    def generate(self, bundle: PromptBundle, question: str) -> str: ...


class ExtractiveLLM:
    def generate(self, bundle: PromptBundle, question: str) -> str:
        if not bundle.items:
            return NO_ANSWER
        q = set(tokenize(question))
        scored = []
        for item in bundle.items:
            for s in _SENT.split(item.text):
                ov = len(q & set(tokenize(s)))
                if ov:
                    scored.append((ov, item.cid, s.strip()))
        if not scored:  # no lexical overlap (paraphrase): fall back to the best chunk's lead sentences
            lead = _SENT.split(bundle.items[0].text)[:2]
            return " ".join(f"{s.strip()} [c1]" for s in lead)
        scored.sort(key=lambda x: -x[0])
        top = scored[:3]
        top.sort(key=lambda x: (x[1], bundle.items[int(x[1][1:]) - 1].text.find(x[2])))
        return " ".join(f"{s} [{cid}]" for _, cid, s in top)


class AnthropicLLM:
    def __init__(self, cfg: LLMCfg):
        import anthropic  # lazy: optional dependency
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.client = anthropic.Anthropic()
        self.cfg = cfg

    def generate(self, bundle: PromptBundle, question: str) -> str:
        msg = self.client.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            system=bundle.system,
            messages=[{"role": "user", "content": bundle.user}],
        )
        bundle.usage = {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens}
        return "".join(b.text for b in msg.content if b.type == "text").strip()


def make_llm(cfg: LLMCfg) -> LLM:
    return AnthropicLLM(cfg) if cfg.provider == "anthropic" else ExtractiveLLM()
