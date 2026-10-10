"""Output guardrail: runs on the raw model output, BEFORE re-hydration.

Checks
  1. canary / system-prompt leakage
  2. residual raw PII (the model must only ever emit privacy tokens, never real values)
  3. citation validity (cited ids must exist in the context actually supplied)
  4. groundedness proxy: share of answer content-words that appear in the supplied context
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..privacy.pseudonymise import Pseudonymiser
from ..retrieval.text import tokenize

NO_ANSWER = "I don't have enough information in the knowledge base to answer that."
_CITE = re.compile(r"\[(c\d+)\]")


@dataclass
class OutputVerdict:
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    groundedness: float = 1.0


def check_output(answer: str, context_text: str, valid_ids: set[str], canary: str,
                 system_prompt: str, pseudo: Pseudonymiser,
                 min_groundedness: float = 0.55) -> OutputVerdict:
    reasons: list[str] = []
    if canary in answer:
        reasons.append("canary_leak")
    # leaking a long verbatim slice of the system prompt
    for line in system_prompt.splitlines():
        line = line.strip()
        if len(line) > 40 and line in answer:
            reasons.append("system_prompt_leak")
            break
    if pseudo.residual(answer):
        reasons.append("residual_pii")

    if answer.strip() == NO_ANSWER:
        return OutputVerdict(not reasons, reasons, 1.0)

    cited = set(_CITE.findall(answer))
    if not cited:
        reasons.append("no_citation")
    elif not cited <= valid_ids:
        reasons.append("invalid_citation")

    words = [w for w in tokenize(_CITE.sub("", answer)) if "_" not in w]
    ctx = set(tokenize(context_text))
    g = (sum(w in ctx for w in words) / len(words)) if words else 1.0
    if g < min_groundedness:
        reasons.append("low_groundedness")
    return OutputVerdict(not reasons, reasons, round(g, 3))
