"""Input guardrail: runs on the RAW user query, before anything else.

Two families of checks:
* security   - injection / exfiltration / oversize (see injection.py)
* scope      - this assistant explains platform POLICY and PROCEDURE. Requests for medical advice,
               diagnosis or dosing are refused up front rather than left to "nothing relevant was
               retrieved" (that stopped working as soon as clinical reports entered the corpus).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .injection import detect, normalise

MAX_QUERY_CHARS = 800

# Advice-seeking phrasing only. Policy questions that merely mention medication or diagnoses
# ("how are diagnoses coded?", "are dosage fields validated?") must keep working.
_CLINICAL_ADVICE = [
    re.compile(r"\bprescribe\b", re.I),
    re.compile(r"\bshould (?:I|we|the patient)\b.{0,30}\b(?:take|give|administer|treat|start|stop|use)\b",
               re.I | re.S),
    re.compile(r"\b(?:what|which|how much)\b.{0,20}\b(?:dose|dosage|dosing|medication|drug|medicine|"
               r"treatment|therapy)\b.{0,40}\b(?:should|to give|to prescribe|to take|to use)\b",
               re.I | re.S),
    re.compile(r"\b(?:what(?:'s| is) the diagnosis|diagnose (?:me|him|her|my|this|the patient)|"
               r"what is wrong with)\b", re.I),
    re.compile(r"\bmy patient (?:has|is|presents|complains|suffers)\b", re.I),
]


@dataclass
class InputVerdict:
    allowed: bool
    query: str
    reasons: list[str] = field(default_factory=list)


def check_input(query: str) -> InputVerdict:
    q = normalise(query).strip()
    if not q:
        return InputVerdict(False, q, ["empty_query"])
    if len(q) > MAX_QUERY_CHARS:
        return InputVerdict(False, q[:MAX_QUERY_CHARS], ["query_too_long"])
    hits = detect(q)
    if any(p.search(q) for p in _CLINICAL_ADVICE):
        hits = hits + ["clinical_advice"]
    if hits:
        return InputVerdict(False, q, hits)
    return InputVerdict(True, q)
