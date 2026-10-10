"""Prompt-injection / exfiltration heuristics shared by the input guard and the context sanitiser.

Heuristics are a cheap first layer, not a complete defence. They are backed by: structural prompt
isolation (untrusted data fenced + labelled), pseudonymisation (nothing sensitive to leak), the
output guard, and adversarial testing with promptfoo (see promptfoo/).
"""
from __future__ import annotations

import re
import unicodedata

_PATTERNS: dict[str, re.Pattern] = {
    "override_instructions": re.compile(
        r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|above|earlier|all|any|your)\b"
        r".{0,40}\b(instruction|prompt|rule|guideline|polic)", re.I | re.S),
    "reveal_prompt": re.compile(
        r"\b(reveal|show|print|repeat|display|output|leak|tell me)\b.{0,40}\b(system|hidden|initial|original)\b"
        r".{0,20}\b(prompt|instruction|message|rule)", re.I | re.S),
    "role_hijack": re.compile(
        r"\b(you are now|act as|pretend (to be|you are)|developer mode|jailbreak|\bDAN\b)", re.I),
    "mass_exfiltration": re.compile(
        r"\b(list|show|give|dump|export|reveal|send|print|tell)\b.{0,30}\b(every|all|each)\b.{0,30}"
        r"\b(customers?|users?|employees?|patients?|clinicians?|doctors?|researchers?)\b.{0,50}"
        r"\b(email|phone|address|password|card|ssn|data|id|identifier|number|record|name|diagnos)",
        re.I | re.S),
    "mass_exfiltration_b": re.compile(
        r"\b(list|show|give|dump|export|reveal|send|print|tell)\b.{0,40}"
        r"\b(id|identifier|number|record|email|phone|diagnos\w*|name)s?\b.{0,25}"
        r"\b(every|all|each)\b.{0,25}\b(customers?|users?|patients?|clinicians?|doctors?|employees?)\b",
        re.I | re.S),
    "raw_pii_request": re.compile(
        r"\b(real|actual|original|unmasked|un-?redacted)\b.{0,30}\b(name|email|phone|number|identity|value)",
        re.I | re.S),
    "encoded_payload": re.compile(r"\b[A-Za-z0-9+/]{80,}={0,2}\b"),
    "delimiter_break": re.compile(r"</?\s*(context|system|assistant|instructions?)\s*>", re.I),
}


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")


_SPACED = re.compile(r"(?<![A-Za-z])(?:[A-Za-z] ){3,}[A-Za-z](?![A-Za-z])")


def despace(text: str) -> str:
    """Undo letter-spacing obfuscation: 'i g n o r e   p r e v' -> 'ignore prev'."""
    marked = re.sub(r"\s{2,}", " \x00 ", text)           # wide gaps are the real word breaks
    marked = _SPACED.sub(lambda m: m.group().replace(" ", ""), marked)
    return marked.replace(" \x00 ", " ")


def detect(text: str) -> list[str]:
    t = normalise(text)
    variants = {t, despace(t)}
    return sorted({name for v in variants for name, pat in _PATTERNS.items() if pat.search(v)})
