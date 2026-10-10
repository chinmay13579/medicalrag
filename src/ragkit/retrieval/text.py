"""Shared tokenisation: accent folding, lowercase, stopword removal, light suffix stemming.

PII tokens such as ``[PERSON_ab12cd]`` survive as a single term ("person_ab12cd") so the same
person matches across documents and the query without exposing the name.
"""
from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(r"\[[A-Z]+_[0-9a-f]{6}\]|[a-z0-9]+(?:\.[0-9]+)?", re.I)

STOPWORDS = frozenset("""
a an the and or but if of to in on at by for with from as is are was were be been being do does did
i you we they it this that these those what which who whom how when where why can could should would
will shall may might must not no yes any all each per than then so such into out up down over under
about after before between during within without there their our your my me us has have had
""".split())


def stem(w: str) -> str:
    if "_" in w:          # privacy placeholder such as person_ab12cd: keep verbatim
        return w
    for suf in ("ations", "ation", "ings", "ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def tokenize(text: str, keep_stop: bool = False) -> list[str]:
    out = []
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    for m in _TOKEN.finditer(text):
        t = m.group().lower().strip("[]")
        if len(t) < 2 or (not keep_stop and t in STOPWORDS):
            continue
        out.append(stem(t))
    return out
