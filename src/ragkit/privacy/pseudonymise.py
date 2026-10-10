"""Deterministic, keyed pseudonymisation for health-domain text.

Design
------
* Token = ``[KIND_<6 hex of HMAC-SHA256(key, kind|normalised value)>]``.
  - Deterministic: the same patient/clinician gets the same token in every document and in the
    user's query, so retrieval still works on pseudonymised text and no mapping table is needed at
    index time.
  - Keyed: without PSEUDO_KEY a token cannot be brute-forced back to a value from a name list.
* The *index* only ever contains pseudonymised text. The LLM only ever sees pseudonymised text.
* A per-request ``Vault`` records token -> original for values the *user typed*, so the final answer
  can be re-hydrated for that user (policy ``rehydrate: query_only``). Values that exist only in
  retrieved documents are never re-identified.
* Names are matched accent-insensitively ("Lubomir Horvath" == "Ľubomír Horváth") because Slovak /
  Czech text is routinely typed without diacritics.
* Detected: e-mail, phone, Slovak/Czech birth number (rodné číslo), India ABHA / Aadhaar numbers,
  MRN, Luhn-valid card numbers, SSN, IPv4, dates of birth (cue-based), persons (honorific + cue
  phrases + corpus gazetteer). For production, add a NER detector (Presidio / spaCy) next to these.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import unicodedata
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    kind: str
    canonical: str   # normalised value used for hashing
    original: str


@dataclass
class Vault:
    """Per-request token -> original mapping. Never persisted, never logged."""
    mapping: dict[str, str] = field(default_factory=dict)

    def rehydrate(self, text: str) -> str:
        for tok, val in self.mapping.items():
            text = text.replace(tok, val)
        return text


def fold(s: str) -> str:
    """Lower-case and strip diacritics: 'Ľubomír' -> 'lubomir'."""
    d = unicodedata.normalize("NFKD", s)
    return "".join(c for c in d if not unicodedata.combining(c)).lower()


_UP = "A-ZÁÄČĎÉĚÍĽĹŇÓÔŔŘŠŤÚŮÝŽ"
_LO = "a-záäčďéěíľĺňóôŕřšťúůýž"
_NAME = rf"[{_UP}][{_LO}]+(?:[-'][{_UP}][{_LO}]+)?"
_FULLNAME = rf"{_NAME}\s+{_NAME}"

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# shared role mailboxes are public contacts, not personal data
ROLE_MAILBOXES = frozenset({"privacy", "support", "billing", "security", "info", "help", "dpo"})
_PHONE = re.compile(r"(?<![\w+])\+\d[\d\s().-]{6,}\d|\b\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")
_NATID = re.compile(r"\b\d{6}/\d{3,4}\b")                       # SK/CZ rodné číslo
_ABHA = re.compile(r"\b\d{2}-\d{4}-\d{4}-\d{4}\b")              # India ABHA number
_AADHAAR = re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")
_MRN = re.compile(r"\bMRN[-\s]?\d{5,}\b", re.I)
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_DOB = re.compile(r"(?:\bborn|\bDOB|\bdate of birth)\s*:?\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", re.I)
_WORD = re.compile(rf"[{_UP}{_LO}A-Za-z]+(?:-[{_UP}{_LO}A-Za-z]+)*")  # apostrophe excluded: "Horváth's"

_HONORIFIC = re.compile(
    rf"\b(?:Mr|Mrs|Ms|Dr|Prof|MUDr|Ing|Mgr)\.?\s+({_NAME}(?:\s+{_NAME})?)")
# cue phrases used in the corpus; harvested names feed the gazetteer
_CUES = re.compile(
    rf"(?:\bPatient|\bClinician|\bResearcher|\bcommander:|\bOfficer:|\blead:|\bon-call, currently)"
    rf"\s+(?:(?:Dr|Prof|MUDr)\.?\s+)?({_FULLNAME})\b")


def _luhn(num: str) -> bool:
    d = [int(c) for c in num[::-1]]
    return (sum(d[0::2]) + sum(sum(divmod(2 * x, 10)) for x in d[1::2])) % 10 == 0


def harvest_names(text: str) -> set[str]:
    """Find full person names via cue phrases; used to build the gazetteer at ingestion."""
    body = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    names = {m.group(1) for m in _CUES.finditer(body)}
    names |= {m.group(1) for m in _HONORIFIC.finditer(body) if " " in m.group(1)}
    return names


class Pseudonymiser:
    def __init__(self, key: str, known_names: set[str] | None = None):
        self._key = key.encode()
        self.gazetteer: dict[str, str] = {}  # folded surface form -> canonical full name
        for n in known_names or ():
            self.add_name(n)

    # ---- gazetteer -------------------------------------------------------------------------
    def add_name(self, full: str) -> None:
        full = " ".join(full.split())
        self.gazetteer[fold(full)] = full
        for part in full.split():  # "Jana" or "Nováková" alone resolve to the same person token
            if len(part) > 2:
                self.gazetteer.setdefault(fold(part), full)

    @property
    def known_names(self) -> set[str]:
        return set(self.gazetteer.values())

    # ---- detection -------------------------------------------------------------------------
    def scan(self, text: str) -> list[Span]:
        spans: list[Span] = []

        def add(rx: re.Pattern, kind: str, group: int = 0, canon=lambda s: s) -> None:
            for m in rx.finditer(text):
                spans.append(Span(m.start(group), m.end(group), kind, canon(m.group(group)),
                                  m.group(group)))

        for m in _EMAIL.finditer(text):
            if m.group().split("@")[0].lower() not in ROLE_MAILBOXES:
                spans.append(Span(m.start(), m.end(), "EMAIL", m.group().lower(), m.group()))
        add(_NATID, "NATID")
        add(_ABHA, "ABHA")
        add(_AADHAAR, "AADHAAR", canon=lambda s: re.sub(r"\D", "", s))
        add(_MRN, "MRN", canon=lambda s: re.sub(r"\W", "", s).upper())
        add(_SSN, "SSN")
        add(_DOB, "DOB", group=1)
        add(_IPV4, "IP")
        for m in _CARD.finditer(text):
            digits = re.sub(r"\D", "", m.group())
            if 13 <= len(digits) <= 19 and _luhn(digits):
                spans.append(Span(m.start(), m.end(), "CARD", digits, m.group()))
        for m in _PHONE.finditer(text):
            digits = re.sub(r"\D", "", m.group())
            if len(digits) >= 8:
                spans.append(Span(m.start(), m.end(), "PHONE", digits, m.group()))
        for m in _HONORIFIC.finditer(text):
            spans.append(Span(m.start(1), m.end(1), "PERSON", self._person_key(m.group(1)),
                              m.group(1)))
        self._gazetteer_spans(text, spans)
        return self._resolve(spans)

    def _person_key(self, surface: str) -> str:
        """Canonical key for a person: the gazetteer's full name if known, else the surface."""
        return fold(self.gazetteer.get(fold(surface), surface))

    def _gazetteer_spans(self, text: str, spans: list[Span]) -> None:
        """Greedy left-to-right: prefer a 2-word full name, else a single capitalised known word."""
        words = list(_WORD.finditer(text))
        i = 0
        while i < len(words):
            w = words[i]
            if not w.group()[0].isupper():
                i += 1
                continue
            if (i + 1 < len(words) and words[i + 1].group()[0].isupper()
                    and text[w.end():words[i + 1].start()].isspace()):
                pair = f"{w.group()} {words[i + 1].group()}"
                if fold(pair) in self.gazetteer:
                    end = words[i + 1].end()
                    spans.append(Span(w.start(), end, "PERSON", self._person_key(pair),
                                      text[w.start():end]))
                    i += 2
                    continue
            if fold(w.group()) in self.gazetteer:
                spans.append(Span(w.start(), w.end(), "PERSON", self._person_key(w.group()),
                                  w.group()))
            i += 1

    @staticmethod
    def _resolve(spans: list[Span]) -> list[Span]:
        """Drop overlaps, preferring earlier-start then longer spans."""
        spans = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
        out: list[Span] = []
        for s in spans:
            if out and s.start < out[-1].end:
                continue
            out.append(s)
        return out

    # ---- tokenisation ----------------------------------------------------------------------
    def token(self, kind: str, canonical: str) -> str:
        h = hmac.new(self._key, f"{kind}|{canonical}".encode(), hashlib.sha256).hexdigest()[:6]
        return f"[{kind}_{h}]"

    def pseudonymise(self, text: str, vault: Vault | None = None) -> tuple[str, Vault]:
        vault = vault if vault is not None else Vault()
        out, last = [], 0
        for s in self.scan(text):
            tok = self.token(s.kind, s.canonical)
            vault.mapping.setdefault(tok, s.original)
            out.append(text[last:s.start])
            out.append(tok)
            last = s.end
        out.append(text[last:])
        return "".join(out), vault

    def residual(self, text: str) -> list[Span]:
        """Spans of PII still present in already-pseudonymised text (defence in depth)."""
        return self.scan(text)
