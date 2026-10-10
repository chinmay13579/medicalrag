"""De-identification of medical reports (lab, radiology, discharge summaries).

Pseudonymisation vs. anonymisation
----------------------------------
``pseudonymise.py`` replaces identifiers with *reversible-by-key, linkable* tokens (used for queries
and ordinary documents). This module is the *irreversible* path used for clinical reports:

* direct identifiers  -> generic category markers with NO hash, so two mentions can't be linked:
  ``[PATIENT] [CLINICIAN] [PERSON] [RELATIVE] [NATIONAL_ID] [ABHA] [MRN] [EMAIL] [PHONE]
  [FACILITY] [ACCESSION] [ADDRESS]``
* quasi-identifiers   -> generalised, following docs/data policy (retention_and_deidentification.md):
  date of birth -> year (suppressed if the person is over 89), every other date -> year-month,
  age -> 5-year band (``90+`` above 89), postal code -> first 3 digits (``130**``)
* clinical content    -> kept (analytes, values, units, reference ranges, LOINC/ICD codes, findings)

The result is labelled **de-identified, not anonymous**: free-text narrative can always carry
indirect identifiers (rare conditions, occupations, small places). Under the platform's own policy,
data counts as anonymised only after DPO sign-off of a re-identification risk assessment.

Safety net: after transformation the text is re-scanned for structured identifiers and for every
original value that was removed (word-boundary, accent-insensitive). Any hit => ``needs_review``.
The log records categories and counts only, never the original values.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .pseudonymise import _NAME, Pseudonymiser, fold

NOTE = ("De-identified (best effort), not anonymous: free text may still contain indirect "
        "identifiers. Requires DPO sign-off before being treated as anonymised.")

# labelled header fields -> replacement marker
FIELD_RULES: dict[str, str] = {
    "patient name": "[PATIENT]", "name": "[PATIENT]", "patient": "[PATIENT]",
    "ordering physician": "[CLINICIAN]", "referring physician": "[CLINICIAN]",
    "attending physician": "[CLINICIAN]", "requesting clinician": "[CLINICIAN]",
    "reported by": "[CLINICIAN]", "reviewed by": "[CLINICIAN]", "radiologist": "[CLINICIAN]",
    "pathologist": "[CLINICIAN]", "surgeon": "[CLINICIAN]", "authorised by": "[CLINICIAN]",
    "national id": "[NATIONAL_ID]", "birth number": "[NATIONAL_ID]",
    "abha number": "[ABHA]", "aadhaar": "[ID]", "insurance number": "[ID]",
    "mrn": "[MRN]", "phone": "[PHONE]", "mobile": "[PHONE]", "email": "[EMAIL]",
    "facility": "[FACILITY]", "hospital": "[FACILITY]", "clinic": "[FACILITY]",
    "laboratory": "[FACILITY]",
    "accession no": "[ACCESSION]", "report id": "[ACCESSION]", "specimen id": "[ACCESSION]",
    "order no": "[ACCESSION]",
}
PATIENT_FIELDS = {"patient name", "name", "patient"}
CLINICIAN_FIELDS = {k for k, v in FIELD_RULES.items() if v == "[CLINICIAN]"}

_FIELD = re.compile(
    r"^(?P<lead>[ \t]*(?:[-*][ \t]+)?)(?P<label>[A-Za-z][A-Za-z ./]{1,30}?):[ \t]*(?P<value>\S.*?)[ \t]*$",
    re.M)
_TITLE = re.compile(r"^(?:Dr|MUDr|Prof|Mr|Mrs|Ms|Ing|Mgr)\.?\s+", re.I)
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
           "october", "november", "december"]
_TIME = r"(?:\s+\d{1,2}:\d{2})?"
_DATE_NUM = re.compile(rf"\b(\d{{1,2}})[./](\d{{1,2}})[./](\d{{4}}){_TIME}\b")
_DATE_ISO = re.compile(rf"\b(\d{{4}})-(\d{{2}})-(\d{{2}}){_TIME}\b")
_DATE_TXT = re.compile(rf"\b(\d{{1,2}})\s+({'|'.join(m.capitalize() for m in _MONTHS)})\s+(\d{{4}})\b")
_AGE = re.compile(r"(?<![\w.])(\d{1,3})(?=[- ]year[- ]old)|\b(?:Age|Aged)\s*:?\s*(\d{1,3})\b")
_ACCESSION = re.compile(r"\bACC-\d{4}-\d{3,}\b")
_POSTAL = re.compile(r"\b(\d{3})\s?\d{2,3}\b")
_RELATIVE = re.compile(
    rf"(?i:\b(?:daughter|son|wife|husband|mother|father|sister|brother|spouse|partner|niece|nephew))"
    rf"\s+({_NAME}(?:\s+{_NAME})?)")

_KIND_LABEL = {"EMAIL": "[EMAIL]", "PHONE": "[PHONE]", "NATID": "[NATIONAL_ID]", "ABHA": "[ABHA]",
               "AADHAAR": "[ID]", "MRN": "[MRN]", "SSN": "[ID]", "CARD": "[ID]", "IP": "[IP]"}
_STRUCTURED = set(_KIND_LABEL)


@dataclass
class AnonResult:
    text: str
    log: dict[str, int]                     # category -> count; never contains original values
    leaks: list[str] = field(default_factory=list)   # categories of anything that survived
    status: str = "deidentified"            # deidentified | needs_review
    note: str = NOTE


def age_band(n: int) -> str:
    return "90+" if n >= 90 else f"{n // 5 * 5}-{n // 5 * 5 + 4}"


def _strip_title(v: str) -> str:
    return _TITLE.sub("", v).strip()


def _month_no(name: str) -> int:
    return _MONTHS.index(name.lower()) + 1


class Anonymiser:
    def anonymise(self, text: str) -> tuple[AnonResult, set[str]]:
        """Return (result, person_names). The names are returned separately so the caller can
        feed them to the query-side pseudonymiser gazetteer; they are never stored in the result."""
        spans: list[tuple[int, int, str, str, int]] = []   # start, end, replacement, category, prio
        sensitive: dict[str, str] = {}                      # original value -> category
        names: set[str] = set()
        patient_keys: set[str] = set()
        clinician_keys: set[str] = set()

        # ---- reference year for age-aware DOB handling ------------------------------------
        date_hits = self._dates(text)
        years = [y for *_, y in date_hits]
        ref_year = max(years) if years else None

        # ---- 1. labelled header fields ----------------------------------------------------
        for m in _FIELD.finditer(text):
            label, val = m.group("label").strip().lower(), m.group("value")
            s, e = m.span("value")
            if label in FIELD_RULES:
                spans.append((s, e, FIELD_RULES[label], FIELD_RULES[label].strip("[]"), 0))
                sensitive[val] = FIELD_RULES[label].strip("[]")
                if label in PATIENT_FIELDS | CLINICIAN_FIELDS:
                    full = _strip_title(val)
                    names.add(full)
                    (patient_keys if label in PATIENT_FIELDS else clinician_keys).add(fold(full))
                    for part in full.split():
                        if len(part) >= 3:
                            sensitive[part] = FIELD_RULES[label].strip("[]")
            elif label == "date of birth":
                spans.append((s, e, self._dob(val, ref_year), "DOB", 0))
                sensitive[val] = "DOB"
            elif label == "address":
                pm = _POSTAL.search(val)
                repl = "[ADDRESS]" + (f" (postal area {pm.group(1)}**)" if pm else "")
                spans.append((s, e, repl, "ADDRESS", 0))
                sensitive[val] = "ADDRESS"
                if pm:
                    sensitive[pm.group(0)] = "ADDRESS"
                for w in re.findall(r"[A-Za-z]{5,}", val):
                    sensitive[w] = "ADDRESS"

        # ---- 2. relatives mentioned in free text -----------------------------------------
        for m in _RELATIVE.finditer(text):
            spans.append((m.start(1), m.end(1), "[RELATIVE]", "RELATIVE", 1))
            names.add(m.group(1))
            sensitive[m.group(1)] = "RELATIVE"
            for part in m.group(1).split():
                if len(part) >= 3:
                    sensitive[part] = "RELATIVE"

        # ---- 3. pattern + gazetteer detection over the whole text -------------------------
        scanner = Pseudonymiser("anonymiser-scan-only", {n for n in names if " " in n})
        for sp in scanner.scan(text):
            if sp.kind == "PERSON":
                key = sp.canonical
                before = text[max(0, sp.start - 8):sp.start]
                if key in patient_keys:
                    repl, cat = "[PATIENT]", "PATIENT"
                elif key in clinician_keys or re.search(r"(?i)\b(?:dr|mudr|prof)\.?\s*$", before):
                    repl, cat = "[CLINICIAN]", "CLINICIAN"
                else:
                    repl, cat = "[PERSON]", "PERSON"
                spans.append((sp.start, sp.end, repl, cat, 2))
                sensitive[sp.original] = cat
            elif sp.kind == "DOB":
                spans.append((sp.start, sp.end, self._dob(sp.original, ref_year), "DOB", 1))
                sensitive[sp.original] = "DOB"
            elif sp.kind in _KIND_LABEL:
                spans.append((sp.start, sp.end, _KIND_LABEL[sp.kind], sp.kind, 2))
                sensitive[sp.original] = sp.kind
        for m in _ACCESSION.finditer(text):
            spans.append((m.start(), m.end(), "[ACCESSION]", "ACCESSION", 2))
            sensitive[m.group()] = "ACCESSION"

        # ---- 4. generalise quasi-identifiers ---------------------------------------------
        for s, e, ym, _y in date_hits:
            spans.append((s, e, ym, "DATE", 3))
            sensitive[text[s:e].strip()] = "DATE"
        for m in _AGE.finditer(text):
            g = 1 if m.group(1) else 2
            spans.append((m.start(g), m.end(g), age_band(int(m.group(g))), "AGE", 3))

        # ---- 5. apply (field > relative/dob > detectors > generalisations) ----------------
        spans.sort(key=lambda t: (t[0], t[4], -(t[1] - t[0])))
        out, last, log = [], 0, Counter()
        for s, e, repl, cat, _p in spans:
            if s < last:
                continue
            out.append(text[last:s])
            out.append(repl)
            log[cat] += 1
            last = e
        out.append(text[last:])
        result_text = "".join(out)

        leaks = self._residual(result_text, scanner, sensitive)
        res = AnonResult(result_text, dict(sorted(log.items())), leaks,
                         "deidentified" if not leaks else "needs_review")
        return res, names

    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _dates(text: str) -> list[tuple[int, int, str, int]]:
        hits = []
        for m in _DATE_NUM.finditer(text):
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 1 <= mo <= 12 and 1 <= d <= 31:
                hits.append((m.start(), m.end(), f"{y}-{mo:02d}", y))
        for m in _DATE_ISO.finditer(text):
            y, mo = int(m.group(1)), int(m.group(2))
            if 1 <= mo <= 12:
                hits.append((m.start(), m.end(), f"{y}-{mo:02d}", y))
        for m in _DATE_TXT.finditer(text):
            y = int(m.group(3))
            hits.append((m.start(), m.end(), f"{y}-{_month_no(m.group(2)):02d}", y))
        return hits

    @staticmethod
    def _dob(raw: str, ref_year: int | None) -> str:
        ym = re.search(r"(?:19|20)\d{2}", raw)
        if not ym:
            return "[DOB]"
        y = int(ym.group())
        if ref_year is not None and ref_year - y > 89:
            return "[DOB]"          # year of birth would itself identify a very old person
        return str(y)

    @staticmethod
    def _residual(out: str, scanner: Pseudonymiser, sensitive: dict[str, str]) -> list[str]:
        leaks: set[str] = set()
        for sp in scanner.scan(out):
            if sp.kind in _STRUCTURED or sp.kind == "PERSON":
                leaks.add(f"residual_{sp.kind.lower()}")
        folded = fold(out)
        for value, cat in sensitive.items():
            v = fold(value).strip()
            if len(v) >= 3 and re.search(rf"(?<!\w){re.escape(v)}(?!\w)", folded):
                leaks.add(f"leak_{cat.lower()}")
        return sorted(leaks)
