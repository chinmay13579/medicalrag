"""Ingestion boundary: raw docs -> protected chunks. Raw identifiers never leave this function.

* ``data/raw/*.md``          policy / case documents -> pseudonymised (keyed, linkable tokens)
* ``data/raw/reports/*.md``  clinical reports        -> de-identified (irreversible markers,
                                                         generalised dates/ages/postcodes)
Names found in either path are fed to the *query-side* gazetteer so a name typed by a user is masked
before it reaches retrieval or the LLM, even though the indexed report text no longer contains it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings
from ..privacy.anonymise import AnonResult, Anonymiser
from ..privacy.pseudonymise import Pseudonymiser, harvest_names
from ..types import Chunk
from .chunking import chunk_document


@dataclass
class Corpus:
    chunks: list[Chunk]
    pseudo: Pseudonymiser
    anon: dict[str, AnonResult] = field(default_factory=dict)   # doc_id -> report de-id result


def load_corpus(settings: Settings) -> Corpus:
    data_dir = settings.path(settings.data_dir)
    docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted(data_dir.glob("*.md"))}
    reports = {p.stem: p.read_text(encoding="utf-8") for p in sorted((data_dir / "reports").glob("*.md"))}
    if not docs and not reports:
        raise FileNotFoundError(f"no .md documents in {data_dir}")

    names: set[str] = set()
    for text in docs.values():
        names |= harvest_names(text)

    anon = Anonymiser()
    anon_results: dict[str, AnonResult] = {}
    safe_reports: dict[str, str] = {}
    for doc_id, text in reports.items():
        result, report_names = anon.anonymise(text)
        anon_results[doc_id] = result
        safe_reports[doc_id] = result.text
        names |= report_names
    pseudo = Pseudonymiser(settings.pseudo_key, names)

    c = settings.chunking
    chunks: list[Chunk] = []
    for doc_id, text in docs.items():
        safe, _vault = pseudo.pseudonymise(text)   # vault discarded: index holds no PII
        chunks += chunk_document(doc_id, safe, c.target_words, c.overlap_sentences, c.min_words)
    for doc_id, safe in safe_reports.items():
        chunks += chunk_document(doc_id, safe, c.target_words, c.overlap_sentences, c.min_words,
                                 anonymised=True)
    return Corpus(chunks, pseudo, anon_results)


def dump_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch.__dict__, ensure_ascii=False) + "\n")
