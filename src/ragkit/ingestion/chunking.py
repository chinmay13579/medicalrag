"""Structure-aware chunking.

Strategy
--------
1. Split each markdown file on headings: a "section" is the natural unit of meaning in policy docs.
2. If a section exceeds `target_words`, split on sentence boundaries with `overlap_sentences`
   of overlap, so facts spanning a boundary stay retrievable.
3. A tiny trailing piece is merged into the previous chunk.
4. The heading path ("Refund Policy > Exceptions") is stored as `section` and prepended to the
   indexed text, so a query like "refund exceptions" matches even if the body never says it.
"""
from __future__ import annotations

import re

from ..types import Chunk

_HEADING = re.compile(r"^(#{1,3})\s+(.*)$")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\"(])")


_LIST_ITEM = re.compile(r"^[-*]\s+")
_LABELLED = re.compile(r"^[A-Za-z][A-Za-z ./]{1,30}:\s")


def _as_sentence(line: str) -> str:
    """Report-style lines ('- Hemoglobin: 9.8 g/dL ...') carry no full stop; add one so the
    sentence splitter can break long result tables at item boundaries."""
    if not line:
        return line
    if _LIST_ITEM.match(line):
        line = _LIST_ITEM.sub("", line)
    elif not _LABELLED.match(line):
        return line
    return line if line[-1] in ".!?" else line + "."


def split_sections(md: str) -> list[tuple[str, str]]:
    """Return [(heading_path, body)]; h1 is the doc title, h2/h3 the section."""
    state = {"title": "", "section": ""}
    buf: list[str] = []
    out: list[tuple[str, str]] = []

    def flush() -> None:
        body = " ".join(" ".join(buf).split())
        if body:
            t, s = state["title"], state["section"]
            out.append((f"{t} > {s}" if s else t, body))
        buf.clear()

    for line in md.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            if len(m.group(1)) == 1:
                state["title"], state["section"] = m.group(2).strip(), ""
            else:
                state["section"] = m.group(2).strip()
        else:
            buf.append(_as_sentence(line.strip()))
    flush()
    return out


def _wc(s: str) -> int:
    return len(s.split())


def split_long(body: str, target_words: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    cur: list[str] = []
    for s in _SENT.split(body):
        if cur and _wc(" ".join(cur)) + _wc(s) > target_words:
            pieces.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
        cur.append(s)
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def chunk_document(doc_id: str, md: str, target_words: int = 90, overlap: int = 1,
                   min_words: int = 12, anonymised: bool = False) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path, body in split_sections(md):
        pieces = split_long(body, target_words, overlap) if _wc(body) > target_words else [body]
        if len(pieces) > 1 and _wc(pieces[-1]) < min_words:
            pieces[-2] = pieces[-2] + " " + pieces[-1]
            pieces.pop()
        for p in pieces:
            chunks.append(Chunk(chunk_id=f"{doc_id}#{len(chunks)}", doc_id=doc_id,
                                section=path, text=p, n_words=_wc(p),
                                anonymised=anonymised))
    return chunks
