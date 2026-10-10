"""Show what the de-identification stage does to a medical report.

    python scripts/anonymise_report.py                      # all sample reports, summary only
    python scripts/anonymise_report.py lab_report_hematology  # before / after + transformation log

The ORIGINAL is printed only to make the demo readable (the sample reports are synthetic). Never do
this with real data; in the pipeline the original text never leaves ingestion and the log contains
categories and counts only.
"""
from __future__ import annotations

import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragkit.privacy.anonymise import Anonymiser  # noqa: E402


def main() -> None:
    reports = {p.stem: p for p in sorted((ROOT / "data/raw/reports").glob("*.md"))}
    chosen = sys.argv[1:] or list(reports)
    verbose = bool(sys.argv[1:])
    for name in chosen:
        original = reports[name].read_text(encoding="utf-8")
        res, _names = Anonymiser().anonymise(original)
        print(f"\n=== {name}: {res.status} | leaks={res.leaks or 'none'}")
        print("    transformations:", ", ".join(f"{k}x{v}" for k, v in res.log.items()))
        if verbose:
            for line in difflib.unified_diff(original.splitlines(), res.text.splitlines(),
                                             "original", "de-identified", n=0, lineterm=""):
                if line.startswith(("---", "+++", "@@")):
                    continue
                print("   ", line)
            print(f"\n    NOTE: {res.note}")


if __name__ == "__main__":
    main()
