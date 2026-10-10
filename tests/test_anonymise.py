import json
import re
from pathlib import Path

import pytest

from ragkit.pipeline import RagPipeline
from ragkit.privacy.anonymise import Anonymiser, age_band

ROOT = Path(__file__).resolve().parents[1]
REPORTS = {p.stem: p.read_text(encoding="utf-8") for p in sorted((ROOT / "data/raw/reports").glob("*.md"))}
ANNOT = [json.loads(ln) for ln in (ROOT / "data/eval/report_annotations.jsonl").read_text(encoding="utf-8").splitlines()]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


@pytest.fixture(scope="module")
def pipe():
    return RagPipeline()


@pytest.mark.parametrize("a", ANNOT, ids=lambda a: a["doc_id"])
def test_every_planted_identifier_is_removed_and_clinical_content_kept(a):
    res, _ = Anonymiser().anonymise(REPORTS[a["doc_id"]])
    out = norm(res.text)
    leaked = [p for p in a["phi"] if re.search(rf"(?<!\w){re.escape(norm(p))}(?!\w)", out)]
    assert leaked == []
    assert [k for k in a["keep"] if norm(k) not in out] == []
    assert res.status == "deidentified" and res.leaks == []


def test_generalisation_rules():
    res, _ = Anonymiser().anonymise(REPORTS["lab_report_hematology"])
    t = res.text
    assert "Date of birth: 1978" in t           # DOB -> year
    assert "Age: 45-49" in t                     # age -> 5-year band
    assert "Collected: 2025-03" in t             # other dates -> year-month
    assert "postal area 130**" in t              # postcode -> first 3 digits
    assert "14:40" not in t                      # time of day removed with the date


def test_very_old_patient_is_not_identifiable_by_birth_year_or_exact_age():
    res, _ = Anonymiser().anonymise(REPORTS["radiology_chest_xray"])
    assert "Age: 90+" in res.text and "Date of birth: [DOB]" in res.text
    assert "1933" not in res.text and "91" not in res.text


def test_age_bands():
    assert [age_band(n) for n in (0, 4, 5, 47, 89, 90, 101)] == [
        "0-4", "0-4", "5-9", "45-49", "85-89", "90+", "90+"]


def test_markers_are_unlinkable_not_hashed():
    res, _ = Anonymiser().anonymise(REPORTS["lab_report_hematology"])
    assert not re.search(r"\[[A-Z_]+_[0-9a-f]{6}\]", res.text)   # no HMAC-style tokens
    assert res.text.count("[PATIENT]") == 2                          # same marker for every mention


def test_narrative_mentions_are_caught_by_role():
    res, _ = Anonymiser().anonymise(REPORTS["lab_report_hematology"])
    assert "Mrs [PATIENT]" in res.text and "Dr. [CLINICIAN]" in res.text
    res, _ = Anonymiser().anonymise(REPORTS["discharge_summary"])
    assert "husband [RELATIVE]" in res.text


def test_log_has_counts_but_no_original_values():
    res, _ = Anonymiser().anonymise(REPORTS["lab_report_hematology"])
    blob = json.dumps(res.log) + json.dumps(res.leaks)
    assert "Novakova" not in blob and "785214" not in blob
    assert res.log["PATIENT"] == 2 and res.log["DATE"] == 4


def test_residual_scan_flags_unhandled_leak():
    text = "- Patient name: Jan Test\n- Address: 14 Marine Drive, Mumbai 400 001\n\nHe lives near Marine Drive."
    res, _ = Anonymiser().anonymise(text)
    assert res.status == "needs_review" and "leak_address" in res.leaks


def test_report_chunks_are_split_by_result_line_and_flagged(pipe):
    chunks = [c for c in pipe.chunks if c.doc_id == "lab_report_hematology"]
    assert len(chunks) >= 2 and all(c.anonymised for c in chunks)
    assert max(c.n_words for c in chunks) <= 110
    assert not any(c.anonymised for c in pipe.chunks if c.doc_id == "incident_response")


def test_index_has_no_report_identifiers(pipe):
    # per report: a city such as "Mumbai" is legitimately public in the residency policy document
    for a in ANNOT:
        blob = norm(" ".join(c.text for c in pipe.chunks if c.doc_id == a["doc_id"]))
        for p in a["phi"]:
            assert not re.search(rf"(?<!\w){re.escape(norm(p))}(?!\w)", blob), (a["doc_id"], p)


def test_named_person_query_never_gets_an_anonymised_report(pipe):
    _, _, bundle = pipe.build("What was Jana Novakova's hemoglobin?")
    assert any(d["reason"] == "anonymised_report_vs_identified_query" for d in bundle.dropped)
    assert "9.8" not in bundle.user and "Novakova" not in bundle.system + bundle.user


def test_anonymous_report_question_is_answered(pipe):
    a = pipe.answer("What was the hemoglobin result in the hematology lab report?")
    assert a.status == "ok" and "9.8" in a.answer


def test_cross_report_phi_not_in_answers(pipe):
    for q, bad in [("What is the home address of the patient in the biochemistry report?", "Marine"),
                   ("Who was the radiologist for the chest X-ray?", "Balazova"),
                   ("What is the national ID of the patient in the hematology report?", "785214")]:
        assert bad not in pipe.answer(q).answer
