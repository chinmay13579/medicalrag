"""Evaluation harness.  Usage:  python eval/run_eval.py [--k 3] [--no-gate] [--out eval/reports]

Three layers, each answering a different question:
  1. RETRIEVAL ablation  - does each stage (bm25 -> dense -> RRF -> re-rank) help?  (answerable + pii)
  2. END-TO-END quality  - correct answer, valid citation, correct refusals.
  3. SAFETY              - zero PII in prompts/answers; injections blocked.
Exits non-zero if any gate in eval/thresholds.<profile>.json is violated, so it can run in CI.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import logging  # noqa: E402

import metrics as M  # noqa: E402

from ragkit.pipeline import RagPipeline  # noqa: E402

MODES = ["bm25", "dense", "hybrid", "hybrid+rerank"]


def load_cases(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def retrieval_eval(pipe: RagPipeline, cases: list[dict], k: int, hard_only: bool = False) -> dict:
    labelled = [c for c in cases if c["gold_spans"] and (c.get("hard") or not hard_only)]
    out = {}
    for mode in MODES:
        rows = {"hit@k": [], "span_recall@k": [], "mrr": [], "ndcg@k": [], "precision@k": []}
        for c in labelled:
            hits = pipe.retrieve(c["query"], k=k, mode=mode)
            texts = [h.chunk.text for h in hits]
            rel = M.rel_vector(texts, c["gold_spans"])
            n_rel_total = sum(M.is_relevant(ch.text, c["gold_spans"]) for ch in pipe.chunks)
            rows["hit@k"].append(M.hit_at_k(rel, k))
            rows["span_recall@k"].append(M.span_recall_at_k(texts, c["gold_spans"], k))
            rows["mrr"].append(M.mrr(rel))
            rows["ndcg@k"].append(M.ndcg_at_k(rel, k, n_rel_total))
            rows["precision@k"].append(sum(rel) / k)
        out[mode] = {m: M.mean(v) for m, v in rows.items()}
    return out


def e2e_eval(pipe: RagPipeline, cases: list[dict]) -> tuple[dict, list[dict]]:
    chunk_by_id = {c.chunk_id: c for c in pipe.chunks}
    per_case, agg = [], {k: [] for k in (
        "keyword_coverage", "citation_hit", "status_accuracy", "refusal_accuracy",
        "injection_block_rate", "prompt_pii_leaks", "answer_pii_leaks", "context_tokens",
        "context_chunks", "groundedness")}
    for c in cases:
        expect = c.get("expect_status", ["ok"])
        ans = pipe.answer(c["query"], debug=True)

        prompt_text = ""
        if ans.prompt:
            prompt_text = ans.prompt["system"] + "\n" + ans.prompt["user"]
        leaks_p = M.contains_any(prompt_text, c.get("forbidden_in_prompt", []))
        leaks_a = M.contains_any(ans.answer, c.get("forbidden_in_answer", []))
        ok_status = ans.status in expect
        row = {"id": c["id"], "type": c["type"], "status": ans.status, "expected": expect,
               "status_ok": ok_status, "prompt_leaks": leaks_p, "answer_leaks": leaks_a}

        agg["status_accuracy"].append(float(ok_status))
        agg["prompt_pii_leaks"].append(float(bool(leaks_p)))
        agg["answer_pii_leaks"].append(float(bool(leaks_a)))
        if c["type"] in ("answerable", "pii") and c["gold_spans"]:
            kc = M.keyword_coverage(ans.answer, c.get("keywords", []))
            cited = [chunk_by_id[x["chunk_id"]].text for x in ans.citations]
            ch = float(any(M.is_relevant(t, c["gold_spans"]) for t in cited))
            row.update(keyword_coverage=kc, citation_hit=ch)
            agg["keyword_coverage"].append(kc)
            agg["citation_hit"].append(ch)
            if ans.prompt:
                agg["context_tokens"].append(ans.prompt["context_tokens"])
                agg["context_chunks"].append(len(ans.citations))
            if "groundedness" in ans.guardrails:
                agg["groundedness"].append(ans.guardrails["groundedness"])
        if c["type"] == "unanswerable":
            agg["refusal_accuracy"].append(float(ok_status))
        if c["type"] == "injection":
            agg["injection_block_rate"].append(float(ok_status and not leaks_a))
        per_case.append(row)
    summary = {k: M.mean(v) for k, v in agg.items() if v}
    return summary, per_case


def anonymisation_eval(pipe: RagPipeline, annotations: list[dict]) -> dict:
    """Measure the report de-identification stage against planted ground truth.

    phi_recall        share of planted identifiers that are ABSENT from the indexed chunk text
    utility_retention share of clinical / generalised strings that SURVIVE de-identification
    needs_review_rate share of reports whose own residual scan flagged something
    """
    removed = total_phi = kept = total_keep = review = 0
    per_doc = []
    for a in annotations:
        res = pipe.anon_results[a["doc_id"]]
        index_text = " ".join(c.text for c in pipe.chunks if c.doc_id == a["doc_id"])
        leaked = [p for p in a["phi"] if M.contains_any(index_text, [p])]
        lost = [k for k in a["keep"] if M.norm(k) not in M.norm(res.text)]
        removed += len(a["phi"]) - len(leaked)
        total_phi += len(a["phi"])
        kept += len(a["keep"]) - len(lost)
        total_keep += len(a["keep"])
        review += res.status != "deidentified"
        per_doc.append({"doc": a["doc_id"], "status": res.status, "n_leaked": len(leaked),
                        "n_utility_lost": len(lost), "transformations": res.log,
                        "leaked_idx": [a["phi"].index(p) for p in leaked],
                        "lost_idx": [a["keep"].index(k) for k in lost]})
    return {"phi_recall": round(removed / total_phi, 4), "utility_retention": round(kept / total_keep, 4),
            "needs_review_rate": round(review / len(annotations), 4), "docs": per_doc}


def gate(report: dict, thresholds: dict) -> list[str]:
    flat = {**report["e2e"], **{f"retrieval.{k}": v for k, v in
                                report["retrieval"]["hybrid+rerank"].items()},
            **{f"anon.{k}": v for k, v in report["anon"].items() if k != "docs"}}
    fails = []
    for key, rule in thresholds.items():
        val = flat.get(key)
        if val is None:
            fails.append(f"{key}: missing")
        elif "min" in rule and val < rule["min"]:
            fails.append(f"{key}: {val} < min {rule['min']}")
        elif "max" in rule and val > rule["max"]:
            fails.append(f"{key}: {val} > max {rule['max']}")
    return fails


def md_table(retrieval: dict, k: int) -> str:
    cols = list(next(iter(retrieval.values())).keys())
    lines = ["| mode | " + " | ".join(c.replace("@k", f"@{k}") for c in cols) + " |",
             "|---|" + "---|" * len(cols)]
    for mode, m in retrieval.items():
        lines.append(f"| {mode} | " + " | ".join(f"{m[c]:.3f}" for c in cols) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--cases", default=str(ROOT / "data/eval/eval_set.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "eval/reports"))
    profile = Path(os.environ.get("RAGKIT_CONFIG", "default")).stem   # default | semantic
    ap.add_argument("--thresholds", default=str(ROOT / f"eval/thresholds.{profile}.json"))
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="print per-case rows")
    args = ap.parse_args()

    logging.getLogger("ragkit").setLevel(logging.ERROR)   # keep eval output readable
    pipe = RagPipeline()
    cases = load_cases(Path(args.cases))
    report = {"n_cases": len(cases), "k": args.k,
              "retrieval": retrieval_eval(pipe, cases, args.k),
              "retrieval_hard": retrieval_eval(pipe, cases, args.k, hard_only=True)}
    report["e2e"], report["cases"] = e2e_eval(pipe, cases)
    ann_path = Path(args.cases).with_name("report_annotations.jsonl")
    report["anon"] = anonymisation_eval(pipe, load_cases(ann_path))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n# Retrieval ablation (k={args.k}, {sum(1 for c in cases if c['gold_spans'])} labelled queries)\n")
    print(md_table(report["retrieval"], args.k))
    print("\n# End-to-end + safety\n")
    for k, v in report["e2e"].items():
        print(f"  {k:<22} {v}")
    a = report["anon"]
    print(f"\n# Report de-identification ({len(a['docs'])} reports)\n")
    for k in ("phi_recall", "utility_retention", "needs_review_rate"):
        print(f"  {k:<22} {a[k]}")
    for d in a["docs"]:
        print(f"   - {d['doc']:<26} {d['status']:<13} leaked={d['n_leaked']} utility_lost={d['n_utility_lost']}")
    bad = [r for r in report["cases"] if not r["status_ok"] or r["prompt_leaks"] or r["answer_leaks"]
           or r.get("keyword_coverage", 1) < 1 or r.get("citation_hit", 1) < 1]
    if args.verbose or bad:
        print(f"\n# Cases needing attention ({len(bad)})\n")
        for r in (report["cases"] if args.verbose else bad):
            print("  ", json.dumps(r))

    if args.no_gate:
        return 0
    th = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    fails = gate(report, th)
    print("\n# Gates:", "PASS" if not fails else "FAIL")
    for f in fails:
        print("  x", f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
