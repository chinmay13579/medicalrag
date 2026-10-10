import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import metrics as M  # noqa: E402


def test_rank_metrics():
    rel = [0, 1, 0]
    assert M.hit_at_k(rel, 1) == 0.0 and M.hit_at_k(rel, 2) == 1.0
    assert M.mrr(rel) == 0.5
    assert 0 < M.ndcg_at_k(rel, 3, 1) < 1
    assert M.ndcg_at_k([1, 0, 0], 3, 1) == 1.0


def test_span_matching_is_whitespace_and_case_insensitive():
    assert M.is_relevant("Refunds take  5 to 10\nBusiness days.", ["5 to 10 business days"])
    assert M.span_recall_at_k(["a b", "c d"], ["a b", "c d", "zzz"], 2) == 2 / 3


def test_leak_matcher_uses_word_boundaries():
    assert M.contains_any("the customer asked", ["Tom"]) == []
    assert M.contains_any("hello Tom,", ["Tom"]) == ["Tom"]
