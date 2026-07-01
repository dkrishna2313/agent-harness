"""Tests for the LLM output boundary normalization helper (PH1)."""

from __future__ import annotations

from research_agent.llm_normalize import normalize_llm_items


# ---------------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------------

def test_valid_object_list():
    raw = [{"evidence_id": "E1", "relevance_score": 0.9}, {"evidence_id": "E2"}]
    items, diag = normalize_llm_items(raw, required_fields=("evidence_id",), component="reranker")
    assert len(items) == 2
    assert diag == {
        "component": "reranker", "items_received": 2,
        "items_valid": 2, "items_dropped": 0, "fallback_used": False,
    }


def test_malformed_object_missing_required_field():
    raw = [{"evidence_id": "E1"}, {"relevance_score": 0.5}]  # 2nd lacks evidence_id
    items, diag = normalize_llm_items(raw, required_fields=("evidence_id",), component="reranker")
    assert len(items) == 1
    assert diag["items_dropped"] == 1


def test_string_instead_of_object_coerced():
    raw = ["E1", "E2"]
    items, diag = normalize_llm_items(
        raw, required_fields=("evidence_id",), coerce_str_key="evidence_id", component="reranker"
    )
    assert items == [{"evidence_id": "E1"}, {"evidence_id": "E2"}]
    assert diag["items_valid"] == 2


def test_string_instead_of_object_dropped_when_no_coercion():
    raw = ["E1", "E2"]
    items, diag = normalize_llm_items(raw, required_fields=("evidence_id",), component="reranker")
    assert items == []
    assert diag["items_dropped"] == 2


def test_empty_list():
    items, diag = normalize_llm_items([], component="reranker")
    assert items == []
    assert diag["items_received"] == 0
    assert diag["items_valid"] == 0


def test_mixed_valid_and_invalid():
    raw = [
        {"evidence_id": "E1"},          # valid
        "E2",                           # coerced → valid
        {"relevance_score": 0.3},       # missing evidence_id → dropped
        42,                             # non-dict/str → dropped
        {"evidence_id": None},          # None required field → dropped
    ]
    items, diag = normalize_llm_items(
        raw, required_fields=("evidence_id",), coerce_str_key="evidence_id", component="reranker"
    )
    ids = [i["evidence_id"] for i in items]
    assert ids == ["E1", "E2"]
    assert diag["items_received"] == 5
    assert diag["items_valid"] == 2
    assert diag["items_dropped"] == 3


def test_non_list_input_does_not_raise():
    for bad in (None, "just a string", {"evidence_id": "E1"}, 123):
        items, diag = normalize_llm_items(bad, component="reranker")
        assert items == []
        assert isinstance(diag, dict)
        assert diag["items_valid"] == 0


def test_no_required_fields_keeps_all_dicts():
    raw = [{"a": 1}, {"b": 2}, "x"]
    items, diag = normalize_llm_items(raw, component="c")
    assert len(items) == 2  # the string is dropped (no coercion)
