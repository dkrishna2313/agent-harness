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


# ---------------------------------------------------------------------------
# PH1a — single-object normalization (normalize_llm_object)
# ---------------------------------------------------------------------------

from research_agent.llm_normalize import normalize_llm_object  # noqa: E402


def test_object_valid_dict():
    obj, diag = normalize_llm_object({"recommended_option_id": "O1"},
                                     required_fields=("recommended_option_id",), component="decision_analysis")
    assert obj == {"recommended_option_id": "O1"}
    assert diag["items_received"] == 1 and diag["items_valid"] == 1


def test_object_stringified_json_deserialized():
    import json
    raw = json.dumps({"recommended_option_id": "O1", "rationale": "x"})
    obj, diag = normalize_llm_object(raw, required_fields=("recommended_option_id",), component="decision_analysis")
    assert obj == {"recommended_option_id": "O1", "rationale": "x"}
    assert diag["items_valid"] == 1


def test_object_plain_string_dropped():
    obj, diag = normalize_llm_object("just a sentence, not JSON",
                                     required_fields=("recommended_option_id",), component="decision_analysis")
    assert obj is None
    assert diag["items_valid"] == 0 and diag["items_dropped"] == 1


def test_object_stringified_json_but_not_object():
    obj, diag = normalize_llm_object("[1, 2, 3]", component="c")  # JSON array, not object
    assert obj is None
    assert diag["items_dropped"] == 1


def test_object_missing_required_field():
    obj, diag = normalize_llm_object({"analysis_id": "DA-1"},
                                     required_fields=("recommended_option_id",), component="c")
    assert obj is None
    assert diag["items_dropped"] == 1


def test_object_none_input():
    obj, diag = normalize_llm_object(None, component="c")
    assert obj is None
    assert diag["items_received"] == 0
    assert diag["items_dropped"] == 0


def test_object_never_raises():
    for bad in (123, [1], {"x": 1}, "", "  ", "{bad json"):
        obj, diag = normalize_llm_object(bad, required_fields=("k",), component="c")
        assert obj is None or isinstance(obj, dict)
        assert isinstance(diag, dict)
