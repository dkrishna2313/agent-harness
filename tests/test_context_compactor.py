"""Tests for the PH3.2 Shared Context Compactor.

Verifies: deterministic compaction, unused-section removal, duplicate-section
detection, non-destructive semantics (kept content is never rewritten), the
documented AgentContext profiles, and the measurement-only integration seam
(record_context_compaction / measure_and_record never mutate context).
"""

from __future__ import annotations

import types

import pytest

from functional_agents.context_compactor import (
    AGENT_CONTEXT_PROFILES,
    SECTION_ORDER,
    CompactionResult,
    build_context_sections,
    compact_context,
    compact_context_for_agent,
    estimate_tokens,
    measure_and_record,
)
from functional_agents.performance import PerformanceTracker


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

def test_estimate_tokens_empty_is_zero():
    assert estimate_tokens(None) == 0
    assert estimate_tokens("") == 0
    assert estimate_tokens({}) == 0
    assert estimate_tokens([]) == 0


def test_estimate_tokens_scales_with_length():
    short = estimate_tokens("a" * 40)
    long = estimate_tokens("a" * 400)
    assert long > short
    assert short >= 1


def test_estimate_tokens_deterministic_for_dicts_regardless_of_key_order():
    a = {"x": 1, "y": 2}
    b = {"y": 2, "x": 1}
    assert estimate_tokens(a) == estimate_tokens(b)


# ---------------------------------------------------------------------------
# compact_context — core behaviour
# ---------------------------------------------------------------------------

def test_compact_context_drops_unused_sections():
    sections = {"a": "keep me", "b": "drop me", "c": "also drop"}
    result = compact_context(sections, profile=["a"])
    assert result.kept_sections == ("a",)
    assert result.removed_unused == ("b", "c")
    assert result.compacted == {"a": "keep me"}


def test_compact_context_preserves_kept_content_unmodified():
    payload = {"nested": [1, 2, {"k": "v"}]}
    sections = {"a": payload}
    result = compact_context(sections, profile=["a"])
    assert result.compacted["a"] is payload  # same object, untouched


def test_compact_context_preserves_original_ordering():
    sections = {"z": "1", "a": "2", "m": "3"}
    result = compact_context(sections, profile=["a", "m", "z"])
    # order follows the input dict's insertion order, not the profile's order
    assert result.kept_sections == ("z", "a", "m")


def test_compact_context_detects_and_removes_duplicates():
    sections = {"first": {"x": 1}, "second": {"x": 1}, "third": {"x": 2}}
    result = compact_context(sections, profile=["first", "second", "third"])
    assert result.kept_sections == ("first", "third")
    assert len(result.removed_duplicate) == 1
    dup = result.removed_duplicate[0]
    assert dup.name == "second"
    assert dup.duplicate_of == "first"


def test_compact_context_empty_sections_not_treated_as_duplicates():
    sections = {"a": "", "b": None, "c": {}}
    result = compact_context(sections, profile=["a", "b", "c"])
    # Empty/falsy content must not be flagged as duplicating each other
    assert result.removed_duplicate == ()


def test_compact_context_is_deterministic():
    sections = {"a": {"x": [1, 2, 3]}, "b": {"x": [1, 2, 3]}, "c": "unique"}
    r1 = compact_context(sections, profile=["a", "b", "c"])
    r2 = compact_context(sections, profile=["a", "b", "c"])
    assert r1.to_dict() == r2.to_dict()


def test_compact_context_rejects_non_dict_sections():
    with pytest.raises(TypeError):
        compact_context("not a dict", profile=["a"])


def test_compaction_result_reduction_pct_and_tokens_saved():
    sections = {"a": "x" * 400, "b": "y" * 400}
    result = compact_context(sections, profile=["a"])
    assert result.tokens_saved == result.original_tokens - result.compacted_tokens
    assert 0 < result.reduction_pct <= 100


def test_compaction_result_reduction_pct_zero_when_original_zero():
    result = compact_context({}, profile=[])
    assert result.reduction_pct == 0.0
    assert result.original_tokens == 0


def test_to_dict_shape():
    sections = {"a": "keep", "b": "drop"}
    d = compact_context(sections, profile=["a"]).to_dict()
    assert set(d) == {
        "original_sections", "kept_sections", "removed_unused", "removed_duplicate",
        "original_tokens", "compacted_tokens", "tokens_saved", "reduction_pct",
    }


# ---------------------------------------------------------------------------
# build_context_sections — read-only AgentContext extraction
# ---------------------------------------------------------------------------

def _make_context(**overrides):
    base = dict(
        question="What are the power constraints?",
        profiles=["ai_data_centers"],
        decision_model={"objective": "assess feasibility"},
        research_strategy={"required_evidence": ["power"]},
        plan={"subquestions": ["q1"], "investigation_areas": ["power"]},
        evidence_notes=[{"evidence_items": [{"evidence_id": "E1"}]}],
        hypotheses=[{"id": "H1"}],
        surviving_hypotheses=[{"id": "H1"}],
        hypothesis_challenges=[],
        strategic_synthesis={},
        research_object={"contradictions": []},
        validated_contradictions=[],
        recommendations=[],
        recommendation_portfolio={},
        assumptions=[],
        risks=[],
        opportunities=[],
        strategic_options=[],
        decision_analysis={},
        executive_confidence={},
        artifacts={},
        trace={},
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def test_build_context_sections_covers_full_catalog_by_default():
    ctx = _make_context()
    sections = build_context_sections(ctx)
    assert set(sections) == set(SECTION_ORDER)


def test_build_context_sections_never_mutates_context():
    ctx = _make_context()
    before = dict(vars(ctx))
    build_context_sections(ctx)
    assert vars(ctx) == before


def test_build_context_sections_handles_missing_attrs_gracefully():
    ctx = types.SimpleNamespace(question="q")  # everything else absent
    sections = build_context_sections(ctx)
    assert sections["question"] == "q"
    assert sections["decision_model"] == {}
    assert sections["hypotheses"] == []


def test_build_context_sections_subset():
    ctx = _make_context()
    sections = build_context_sections(ctx, section_names=["question", "plan"])
    assert set(sections) == {"question", "plan"}


def test_evidence_notes_section_takes_first_note():
    ctx = _make_context(evidence_notes=[{"evidence_items": ["a"]}, {"evidence_items": ["b"]}])
    sections = build_context_sections(ctx, section_names=["evidence_notes"])
    assert sections["evidence_notes"] == {"evidence_items": ["a"]}


def test_contradictions_prefers_research_object():
    ctx = _make_context(
        research_object={"contradictions": ["from_ro"]},
        validated_contradictions=["from_validated"],
    )
    sections = build_context_sections(ctx, section_names=["contradictions"])
    assert sections["contradictions"] == ["from_ro"]


def test_contradictions_falls_back_to_validated_when_no_research_object():
    ctx = _make_context(research_object=None, validated_contradictions=["from_validated"])
    sections = build_context_sections(ctx, section_names=["contradictions"])
    assert sections["contradictions"] == ["from_validated"]


# ---------------------------------------------------------------------------
# AGENT_CONTEXT_PROFILES — documented, code-derived, and internally consistent
# ---------------------------------------------------------------------------

def test_all_profile_sections_are_known_accessors():
    for agent, profile in AGENT_CONTEXT_PROFILES.items():
        for section in profile:
            assert section in SECTION_ORDER, f"{agent} references unknown section {section!r}"


def test_no_agent_profile_lists_its_own_output_as_input():
    # Sanity guard against accidental self-reference (would be a modelling bug).
    OWN_OUTPUT = {
        "PlannerAgent": "plan",
        "HypothesisAgent": "hypotheses",
        "RecommendationAgent": "recommendations",
    }
    for agent, own_output in OWN_OUTPUT.items():
        assert own_output not in AGENT_CONTEXT_PROFILES[agent]


def test_named_agents_have_profiles():
    for agent in ("PlannerAgent", "EvidenceAgent", "HypothesisAgent",
                  "RecommendationAgent", "ReportAgent"):
        assert agent in AGENT_CONTEXT_PROFILES
        assert len(AGENT_CONTEXT_PROFILES[agent]) > 0


def test_compact_context_for_agent_unknown_agent_returns_none():
    ctx = _make_context()
    assert compact_context_for_agent(ctx, "NotARealAgent") is None


def test_compact_context_for_agent_matches_profile():
    ctx = _make_context()
    result = compact_context_for_agent(ctx, "EvidenceAgent")
    assert set(result.kept_sections) <= set(AGENT_CONTEXT_PROFILES["EvidenceAgent"])
    assert "decision_model" in result.removed_unused  # Evidence doesn't use it


def test_recommendation_profile_is_richest_llm_agent():
    # Recommendation reads the most upstream sections of any hardened agent —
    # matches recommendation_agent.py's _execute() argument list.
    profile = AGENT_CONTEXT_PROFILES["RecommendationAgent"]
    for expected in ("hypotheses", "evidence_notes", "decision_model",
                      "research_strategy", "strategic_synthesis"):
        assert expected in profile


def test_report_profile_covers_most_of_decision_graph():
    profile = AGENT_CONTEXT_PROFILES["ReportAgent"]
    assert len(profile) >= 15  # Report legitimately needs nearly everything


# ---------------------------------------------------------------------------
# measure_and_record — the PH3.2 integration seam
# ---------------------------------------------------------------------------

def test_measure_and_record_noop_without_tracker():
    ctx = _make_context(trace={})
    result = measure_and_record(ctx, "PlannerAgent")
    assert result is not None  # still computes...
    assert ctx.trace == {}     # ...but records nothing without a tracker


def test_measure_and_record_noop_for_unknown_agent():
    tracker = PerformanceTracker()
    ctx = _make_context(trace={"_perf_tracker": tracker})
    result = measure_and_record(ctx, "NotARealAgent")
    assert result is None
    assert tracker.flush_context_compaction() is None


def test_measure_and_record_writes_to_tracker():
    tracker = PerformanceTracker()
    ctx = _make_context(trace={"_perf_tracker": tracker})
    result = measure_and_record(ctx, "PlannerAgent")
    recorded = tracker.flush_context_compaction()
    assert recorded == result.to_dict()


def test_measure_and_record_never_mutates_context():
    tracker = PerformanceTracker()
    ctx = _make_context(trace={"_perf_tracker": tracker})
    before = {k: v for k, v in vars(ctx).items() if k != "trace"}
    measure_and_record(ctx, "RecommendationAgent")
    after = {k: v for k, v in vars(ctx).items() if k != "trace"}
    assert before == after
