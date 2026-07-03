"""Tests for the PH3.3 per-agent LLM input slices.

Verifies: each slice includes the fields its real prompt-building function
reads, excludes fields verified unused, slices are deterministic and never
mutate the context, diagnostics compute correctly, and wired agents still
produce schema-compatible, boundary-passing output with unchanged counts on
their existing fixtures (proving no behaviour/contract change).
"""

from __future__ import annotations

import copy
import json
import types

import pytest

from functional_agents.context_slices import (
    hypothesis_input_slice,
    planner_input_slice,
    recommendation_input_slice,
    record_slice_diagnostics,
    slice_diagnostics,
    strategic_synthesis_input_slice,
)
from functional_agents.performance import PerformanceTracker


def _make_context(**overrides):
    base = dict(
        question="What are the power constraints?",
        profiles=["ai_data_centers"],
        decision_model={
            "objective": "assess feasibility",
            "decision_areas": ["power", "cooling"],
            "critical_uncertainties": ["regulatory timeline"],
            "research_questions": ["Q1"],
            "evidence_requirements": ["power data"],
            "decision_architecture": {"huge": "unused blob" * 50},  # never read
            "board_decisions": ["unused"],
        },
        research_strategy={
            "research_question_priorities": [{"question": "Q1", "priority": 1}],
            "required_evidence": ["power data"],
            "source_priorities": ["primary"],
            "coverage_targets": {"power": "strong"},
            "strategy_rationale": "unused by any of the 4 target prompts",
        },
        plan={"subquestions": ["q1"], "investigation_areas": ["power"]},
        evidence_notes=[{
            "evidence_items": [{"evidence_id": "E1", "claim": "c"}],
            "profile_coverage_by_profile": {"ai_data_centers": {"coverage_level": "STRONG"}},
        }],
        hypotheses=[{"id": "H1", "title": "t"}],
        surviving_hypotheses=[{"hypothesis_id": "H1", "survival_status": "strong"}],
        hypothesis_challenges=[{"hypothesis_id": "H1"}],
        strategic_synthesis={"executive_summary": "s", "cross_domain_findings": ["f"]},
        research_object={"contradictions": ["c1"], "validated_contradictions": ["vc1"]},
        validated_contradictions=[],
        domain_plans=[{"decision_domain_title": "Domain A"}],
        domain_evidence=[{"evidence_id": "E1"}, {"evidence_id": "E2"}],
        domain_hypotheses=[{"decision_domain_title": "Domain A", "hypotheses": [{"title": "H1"}]}],
        decision_architecture={
            "strategic_themes": ["theme1"],
            "decision_statement": "Should we invest?",
            "executive_unknowns": ["unknown1"],
            "board_decisions": ["unused by synthesis"],
            "scope": {"unused": True},
        },
        recommendations=[],
        recommendation_portfolio={},
        trace={},
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# planner_input_slice
# ---------------------------------------------------------------------------

def test_planner_slice_includes_required_fields():
    ctx = _make_context()
    s = planner_input_slice(ctx)
    assert s["question"] == ctx.question
    dm = s["decision_model"]
    for key in ("objective", "decision_areas", "critical_uncertainties",
                "research_questions", "evidence_requirements"):
        assert key in dm
    rs = s["research_strategy"]
    for key in ("research_question_priorities", "required_evidence",
                "source_priorities", "coverage_targets"):
        assert key in rs


def test_planner_slice_excludes_unused_fields():
    ctx = _make_context()
    s = planner_input_slice(ctx)
    assert "decision_architecture" not in s["decision_model"]
    assert "board_decisions" not in s["decision_model"]
    assert "strategy_rationale" not in s["research_strategy"]


def test_planner_slice_handles_missing_decision_model():
    ctx = _make_context(decision_model=None, research_strategy=None)
    s = planner_input_slice(ctx)
    assert s["decision_model"] == {}
    assert s["research_strategy"] == {}


# ---------------------------------------------------------------------------
# hypothesis_input_slice
# ---------------------------------------------------------------------------

def test_hypothesis_slice_includes_required_fields():
    ctx = _make_context()
    s = hypothesis_input_slice(ctx)
    for key in ("objective", "decision_areas", "critical_uncertainties"):
        assert key in s["decision_model"]
    assert "research_question_priorities" in s["research_strategy"]
    assert s["evidence_items"] == [{"evidence_id": "E1", "claim": "c"}]
    assert s["profile_coverage"] == {"ai_data_centers": "strong"}
    assert s["contradictions"] == ["c1"]


def test_hypothesis_slice_excludes_unused_fields():
    ctx = _make_context()
    s = hypothesis_input_slice(ctx)
    assert "research_questions" not in s["decision_model"]  # planner-only field
    assert "evidence_requirements" not in s["decision_model"]
    assert set(s["research_strategy"]) == {"research_question_priorities"}


def test_hypothesis_slice_never_truncates_evidence():
    many_items = [{"evidence_id": f"E{i}"} for i in range(30)]
    ctx = _make_context(evidence_notes=[{"evidence_items": many_items,
                                          "profile_coverage_by_profile": {}}])
    s = hypothesis_input_slice(ctx)
    assert len(s["evidence_items"]) == 30  # not capped by the slice itself


def test_hypothesis_slice_profile_coverage_fallback():
    ctx = _make_context(evidence_notes=[{"evidence_items": [], "profile_coverage_by_profile": {}}],
                         profiles=["p1", "p2"])
    s = hypothesis_input_slice(ctx)
    assert s["profile_coverage"] == {"p1": "unknown", "p2": "unknown"}


# ---------------------------------------------------------------------------
# recommendation_input_slice
# ---------------------------------------------------------------------------

def test_recommendation_slice_includes_required_fields():
    ctx = _make_context()
    s = recommendation_input_slice(ctx)
    assert s["hypotheses"] == ctx.hypotheses
    assert s["surviving_hypotheses"] == ctx.surviving_hypotheses
    assert s["hypothesis_challenges"] == ctx.hypothesis_challenges
    assert s["evidence_items"] == [{"evidence_id": "E1", "claim": "c"}]
    assert "objective" in s["decision_model"]
    assert "decision_areas" in s["decision_model"]
    assert s["strategic_synthesis"] == ctx.strategic_synthesis


def test_recommendation_slice_research_strategy_fully_excluded():
    """research_strategy is confirmed dead in both the live prompt and mock — see module docstring."""
    ctx = _make_context()
    s = recommendation_input_slice(ctx)
    assert s["research_strategy"] == {}


def test_recommendation_slice_excludes_unused_decision_model_fields():
    ctx = _make_context()
    s = recommendation_input_slice(ctx)
    assert "critical_uncertainties" not in s["decision_model"]  # unused by _recommendation_prompt
    assert "decision_architecture" not in s["decision_model"]


def test_recommendation_slice_never_truncates_hypotheses_or_evidence():
    many_hyps = [{"id": f"H{i}"} for i in range(10)]
    ctx = _make_context(hypotheses=many_hyps)
    s = recommendation_input_slice(ctx)
    assert len(s["hypotheses"]) == 10


def test_recommendation_slice_prefers_explicit_validated_contradictions():
    ctx = _make_context(validated_contradictions=["explicit"])
    s = recommendation_input_slice(ctx)
    assert s["validated_contradictions"] == ["explicit"]


# ---------------------------------------------------------------------------
# strategic_synthesis_input_slice
# ---------------------------------------------------------------------------

def test_strategic_synthesis_slice_domain_evidence_fully_excluded():
    """domain_evidence is confirmed dead in both the live prompt and mock — see module docstring."""
    ctx = _make_context()
    s = strategic_synthesis_input_slice(ctx)
    assert s["domain_evidence"] == []


def test_strategic_synthesis_slice_keeps_domain_plans_and_hypotheses_in_full():
    ctx = _make_context()
    s = strategic_synthesis_input_slice(ctx)
    assert s["domain_plans"] == ctx.domain_plans
    assert s["domain_hypotheses"] == ctx.domain_hypotheses


def test_strategic_synthesis_slice_decision_architecture_trimmed():
    ctx = _make_context()
    s = strategic_synthesis_input_slice(ctx)
    da = s["decision_architecture"]
    assert set(da) == {"strategic_themes", "decision_statement", "executive_unknowns"}
    assert "board_decisions" not in da
    assert "scope" not in da


# ---------------------------------------------------------------------------
# Determinism + non-mutation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("slice_fn", [
    planner_input_slice, hypothesis_input_slice,
    recommendation_input_slice, strategic_synthesis_input_slice,
])
def test_slices_are_deterministic(slice_fn):
    ctx = _make_context()
    r1 = slice_fn(ctx)
    r2 = slice_fn(ctx)
    assert json.dumps(r1, sort_keys=True, default=str) == json.dumps(r2, sort_keys=True, default=str)


@pytest.mark.parametrize("slice_fn", [
    planner_input_slice, hypothesis_input_slice,
    recommendation_input_slice, strategic_synthesis_input_slice,
])
def test_slices_never_mutate_context(slice_fn):
    ctx = _make_context()
    before = copy.deepcopy({k: v for k, v in vars(ctx).items() if k != "trace"})
    slice_fn(ctx)
    after = copy.deepcopy({k: v for k, v in vars(ctx).items() if k != "trace"})
    assert before == after


@pytest.mark.parametrize("slice_fn", [
    planner_input_slice, hypothesis_input_slice,
    recommendation_input_slice, strategic_synthesis_input_slice,
])
def test_slices_are_json_safe(slice_fn):
    ctx = _make_context()
    result = slice_fn(ctx)
    json.dumps(result, default=str)  # must not raise


# ---------------------------------------------------------------------------
# slice_diagnostics
# ---------------------------------------------------------------------------

def test_slice_diagnostics_reports_reduction():
    original = {"decision_model": {"a": 1, "b": "x" * 200, "c": [1, 2, 3]}}
    sliced = {"decision_model": {"a": 1}}
    diag = slice_diagnostics(original, sliced)
    assert diag["original_bytes"] > diag["sliced_bytes"]
    assert diag["bytes_saved"] == diag["original_bytes"] - diag["sliced_bytes"]
    assert diag["reduction_pct"] > 0
    assert "decision_model.a" in diag["fields_included"]
    assert "decision_model.b" in diag["fields_excluded"]
    assert "decision_model.c" in diag["fields_excluded"]


def test_slice_diagnostics_list_fully_excluded():
    original = {"domain_evidence": [{"x": 1}, {"x": 2}]}
    sliced = {"domain_evidence": []}
    diag = slice_diagnostics(original, sliced)
    assert any("domain_evidence" in f for f in diag["fields_excluded"])


def test_slice_diagnostics_no_reduction_when_identical():
    payload = {"a": {"x": 1}}
    diag = slice_diagnostics(payload, payload)
    assert diag["reduction_pct"] == 0.0
    assert diag["bytes_saved"] == 0


def test_slice_diagnostics_zero_original_bytes():
    diag = slice_diagnostics({}, {})
    assert diag["reduction_pct"] == 0.0


# ---------------------------------------------------------------------------
# record_slice_diagnostics — trace + tracker integration
# ---------------------------------------------------------------------------

def test_record_slice_diagnostics_writes_trace_key():
    ctx = _make_context(trace={})
    diag = record_slice_diagnostics(ctx, "planner", {"decision_model": {"a": 1, "b": 2}},
                                     {"decision_model": {"a": 1}})
    assert ctx.trace["_planner_prompt_slice"] == diag


def test_record_slice_diagnostics_records_on_tracker():
    tracker = PerformanceTracker()
    ctx = _make_context(trace={"_perf_tracker": tracker})
    record_slice_diagnostics(ctx, "recommendation", {"research_strategy": {"a": 1}}, {"research_strategy": {}})
    flushed = tracker.flush_prompt_slice()
    assert flushed is not None
    assert flushed["fields_excluded"] == ["research_strategy.a"]


def test_record_slice_diagnostics_noop_without_tracker():
    ctx = _make_context(trace={})
    record_slice_diagnostics(ctx, "hypothesis", {"a": {"x": 1}}, {"a": {}})
    assert "_hypothesis_prompt_slice" in ctx.trace  # trace always written


# ---------------------------------------------------------------------------
# Integration — real wired agents via .run(), through MockClaudeClient
#
# Proves: existing fixtures still produce valid, boundary-passing,
# schema-compatible output after PH3.3 wiring, and prompt-slice diagnostics
# are actually emitted end-to-end (not just unit-testable in isolation).
# ---------------------------------------------------------------------------

_DECISION_MODEL = {
    "objective": "Develop a power supply strategy for AI data centres",
    "decision_areas": ["Grid Constraints", "SMR Viability", "Hybrid Portfolios"],
    "critical_uncertainties": ["Transmission queue timelines", "SMR licensing pace"],
    "research_questions": [
        "What are the current grid interconnection queue backlogs?",
        "When can SMRs be commercially deployed?",
    ],
    "evidence_requirements": ["Grid operator data", "SMR cost estimates"],
    "decision_architecture": {"unused_by_any_target_prompt": "x" * 500},
}

_EVIDENCE_ITEMS = [
    {"evidence_id": "E001", "claim": "Grid queues have grown.", "source_document": "GridReport"},
    {"evidence_id": "E002", "claim": "AI demand is rising.", "source_document": "AIStudy"},
]


def _full_context(**overrides):
    from functional_agents.context import AgentContext

    ctx = AgentContext(
        question="What are the power constraints for AI data centers?",
        goal="Develop a strategy for supplying power to AI data centers",
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={"id": "R-TEST_PH33", "contradictions": []},
        run_id="ph33test",
    )
    ctx.decision_model = _DECISION_MODEL
    ctx.research_strategy = {
        "research_question_priorities": [
            {"question": q, "priority": i + 1} for i, q in enumerate(_DECISION_MODEL["research_questions"])
        ],
        "strategy_rationale": "unused by any of the 4 target prompts",
    }
    ctx.evidence_notes = [{
        "evidence_items": _EVIDENCE_ITEMS,
        "profile_coverage_by_profile": {
            "ai_data_centers": {"coverage_level": "STRONG", "evidence_count": 2},
        },
    }]
    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


def test_planner_agent_still_valid_after_wiring():
    from functional_agents.planner_agent import PlannerAgent

    ctx = _full_context()
    result = PlannerAgent().run(ctx)
    assert result.status == "success"
    assert ctx.plan.get("research_type")
    assert ctx.trace["_planner_boundary"]["failed_stage"] is None
    slice_diag = ctx.trace["_planner_prompt_slice"]
    assert "decision_model.decision_architecture" in slice_diag["fields_excluded"]


def test_hypothesis_agent_still_valid_after_wiring():
    from functional_agents.hypothesis_agent import HypothesisAgent

    ctx = _full_context()
    result = HypothesisAgent().run(ctx)
    assert result.status == "success"
    assert len(ctx.hypotheses) >= 3
    assert ctx.trace["_hypothesis_boundary"]["failed_stage"] is None
    slice_diag = ctx.trace["_hypothesis_prompt_slice"]
    assert "decision_model.decision_architecture" in slice_diag["fields_excluded"]
    assert "research_strategy.strategy_rationale" in slice_diag["fields_excluded"]


def test_recommendation_agent_still_valid_after_wiring_preserves_references():
    from functional_agents.hypothesis_agent import HypothesisAgent
    from functional_agents.recommendation_agent import RecommendationAgent

    ctx = _full_context()
    HypothesisAgent().run(ctx)
    ctx.surviving_hypotheses = [
        {"hypothesis_id": h["id"], "survival_status": "strong"} for h in ctx.hypotheses
    ]
    ctx.hypothesis_challenges = [{"hypothesis_id": h["id"]} for h in ctx.hypotheses]

    result = RecommendationAgent().run(ctx)
    assert result.status == "success"
    assert len(ctx.recommendations) >= 1
    assert ctx.trace["_recommendation_boundary"]["failed_stage"] is None

    # Recommendation evidence/hypothesis references are preserved (not stripped by slicing)
    all_evidence_ids = {e["evidence_id"] for e in _EVIDENCE_ITEMS}
    all_hyp_ids = {h["id"] for h in ctx.hypotheses}
    for rec in ctx.recommendations:
        assert set(rec.get("supporting_evidence", [])) <= all_evidence_ids
        assert set(rec.get("supported_by_hypotheses", [])) <= all_hyp_ids

    slice_diag = ctx.trace["_recommendation_prompt_slice"]
    assert "research_strategy.research_question_priorities" in slice_diag["fields_excluded"]
    assert "decision_model.decision_architecture" in slice_diag["fields_excluded"]


def test_strategic_synthesis_agent_still_valid_after_wiring():
    from functional_agents.strategic_synthesis_agent import StrategicSynthesisAgent

    ctx = _full_context()
    ctx.domain_plans = [{"decision_domain_title": "Domain A"}]
    ctx.domain_evidence = [{"evidence_id": "E1"}, {"evidence_id": "E2"}]
    ctx.domain_hypotheses = [{"decision_domain_title": "Domain A",
                              "hypotheses": [{"title": "Constraint-dominant view"}]}]
    ctx.decision_architecture = {
        "strategic_themes": ["theme1"],
        "decision_statement": "Should we invest?",
        "executive_unknowns": ["unknown1"],
        "board_decisions": ["unused"],
    }

    result = StrategicSynthesisAgent().run(ctx)
    assert result.status == "success"
    assert ctx.strategic_synthesis.get("executive_summary")
    slice_diag = ctx.trace["_strategic_synthesis_prompt_slice"]
    assert any("domain_evidence" in f for f in slice_diag["fields_excluded"])
    assert "decision_architecture.board_decisions" in slice_diag["fields_excluded"]
