"""Tests for RecommendationAgent (J6.5)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

@pytest.fixture(autouse=True)
def _mock_yaml():
    sys.modules.setdefault("yaml", MagicMock())
    yield


from functional_agents.base import FunctionalAgent
from functional_agents.context import AgentContext, AgentResult, WorkflowState
from functional_agents.recommendation_agent import RecommendationAgent
from functional_agents.qa_agent import _validate_recommendations
from research_agent.claude_client import (
    MockClaudeClient,
    RecommendationItem,
    RecommendationPayload,
    RecommendationPortfolio,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_HYPOTHESES = [
    {
        "id": "H1", "title": "Constraints dominate", "summary": "S1",
        "supporting_evidence": ["E001", "E002"], "contradicting_evidence": ["E005"],
        "evidence_gaps": ["Operational data"], "confidence": "medium",
        "confidence_rationale": "Some data.", "decision_implications": ["Act A"],
        "disconfirming_evidence_needed": ["Cost data"],
    },
    {
        "id": "H2", "title": "Technology unlocks opportunity", "summary": "S2",
        "supporting_evidence": ["E003"], "contradicting_evidence": [],
        "evidence_gaps": ["Cost projections"], "confidence": "low",
        "confidence_rationale": "Thin data.",
        "decision_implications": ["Act B"],
        "disconfirming_evidence_needed": ["Regulatory data"],
    },
    {
        "id": "H3", "title": "Portfolio strategy wins", "summary": "S3",
        "supporting_evidence": ["E004"], "contradicting_evidence": [],
        "evidence_gaps": ["Comparative studies"], "confidence": "medium",
        "confidence_rationale": "General theory.",
        "decision_implications": ["Diversify"],
        "disconfirming_evidence_needed": ["Dominance evidence"],
    },
]

_SURVIVING = [
    {"hypothesis_id": "H1", "survival_status": "moderate", "reason": "Survives with caveats."},
    {"hypothesis_id": "H2", "survival_status": "weak", "reason": "Major gaps remain."},
    {"hypothesis_id": "H3", "survival_status": "strong", "reason": "Core logic intact."},
]

_CHALLENGES = [
    {
        "hypothesis_id": "H1", "challenge_summary": "Challenge H1",
        "hidden_assumptions": ["Assumption 1"], "weak_evidence": ["Vendor projection"],
        "contradicting_evidence": [], "missing_evidence": ["Operating data"],
        "falsification_tests": ["If X then downgrade"], "robustness": "medium",
    },
    {
        "hypothesis_id": "H2", "challenge_summary": "Challenge H2",
        "hidden_assumptions": ["Assumption 2"], "weak_evidence": ["Single source"],
        "contradicting_evidence": [], "missing_evidence": ["Cost data"],
        "falsification_tests": ["If regulatory blocks"], "robustness": "low",
    },
    {
        "hypothesis_id": "H3", "challenge_summary": "Challenge H3",
        "hidden_assumptions": ["Assumption 3"], "weak_evidence": [],
        "contradicting_evidence": [], "missing_evidence": ["Comparative studies"],
        "falsification_tests": ["If dominant path found"], "robustness": "high",
    },
]

_EVIDENCE = [
    {"evidence_id": "E001", "claim": "Claim 1", "source_document": "doc_a.txt"},
    {"evidence_id": "E002", "claim": "Claim 2", "source_document": "doc_b.txt"},
    {"evidence_id": "E003", "claim": "Claim 3", "source_document": "doc_c.txt"},
    {"evidence_id": "E004", "claim": "Claim 4", "source_document": "doc_d.txt"},
]


def _make_context() -> AgentContext:
    ctx = AgentContext(
        question="What strategy to pursue?",
        profiles=["test_profile"],
        execution_profile="test_profile",
        research_object={"research_id": "R-TEST_020"},
        run_id="testrec01",
    )
    ctx.hypotheses = _HYPOTHESES
    ctx.surviving_hypotheses = _SURVIVING
    ctx.hypothesis_challenges = _CHALLENGES
    ctx.evidence_notes = [{"evidence_items": _EVIDENCE, "profile_coverage_by_profile": {}}]
    return ctx


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def test_recommendation_agent_inherits_functional_agent():
    assert issubclass(RecommendationAgent, FunctionalAgent)


def test_recommendation_agent_run_returns_agent_result():
    result = RecommendationAgent(client=MockClaudeClient()).run(_make_context())
    assert isinstance(result, AgentResult)


def test_recommendation_agent_status_success():
    result = RecommendationAgent(client=MockClaudeClient()).run(_make_context())
    assert result.status == "success"


def test_recommendation_agent_metrics_duration():
    result = RecommendationAgent(client=MockClaudeClient()).run(_make_context())
    assert "duration_seconds" in result.metrics
    assert result.metrics["duration_seconds"] >= 0.0


def test_recommendation_agent_trace_required_keys():
    result = RecommendationAgent(client=MockClaudeClient()).run(_make_context())
    for key in ("agent", "run_id", "duration_seconds", "status"):
        assert key in result.trace


# ---------------------------------------------------------------------------
# MockClaudeClient.generate_recommendations — structure
# ---------------------------------------------------------------------------

def test_mock_generate_recommendations_returns_payload():
    payload = MockClaudeClient().generate_recommendations(
        hypotheses=_HYPOTHESES, surviving_hypotheses=_SURVIVING,
        hypothesis_challenges=_CHALLENGES, evidence_items=_EVIDENCE,
        decision_model={}, research_strategy={},
    )
    assert isinstance(payload, RecommendationPayload)


def test_mock_recommendation_count_matches_hypotheses():
    payload = MockClaudeClient().generate_recommendations(
        hypotheses=_HYPOTHESES, surviving_hypotheses=_SURVIVING,
        hypothesis_challenges=_CHALLENGES, evidence_items=_EVIDENCE,
        decision_model={}, research_strategy={},
    )
    assert len(payload.recommendations) == len(_HYPOTHESES)


def test_mock_recommendation_items_have_required_fields():
    payload = MockClaudeClient().generate_recommendations(
        hypotheses=_HYPOTHESES, surviving_hypotheses=_SURVIVING,
        hypothesis_challenges=_CHALLENGES, evidence_items=_EVIDENCE,
        decision_model={}, research_strategy={},
    )
    for r in payload.recommendations:
        assert r.id
        assert r.title
        assert r.summary
        assert r.priority in ("high", "medium", "low")
        assert r.time_horizon in ("near_term", "medium_term", "long_term")
        assert r.confidence in ("high", "medium", "low")
        assert isinstance(r.supported_by_hypotheses, list) and len(r.supported_by_hypotheses) > 0
        assert isinstance(r.key_risks, list) and len(r.key_risks) > 0
        assert isinstance(r.trigger_conditions, list) and len(r.trigger_conditions) > 0


def test_mock_portfolio_has_time_horizons():
    payload = MockClaudeClient().generate_recommendations(
        hypotheses=_HYPOTHESES, surviving_hypotheses=_SURVIVING,
        hypothesis_challenges=_CHALLENGES, evidence_items=_EVIDENCE,
        decision_model={}, research_strategy={},
    )
    p = payload.recommendation_portfolio
    assert isinstance(p.near_term, list)
    assert isinstance(p.medium_term, list)
    assert isinstance(p.long_term, list)
    # All recommendation IDs appear in the portfolio
    all_ids = set(r.id for r in payload.recommendations)
    portfolio_ids = set(p.near_term) | set(p.medium_term) | set(p.long_term)
    assert all_ids == portfolio_ids


def test_mock_synthesis_note_not_empty():
    payload = MockClaudeClient().generate_recommendations(
        hypotheses=_HYPOTHESES, surviving_hypotheses=_SURVIVING,
        hypothesis_challenges=_CHALLENGES, evidence_items=_EVIDENCE,
        decision_model={}, research_strategy={},
    )
    assert payload.synthesis_note


# ---------------------------------------------------------------------------
# Context persistence
# ---------------------------------------------------------------------------

def test_agent_writes_recommendations_to_context():
    ctx = _make_context()
    RecommendationAgent(client=MockClaudeClient()).run(ctx)
    assert len(ctx.recommendations) == 3


def test_agent_writes_portfolio_to_context():
    ctx = _make_context()
    RecommendationAgent(client=MockClaudeClient()).run(ctx)
    assert isinstance(ctx.recommendation_portfolio, dict)
    assert "near_term" in ctx.recommendation_portfolio


def test_agent_persists_to_research_object():
    ctx = _make_context()
    RecommendationAgent(client=MockClaudeClient()).run(ctx)
    assert "recommendations" in ctx.research_object
    assert "recommendation_portfolio" in ctx.research_object
    assert len(ctx.research_object["recommendations"]) == 3


def test_agent_writes_trace_key():
    ctx = _make_context()
    RecommendationAgent(client=MockClaudeClient()).run(ctx)
    assert "_recommendations" in ctx.trace
    t = ctx.trace["_recommendations"]
    assert "recommendations" in t
    assert "recommendation_portfolio" in t
    assert "synthesis_note" in t


def test_agent_appends_agent_history():
    ctx = _make_context()
    RecommendationAgent(client=MockClaudeClient()).run(ctx)
    assert any(h["agent"] == "RecommendationAgent" for h in ctx.agent_history)


# ---------------------------------------------------------------------------
# No-op when hypotheses absent
# ---------------------------------------------------------------------------

def test_no_hypotheses_returns_warning():
    ctx = _make_context()
    ctx.hypotheses = []
    result = RecommendationAgent(client=MockClaudeClient()).run(ctx)
    assert result.status == "warning"
    assert ctx.recommendations == []
    assert ctx.recommendation_portfolio == {}


# ---------------------------------------------------------------------------
# No client falls back to mock
# ---------------------------------------------------------------------------

def test_no_client_uses_mock():
    ctx = _make_context()
    RecommendationAgent(client=None).run(ctx)
    assert len(ctx.recommendations) == 3


# ---------------------------------------------------------------------------
# QA validation helper
# ---------------------------------------------------------------------------

def _sample_recs(ids: list[str]) -> list[dict]:
    return [
        {
            "id": rid,
            "title": f"Rec {rid}",
            "summary": "Do this.",
            "priority": "high",
            "time_horizon": "near_term",
            "supported_by_hypotheses": ["H1"],
            "supporting_evidence": ["E001"],
            "key_risks": ["Risk A"],
            "trigger_conditions": ["Trigger A"],
            "confidence": "medium",
            "confidence_rationale": "Based on H1.",
        }
        for rid in ids
    ]


def _sample_portfolio(ids: list[str]) -> dict:
    return {"near_term": ids, "medium_term": [], "long_term": []}


def test_validate_recommendations_all_valid():
    ids = ["R1", "R2"]
    result = _validate_recommendations(_sample_recs(ids), _sample_portfolio(ids))
    assert result["recommendations_present"] is True
    assert result["all_have_evidence"] is True
    assert result["all_have_hypothesis_links"] is True
    assert result["all_have_confidence"] is True
    assert result["all_have_time_horizon"] is True
    assert result["portfolio_present"] is True
    assert result["issues"] == []


def test_validate_recommendations_empty():
    result = _validate_recommendations([], {})
    assert result["recommendations_present"] is False
    assert result["issues"] == ["No recommendations generated"]


def test_validate_recommendations_missing_evidence():
    recs = _sample_recs(["R1"])
    recs[0]["supporting_evidence"] = []
    result = _validate_recommendations(recs, _sample_portfolio(["R1"]))
    assert result["all_have_evidence"] is False
    assert len(result["issues"]) > 0


def test_validate_recommendations_missing_hypothesis_link():
    recs = _sample_recs(["R1"])
    recs[0]["supported_by_hypotheses"] = []
    result = _validate_recommendations(recs, _sample_portfolio(["R1"]))
    assert result["all_have_hypothesis_links"] is False


def test_validate_recommendations_invalid_confidence():
    recs = _sample_recs(["R1"])
    recs[0]["confidence"] = "very_high"
    result = _validate_recommendations(recs, _sample_portfolio(["R1"]))
    assert result["all_have_confidence"] is False


def test_validate_recommendations_invalid_time_horizon():
    recs = _sample_recs(["R1"])
    recs[0]["time_horizon"] = "immediate"
    result = _validate_recommendations(recs, _sample_portfolio(["R1"]))
    assert result["all_have_time_horizon"] is False


def test_validate_recommendations_empty_portfolio():
    recs = _sample_recs(["R1"])
    result = _validate_recommendations(recs, {"near_term": [], "medium_term": [], "long_term": []})
    assert result["portfolio_present"] is False
    assert any("portfolio" in i.lower() for i in result["issues"])


# ---------------------------------------------------------------------------
# Orchestrator routing
# ---------------------------------------------------------------------------

def _make_stub(name: str):
    from functional_agents.base import FunctionalAgent
    from functional_agents.context import AgentContext
    def _execute(self, ctx: AgentContext) -> AgentContext:
        self._record(ctx, status="success", summary=f"{name} stub")
        return ctx
    return type(name, (FunctionalAgent,), {"_execute": _execute})


def test_orchestrator_includes_recommendation_state():
    from functional_agents.orchestrator import AgentOrchestrator
    _Stub = _make_stub("_Stub")
    ctx = AgentContext(
        question="test", profiles=["p"], execution_profile="p",
        research_object={"research_id": "R-TEST_030"}, run_id="orch001",
    )
    orch = AgentOrchestrator(
        planner_factory=_Stub, evidence_factory=_Stub, qa_factory=_Stub,
        report_factory=_Stub, hypothesis_factory=_Stub,
        challenge_factory=_Stub, recommendation_factory=_Stub,
    )
    result_ctx = orch.run(ctx)
    assert result_ctx.workflow_path
    assert WorkflowState.RECOMMENDATION == "RECOMMENDATION"


def test_orchestrator_skips_recommendation_when_factory_none():
    from functional_agents.orchestrator import AgentOrchestrator
    _Stub = _make_stub("_Stub2")
    ctx = AgentContext(
        question="test", profiles=["p"], execution_profile="p",
        research_object={"research_id": "R-TEST_031"}, run_id="orch002",
    )
    orch = AgentOrchestrator(
        planner_factory=_Stub, evidence_factory=_Stub, qa_factory=_Stub,
        report_factory=_Stub, hypothesis_factory=_Stub,
        challenge_factory=_Stub, recommendation_factory=None,
    )
    result_ctx = orch.run(ctx)
    assert result_ctx.workflow_path


# ---------------------------------------------------------------------------
# ReportAgent — recommendations section markdown
# ---------------------------------------------------------------------------

def test_report_recommendations_section_has_table():
    from functional_agents.report_agent import _build_recommendations_section
    md = _build_recommendations_section(_sample_recs(["R1"]), _sample_portfolio(["R1"]))
    assert "## Strategic Recommendations" in md
    assert "| Recommendation |" in md
    assert "| Priority |" in md
    assert "| Confidence |" in md
    assert "| Time Horizon |" in md


def test_report_recommendations_section_has_detail():
    from functional_agents.report_agent import _build_recommendations_section
    md = _build_recommendations_section(_sample_recs(["R1"]), _sample_portfolio(["R1"]))
    assert "Key Risks" in md
    assert "Trigger Conditions" in md
    assert "Supported by" in md
    assert "Evidence" in md


def test_report_recommendations_section_has_portfolio():
    from functional_agents.report_agent import _build_recommendations_section
    md = _build_recommendations_section(_sample_recs(["R1"]), _sample_portfolio(["R1"]))
    assert "Recommendation Portfolio" in md
    assert "Near-term" in md


def test_report_recommendations_section_empty_when_none():
    from functional_agents.report_agent import _build_recommendations_section
    assert _build_recommendations_section([], {}) == ""
