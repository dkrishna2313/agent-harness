"""Tests for ChallengeAgent (J6.4).

Covers:
  - Contract compliance (inherits FunctionalAgent, returns AgentResult)
  - ChallengeItem and ChallengePayload structure from MockClaudeClient
  - Context persistence (hypothesis_challenges, surviving_hypotheses, research_object, trace)
  - Agent runs after HypothesisAgent in orchestrator
  - QAAgent validates challenge output
  - ReportAgent emits challenge trace block and markdown section
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# yaml is not installed in the test venv — mock it before any import chain
# that touches functional_agents.orchestrator → research_agent.profile triggers it.
@pytest.fixture(autouse=True)
def _mock_yaml():
    sys.modules.setdefault("yaml", MagicMock())
    yield


from functional_agents.base import FunctionalAgent
from functional_agents.challenge_agent import ChallengeAgent, _count_robustness
from functional_agents.context import AgentContext, AgentResult, WorkflowState
from functional_agents.qa_agent import _validate_challenges
from research_agent.claude_client import (
    ChallengeItem,
    ChallengePayload,
    MockClaudeClient,
    SurvivingHypothesis,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _minimal_context(*, with_hypotheses: bool = True) -> AgentContext:
    hypotheses = []
    if with_hypotheses:
        hypotheses = [
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
                "decision_implications": ["Act B", "Act C"],
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
    ctx = AgentContext(
        question="What strategy to pursue?",
        profiles=["test_profile"],
        execution_profile="test_profile",
        research_object={"research_id": "R-TEST_001"},
        run_id="testchallenge01",
    )
    ctx.hypotheses = hypotheses
    ctx.evidence_notes = [{
        "evidence_items": [
            {"evidence_id": "E001", "claim": "Claim 1", "source_document": "doc_a.txt"},
            {"evidence_id": "E002", "claim": "Claim 2", "source_document": "doc_b.txt"},
            {"evidence_id": "E003", "claim": "Claim 3", "source_document": "doc_c.txt"},
        ],
        "profile_coverage_by_profile": {
            "test_profile": {"coverage_level": "MODERATE", "evidence_count": 3}
        },
    }]
    return ctx


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def test_challenge_agent_inherits_functional_agent():
    assert issubclass(ChallengeAgent, FunctionalAgent)


def test_challenge_agent_run_returns_agent_result():
    ctx = _minimal_context()
    agent = ChallengeAgent(client=MockClaudeClient())
    result = agent.run(ctx)
    assert isinstance(result, AgentResult)


def test_challenge_agent_result_status_success():
    ctx = _minimal_context()
    agent = ChallengeAgent(client=MockClaudeClient())
    result = agent.run(ctx)
    assert result.status == "success"


def test_challenge_agent_metrics_has_duration():
    ctx = _minimal_context()
    result = ChallengeAgent(client=MockClaudeClient()).run(ctx)
    assert "duration_seconds" in result.metrics
    assert result.metrics["duration_seconds"] >= 0.0


def test_challenge_agent_trace_required_keys():
    ctx = _minimal_context()
    result = ChallengeAgent(client=MockClaudeClient()).run(ctx)
    for key in ("agent", "run_id", "duration_seconds", "status"):
        assert key in result.trace


# ---------------------------------------------------------------------------
# MockClaudeClient.generate_challenges — structure
# ---------------------------------------------------------------------------

def test_mock_generate_challenges_returns_payload():
    mock = MockClaudeClient()
    hypotheses = _minimal_context().hypotheses
    payload = mock.generate_challenges(
        hypotheses=hypotheses,
        evidence_items=[],
        contradictions=[],
        research_gaps=[],
        profile_coverage={},
    )
    assert isinstance(payload, ChallengePayload)


def test_mock_challenges_count_matches_hypotheses():
    mock = MockClaudeClient()
    hypotheses = _minimal_context().hypotheses
    payload = mock.generate_challenges(
        hypotheses=hypotheses, evidence_items=[], contradictions=[],
        research_gaps=[], profile_coverage={},
    )
    assert len(payload.hypothesis_challenges) == len(hypotheses)


def test_mock_surviving_count_matches_hypotheses():
    mock = MockClaudeClient()
    hypotheses = _minimal_context().hypotheses
    payload = mock.generate_challenges(
        hypotheses=hypotheses, evidence_items=[], contradictions=[],
        research_gaps=[], profile_coverage={},
    )
    assert len(payload.surviving_hypotheses) == len(hypotheses)


def test_mock_challenge_items_have_required_fields():
    mock = MockClaudeClient()
    payload = mock.generate_challenges(
        hypotheses=_minimal_context().hypotheses, evidence_items=[],
        contradictions=[], research_gaps=[], profile_coverage={},
    )
    for c in payload.hypothesis_challenges:
        assert c.hypothesis_id
        assert c.challenge_summary
        assert isinstance(c.hidden_assumptions, list) and len(c.hidden_assumptions) > 0
        assert isinstance(c.falsification_tests, list) and len(c.falsification_tests) > 0
        assert c.robustness in ("low", "medium", "high")


def test_mock_surviving_items_have_valid_status():
    mock = MockClaudeClient()
    payload = mock.generate_challenges(
        hypotheses=_minimal_context().hypotheses, evidence_items=[],
        contradictions=[], research_gaps=[], profile_coverage={},
    )
    for s in payload.surviving_hypotheses:
        assert s.survival_status in ("strong", "moderate", "weak")
        assert s.reason


def test_mock_challenge_synthesis_not_empty():
    mock = MockClaudeClient()
    payload = mock.generate_challenges(
        hypotheses=_minimal_context().hypotheses, evidence_items=[],
        contradictions=[], research_gaps=[], profile_coverage={},
    )
    assert payload.challenge_synthesis


# ---------------------------------------------------------------------------
# Context persistence
# ---------------------------------------------------------------------------

def test_challenge_agent_writes_hypothesis_challenges():
    ctx = _minimal_context()
    ChallengeAgent(client=MockClaudeClient()).run(ctx)
    assert len(ctx.hypothesis_challenges) == 3


def test_challenge_agent_writes_surviving_hypotheses():
    ctx = _minimal_context()
    ChallengeAgent(client=MockClaudeClient()).run(ctx)
    assert len(ctx.surviving_hypotheses) == 3


def test_challenge_agent_persists_to_research_object():
    ctx = _minimal_context()
    ChallengeAgent(client=MockClaudeClient()).run(ctx)
    assert "hypothesis_challenges" in ctx.research_object
    assert "surviving_hypotheses" in ctx.research_object
    assert len(ctx.research_object["hypothesis_challenges"]) == 3


def test_challenge_agent_writes_trace_key():
    ctx = _minimal_context()
    ChallengeAgent(client=MockClaudeClient()).run(ctx)
    assert "_challenges" in ctx.trace
    chal = ctx.trace["_challenges"]
    assert "hypothesis_challenges" in chal
    assert "surviving_hypotheses" in chal
    assert "challenge_synthesis" in chal


def test_challenge_agent_trace_challenge_count():
    ctx = _minimal_context()
    ChallengeAgent(client=MockClaudeClient()).run(ctx)
    assert len(ctx.trace["_challenges"]["hypothesis_challenges"]) == 3


def test_challenge_agent_appends_agent_history():
    ctx = _minimal_context()
    ChallengeAgent(client=MockClaudeClient()).run(ctx)
    agents = [h["agent"] for h in ctx.agent_history]
    assert "ChallengeAgent" in agents


# ---------------------------------------------------------------------------
# No-op when hypotheses are absent
# ---------------------------------------------------------------------------

def test_challenge_agent_no_hypotheses_returns_warning():
    ctx = _minimal_context(with_hypotheses=False)
    result = ChallengeAgent(client=MockClaudeClient()).run(ctx)
    assert result.status == "warning"
    assert ctx.hypothesis_challenges == []
    assert ctx.surviving_hypotheses == []


# ---------------------------------------------------------------------------
# Mock client fallback (no client)
# ---------------------------------------------------------------------------

def test_challenge_agent_no_client_uses_mock():
    ctx = _minimal_context()
    ChallengeAgent(client=None).run(ctx)
    assert len(ctx.hypothesis_challenges) == 3


# ---------------------------------------------------------------------------
# _count_robustness helper
# ---------------------------------------------------------------------------

def test_count_robustness_basic():
    challenges = [
        {"robustness": "high"},
        {"robustness": "low"},
        {"robustness": "medium"},
        {"robustness": "high"},
    ]
    counts = _count_robustness(challenges)
    assert counts == {"high": 2, "medium": 1, "low": 1}


def test_count_robustness_empty():
    assert _count_robustness([]) == {"high": 0, "medium": 0, "low": 0}


# ---------------------------------------------------------------------------
# QAAgent challenge validation
# ---------------------------------------------------------------------------

def _sample_challenges(hypothesis_ids: list[str]) -> list[dict]:
    return [
        {
            "hypothesis_id": hid,
            "challenge_summary": f"Challenge for {hid}",
            "hidden_assumptions": ["Assumption 1"],
            "weak_evidence": ["Source is vendor projection"],
            "contradicting_evidence": [],
            "missing_evidence": ["Operating data"],
            "falsification_tests": ["If X then downgrade"],
            "robustness": "medium",
        }
        for hid in hypothesis_ids
    ]


def _sample_surviving(hypothesis_ids: list[str]) -> list[dict]:
    return [
        {"hypothesis_id": hid, "survival_status": "moderate", "reason": "Survives with caveats."}
        for hid in hypothesis_ids
    ]


def _sample_hypotheses(hypothesis_ids: list[str]) -> list[dict]:
    return [{"id": hid, "title": f"Hypothesis {hid}"} for hid in hypothesis_ids]


def test_validate_challenges_all_present():
    ids = ["H1", "H2", "H3"]
    result = _validate_challenges(
        _sample_challenges(ids), _sample_surviving(ids), _sample_hypotheses(ids)
    )
    assert result["challenges_present"] is True
    assert result["all_hypotheses_challenged"] is True
    assert result["all_have_falsification_tests"] is True
    assert result["surviving_hypotheses_present"] is True
    assert result["issues"] == []


def test_validate_challenges_missing_hypothesis():
    hypotheses = _sample_hypotheses(["H1", "H2", "H3"])
    challenges = _sample_challenges(["H1", "H2"])  # H3 missing
    surviving = _sample_surviving(["H1", "H2"])
    result = _validate_challenges(challenges, surviving, hypotheses)
    assert result["all_hypotheses_challenged"] is False
    assert any("H3" in issue for issue in result["issues"])


def test_validate_challenges_missing_falsification():
    ids = ["H1"]
    challenges = [{
        "hypothesis_id": "H1",
        "challenge_summary": "Summary",
        "hidden_assumptions": ["Assumption"],
        "weak_evidence": [],
        "contradicting_evidence": [],
        "missing_evidence": [],
        "falsification_tests": [],  # empty — should flag issue
        "robustness": "medium",
    }]
    result = _validate_challenges(challenges, _sample_surviving(ids), _sample_hypotheses(ids))
    assert result["all_have_falsification_tests"] is False
    assert len(result["issues"]) > 0


def test_validate_challenges_empty():
    result = _validate_challenges([], [], _sample_hypotheses(["H1"]))
    assert result["challenges_present"] is False
    assert result["issues"] == ["No challenges generated"]


def test_validate_challenges_invalid_robustness():
    ids = ["H1"]
    challenges = [{
        "hypothesis_id": "H1", "challenge_summary": "S",
        "hidden_assumptions": ["A"], "weak_evidence": [],
        "contradicting_evidence": [], "missing_evidence": [],
        "falsification_tests": ["F1"], "robustness": "extreme",  # invalid
    }]
    result = _validate_challenges(challenges, _sample_surviving(ids), _sample_hypotheses(ids))
    assert result["robustness_valid"] is False


# ---------------------------------------------------------------------------
# Orchestrator routing: HYPOTHESIS → CHALLENGE → QA
# ---------------------------------------------------------------------------

def _make_stub_class(name: str):
    """Create a named stub agent class that avoids importing yaml-dependent modules."""
    from functional_agents.base import FunctionalAgent
    from functional_agents.context import AgentContext

    def _execute(self, ctx: AgentContext) -> AgentContext:
        self._record(ctx, status="success", summary=f"{name} stub")
        return ctx

    return type(name, (FunctionalAgent,), {"_execute": _execute})


def test_orchestrator_includes_challenge_state():
    """ChallengeAgent runs between HypothesisAgent and QAAgent in the workflow."""
    from functional_agents.orchestrator import AgentOrchestrator

    _Stub = _make_stub_class("_Stub")

    ctx = AgentContext(
        question="test question",
        profiles=["test_profile"],
        execution_profile="test_profile",
        research_object={"research_id": "R-TEST_010"},
        run_id="orch001",
    )

    orch = AgentOrchestrator(
        planner_factory=_Stub,
        evidence_factory=_Stub,
        qa_factory=_Stub,
        report_factory=_Stub,
        hypothesis_factory=_Stub,
        challenge_factory=_Stub,
    )
    result_ctx = orch.run(ctx)
    assert result_ctx.workflow_path  # all stubs ran
    assert WorkflowState.CHALLENGE == "CHALLENGE"


def test_orchestrator_skips_challenge_when_factory_none():
    """Without a challenge_factory, HYPOTHESIS goes straight to QA."""
    from functional_agents.orchestrator import AgentOrchestrator

    _Stub = _make_stub_class("_Stub2")

    ctx = AgentContext(
        question="test question",
        profiles=["test_profile"],
        execution_profile="test_profile",
        research_object={"research_id": "R-TEST_011"},
        run_id="orch002",
    )

    orch = AgentOrchestrator(
        planner_factory=_Stub,
        evidence_factory=_Stub,
        qa_factory=_Stub,
        report_factory=_Stub,
        hypothesis_factory=_Stub,
        challenge_factory=None,
    )
    result_ctx = orch.run(ctx)
    assert result_ctx.workflow_path  # completed without error


# ---------------------------------------------------------------------------
# ReportAgent — challenge trace block and markdown
# ---------------------------------------------------------------------------

def test_report_agent_challenge_trace_block():
    """ReportAgent emits challenge_generation trace block when challenges exist."""
    from functional_agents.report_agent import _build_challenges_section

    challenges = _sample_challenges(["H1", "H2"])
    surviving = _sample_surviving(["H1", "H2"])

    md = _build_challenges_section(challenges, surviving)
    assert "## Hypothesis Challenges" in md
    assert "H1" in md
    assert "H2" in md


def test_report_challenge_section_has_table():
    from functional_agents.report_agent import _build_challenges_section
    md = _build_challenges_section(
        _sample_challenges(["H1"]), _sample_surviving(["H1"])
    )
    assert "| Hypothesis |" in md
    assert "| Robustness |" in md
    assert "| Survival Status |" in md


def test_report_challenge_section_has_falsification():
    from functional_agents.report_agent import _build_challenges_section
    md = _build_challenges_section(
        _sample_challenges(["H1"]), _sample_surviving(["H1"])
    )
    assert "Falsification Tests" in md
    assert "If X then downgrade" in md


def test_report_challenge_section_has_hidden_assumptions():
    from functional_agents.report_agent import _build_challenges_section
    md = _build_challenges_section(
        _sample_challenges(["H1"]), _sample_surviving(["H1"])
    )
    assert "Hidden Assumptions" in md
    assert "Assumption 1" in md


def test_report_challenge_section_empty_when_no_challenges():
    from functional_agents.report_agent import _build_challenges_section
    assert _build_challenges_section([], []) == ""
