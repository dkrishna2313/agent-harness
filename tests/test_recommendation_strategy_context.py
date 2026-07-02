"""Tests for Strategic-Synthesis-informed recommendations (J10.8)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _mock_yaml():
    sys.modules.setdefault("yaml", MagicMock())
    yield


from functional_agents.context import AgentContext
from functional_agents.recommendation_agent import RecommendationAgent
from research_agent.claude_client import (
    _recommendation_prompt,
    _SYNTH_LIST_CAP,
    _SYNTH_SUMMARY_MAX_CHARS,
)

_HYPS = [
    {"id": "H1", "title": "H one", "summary": "s", "supporting_evidence": ["E001"],
     "confidence": "medium"},
]
_SURV = [{"hypothesis_id": "H1", "survival_status": "strong", "reason": "ok"}]
_CHAL = [{"hypothesis_id": "H1", "challenge_summary": "c", "robustness": "high"}]
_EV = [{"evidence_id": "E001", "claim": "Claim 1", "source_document": "doc.txt"}]

_SYNTHESIS = {
    "executive_summary": "Integrated cross-domain view.",
    "cross_domain_findings": [f"finding {i}" for i in range(8)],
    "cross_domain_dependencies": [f"A{i} depends on B{i}" for i in range(8)],
    "cross_domain_conflicts": [f"conflict {i}" for i in range(8)],
    "strategic_levers": [f"lever {i}" for i in range(8)],
    "dominant_constraints": [f"constraint {i}" for i in range(8)],
    "emerging_themes": [f"theme {i}" for i in range(8)],
}


def _ctx(*, with_synthesis: bool) -> AgentContext:
    ctx = AgentContext(
        question="What strategy?",
        profiles=["test_profile"],
        execution_profile="test_profile",
        research_object={"research_id": "R-RSC"},
        run_id="rsc01",
    )
    ctx.hypotheses = _HYPS
    ctx.surviving_hypotheses = _SURV
    ctx.hypothesis_challenges = _CHAL
    ctx.evidence_notes = [{"evidence_items": _EV, "profile_coverage_by_profile": {}}]
    if with_synthesis:
        ctx.strategic_synthesis = _SYNTHESIS
    return ctx


# ---------------------------------------------------------------------------
# Prompt: bounded synthesis section
# ---------------------------------------------------------------------------

def test_prompt_includes_synthesis_when_present():
    prompt = _recommendation_prompt(
        _HYPS, _SURV, _CHAL, _EV, {"objective": "obj"}, {},
        strategic_synthesis=_SYNTHESIS,
    )
    assert "Strategic Synthesis" in prompt
    assert "Integrated cross-domain view." in prompt


def test_prompt_omits_synthesis_when_absent():
    prompt = _recommendation_prompt(_HYPS, _SURV, _CHAL, _EV, {"objective": "obj"}, {})
    assert "Strategic Synthesis" not in prompt


def test_prompt_caps_lists():
    prompt = _recommendation_prompt(
        _HYPS, _SURV, _CHAL, _EV, {"objective": "obj"}, {},
        strategic_synthesis=_SYNTHESIS,
    )
    # Only the first _SYNTH_LIST_CAP findings appear; the (cap)th index does not.
    assert "finding 0" in prompt
    assert f"finding {_SYNTH_LIST_CAP}" not in prompt


def test_prompt_caps_summary_length():
    long_summary = "x" * 2000
    synth = {**_SYNTHESIS, "executive_summary": long_summary}
    prompt = _recommendation_prompt(
        _HYPS, _SURV, _CHAL, _EV, {"objective": "obj"}, {}, strategic_synthesis=synth,
    )
    assert ("x" * _SYNTH_SUMMARY_MAX_CHARS) in prompt
    assert ("x" * (_SYNTH_SUMMARY_MAX_CHARS + 1)) not in prompt


# ---------------------------------------------------------------------------
# Agent: consumes synthesis, records diagnostics, falls back cleanly
# ---------------------------------------------------------------------------

def test_agent_generates_recommendations_with_synthesis():
    ctx = _ctx(with_synthesis=True)
    RecommendationAgent().run(ctx)
    assert ctx.recommendations  # generated successfully


def test_diagnostics_present_with_synthesis():
    ctx = _ctx(with_synthesis=True)
    RecommendationAgent().run(ctx)
    diag = ctx.trace["_recommendation_strategy_context"]
    assert diag["strategic_synthesis_available"] is True
    assert diag["strategic_synthesis_used"] is True
    # Counts capped at 5.
    assert diag["cross_domain_findings_used"] == 5
    assert diag["dependencies_used"] == 5
    assert diag["conflicts_used"] == 5
    assert diag["strategic_levers_used"] == 5
    assert diag["dominant_constraints_used"] == 5
    assert diag["emerging_themes_used"] == 5


def test_fallback_when_synthesis_absent():
    ctx = _ctx(with_synthesis=False)
    result = RecommendationAgent().run(ctx)
    assert result.status == "success"
    assert ctx.recommendations
    diag = ctx.trace["_recommendation_strategy_context"]
    assert diag["strategic_synthesis_available"] is False
    assert diag["strategic_synthesis_used"] is False
    assert diag["cross_domain_findings_used"] == 0


def test_recommendation_schema_unchanged():
    """Recommendation dicts carry the existing fields; no new required schema."""
    ctx = _ctx(with_synthesis=True)
    RecommendationAgent().run(ctx)
    rec = ctx.recommendations[0]
    for key in ("id", "title", "summary", "priority", "time_horizon",
                "supporting_evidence", "confidence"):
        assert key in rec


def test_client_receives_synthesis_kwarg():
    """The agent passes strategic_synthesis through to the client."""
    captured = {}

    class _CaptureClient:
        is_mock = False
        def generate_recommendations(self, *args, **kwargs):
            captured["synthesis"] = kwargs.get("strategic_synthesis")
            from research_agent.claude_client import MockClaudeClient
            return MockClaudeClient().generate_recommendations(
                hypotheses=_HYPS, surviving_hypotheses=_SURV,
                hypothesis_challenges=_CHAL, evidence_items=_EV,
                decision_model={}, research_strategy={},
            )

    ctx = _ctx(with_synthesis=True)
    RecommendationAgent(client=_CaptureClient()).run(ctx)
    assert captured["synthesis"] == _SYNTHESIS
