"""PH1a — DecisionAnalysis LLM boundary normalization tests."""

from __future__ import annotations

import json

import pytest

from research_agent.claude_client import ClaudeClient, DecisionAnalysisPayload
from functional_agents.context import AgentContext
from functional_agents.decision_analysis_agent import DecisionAnalysisAgent


def _valid_analysis() -> dict:
    return {
        "analysis_id": "DA-001",
        "recommended_option_id": "OPT-1",
        "executive_summary": "Option 1 wins on speed and cost.",
        "comparison_dimensions": ["speed", "cost"],
        "option_rankings": ["OPT-1", "OPT-2"],
        "decision_matrix": [],
        "key_tradeoffs": ["Higher speed → higher cost"],
        "key_uncertainties": ["power draw"],
        "sensitivity_analysis": "If A-1 fails, OPT-2 wins.",
        "confidence_summary": "Medium confidence.",
        "rationale": "OPT-1 dominates on the weighted dimensions.",
        "confidence": "Medium",
    }


def _client(monkeypatch, payload) -> ClaudeClient:
    # anthropic_client set → __init__ does not require ANTHROPIC_API_KEY.
    c = ClaudeClient(anthropic_client=object())
    monkeypatch.setattr(c, "_call_json", lambda **kw: payload)
    return c


# ---------------------------------------------------------------------------
# Client-level normalization
# ---------------------------------------------------------------------------

def test_valid_object_payload(monkeypatch):
    c = _client(monkeypatch, {"analysis": _valid_analysis()})
    result = c.generate_decision_analysis([], [], [], [], [], {})
    assert isinstance(result, DecisionAnalysisPayload)
    assert result.analysis.recommended_option_id == "OPT-1"
    assert result.normalization["items_valid"] == 1
    assert result.normalization["fallback_used"] is False


def test_stringified_json_payload_recovered(monkeypatch):
    """The exact PH1a failure: analysis arrives as a stringified JSON object."""
    c = _client(monkeypatch, {"analysis": json.dumps(_valid_analysis())})
    result = c.generate_decision_analysis([], [], [], [], [], {})
    assert result.analysis.recommended_option_id == "OPT-1"
    assert result.normalization["items_valid"] == 1
    assert result.normalization["component"] == "decision_analysis"


def test_plain_string_payload_raises_for_agent_to_catch(monkeypatch):
    c = _client(monkeypatch, {"analysis": "the option one is best"})
    with pytest.raises(Exception):
        c.generate_decision_analysis([], [], [], [], [], {})


def test_missing_required_field_raises(monkeypatch):
    bad = _valid_analysis()
    del bad["recommended_option_id"]
    c = _client(monkeypatch, {"analysis": bad})
    with pytest.raises(Exception):
        c.generate_decision_analysis([], [], [], [], [], {})


# ---------------------------------------------------------------------------
# Agent-level graceful fallback (no runtime exception reaches the pipeline)
# ---------------------------------------------------------------------------

class _RaisingClient:
    is_mock = False

    def generate_decision_analysis(self, **kwargs):
        raise ValueError("1 validation error for DecisionAnalysisPayload / analysis input_type=str")


class _GoodClient:
    is_mock = False

    def generate_decision_analysis(self, **kwargs):
        return DecisionAnalysisPayload.model_validate({
            "analysis": _valid_analysis(),
            "normalization": {"component": "decision_analysis", "items_received": 1,
                              "items_valid": 1, "items_dropped": 0, "fallback_used": False},
        })


def _ctx() -> AgentContext:
    return AgentContext(
        question="q",
        strategic_options=[{"option_id": "OPT-1", "title": "Option 1"}],
        assumptions=[], risks=[], opportunities=[], recommendations=[],
        decision_model={"decision_model_id": None},
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={"id": "R-DA"},
        run_id="da001",
    )


def test_agent_degrades_gracefully_on_malformed(monkeypatch):
    ctx = _ctx()
    result = DecisionAnalysisAgent(client=_RaisingClient()).run(ctx)
    assert result.status == "success"                 # no runtime exception
    assert ctx.decision_analysis                        # mock analysis populated
    diags = ctx.trace.get("_llm_normalization", [])
    assert any(d.get("fallback_used") for d in diags)


def test_agent_records_normalization_on_success(monkeypatch):
    ctx = _ctx()
    DecisionAnalysisAgent(client=_GoodClient()).run(ctx)
    diags = ctx.trace.get("_llm_normalization", [])
    assert diags and diags[-1]["component"] == "decision_analysis"
    assert diags[-1]["fallback_used"] is False


def test_agent_decision_analysis_behavior_unchanged(monkeypatch):
    """Valid path yields the same analysis content the client produced."""
    ctx = _ctx()
    DecisionAnalysisAgent(client=_GoodClient()).run(ctx)
    assert ctx.decision_analysis["recommended_option_id"] == "OPT-1"
    assert ctx.decision_analysis["key_tradeoffs"] == ["Higher speed → higher cost"]
