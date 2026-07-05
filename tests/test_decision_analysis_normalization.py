"""PH1a — DecisionAnalysis LLM boundary normalization tests.

J11.2 extends this file with tests for the _normalize_tool_input production path:
the fix that allows a stringified-JSON object to be decoded BEFORE _validate_payload
(the actual production failure scenario).
"""

from __future__ import annotations

import json

import pytest

from research_agent.claude_client import ClaudeClient, DecisionAnalysisPayload, _normalize_tool_input
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


# ---------------------------------------------------------------------------
# J11.2 — _normalize_tool_input production path (the actual root-cause fix)
# ---------------------------------------------------------------------------

class TestNormalizeToolInput:
    """Direct unit tests for _normalize_tool_input (J11.2 fix)."""

    def test_decodes_stringified_object(self):
        """The exact production failure: analysis arrives as a {}-JSON string."""
        raw = {"analysis": json.dumps(_valid_analysis())}
        result = _normalize_tool_input(raw)
        assert isinstance(result["analysis"], dict)
        assert result["analysis"]["recommended_option_id"] == "OPT-1"

    def test_decodes_stringified_list(self):
        """Existing []-list behaviour unchanged."""
        raw = {"items": json.dumps([{"id": "x"}])}
        result = _normalize_tool_input(raw)
        assert result["items"] == [{"id": "x"}]

    def test_leaves_plain_strings_unchanged(self):
        """A string like 'OPT-A' must not be decoded."""
        raw = {"recommended_option_id": "OPT-A"}
        result = _normalize_tool_input(raw)
        assert result["recommended_option_id"] == "OPT-A"

    def test_leaves_actual_dicts_unchanged(self):
        """Already-parsed dicts pass through unmodified."""
        inner = {"key": "value"}
        raw = {"analysis": inner}
        result = _normalize_tool_input(raw)
        assert result["analysis"] is inner

    def test_leaves_invalid_json_string_unchanged(self):
        """{not valid json} is not decoded, stays a string."""
        raw = {"analysis": "{not valid json}"}
        result = _normalize_tool_input(raw)
        assert isinstance(result["analysis"], str)

    def test_non_dict_passthrough(self):
        """Non-dict input passes through unmodified."""
        assert _normalize_tool_input("string") == "string"
        assert _normalize_tool_input(None) is None
        assert _normalize_tool_input([1, 2]) == [1, 2]


class TestProductionPathEndToEnd:
    """Integration tests for the full _call_json → _normalize_tool_input → _validate_payload path.

    These tests simulate the production failure scenario (analysis as a string coming
    out of _response_tool_input) and verify the fix makes _validate_payload succeed.
    """

    def _make_client_with_tool_response(self, monkeypatch, tool_input_payload):
        """Return a ClaudeClient whose _call_json will receive the given raw tool input."""
        import research_agent.claude_client as cc_mod

        # Patch _response_tool_input to return the stringified payload (as Claude would)
        monkeypatch.setattr(cc_mod, "_response_tool_input", lambda resp: tool_input_payload)

        class FakeUsage:
            input_tokens = 100
            output_tokens = 100
            cache_read_input_tokens = None
            cache_creation_input_tokens = None

        class FakeResponse:
            stop_reason = "tool_use"
            usage = FakeUsage()

        class FakeMessages:
            def create(self, **kwargs):
                return FakeResponse()

        class FakeAnthropic:
            messages = FakeMessages()

        return ClaudeClient(anthropic_client=FakeAnthropic())

    def test_stringified_analysis_succeeds_through_validate_payload(self, monkeypatch):
        """The J11.2 production fix: {}-stringified analysis decodes before _validate_payload."""
        c = self._make_client_with_tool_response(
            monkeypatch,
            {"analysis": json.dumps(_valid_analysis())}
        )
        result = c.generate_decision_analysis([], [], [], [], [], {})
        assert isinstance(result, DecisionAnalysisPayload)
        assert result.analysis.recommended_option_id == "OPT-1"

    def test_valid_dict_analysis_still_works(self, monkeypatch):
        """Normal path (analysis already a dict) is unaffected by the fix."""
        c = self._make_client_with_tool_response(
            monkeypatch,
            {"analysis": _valid_analysis()}
        )
        result = c.generate_decision_analysis([], [], [], [], [], {})
        assert result.analysis.recommended_option_id == "OPT-1"

    def test_agent_gets_live_result_not_mock(self, monkeypatch):
        """With the fix, DecisionAnalysisAgent receives a live result; fallback NOT triggered."""
        import research_agent.claude_client as cc_mod
        monkeypatch.setattr(cc_mod, "_response_tool_input", lambda resp: {"analysis": json.dumps(_valid_analysis())})

        class FakeUsage:
            input_tokens, output_tokens = 100, 100
            cache_read_input_tokens = cache_creation_input_tokens = None

        class FakeResponse:
            stop_reason = "tool_use"
            usage = FakeUsage()

        class FakeAnthropic:
            class messages:
                @staticmethod
                def create(**kwargs):
                    return FakeResponse()

        ctx = _ctx()
        ctx.research_object = {"id": "R-TEST"}
        client = ClaudeClient(anthropic_client=FakeAnthropic())
        DecisionAnalysisAgent(client=client).run(ctx)

        # normalization fallback must NOT have been used
        diags = ctx.trace.get("_llm_normalization", [])
        assert diags, "normalization diagnostics should be present"
        assert not diags[-1]["fallback_used"], "fallback should NOT be triggered with the fix"
        assert ctx.decision_analysis["recommended_option_id"] == "OPT-1"
