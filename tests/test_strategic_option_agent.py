"""J7.1 – StrategicOptionAgent tests.

Covers:
  - Public helpers: build_strategic_options, compute_option_comparison,
    compute_scenario_robustness, identify_preferred_option,
    build_option_portfolio, generate_strategic_options
  - Agent contract: inherits FunctionalAgent, run() returns AgentResult
  - Agent _execute(): writes to context, trace, RO
  - Structural assertions on schema and option count
"""

from __future__ import annotations

import pytest

from functional_agents.base import FunctionalAgent
from functional_agents.context import AgentContext, AgentResult, WorkflowState
from functional_agents.strategic_option_agent import (
    StrategicOptionAgent,
    build_strategic_options,
    build_option_portfolio,
    compute_option_comparison,
    compute_scenario_robustness,
    generate_strategic_options,
    identify_preferred_option,
    COMPARISON_CRITERIA,
    _OPTION_TEMPLATES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROFILES = ["ai_data_centers", "transmission"]


def _ctx(
    recommendations=None,
    scenarios=None,
    profiles=None,
) -> AgentContext:
    return AgentContext(
        question="Develop a strategy for AI infrastructure investment.",
        profiles=profiles or list(_PROFILES),
        execution_profile=(profiles or _PROFILES)[0],
        research_object={"id": "R-TEST_001"},
        run_id="testrun001",
        recommendations=recommendations or [],
        scenarios=scenarios or [],
    )


def _rec(rec_id="R1", title="Deploy GPUs with liquid cooling", profiles=None):
    return {
        "id": rec_id,
        "title": title,
        "contributing_profiles": profiles or [],
    }


def _scenario(label="base", scenario_type="base"):
    return {"label": label, "scenario_type": scenario_type}


# ---------------------------------------------------------------------------
# WorkflowState constant
# ---------------------------------------------------------------------------

def test_workflow_state_strategic_options_constant():
    assert WorkflowState.STRATEGIC_OPTIONS == "STRATEGIC_OPTIONS"


# ---------------------------------------------------------------------------
# _OPTION_TEMPLATES structure
# ---------------------------------------------------------------------------

def test_option_templates_count():
    assert len(_OPTION_TEMPLATES) >= 3


def test_option_templates_have_required_keys():
    required = {"option_id", "title", "strategic_logic", "where_to_play",
                "how_to_win", "required_capabilities", "key_investments",
                "dependencies", "risks", "trigger_conditions",
                "time_horizon", "posture", "comparison_scores"}
    for tmpl in _OPTION_TEMPLATES:
        missing = required - set(tmpl.keys())
        assert not missing, f"{tmpl['option_id']} missing: {missing}"


def test_option_templates_distinct_postures():
    postures = {t["posture"] for t in _OPTION_TEMPLATES}
    assert len(postures) == len(_OPTION_TEMPLATES)


def test_comparison_criteria_non_empty():
    assert len(COMPARISON_CRITERIA) >= 4


# ---------------------------------------------------------------------------
# build_strategic_options
# ---------------------------------------------------------------------------

def test_build_strategic_options_count():
    ctx = _ctx()
    opts = build_strategic_options(ctx)
    assert len(opts) >= 3


def test_build_strategic_options_have_option_id():
    opts = build_strategic_options(_ctx())
    for o in opts:
        assert "option_id" in o and o["option_id"]


def test_build_strategic_options_contributing_profiles():
    opts = build_strategic_options(_ctx())
    for o in opts:
        assert set(_PROFILES) == set(o["contributing_profiles"])


def test_build_strategic_options_have_supporting_recommendations():
    ctx = _ctx(recommendations=[_rec("R1"), _rec("R2")])
    opts = build_strategic_options(ctx)
    for o in opts:
        assert isinstance(o["supporting_recommendations"], list)


def test_build_strategic_options_no_recs_fallback():
    ctx = _ctx(recommendations=[])
    opts = build_strategic_options(ctx)
    # Should still produce options; supporting_recommendations may be empty
    assert len(opts) >= 3


def test_build_strategic_options_have_schema_fields():
    required = {"option_id", "title", "strategic_logic", "where_to_play",
                "how_to_win", "required_capabilities", "key_investments",
                "dependencies", "risks", "trigger_conditions",
                "supporting_recommendations", "supporting_evidence",
                "contributing_profiles"}
    opts = build_strategic_options(_ctx())
    for o in opts:
        missing = required - set(o.keys())
        assert not missing, f"{o['option_id']} missing: {missing}"


# ---------------------------------------------------------------------------
# compute_option_comparison
# ---------------------------------------------------------------------------

def test_compute_option_comparison_has_criteria():
    opts = build_strategic_options(_ctx())
    cmp = compute_option_comparison(opts)
    assert "criteria" in cmp
    assert set(cmp["criteria"]) == set(COMPARISON_CRITERIA)


def test_compute_option_comparison_has_scores():
    opts = build_strategic_options(_ctx())
    cmp = compute_option_comparison(opts)
    assert "scores" in cmp
    for oid in [o["option_id"] for o in opts]:
        assert oid in cmp["scores"]


def test_compute_option_comparison_score_values():
    opts = build_strategic_options(_ctx())
    cmp = compute_option_comparison(opts)
    valid_values = {"high", "medium", "low"}
    for oid, score_dict in cmp["scores"].items():
        for criterion, value in score_dict.items():
            assert value in valid_values, f"{oid}.{criterion} = {value!r}"


def test_compute_option_comparison_all_criteria_present():
    opts = build_strategic_options(_ctx())
    cmp = compute_option_comparison(opts)
    for oid, score_dict in cmp["scores"].items():
        for c in COMPARISON_CRITERIA:
            assert c in score_dict, f"{oid} missing criterion {c}"


# ---------------------------------------------------------------------------
# compute_scenario_robustness
# ---------------------------------------------------------------------------

def test_compute_scenario_robustness_with_scenarios():
    opts = build_strategic_options(_ctx())
    scenarios = [_scenario("base", "base"), _scenario("upside", "upside")]
    rob = compute_scenario_robustness(opts, scenarios)
    for o in opts:
        assert o["option_id"] in rob


def test_compute_scenario_robustness_values_in_range():
    opts = build_strategic_options(_ctx())
    rob = compute_scenario_robustness(opts, [])
    for oid, sc_scores in rob.items():
        for sc, score in sc_scores.items():
            assert 0.0 <= score <= 1.0, f"{oid}.{sc} = {score}"


def test_compute_scenario_robustness_without_scenarios_uses_canonical():
    opts = build_strategic_options(_ctx())
    rob = compute_scenario_robustness(opts, [])
    # Should still produce some scores per option
    for o in opts:
        assert len(rob[o["option_id"]]) > 0


def test_compute_scenario_robustness_with_downside():
    opts = build_strategic_options(_ctx())
    scenarios = [_scenario("transmission_delay", "downside")]
    rob = compute_scenario_robustness(opts, scenarios)
    # O2 (aggressive build) should score lower than O1 (grid-first) on transmission delay
    o1_score = rob.get("O1", {}).get("transmission_delay", 0)
    o2_score = rob.get("O2", {}).get("transmission_delay", 0)
    assert o1_score > o2_score


# ---------------------------------------------------------------------------
# identify_preferred_option
# ---------------------------------------------------------------------------

def test_identify_preferred_option_returns_dict():
    opts = build_strategic_options(_ctx())
    rob = compute_scenario_robustness(opts, [])
    pref = identify_preferred_option(opts, rob)
    assert isinstance(pref, dict)


def test_identify_preferred_option_has_option_id():
    opts = build_strategic_options(_ctx())
    rob = compute_scenario_robustness(opts, [])
    pref = identify_preferred_option(opts, rob)
    assert "option_id" in pref


def test_identify_preferred_option_has_rationale():
    opts = build_strategic_options(_ctx())
    rob = compute_scenario_robustness(opts, [])
    pref = identify_preferred_option(opts, rob)
    assert "rationale" in pref and pref["rationale"]


def test_identify_preferred_option_empty_options():
    pref = identify_preferred_option([], {})
    assert pref["option_id"] is None


# ---------------------------------------------------------------------------
# build_option_portfolio
# ---------------------------------------------------------------------------

def test_build_option_portfolio_has_time_horizons():
    opts = build_strategic_options(_ctx())
    portfolio = build_option_portfolio(opts)
    assert "near_term" in portfolio
    assert "medium_term" in portfolio
    assert "long_term" in portfolio


def test_build_option_portfolio_all_options_placed():
    opts = build_strategic_options(_ctx())
    portfolio = build_option_portfolio(opts)
    all_placed = (
        portfolio["near_term"] + portfolio["medium_term"] + portfolio["long_term"]
    )
    for o in opts:
        assert o["option_id"] in all_placed


def test_build_option_portfolio_values_are_lists():
    opts = build_strategic_options(_ctx())
    portfolio = build_option_portfolio(opts)
    for horizon, ids in portfolio.items():
        assert isinstance(ids, list)


# ---------------------------------------------------------------------------
# generate_strategic_options
# ---------------------------------------------------------------------------

def test_generate_strategic_options_has_all_keys():
    ctx = _ctx()
    output = generate_strategic_options(ctx)
    for key in ("strategic_options", "strategic_option_comparison",
                "option_scenario_robustness", "preferred_option",
                "strategic_option_portfolio", "option_count"):
        assert key in output, f"missing: {key}"


def test_generate_strategic_options_count():
    ctx = _ctx()
    output = generate_strategic_options(ctx)
    assert output["option_count"] >= 3
    assert len(output["strategic_options"]) >= 3


def test_generate_strategic_options_distinct_options():
    ctx = _ctx()
    output = generate_strategic_options(ctx)
    ids = [o["option_id"] for o in output["strategic_options"]]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# StrategicOptionAgent – contract
# ---------------------------------------------------------------------------

def test_agent_inherits_functional_agent():
    assert issubclass(StrategicOptionAgent, FunctionalAgent)


def test_agent_run_returns_agent_result():
    result = StrategicOptionAgent().run(_ctx())
    assert isinstance(result, AgentResult)


def test_agent_run_status_success():
    result = StrategicOptionAgent().run(_ctx())
    assert result.status == "success"


# ---------------------------------------------------------------------------
# StrategicOptionAgent – context writes
# ---------------------------------------------------------------------------

def test_agent_writes_strategic_options_to_context():
    result = StrategicOptionAgent().run(_ctx())
    assert isinstance(result.context.strategic_options, list)
    assert len(result.context.strategic_options) >= 3


def test_agent_writes_comparison_to_context():
    result = StrategicOptionAgent().run(_ctx())
    cmp = result.context.strategic_option_comparison
    assert isinstance(cmp, dict)
    assert "criteria" in cmp
    assert "scores" in cmp


def test_agent_writes_robustness_to_context():
    result = StrategicOptionAgent().run(_ctx())
    rob = result.context.option_scenario_robustness
    assert isinstance(rob, dict)
    assert len(rob) >= 3


def test_agent_writes_preferred_option_to_context():
    result = StrategicOptionAgent().run(_ctx())
    pref = result.context.preferred_option
    assert isinstance(pref, dict)
    assert "option_id" in pref


def test_agent_writes_portfolio_to_context():
    result = StrategicOptionAgent().run(_ctx())
    portfolio = result.context.strategic_option_portfolio
    assert isinstance(portfolio, dict)
    assert "near_term" in portfolio


def test_options_cite_contributing_profiles():
    result = StrategicOptionAgent().run(_ctx())
    for opt in result.context.strategic_options:
        assert len(opt.get("contributing_profiles", [])) > 0


# ---------------------------------------------------------------------------
# StrategicOptionAgent – research object
# ---------------------------------------------------------------------------

def test_agent_writes_to_research_object():
    result = StrategicOptionAgent().run(_ctx())
    ro = result.context.research_object
    assert "strategic_option_generation" in ro


def test_research_object_has_required_keys():
    result = StrategicOptionAgent().run(_ctx())
    sog = result.context.research_object["strategic_option_generation"]
    for key in ("strategic_options", "strategic_option_comparison",
                "option_scenario_robustness", "preferred_option",
                "strategic_option_portfolio", "option_count"):
        assert key in sog, f"RO missing: {key}"


# ---------------------------------------------------------------------------
# StrategicOptionAgent – trace
# ---------------------------------------------------------------------------

def test_agent_writes_trace_block():
    result = StrategicOptionAgent().run(_ctx())
    assert "_strategic_options" in result.context.trace


def test_trace_has_agent_field():
    result = StrategicOptionAgent().run(_ctx())
    trace = result.context.trace["_strategic_options"]
    assert trace["agent"] == "StrategicOptionAgent"


def test_trace_has_option_count():
    result = StrategicOptionAgent().run(_ctx())
    trace = result.context.trace["_strategic_options"]
    assert "option_count" in trace
    assert trace["option_count"] >= 3


def test_trace_has_strategic_options():
    result = StrategicOptionAgent().run(_ctx())
    trace = result.context.trace["_strategic_options"]
    assert "strategic_options" in trace
    assert isinstance(trace["strategic_options"], list)


def test_trace_has_preferred_option():
    result = StrategicOptionAgent().run(_ctx())
    trace = result.context.trace["_strategic_options"]
    assert "preferred_option" in trace
    assert "option_id" in trace["preferred_option"]


def test_trace_has_comparison_and_robustness():
    result = StrategicOptionAgent().run(_ctx())
    trace = result.context.trace["_strategic_options"]
    assert "strategic_option_comparison" in trace
    assert "option_scenario_robustness" in trace


def test_trace_has_portfolio():
    result = StrategicOptionAgent().run(_ctx())
    trace = result.context.trace["_strategic_options"]
    assert "strategic_option_portfolio" in trace
    portfolio = trace["strategic_option_portfolio"]
    assert "near_term" in portfolio
