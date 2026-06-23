"""Tests for J6.8 ScenarioAgent.

Covers:
- generate_scenarios() returns 3 scenarios with required structure
- Scenario assumptions cover all 7 dimensions
- Critical uncertainties present in each scenario
- Evidence IDs linked when available in RO
- stress_test_recommendations() returns one record per rec
- scenario_fit has base_case / upside_case / downside_case
- robustness_score in [0, 1]
- scenario_risks populated for downside
- scenario_adjustments generated for medium/weak downside recs
- _compute_robustness_score correct for known inputs
- ScenarioAgent writes context.scenarios / scenario_analysis
- QA validation block written by ScenarioAgent (then overwritten by QAAgent's _validate_scenario)
- Research object updated
- Trace written
- QAAgent._validate_scenario() function
- WorkflowState.SCENARIO exists
- AgentContext.scenarios / scenario_analysis fields exist
- Report section _build_scenario_section() generates markdown table
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("yaml", MagicMock())

from functional_agents.scenario_agent import (
    ScenarioAgent,
    generate_scenarios,
    stress_test_recommendations,
    _compute_scenario_fit,
    _compute_robustness_score,
    _SCENARIO_TEMPLATES,
    _FIT_TO_SCORE,
)
from functional_agents.context import AgentContext, WorkflowState
from functional_agents.report_agent import _build_scenario_section


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ctx(recs=None, ev_ids=None) -> AgentContext:
    ctx = AgentContext(goal="test")
    ev = [{"evidence_id": eid} for eid in (ev_ids or [])]
    ctx.research_object = {"evidence": ev}
    ctx.recommendations = recs or []
    ctx.hypotheses = []
    ctx.hypothesis_challenges = []
    return ctx


def _rec(rid: str, *, has_power=False, has_cooling=False, has_capital=False,
         has_risks=True, near_term=True) -> dict:
    keywords = []
    if has_power:
        keywords.append("power grid energy transmission utility")
    if has_cooling:
        keywords.append("liquid cooling thermal pue")
    if has_capital:
        keywords.append("capital investment fund budget")
    summary = "Must deploy infrastructure. " + " ".join(keywords) + " Now."
    risks = (
        ["Capital cost risk exceeding projections", "Operational delay risk from vendors"]
        if has_risks else []
    )
    return {
        "id": rid,
        "title": f"Recommendation {rid}",
        "summary": summary,
        "priority": "high",
        "time_horizon": "near_term" if near_term else "long_term",
        "supported_by_hypotheses": ["H1"],
        "supporting_evidence": ["E001"],
        "key_risks": risks,
        "trigger_conditions": [],
        "confidence": "high",
        "confidence_rationale": "Evidence supports this recommendation.",
    }


# ---------------------------------------------------------------------------
# WorkflowState
# ---------------------------------------------------------------------------

def test_workflow_state_scenario_exists():
    assert WorkflowState.SCENARIO == "SCENARIO"


# ---------------------------------------------------------------------------
# AgentContext fields
# ---------------------------------------------------------------------------

def test_context_scenarios_field_default():
    ctx = AgentContext(goal="test")
    assert isinstance(ctx.scenarios, list)
    assert ctx.scenarios == []


def test_context_scenario_analysis_field_default():
    ctx = AgentContext(goal="test")
    assert isinstance(ctx.scenario_analysis, dict)
    assert ctx.scenario_analysis == {}


# ---------------------------------------------------------------------------
# generate_scenarios()
# ---------------------------------------------------------------------------

def test_generate_scenarios_returns_three():
    ctx = _ctx()
    scenarios = generate_scenarios(ctx)
    assert len(scenarios) == 3


def test_generate_scenarios_ids():
    ctx = _ctx()
    scenarios = generate_scenarios(ctx)
    ids = [s["scenario_id"] for s in scenarios]
    assert "S1" in ids
    assert "S2" in ids
    assert "S3" in ids


def test_generate_scenarios_names():
    ctx = _ctx()
    scenarios = generate_scenarios(ctx)
    names = [s["name"] for s in scenarios]
    assert "Base Case" in names
    assert "Upside Case" in names
    assert "Downside Case" in names


def test_generate_scenarios_assumptions_have_seven_keys():
    ctx = _ctx()
    scenarios = generate_scenarios(ctx)
    required_keys = {
        "ai_demand_growth",
        "power_availability",
        "grid_interconnection_timelines",
        "transmission_constraints",
        "cooling_technology_readiness",
        "capital_availability",
        "regulatory_permitting",
    }
    for s in scenarios:
        assert required_keys.issubset(s["assumptions"].keys()), (
            f"Missing assumption keys in {s['scenario_id']}: "
            f"{required_keys - s['assumptions'].keys()}"
        )


def test_generate_scenarios_critical_uncertainties():
    ctx = _ctx()
    for s in generate_scenarios(ctx):
        assert len(s.get("critical_uncertainties", [])) >= 1, (
            f"{s['scenario_id']} has no critical_uncertainties"
        )


def test_generate_scenarios_probability_sums_to_one():
    ctx = _ctx()
    total = sum(s.get("probability", 0) for s in generate_scenarios(ctx))
    assert abs(total - 1.0) < 1e-9


def test_generate_scenarios_links_evidence_ids():
    ctx = _ctx(ev_ids=["E001", "E002", "E003", "E004", "E005", "E006"])
    scenarios = generate_scenarios(ctx)
    all_ids = [eid for s in scenarios for eid in s.get("evidence_ids", [])]
    assert len(all_ids) > 0
    assert all(eid.startswith("E") for eid in all_ids)


def test_generate_scenarios_no_evidence_ids():
    ctx = _ctx()
    scenarios = generate_scenarios(ctx)
    # Should not error; evidence_ids may be empty lists
    for s in scenarios:
        assert isinstance(s.get("evidence_ids", []), list)


# ---------------------------------------------------------------------------
# _compute_scenario_fit()
# ---------------------------------------------------------------------------

def test_fit_labels_valid():
    rec = _rec("R1", has_power=True, has_risks=True)
    fit = _compute_scenario_fit(rec)
    for k in ("base_case", "upside_case", "downside_case"):
        assert fit[k] in ("strong", "medium", "weak"), f"Unexpected fit label: {fit[k]}"


def test_fit_power_grid_rec_base_strong():
    rec = _rec("R1", has_power=True, has_risks=True)
    fit = _compute_scenario_fit(rec)
    assert fit["base_case"] == "strong"


def test_fit_capital_rec_upside_strong():
    rec = _rec("R1", has_capital=True, has_risks=True)
    fit = _compute_scenario_fit(rec)
    assert fit["upside_case"] == "strong"


def test_fit_cooling_rec_upside_strong():
    rec = _rec("R1", has_cooling=True, has_risks=True)
    fit = _compute_scenario_fit(rec)
    assert fit["upside_case"] == "strong"


def test_fit_no_risks_downside_weakens():
    rec_with = _rec("R_WITH", has_power=True, has_risks=True)
    rec_without = _rec("R_WITHOUT", has_power=True, has_risks=False)
    fit_with = _compute_scenario_fit(rec_with)
    fit_without = _compute_scenario_fit(rec_without)
    score_with = _FIT_TO_SCORE[fit_with["downside_case"]]
    score_without = _FIT_TO_SCORE[fit_without["downside_case"]]
    assert score_with >= score_without, (
        f"Risk-aware rec should be at least as robust as risk-unaware: "
        f"{fit_with['downside_case']} vs {fit_without['downside_case']}"
    )


# ---------------------------------------------------------------------------
# _compute_robustness_score()
# ---------------------------------------------------------------------------

def test_robustness_all_strong():
    fit = {"base_case": "strong", "upside_case": "strong", "downside_case": "strong"}
    assert _compute_robustness_score(fit) == 1.0


def test_robustness_all_medium():
    fit = {"base_case": "medium", "upside_case": "medium", "downside_case": "medium"}
    assert abs(_compute_robustness_score(fit) - 0.6) < 0.001


def test_robustness_mixed():
    fit = {"base_case": "strong", "upside_case": "strong", "downside_case": "medium"}
    score = _compute_robustness_score(fit)
    assert 0.6 < score < 1.0


def test_robustness_in_zero_one():
    for rec_args in [
        {"has_power": True, "has_risks": True},
        {"has_cooling": True},
        {"has_capital": False, "has_risks": False},
    ]:
        rec = _rec("R", **rec_args)
        fit = _compute_scenario_fit(rec)
        score = _compute_robustness_score(fit)
        assert 0.0 <= score <= 1.0, f"Robustness out of range: {score}"


# ---------------------------------------------------------------------------
# stress_test_recommendations()
# ---------------------------------------------------------------------------

def test_stress_test_returns_one_per_rec():
    recs = [_rec("R1"), _rec("R2"), _rec("R3")]
    results = stress_test_recommendations(recs, _SCENARIO_TEMPLATES)
    assert len(results) == 3


def test_stress_test_has_required_keys():
    recs = [_rec("R1", has_power=True, has_risks=True)]
    result = stress_test_recommendations(recs, _SCENARIO_TEMPLATES)[0]
    for k in ("recommendation_id", "title", "scenario_fit",
              "robustness_score", "scenario_risks", "scenario_adjustments"):
        assert k in result, f"Missing key: {k}"


def test_stress_test_robustness_score_present():
    recs = [_rec("R1")]
    result = stress_test_recommendations(recs, _SCENARIO_TEMPLATES)[0]
    assert isinstance(result["robustness_score"], float)
    assert 0.0 <= result["robustness_score"] <= 1.0


def test_stress_test_downside_risks_populated():
    recs = [_rec("R1", has_power=True)]
    result = stress_test_recommendations(recs, _SCENARIO_TEMPLATES)[0]
    assert len(result["scenario_risks"]["downside_case"]) >= 1


def test_stress_test_adjustments_for_medium_downside():
    recs = [_rec("R_WEAK", has_power=False, has_risks=False)]
    result = stress_test_recommendations(recs, _SCENARIO_TEMPLATES)[0]
    # No risks → downside_case should be medium or weak → adjustment generated
    downside_fit = result["scenario_fit"]["downside_case"]
    if downside_fit in ("medium", "weak"):
        adj_scenarios = [a["scenario"] for a in result["scenario_adjustments"]]
        assert "downside_case" in adj_scenarios, (
            f"Expected downside_case adjustment for {downside_fit} fit; got {adj_scenarios}"
        )


def test_stress_test_upside_adjustment_for_strong_base_and_upside():
    recs = [_rec("R1", has_capital=True, has_power=True, has_risks=True)]
    result = stress_test_recommendations(recs, _SCENARIO_TEMPLATES)[0]
    if (result["scenario_fit"]["upside_case"] == "strong" and
            result["scenario_fit"]["base_case"] == "strong"):
        adj_scenarios = [a["scenario"] for a in result["scenario_adjustments"]]
        assert "upside_case" in adj_scenarios


# ---------------------------------------------------------------------------
# ScenarioAgent contract
# ---------------------------------------------------------------------------

def test_agent_writes_scenarios():
    ctx = _ctx(recs=[_rec("R1", has_power=True)], ev_ids=["E001", "E002"])
    agent = ScenarioAgent()
    result_ctx = agent._execute(ctx)
    assert len(result_ctx.scenarios) == 3


def test_agent_writes_scenario_analysis():
    ctx = _ctx(recs=[_rec("R1")], ev_ids=["E001"])
    agent = ScenarioAgent()
    result_ctx = agent._execute(ctx)
    sa = result_ctx.scenario_analysis
    assert "scenarios" in sa
    assert "recommendation_stress_test" in sa
    assert "summary" in sa


def test_agent_summary_keys():
    ctx = _ctx(recs=[_rec("R1"), _rec("R2")])
    result_ctx = ScenarioAgent()._execute(ctx)
    summary = result_ctx.scenario_analysis["summary"]
    assert summary["scenario_count"] == 3
    assert summary["recommendations_stress_tested"] == 2
    assert "average_robustness_score" in summary


def test_agent_writes_qa_validation():
    ctx = _ctx(recs=[_rec("R1")])
    result_ctx = ScenarioAgent()._execute(ctx)
    v = result_ctx.qa.get("scenario_validation", {})
    assert v.get("scenarios_present") is True
    assert v.get("scenario_count") == 3
    assert v.get("recommendation_stress_test_present") is True
    assert v.get("robustness_scores_present") is True


def test_agent_writes_research_object():
    ctx = _ctx(recs=[_rec("R1")])
    result_ctx = ScenarioAgent()._execute(ctx)
    assert "scenarios" in result_ctx.research_object
    assert "scenario_analysis" in result_ctx.research_object


def test_agent_writes_trace():
    ctx = _ctx(recs=[_rec("R1")])
    result_ctx = ScenarioAgent()._execute(ctx)
    trace = result_ctx.trace.get("_scenario_analysis", {})
    agent_info = trace.get("scenario_agent", {})
    assert agent_info.get("scenarios_generated") == 3
    assert "recommendations_stress_tested" in agent_info


def test_agent_empty_recommendations():
    ctx = _ctx(recs=[])
    result_ctx = ScenarioAgent()._execute(ctx)
    summary = result_ctx.scenario_analysis["summary"]
    assert summary["recommendations_stress_tested"] == 0
    assert summary["average_robustness_score"] == 0.0


# ---------------------------------------------------------------------------
# _validate_scenario() (QAAgent helper)
# ---------------------------------------------------------------------------

def test_validate_scenario_from_qa_agent():
    from functional_agents.qa_agent import _validate_scenario

    ctx = _ctx(recs=[_rec("R1")])
    result_ctx = ScenarioAgent()._execute(ctx)

    validation = _validate_scenario(result_ctx)
    assert validation["scenarios_present"] is True
    assert validation["scenario_count"] == 3
    assert validation["recommendation_stress_test_present"] is True
    assert validation["robustness_scores_present"] is True


def test_validate_scenario_empty_context():
    from functional_agents.qa_agent import _validate_scenario

    ctx = AgentContext(goal="test")
    validation = _validate_scenario(ctx)
    assert validation["scenarios_present"] is False
    assert validation["scenario_count"] == 0


# ---------------------------------------------------------------------------
# Orchestrator wiring
# ---------------------------------------------------------------------------

def test_orchestrator_accepts_scenario_factory():
    from functional_agents.orchestrator import AgentOrchestrator

    dummy = lambda: MagicMock()
    # Should not raise
    orch = AgentOrchestrator(
        planner_factory=dummy,
        evidence_factory=dummy,
        qa_factory=dummy,
        report_factory=dummy,
        scenario_factory=dummy,
    )
    assert orch._scenario_factory is not None


# ---------------------------------------------------------------------------
# Report section
# ---------------------------------------------------------------------------

def test_build_scenario_section_returns_string():
    ctx = _ctx(recs=[_rec("R1", has_power=True, has_risks=True)])
    result_ctx = ScenarioAgent()._execute(ctx)
    md = _build_scenario_section(result_ctx.scenario_analysis)
    assert isinstance(md, str)
    assert len(md) > 0


def test_build_scenario_section_contains_heading():
    ctx = _ctx(recs=[_rec("R1")])
    result_ctx = ScenarioAgent()._execute(ctx)
    md = _build_scenario_section(result_ctx.scenario_analysis)
    assert "## Scenario Analysis" in md


def test_build_scenario_section_contains_all_scenario_names():
    ctx = _ctx(recs=[_rec("R1")])
    result_ctx = ScenarioAgent()._execute(ctx)
    md = _build_scenario_section(result_ctx.scenario_analysis)
    assert "Base Case" in md
    assert "Upside Case" in md
    assert "Downside Case" in md


def test_build_scenario_section_contains_robustness_table():
    ctx = _ctx(recs=[_rec("R1")])
    result_ctx = ScenarioAgent()._execute(ctx)
    md = _build_scenario_section(result_ctx.scenario_analysis)
    assert "Recommendation Robustness" in md
    assert "R1" in md


def test_build_scenario_section_empty_returns_empty():
    md = _build_scenario_section({})
    assert md == ""


def test_build_scenario_section_downside_adjustments():
    ctx = _ctx(recs=[_rec("R1", has_power=False, has_risks=False)])
    result_ctx = ScenarioAgent()._execute(ctx)
    md = _build_scenario_section(result_ctx.scenario_analysis)
    if "Downside-Case Adjustments" in md:
        assert "R1" in md
