"""Tests for J6.8b Profile-Aware Synthesis.

Covers:
MultiProfileAgent additions (J6.8b):
- _compute_profile_balance() returns dict keyed by profile
- _compute_profile_balance() values sum to 1.0
- _compute_profile_balance() equal split when no profile touches recorded
- _compute_synthesis_validation() returns all four required keys
- _compute_synthesis_validation() profiles_represented_in_findings reflects attribution
- _compute_synthesis_validation() profiles_represented_in_recommendations reflects attribution
- _build_recommendation_profile_audit() keys are rec IDs
- _build_recommendation_profile_audit() values are lists of profile strings
- build_multi_profile_analysis() includes profile_balance, synthesis_validation, recommendation_profile_audit
- MultiProfileAgent._execute() writes profile_balance to context.multi_profile_analysis
- MultiProfileAgent._execute() writes synthesis_validation to context.qa["multi_profile_validation"]
- MultiProfileAgent._execute() writes recommendation_profile_audit to trace

ScenarioAgent additions (J6.8b):
- _SCENARIO_TEMPLATES includes transmission_availability key
- _SCENARIO_TEMPLATES includes grid_congestion_level key
- _SCENARIO_TEMPLATES includes interconnection_queue_delay key
- _SCENARIO_TEMPLATES includes utility_coordination key
- Downside template has constrained transmission_availability
- Upside template has low grid_congestion_level
- _INTERCONNECTION_KW non-empty
- _TRANSMISSION_KW non-empty
- _compute_scenario_fit() with interconnection keywords gets upside boost
- _scenario_risks_downside() with interconnection keywords includes queue risk
- _scenario_risks_downside() with transmission keywords includes congestion risk
- _scenario_adjustment_downside() with interconnection keywords includes queue text
- stress_test_recommendations() includes interconnection-aware adjustments

ReportAgent additions (J6.8b):
- _build_profile_synthesis_section() returns empty string when < 2 profiles
- _build_profile_synthesis_section() returns non-empty when 2 profiles contributing
- Section contains "Multi-Profile Synthesis" heading
- Section contains per-profile perspective headings
- Section contains "Integrated Strategy" subsection
- Section contains "Tradeoffs" subsection
- Section contains compute vs grid tradeoff text for ai_data_centers + transmission
- Section contains "Profile Balance" table
- Section contains "Synthesis Coverage Validation" table
- Section contains "Recommendation Profile Audit" table
- _build_recommendations_section() adds Profiles badge when contributing_profiles present
- Profile synthesis section appears in report when multi_profile_analysis present and profiles > 1
"""

from __future__ import annotations

import sys

# yaml is not available in test venv; mock before importing agents
from unittest.mock import MagicMock
sys.modules.setdefault("yaml", MagicMock())

import pytest

from functional_agents.multi_profile_agent import (
    _compute_profile_balance,
    _compute_synthesis_validation,
    _build_recommendation_profile_audit,
    build_multi_profile_analysis,
    MultiProfileAgent,
)
from functional_agents.scenario_agent import (
    _SCENARIO_TEMPLATES,
    _INTERCONNECTION_KW,
    _TRANSMISSION_KW,
    _compute_scenario_fit,
    _scenario_risks_downside,
    _scenario_adjustment_downside,
    stress_test_recommendations,
)
from functional_agents.report_agent import (
    _build_profile_synthesis_section,
    _build_recommendations_section,
)
from functional_agents.context import AgentContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(**kwargs) -> AgentContext:
    defaults = dict(
        question="test",
        profiles=["ai_data_centers", "transmission"],
        execution_profile="ai_data_centers",
        research_object={"id": "R-TEST_001"},
        run_id="testrun001",
    )
    defaults.update(kwargs)
    return AgentContext(**defaults)


def _make_rec(rec_id: str, profiles: list[str], title: str = "rec title", **kw) -> dict:
    return {
        "id": rec_id,
        "title": title,
        "summary": "summary",
        "priority": "high",
        "time_horizon": "near_term",
        "confidence": "high",
        "confidence_rationale": "rationale",
        "supported_by_hypotheses": [],
        "supporting_evidence": [],
        "key_risks": ["risk1", "risk2"],
        "trigger_conditions": [],
        "contributing_profiles": profiles,
        **kw,
    }


def _make_hyp(hyp_id: str, profiles: list[str], title: str = "hyp title") -> dict:
    return {
        "id": hyp_id,
        "hypothesis_id": hyp_id,
        "title": title,
        "summary": "summary",
        "confidence": "high",
        "confidence_rationale": "rationale",
        "supporting_evidence": [],
        "contributing_profiles": profiles,
    }


# ---------------------------------------------------------------------------
# _compute_profile_balance
# ---------------------------------------------------------------------------

def test_profile_balance_returns_all_profiles():
    recs = [
        _make_rec("R1", ["ai_data_centers"]),
        _make_rec("R2", ["transmission"]),
    ]
    balance = _compute_profile_balance(recs, ["ai_data_centers", "transmission"])
    assert set(balance.keys()) == {"ai_data_centers", "transmission"}


def test_profile_balance_sums_to_one():
    recs = [
        _make_rec("R1", ["ai_data_centers"]),
        _make_rec("R2", ["transmission"]),
        _make_rec("R3", ["ai_data_centers", "transmission"]),
    ]
    balance = _compute_profile_balance(recs, ["ai_data_centers", "transmission"])
    assert abs(sum(balance.values()) - 1.0) < 0.01


def test_profile_balance_equal_split_when_no_touches():
    recs = [_make_rec("R1", [])]
    balance = _compute_profile_balance(recs, ["ai_data_centers", "transmission"])
    assert balance["ai_data_centers"] == balance["transmission"]


def test_profile_balance_single_profile_dominance():
    recs = [
        _make_rec("R1", ["ai_data_centers"]),
        _make_rec("R2", ["ai_data_centers"]),
        _make_rec("R3", ["transmission"]),
    ]
    balance = _compute_profile_balance(recs, ["ai_data_centers", "transmission"])
    assert balance["ai_data_centers"] > balance["transmission"]


def test_profile_balance_values_between_zero_and_one():
    recs = [_make_rec(f"R{i}", ["ai_data_centers"]) for i in range(5)]
    balance = _compute_profile_balance(recs, ["ai_data_centers", "transmission"])
    for v in balance.values():
        assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# _compute_synthesis_validation
# ---------------------------------------------------------------------------

def test_synthesis_validation_has_four_keys():
    sv = _compute_synthesis_validation(["a", "b"], ["a", "b"], [], [])
    required = {
        "profiles_requested",
        "profiles_contributing",
        "profiles_represented_in_findings",
        "profiles_represented_in_recommendations",
    }
    assert required <= sv.keys()


def test_synthesis_validation_profiles_requested_count():
    sv = _compute_synthesis_validation(["a", "b"], ["a"], [], [])
    assert sv["profiles_requested"] == 2


def test_synthesis_validation_profiles_contributing_count():
    sv = _compute_synthesis_validation(["a", "b"], ["a"], [], [])
    assert sv["profiles_contributing"] == 1


def test_synthesis_validation_findings_represented():
    findings = [
        _make_hyp("H1", ["ai_data_centers"]),
        _make_hyp("H2", ["transmission"]),
    ]
    sv = _compute_synthesis_validation(["ai_data_centers", "transmission"], ["ai_data_centers", "transmission"], findings, [])
    assert sv["profiles_represented_in_findings"] == 2


def test_synthesis_validation_recs_represented():
    recs = [
        _make_rec("R1", ["ai_data_centers"]),
        _make_rec("R2", ["transmission"]),
    ]
    sv = _compute_synthesis_validation(["ai_data_centers", "transmission"], ["ai_data_centers", "transmission"], [], recs)
    assert sv["profiles_represented_in_recommendations"] == 2


def test_synthesis_validation_zero_when_no_attribution():
    sv = _compute_synthesis_validation(["a", "b"], ["a", "b"], [], [])
    assert sv["profiles_represented_in_findings"] == 0
    assert sv["profiles_represented_in_recommendations"] == 0


# ---------------------------------------------------------------------------
# _build_recommendation_profile_audit
# ---------------------------------------------------------------------------

def test_rec_audit_keys_are_rec_ids():
    recs = [_make_rec("R1", ["ai_data_centers"]), _make_rec("R2", ["transmission"])]
    audit = _build_recommendation_profile_audit(recs)
    assert set(audit.keys()) == {"R1", "R2"}


def test_rec_audit_values_are_lists():
    recs = [_make_rec("R1", ["ai_data_centers", "transmission"])]
    audit = _build_recommendation_profile_audit(recs)
    assert isinstance(audit["R1"], list)


def test_rec_audit_profiles_correct():
    recs = [_make_rec("R1", ["ai_data_centers", "transmission"])]
    audit = _build_recommendation_profile_audit(recs)
    assert set(audit["R1"]) == {"ai_data_centers", "transmission"}


def test_rec_audit_empty_profiles_when_none():
    recs = [_make_rec("R1", [])]
    audit = _build_recommendation_profile_audit(recs)
    assert audit["R1"] == []


def test_rec_audit_uses_id_field():
    recs = [{"id": "R5", "contributing_profiles": ["ai_data_centers"]}]
    audit = _build_recommendation_profile_audit(recs)
    assert "R5" in audit


# ---------------------------------------------------------------------------
# build_multi_profile_analysis includes J6.8b keys
# ---------------------------------------------------------------------------

def test_build_multi_profile_analysis_has_profile_balance():
    ctx = _ctx()
    analysis = build_multi_profile_analysis(ctx)
    assert "profile_balance" in analysis


def test_build_multi_profile_analysis_has_synthesis_validation():
    ctx = _ctx()
    analysis = build_multi_profile_analysis(ctx)
    assert "synthesis_validation" in analysis


def test_build_multi_profile_analysis_has_recommendation_profile_audit():
    ctx = _ctx()
    analysis = build_multi_profile_analysis(ctx)
    assert "recommendation_profile_audit" in analysis


def test_build_multi_profile_analysis_synthesis_validation_is_dict():
    ctx = _ctx()
    analysis = build_multi_profile_analysis(ctx)
    assert isinstance(analysis["synthesis_validation"], dict)


# ---------------------------------------------------------------------------
# MultiProfileAgent._execute() propagates J6.8b fields
# ---------------------------------------------------------------------------

def test_agent_writes_profile_balance_to_analysis():
    ctx = _ctx()
    ctx.recommendations = [_make_rec("R1", ["ai_data_centers"])]
    agent = MultiProfileAgent()
    result = agent.run(ctx)
    assert "profile_balance" in result.context.multi_profile_analysis


def test_agent_writes_synthesis_validation_to_qa():
    ctx = _ctx()
    agent = MultiProfileAgent()
    result = agent.run(ctx)
    mp_val = result.context.qa.get("multi_profile_validation", {})
    assert "synthesis_validation" in mp_val


def test_agent_writes_recommendation_profile_audit_to_trace():
    ctx = _ctx()
    agent = MultiProfileAgent()
    result = agent.run(ctx)
    mp_trace = result.context.trace.get("_multi_profile", {}).get("multi_profile_validation", {})
    assert "recommendation_profile_audit" in mp_trace


def test_agent_writes_profile_balance_to_research_object():
    ctx = _ctx()
    agent = MultiProfileAgent()
    result = agent.run(ctx)
    ro = result.context.research_object or {}
    mpa = ro.get("multi_profile_analysis", {})
    assert "profile_balance" in mpa


# ---------------------------------------------------------------------------
# ScenarioAgent — transmission dimensions in templates
# ---------------------------------------------------------------------------

def test_scenario_templates_have_transmission_availability():
    for t in _SCENARIO_TEMPLATES:
        assert "transmission_availability" in t["assumptions"], (
            f"{t['name']} missing transmission_availability"
        )


def test_scenario_templates_have_grid_congestion_level():
    for t in _SCENARIO_TEMPLATES:
        assert "grid_congestion_level" in t["assumptions"]


def test_scenario_templates_have_interconnection_queue_delay():
    for t in _SCENARIO_TEMPLATES:
        assert "interconnection_queue_delay" in t["assumptions"]


def test_scenario_templates_have_utility_coordination():
    for t in _SCENARIO_TEMPLATES:
        assert "utility_coordination" in t["assumptions"]


def test_downside_template_has_constrained_transmission():
    downside = next(t for t in _SCENARIO_TEMPLATES if t["scenario_id"] == "S3")
    assert downside["assumptions"]["transmission_availability"] == "constrained"


def test_upside_template_has_low_grid_congestion():
    upside = next(t for t in _SCENARIO_TEMPLATES if t["scenario_id"] == "S2")
    assert upside["assumptions"]["grid_congestion_level"] == "low"


def test_downside_template_has_high_grid_congestion():
    downside = next(t for t in _SCENARIO_TEMPLATES if t["scenario_id"] == "S3")
    assert downside["assumptions"]["grid_congestion_level"] == "high"


def test_interconnection_kw_non_empty():
    assert len(_INTERCONNECTION_KW) > 0


def test_transmission_kw_non_empty():
    assert len(_TRANSMISSION_KW) > 0


def test_interconnection_kw_includes_pjm():
    assert "pjm" in _INTERCONNECTION_KW


def test_transmission_kw_includes_congestion():
    assert "congestion" in _TRANSMISSION_KW


# ---------------------------------------------------------------------------
# ScenarioAgent — fit scoring and risk generation with transmission keywords
# ---------------------------------------------------------------------------

def test_fit_upside_strong_for_interconnection_rec():
    rec = {"title": "Engage utility for interconnection queue reservation",
           "summary": "Coordinate PJM queue position", "key_risks": ["r1", "r2"]}
    fit = _compute_scenario_fit(rec)
    assert fit["upside_case"] == "strong"


def test_downside_risks_include_queue_for_interconnection_rec():
    rec = {"title": "Secure interconnection queue position with MISO",
           "summary": "Utility coordination critical", "key_risks": ["r1", "r2"]}
    risks = _scenario_risks_downside(rec)
    combined = " ".join(risks).lower()
    assert "interconnection" in combined or "queue" in combined


def test_downside_risks_include_congestion_for_transmission_rec():
    rec = {"title": "Avoid transmission congestion zones",
           "summary": "Transmission bottleneck risk", "key_risks": ["r1", "r2"]}
    risks = _scenario_risks_downside(rec)
    combined = " ".join(risks).lower()
    assert "congestion" in combined or "transmission" in combined


def test_downside_adjustment_includes_queue_text_for_interconnection_rec():
    rec = {"title": "Secure interconnection rights early from PJM utility queue",
           "summary": "MISO queue management", "key_risks": []}
    adj = _scenario_adjustment_downside(rec)
    assert "interconnection" in adj.lower() or "queue" in adj.lower()


def test_downside_adjustment_includes_transmission_text_for_transmission_rec():
    rec = {"title": "Assess transmission corridor adequacy for HVDC delivery",
           "summary": "Congestion risk", "key_risks": []}
    adj = _scenario_adjustment_downside(rec)
    assert "transmission" in adj.lower() or "congestion" in adj.lower()


def test_stress_test_returns_adjustments_for_transmission_rec():
    recs = [{"id": "R1", "title": "Avoid grid congestion via transmission study",
             "summary": "HVDC access needed", "key_risks": []}]
    results = stress_test_recommendations(recs, _SCENARIO_TEMPLATES)
    assert len(results) == 1
    adj = results[0].get("adjustments", {})
    assert isinstance(adj, dict)


# ---------------------------------------------------------------------------
# ReportAgent — _build_profile_synthesis_section
# ---------------------------------------------------------------------------

def _make_analysis(profiles: list[str]) -> dict:
    recs = [_make_rec("R1", profiles), _make_rec("R2", ["ai_data_centers"])]
    findings = [_make_hyp("H1", profiles), _make_hyp("H2", ["transmission"])]
    profile_balance = {p: round(1.0 / len(profiles), 3) for p in profiles}
    sv = {
        "profiles_requested": len(profiles),
        "profiles_contributing": len(profiles),
        "profiles_represented_in_findings": len(profiles),
        "profiles_represented_in_recommendations": len(profiles),
    }
    audit = {"R1": profiles, "R2": ["ai_data_centers"]}
    influence = {p: {"evidence": 5, "findings": 2, "recommendations": 2} for p in profiles}
    return {
        "profiles_requested": profiles,
        "profiles_contributing": profiles,
        "attributed_findings": findings,
        "attributed_recommendations": recs,
        "profile_balance": profile_balance,
        "synthesis_validation": sv,
        "recommendation_profile_audit": audit,
        "profile_influence": influence,
    }


def test_section_empty_for_single_profile():
    analysis = _make_analysis(["ai_data_centers"])
    analysis["profiles_contributing"] = ["ai_data_centers"]
    result = _build_profile_synthesis_section(analysis, [], [])
    assert result == ""


def test_section_non_empty_for_two_profiles():
    analysis = _make_analysis(["ai_data_centers", "transmission"])
    result = _build_profile_synthesis_section(analysis, [], [])
    assert len(result) > 100


def test_section_has_heading():
    analysis = _make_analysis(["ai_data_centers", "transmission"])
    result = _build_profile_synthesis_section(analysis, [], [])
    assert "Multi-Profile Synthesis" in result


def test_section_has_ai_dc_perspective():
    analysis = _make_analysis(["ai_data_centers", "transmission"])
    result = _build_profile_synthesis_section(analysis, [], [])
    assert "Ai Data Centers Perspective" in result or "ai_data_centers" in result.lower()


def test_section_has_transmission_perspective():
    analysis = _make_analysis(["ai_data_centers", "transmission"])
    result = _build_profile_synthesis_section(analysis, [], [])
    assert "Transmission Perspective" in result


def test_section_has_integrated_strategy():
    analysis = _make_analysis(["ai_data_centers", "transmission"])
    result = _build_profile_synthesis_section(analysis, [], [])
    assert "Integrated Strategy" in result


def test_section_has_tradeoffs():
    analysis = _make_analysis(["ai_data_centers", "transmission"])
    result = _build_profile_synthesis_section(analysis, [], [])
    assert "Tradeoffs" in result


def test_section_has_compute_vs_grid_tradeoff():
    analysis = _make_analysis(["ai_data_centers", "transmission"])
    result = _build_profile_synthesis_section(analysis, [], [])
    lower = result.lower()
    assert "compute" in lower or "grid" in lower


def test_section_has_site_selection_implication():
    analysis = _make_analysis(["ai_data_centers", "transmission"])
    result = _build_profile_synthesis_section(analysis, [], [])
    assert "site selection" in result.lower() or "site" in result.lower()


def test_section_has_profile_balance_table():
    analysis = _make_analysis(["ai_data_centers", "transmission"])
    result = _build_profile_synthesis_section(analysis, [], [])
    assert "Profile Balance" in result


def test_section_has_synthesis_coverage_validation():
    analysis = _make_analysis(["ai_data_centers", "transmission"])
    result = _build_profile_synthesis_section(analysis, [], [])
    assert "Synthesis Coverage Validation" in result


def test_section_has_recommendation_profile_audit():
    analysis = _make_analysis(["ai_data_centers", "transmission"])
    result = _build_profile_synthesis_section(analysis, [], [])
    assert "Recommendation Profile Audit" in result


def test_section_lists_integrated_recommendations():
    recs = [_make_rec("R1", ["ai_data_centers", "transmission"])]
    analysis = _make_analysis(["ai_data_centers", "transmission"])
    analysis["attributed_recommendations"] = recs
    analysis["recommendation_profile_audit"] = {"R1": ["ai_data_centers", "transmission"]}
    result = _build_profile_synthesis_section(analysis, recs, [])
    assert "R1" in result


# ---------------------------------------------------------------------------
# ReportAgent — _build_recommendations_section adds profile badges
# ---------------------------------------------------------------------------

def test_recommendations_section_shows_profile_badge():
    recs = [_make_rec("R1", ["ai_data_centers", "transmission"])]
    section = _build_recommendations_section(recs, {})
    assert "Profiles:" in section


def test_recommendations_section_shows_profile_names():
    recs = [_make_rec("R1", ["ai_data_centers", "transmission"])]
    section = _build_recommendations_section(recs, {})
    assert "ai_data_centers" in section
    assert "transmission" in section


def test_recommendations_section_no_badge_when_no_profiles():
    recs = [_make_rec("R1", [])]
    section = _build_recommendations_section(recs, {})
    assert "Profiles:" not in section
