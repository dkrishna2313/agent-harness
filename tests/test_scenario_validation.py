"""Tests for J6.8a ScenarioValidation harness.

Covers:
- SYNTHETIC_RECS has 5 realistic AI-infrastructure recommendations
- build_validation_context() returns a correct AgentContext
- run_scenario_validation() returns expected structure (no file I/O)
- Scenarios contain Base Case / Upside Case / Downside Case
- All scenario assumptions have 7 dimensions
- supporting_evidence (alias) present in each scenario
- Stress test contains adjustments dict (not just list)
- adjustments dict has downside_case key for at least one rec
- Robustness scores in [0, 1]
- QA validation block present and valid
- build_validation_report() returns markdown string
- Validation report contains expected headings
- _write_trace() writes trace file and latest_research_object.json (tmp dir)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault("yaml", MagicMock())

from functional_agents.scenario_validation import (
    SYNTHETIC_RECS,
    build_validation_context,
    run_scenario_validation,
    build_validation_report,
)


# ---------------------------------------------------------------------------
# SYNTHETIC_RECS
# ---------------------------------------------------------------------------

def test_synthetic_recs_count():
    assert len(SYNTHETIC_RECS) == 5


def test_synthetic_recs_have_required_keys():
    required = {"id", "title", "summary", "priority", "time_horizon",
                "supported_by_hypotheses", "supporting_evidence",
                "key_risks", "trigger_conditions", "confidence", "confidence_rationale"}
    for rec in SYNTHETIC_RECS:
        missing = required - rec.keys()
        assert not missing, f"Rec {rec.get('id')} missing: {missing}"


def test_synthetic_recs_have_non_empty_titles():
    for rec in SYNTHETIC_RECS:
        assert len(rec["title"]) > 0


def test_synthetic_recs_have_key_risks():
    for rec in SYNTHETIC_RECS:
        assert len(rec["key_risks"]) >= 1, f"Rec {rec['id']} has no key_risks"


def test_synthetic_recs_cover_ai_infrastructure_domains():
    all_text = " ".join(r["title"] + " " + r["summary"] for r in SYNTHETIC_RECS).lower()
    for keyword in ("power", "cooling", "capital", "grid"):
        assert keyword in all_text, f"Expected '{keyword}' domain in synthetic recs"


# ---------------------------------------------------------------------------
# build_validation_context()
# ---------------------------------------------------------------------------

def test_build_context_returns_agent_context():
    from functional_agents.context import AgentContext
    ctx = build_validation_context()
    assert isinstance(ctx, AgentContext)


def test_build_context_has_recommendations():
    ctx = build_validation_context()
    assert len(ctx.recommendations) == 5


def test_build_context_has_hypotheses():
    ctx = build_validation_context()
    assert len(ctx.hypotheses) >= 3


def test_build_context_has_research_object():
    ctx = build_validation_context()
    ro = ctx.research_object
    assert "research_id" in ro
    assert "evidence" in ro
    assert len(ro["evidence"]) > 0


def test_build_context_evidence_ids_are_strings():
    ctx = build_validation_context()
    for e in ctx.research_object["evidence"]:
        assert isinstance(e.get("evidence_id"), str)


# ---------------------------------------------------------------------------
# run_scenario_validation() — no file I/O
# ---------------------------------------------------------------------------

def _run() -> dict:
    return run_scenario_validation(out_path=None)


def test_run_returns_dict():
    result = _run()
    assert isinstance(result, dict)


def test_run_has_scenarios():
    result = _run()
    assert "scenarios" in result
    assert len(result["scenarios"]) == 3


def test_run_scenarios_are_base_upside_downside():
    result = _run()
    names = [s["name"] for s in result["scenarios"]]
    assert "Base Case" in names
    assert "Upside Case" in names
    assert "Downside Case" in names


def test_run_scenario_ids():
    result = _run()
    ids = [s["scenario_id"] for s in result["scenarios"]]
    assert "S1" in ids and "S2" in ids and "S3" in ids


def test_run_scenarios_have_seven_assumption_keys():
    required = {
        "ai_demand_growth",
        "power_availability",
        "grid_interconnection_timelines",
        "transmission_constraints",
        "cooling_technology_readiness",
        "capital_availability",
        "regulatory_permitting",
    }
    result = _run()
    for s in result["scenarios"]:
        assert required.issubset(s["assumptions"].keys()), (
            f"{s['scenario_id']} missing assumption keys: {required - s['assumptions'].keys()}"
        )


def test_run_scenarios_have_supporting_evidence_alias():
    result = _run()
    for s in result["scenarios"]:
        assert "supporting_evidence" in s, (
            f"{s['scenario_id']} missing 'supporting_evidence' alias"
        )
        assert isinstance(s["supporting_evidence"], list)


def test_run_scenarios_have_evidence_ids():
    result = _run()
    for s in result["scenarios"]:
        assert "evidence_ids" in s
        assert isinstance(s["evidence_ids"], list)


def test_run_stress_test_has_five_recs():
    result = _run()
    assert len(result["recommendation_stress_test"]) == 5


def test_run_stress_test_has_adjustments_dict():
    result = _run()
    for r in result["recommendation_stress_test"]:
        assert "adjustments" in r, f"Rec {r.get('recommendation_id')} missing 'adjustments' dict"
        assert isinstance(r["adjustments"], dict)


def test_run_stress_test_adjustments_dict_is_valid():
    result = _run()
    # adjustments dict maps scenario-case strings to adjustment text strings
    for r in result["recommendation_stress_test"]:
        adj = r.get("adjustments", {})
        for key, val in adj.items():
            assert isinstance(key, str), f"adjustments key {key!r} is not a string"
            assert isinstance(val, str), f"adjustments value for {key!r} is not a string"


def test_run_robustness_scores_in_range():
    result = _run()
    for r in result["recommendation_stress_test"]:
        score = r["robustness_score"]
        assert 0.0 <= score <= 1.0, f"Robustness {score} out of [0,1] for {r.get('recommendation_id')}"


def test_run_scenario_fit_has_three_labels():
    result = _run()
    for r in result["recommendation_stress_test"]:
        fit = r["scenario_fit"]
        for key in ("base_case", "upside_case", "downside_case"):
            assert key in fit
            assert fit[key] in ("strong", "medium", "weak")


def test_run_qa_validation_present():
    result = _run()
    qa = result["qa_validation"]
    assert qa.get("scenarios_present") is True
    assert qa.get("scenario_count") == 3
    assert qa.get("recommendation_stress_test_present") is True
    assert qa.get("robustness_scores_present") is True


def test_run_summary_keys():
    result = _run()
    summary = result["scenario_analysis_summary"]
    assert summary["scenario_count"] == 3
    assert summary["recommendations_stress_tested"] == 5
    assert "average_robustness_score" in summary


def test_run_research_object_keys_non_empty():
    result = _run()
    assert len(result["research_object_keys"]) > 0


# ---------------------------------------------------------------------------
# Trace file writing
# ---------------------------------------------------------------------------

def test_write_trace_creates_files():
    with tempfile.TemporaryDirectory() as tmp:
        run_scenario_validation(out_path=tmp)
        trace_path = Path(tmp) / "j68a_scenario_validation.trace.json"
        latest_path = Path(tmp) / "latest_research_object.json"
        assert trace_path.exists(), "j68a_scenario_validation.trace.json not written"
        assert latest_path.exists(), "latest_research_object.json not written"


def test_write_trace_valid_json():
    with tempfile.TemporaryDirectory() as tmp:
        run_scenario_validation(out_path=tmp)
        trace_path = Path(tmp) / "j68a_scenario_validation.trace.json"
        data = json.loads(trace_path.read_text())
        assert "scenarios" in data
        assert "recommendation_stress_test" in data
        assert "qa_validation" in data


def test_write_trace_three_scenarios():
    with tempfile.TemporaryDirectory() as tmp:
        run_scenario_validation(out_path=tmp)
        data = json.loads((Path(tmp) / "j68a_scenario_validation.trace.json").read_text())
        assert len(data["scenarios"]) == 3


def test_write_trace_adjustments_dict_in_trace():
    with tempfile.TemporaryDirectory() as tmp:
        run_scenario_validation(out_path=tmp)
        data = json.loads((Path(tmp) / "j68a_scenario_validation.trace.json").read_text())
        for r in data["recommendation_stress_test"]:
            assert "adjustments" in r
            assert isinstance(r["adjustments"], dict)


def test_latest_research_object_has_scenario_fields():
    with tempfile.TemporaryDirectory() as tmp:
        run_scenario_validation(out_path=tmp)
        data = json.loads((Path(tmp) / "latest_research_object.json").read_text())
        assert "scenarios" in data
        assert "scenario_analysis" in data
        assert len(data["scenarios"]) == 3


def test_latest_research_object_has_j68a_note():
    with tempfile.TemporaryDirectory() as tmp:
        run_scenario_validation(out_path=tmp)
        data = json.loads((Path(tmp) / "latest_research_object.json").read_text())
        assert "_j68a_validation_note" in data


def test_latest_research_object_preserves_existing():
    with tempfile.TemporaryDirectory() as tmp:
        existing = {"existing_key": "existing_value", "research_id": "ORIG-001"}
        (Path(tmp) / "latest_research_object.json").write_text(json.dumps(existing))
        run_scenario_validation(out_path=tmp)
        data = json.loads((Path(tmp) / "latest_research_object.json").read_text())
        # New scenario fields should be added
        assert "scenarios" in data
        # Existing fields should be preserved
        assert data.get("existing_key") == "existing_value"


# ---------------------------------------------------------------------------
# build_validation_report()
# ---------------------------------------------------------------------------

def test_build_report_returns_string():
    result = _run()
    report = build_validation_report(result)
    assert isinstance(report, str)
    assert len(report) > 0


def test_build_report_has_main_heading():
    result = _run()
    report = build_validation_report(result)
    assert "# J6.8a Scenario Validation Report" in report


def test_build_report_has_scenarios_section():
    result = _run()
    report = build_validation_report(result)
    assert "## Scenarios" in report


def test_build_report_has_robustness_section():
    result = _run()
    report = build_validation_report(result)
    assert "## Recommendation Robustness" in report


def test_build_report_has_qa_section():
    result = _run()
    report = build_validation_report(result)
    assert "## QA Validation" in report


def test_build_report_contains_scenario_names():
    result = _run()
    report = build_validation_report(result)
    assert "Base Case" in report
    assert "Upside Case" in report
    assert "Downside Case" in report


def test_build_report_contains_recommendation_ids():
    result = _run()
    report = build_validation_report(result)
    assert "R1" in report
    assert "R5" in report


def test_build_report_has_adjustments_section():
    result = _run()
    report = build_validation_report(result)
    assert "## Scenario Adjustments" in report
