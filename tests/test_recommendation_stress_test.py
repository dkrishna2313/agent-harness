"""Tests for J6.7a Recommendation Improvement Stress Test.

Verifies:
- SYNTHETIC_WEAK_RECS each has the intended weakness (low score in target dimension)
- run_stress_test() returns the required structure
- Every synthetic rec is detected as weak by _detect_weaknesses
- Every synthetic rec is improved (before_score < after_score)
- improvement_loop_validated = True after run
- All four weakness types represented
- Improvement metrics accurate
- History stored for all recs
- Traceability list emitted
- QA validation block correct
- Markdown report section generated
- CLI stress-test command imports without error
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("yaml", MagicMock())

from functional_agents.recommendation_stress_test import (
    SYNTHETIC_WEAK_RECS,
    run_stress_test,
    build_report_section,
)
from functional_agents.recommendation_improvement_agent import _detect_weaknesses
from research_agent.evaluation.recommendation_evaluator import (
    evaluate_recommendations,
    score_single_recommendation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _run():
    """Run the stress test once; cache result per module."""
    if not hasattr(_run, "_cached"):
        _run._cached = run_stress_test()
    return _run._cached


# ---------------------------------------------------------------------------
# Synthetic weak recommendation quality checks
# ---------------------------------------------------------------------------

def test_four_synthetic_recs_exist():
    assert len(SYNTHETIC_WEAK_RECS) == 4


def test_weakness_types_cover_all_four():
    types = {r["_weakness_type"] for r in SYNTHETIC_WEAK_RECS}
    assert "no_tradeoff_awareness" in types
    assert "no_risk_identification" in types
    assert "missing_evidence_links" in types
    assert "low_actionability" in types


def test_stress_tradeoff_has_zero_tradeoff_score():
    rec = next(r for r in SYNTHETIC_WEAK_RECS if r["id"] == "STRESS_TRADEOFF")
    scored = score_single_recommendation(rec)
    assert scored["tradeoff_score"] == 0.0, f"Expected 0.0, got {scored['tradeoff_score']}"


def test_stress_risk_has_zero_risk_score():
    rec = next(r for r in SYNTHETIC_WEAK_RECS if r["id"] == "STRESS_RISK")
    scored = score_single_recommendation(rec)
    assert scored["risk_score"] == 0.0, f"Expected 0.0, got {scored['risk_score']}"


def test_stress_evidence_has_zero_evidence_score():
    rec = next(r for r in SYNTHETIC_WEAK_RECS if r["id"] == "STRESS_EVIDENCE")
    scored = score_single_recommendation(rec)
    assert scored["evidence_support_score"] == 0.0, f"Expected 0.0, got {scored['evidence_support_score']}"


def test_stress_actionability_has_zero_actionability_score():
    rec = next(r for r in SYNTHETIC_WEAK_RECS if r["id"] == "STRESS_ACTIONABILITY")
    scored = score_single_recommendation(rec)
    assert scored["actionability_score"] == 0.0, f"Expected 0.0, got {scored['actionability_score']}"


def test_all_synthetic_recs_detected_as_weak():
    """_detect_weaknesses must return at least one weakness for every synthetic rec."""
    for rec in SYNTHETIC_WEAK_RECS:
        scored = score_single_recommendation(rec)
        weaknesses = _detect_weaknesses(scored)
        assert len(weaknesses) >= 1, (
            f"{rec['id']} expected at least one weakness; got none. "
            f"Scores: {scored}"
        )


def test_stress_tradeoff_detected_as_no_tradeoff():
    rec = next(r for r in SYNTHETIC_WEAK_RECS if r["id"] == "STRESS_TRADEOFF")
    scored = score_single_recommendation(rec)
    weaknesses = _detect_weaknesses(scored)
    assert "tradeoff" in weaknesses, f"Expected 'tradeoff' in {weaknesses}"


def test_stress_risk_detected_as_no_risk():
    rec = next(r for r in SYNTHETIC_WEAK_RECS if r["id"] == "STRESS_RISK")
    scored = score_single_recommendation(rec)
    weaknesses = _detect_weaknesses(scored)
    assert "risk" in weaknesses, f"Expected 'risk' in {weaknesses}"


def test_stress_evidence_detected_as_missing_evidence():
    rec = next(r for r in SYNTHETIC_WEAK_RECS if r["id"] == "STRESS_EVIDENCE")
    scored = score_single_recommendation(rec)
    weaknesses = _detect_weaknesses(scored)
    assert "evidence" in weaknesses, f"Expected 'evidence' in {weaknesses}"


def test_stress_actionability_detected_as_low_actionability():
    rec = next(r for r in SYNTHETIC_WEAK_RECS if r["id"] == "STRESS_ACTIONABILITY")
    scored = score_single_recommendation(rec)
    weaknesses = _detect_weaknesses(scored)
    assert "actionability" in weaknesses, f"Expected 'actionability' in {weaknesses}"


# ---------------------------------------------------------------------------
# run_stress_test() structure
# ---------------------------------------------------------------------------

def test_run_returns_required_keys():
    r = _run()
    required = {
        "synthetic_recommendations",
        "before_evaluation",
        "improvement_result",
        "after_evaluation",
        "comparison",
        "improvement_metrics",
        "recommendation_history",
        "qa_validation",
        "recommendation_improvements",
    }
    assert required.issubset(r.keys()), f"Missing keys: {required - r.keys()}"


def test_comparison_has_four_entries():
    r = _run()
    assert len(r["comparison"]) == 4


def test_all_recs_improved():
    """Every synthetic rec must show a positive delta — core proof of the loop."""
    r = _run()
    for c in r["comparison"]:
        assert c["improved"], (
            f"{c['recommendation_id']} did not improve: "
            f"before={c['before_score']:.3f}, after={c['after_score']:.3f}, delta={c['delta']:.3f}"
        )


def test_before_scores_are_lower_than_after():
    r = _run()
    for c in r["comparison"]:
        assert c["before_score"] < c["after_score"], (
            f"{c['recommendation_id']}: before {c['before_score']:.3f} >= after {c['after_score']:.3f}"
        )


def test_tradeoff_rec_tradeoff_score_improves():
    r = _run()
    c = next(x for x in r["comparison"] if x["recommendation_id"] == "STRESS_TRADEOFF")
    bd = c["before_dimensions"]
    ad = c["after_dimensions"]
    assert ad["tradeoff_score"] > bd["tradeoff_score"], (
        f"Tradeoff score did not improve: {bd['tradeoff_score']} → {ad['tradeoff_score']}"
    )


def test_risk_rec_risk_score_improves():
    r = _run()
    c = next(x for x in r["comparison"] if x["recommendation_id"] == "STRESS_RISK")
    bd = c["before_dimensions"]
    ad = c["after_dimensions"]
    assert ad["risk_score"] > bd["risk_score"], (
        f"Risk score did not improve: {bd['risk_score']} → {ad['risk_score']}"
    )


def test_evidence_rec_evidence_score_improves():
    r = _run()
    c = next(x for x in r["comparison"] if x["recommendation_id"] == "STRESS_EVIDENCE")
    bd = c["before_dimensions"]
    ad = c["after_dimensions"]
    assert ad["evidence_support_score"] > bd["evidence_support_score"], (
        f"Evidence score did not improve: {bd['evidence_support_score']} → {ad['evidence_support_score']}"
    )


def test_actionability_rec_actionability_score_improves():
    r = _run()
    c = next(x for x in r["comparison"] if x["recommendation_id"] == "STRESS_ACTIONABILITY")
    bd = c["before_dimensions"]
    ad = c["after_dimensions"]
    assert ad["actionability_score"] > bd["actionability_score"], (
        f"Actionability score did not improve: {bd['actionability_score']} → {ad['actionability_score']}"
    )


# ---------------------------------------------------------------------------
# QA validation
# ---------------------------------------------------------------------------

def test_improvement_loop_validated_true():
    r = _run()
    assert r["qa_validation"]["improvement_loop_validated"] is True


def test_qa_validation_has_required_keys():
    r = _run()
    qa = r["qa_validation"]
    for k in ("improvement_loop_validated", "recommendations_improved",
              "recommendations_unchanged", "average_score_before",
              "average_score_after", "average_delta"):
        assert k in qa, f"Missing QA key: {k}"


def test_recommendations_improved_equals_four():
    r = _run()
    assert r["qa_validation"]["recommendations_improved"] == 4


def test_average_delta_positive():
    r = _run()
    assert r["qa_validation"]["average_delta"] > 0


# ---------------------------------------------------------------------------
# Improvement metrics
# ---------------------------------------------------------------------------

def test_metrics_present():
    r = _run()
    m = r["improvement_metrics"]
    for k in ("recommendations_improved", "recommendations_unchanged",
              "average_score_before", "average_score_after", "average_delta"):
        assert k in m, f"Missing metric: {k}"


def test_metrics_improved_plus_unchanged_equals_four():
    r = _run()
    m = r["improvement_metrics"]
    assert m["recommendations_improved"] + m["recommendations_unchanged"] == 4


def test_average_score_after_exceeds_before():
    r = _run()
    m = r["improvement_metrics"]
    assert m["average_score_after"] > m["average_score_before"]


# ---------------------------------------------------------------------------
# Recommendation history
# ---------------------------------------------------------------------------

def test_history_has_four_entries():
    r = _run()
    assert len(r["recommendation_history"]) == 4


def test_history_has_version_and_score():
    r = _run()
    for h in r["recommendation_history"]:
        assert "recommendation_id" in h
        assert "version" in h
        assert "score" in h


def test_history_improved_entries_have_v2():
    r = _run()
    for h in r["recommendation_history"]:
        if h.get("improved"):
            assert "version_2_score" in h, f"Missing version_2_score: {h}"
            assert "delta" in h


# ---------------------------------------------------------------------------
# Traceability
# ---------------------------------------------------------------------------

def test_recommendation_improvements_list():
    r = _run()
    ti = r["recommendation_improvements"]
    assert len(ti) == 4
    for item in ti:
        assert "recommendation_id" in item
        assert "before_score" in item
        assert "after_score" in item
        assert "delta" in item


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def test_build_report_section_contains_table():
    r = _run()
    md = build_report_section(r)
    assert "## Recommendation Improvement Validation" in md
    assert "| Recommendation |" in md
    assert "STRESS_TRADEOFF" in md
    assert "STRESS_RISK" in md
    assert "STRESS_EVIDENCE" in md
    assert "STRESS_ACTIONABILITY" in md


def test_build_report_section_shows_improvement_validated():
    r = _run()
    md = build_report_section(r)
    assert "✓" in md


def test_build_report_section_has_dimension_breakdown():
    r = _run()
    md = build_report_section(r)
    assert "### Dimension Score Breakdown" in md
    assert "Evidence" in md
    assert "Tradeoff" in md
    assert "Risk" in md


def test_build_report_section_positive_deltas():
    r = _run()
    md = build_report_section(r)
    assert "+" in md  # at least one positive delta shown


# ---------------------------------------------------------------------------
# CLI import smoke test
# ---------------------------------------------------------------------------

def test_cli_stress_test_command_importable():
    from functional_agents.cli import stress_test_cmd
    assert callable(stress_test_cmd)
