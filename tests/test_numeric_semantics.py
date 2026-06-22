"""Tests for J6.5d Numeric Semantic Classifier.

Covers:
- _classify_value_semantic() returning correct labels
- Gate 8 threshold_vs_measurement suppression
- Gate 8 historical_progression suppression
- Comparison sub-clause detection
- True contradictions preserved (same entity, same context)
- compute_suppression_metrics tracks new semantic reasons
- QA numeric_semantic_validation block
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("yaml", MagicMock())

from research_agent.contradiction import (
    _classify_value_semantic,
    _find_value_position,
    detect_contradictions,
    compute_suppression_metrics,
)
from research_agent.schemas import EvidenceItem, SuppressedComparison


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ev(claim: str, scope: str = "rack") -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"E_{abs(hash(claim)) % 9999:04d}",
        claim=claim,
        category="power",
        source_document="test.pdf",
        evidence_snippet=claim[:60],
        relevance="direct",
        confidence="medium",
        scope=scope,
    )


def _sup(reason: str) -> SuppressedComparison:
    return SuppressedComparison(
        evidence_a_id="E001", evidence_b_id="E002",
        evidence_a_claim="a", evidence_b_claim="b",
        reason=reason, scope_a="rack", scope_b="rack", detail="test",
    )


# ---------------------------------------------------------------------------
# _find_value_position
# ---------------------------------------------------------------------------

def test_find_value_position_basic():
    pos = _find_value_position("rack power is 120 kW", 120.0, "kw")
    assert pos is not None
    assert pos >= 14  # starts at "120"


def test_find_value_position_float():
    pos = _find_value_position("power consumption is 1.5 kW", 1.5, "kw")
    assert pos is not None


def test_find_value_position_missing():
    pos = _find_value_position("rack power is 120 kW", 99.0, "kw")
    assert pos is None


# ---------------------------------------------------------------------------
# _classify_value_semantic
# ---------------------------------------------------------------------------

def test_semantic_primary_plain():
    sem = _classify_value_semantic(
        "rubin nvl72 rack power is rated at 120 kw", 120.0, "kw"
    )
    assert sem == "primary"


def test_semantic_threshold_above():
    sem = _classify_value_semantic(
        "air cooling becomes inadequate for ai clusters with rack power densities above 20 kw; liquid cooling required",
        20.0, "kw",
    )
    assert sem == "threshold"


def test_semantic_threshold_up_to():
    sem = _classify_value_semantic(
        "air cooling is effective up to 30 kw per rack",
        30.0, "kw",
    )
    assert sem == "threshold"


def test_semantic_threshold_limit_of():
    sem = _classify_value_semantic(
        "the thermal limit of 50 kw constrains air-cooled designs",
        50.0, "kw",
    )
    assert sem == "threshold"


def test_semantic_historical_cloud_era():
    sem = _classify_value_semantic(
        "from 20–40 kw (cloud era) to 500–600 kw (current ai training)",
        40.0, "kw",
    )
    assert sem == "historical_comparison"


def test_semantic_historical_traditional_comparison():
    """Value extracted from a 'compared to traditional racks' sub-clause."""
    sem = _classify_value_semantic(
        "ai racks reach 30–100 kw per rack, compared to traditional server racks that typically operate at 7–10 kw per rack",
        10.0, "kw",
    )
    assert sem == "historical_comparison"


def test_semantic_historical_legacy():
    sem = _classify_value_semantic(
        "power requirements have grown from legacy values of 15 kw to today's 120 kw",
        15.0, "kw",
    )
    assert sem == "historical_comparison"


def test_semantic_primary_average():
    """'more than 60 kW' should be primary, not threshold."""
    sem = _classify_value_semantic(
        "ai data centers have average rack power density of more than 60 kw in dedicated ai facilities",
        60.0, "kw",
    )
    assert sem == "primary"


# ---------------------------------------------------------------------------
# Gate 8 in detect_contradictions — threshold suppression
# ---------------------------------------------------------------------------

def test_threshold_vs_measurement_suppressed():
    """'inadequate above 20 kW' vs 'average 60 kW' — threshold, not contradiction."""
    a = _ev("Air cooling becomes inadequate for AI clusters with rack power densities above 20 kW.")
    b = _ev("AI data centers have average rack power density of more than 60 kW in dedicated AI facilities.")
    suppressed: list[SuppressedComparison] = []
    contradictions = detect_contradictions([a, b], out_suppressed=suppressed)
    assert len(contradictions) == 0
    thr_sups = [s for s in suppressed if s.reason == "threshold_vs_measurement"]
    assert len(thr_sups) > 0


def test_threshold_up_to_vs_measurement_suppressed():
    """'effective up to 30 kW' vs '120 kW AI rack' — threshold, not contradiction."""
    a = _ev("Air cooling is effective up to 30 kW per rack for standard workloads.")
    b = _ev("Rubin NVL72 rack power is rated at 120 kW.")
    suppressed: list[SuppressedComparison] = []
    contradictions = detect_contradictions([a, b], out_suppressed=suppressed)
    thr_sups = [s for s in suppressed if s.reason == "threshold_vs_measurement"]
    assert len(thr_sups) > 0
    assert len(contradictions) == 0


# ---------------------------------------------------------------------------
# Gate 8 — historical suppression
# ---------------------------------------------------------------------------

def test_cloud_era_vs_current_suppressed():
    """'cloud era 20–40 kW' vs 'current AI 60 kW' — progression, not contradiction."""
    a = _ev(
        "The shift in rack density—from 20–40 kW (cloud era) to 500–600 kW (current AI training)—"
        "and toward 1 MW (Rubin Ultra) means each conversion step wastes energy.",
        scope="rack",
    )
    b = _ev(
        "AI data centers have average rack power density of more than 60 kW in dedicated AI facilities.",
        scope="rack",
    )
    suppressed: list[SuppressedComparison] = []
    contradictions = detect_contradictions([a, b], out_suppressed=suppressed)
    assert len(contradictions) == 0
    hist_sups = [s for s in suppressed if s.reason == "historical_progression"]
    assert len(hist_sups) > 0


def test_traditional_comparison_clause_suppressed():
    """Value from 'compared to traditional racks at 7-10 kW' should not contradict AI average."""
    a = _ev(
        "AI computing racks can reach power densities of 30–100+ kW per rack, "
        "compared to traditional server racks that typically operate at 7–10 kW per rack.",
        scope="rack",
    )
    b = _ev(
        "AI data centers have average rack power density of more than 60 kW in dedicated AI facilities, "
        "compared to 7–10 kW in standard racks.",
        scope="rack",
    )
    suppressed: list[SuppressedComparison] = []
    contradictions = detect_contradictions([a, b], out_suppressed=suppressed)
    assert len(contradictions) == 0
    hist_sups = [s for s in suppressed if s.reason == "historical_progression"]
    assert len(hist_sups) > 0


def test_legacy_value_vs_current_suppressed():
    """Legacy reference number shouldn't contradict current value."""
    a = _ev("Rack power has grown from legacy values of 15 kW to 120 kW today.", scope="rack")
    b = _ev("Current AI rack power is now 120 kW or more.", scope="rack")
    suppressed: list[SuppressedComparison] = []
    contradictions = detect_contradictions([a, b], out_suppressed=suppressed)
    hist_sups = [s for s in suppressed if s.reason == "historical_progression"]
    # 15 kW (legacy) vs 120 kW (current) should be suppressed
    assert len(hist_sups) > 0
    assert len(contradictions) == 0


# ---------------------------------------------------------------------------
# True contradictions must survive
# ---------------------------------------------------------------------------

def test_same_entity_same_context_contradiction_preserved():
    """Rubin NVL72 at 120 kW vs 180 kW — same product, same context → contradiction."""
    a = _ev("Rubin NVL72 rack power is rated at 120 kW.")
    b = _ev("Rubin NVL72 rack power is estimated at 180 kW.")
    suppressed: list[SuppressedComparison] = []
    contradictions = detect_contradictions([a, b], out_suppressed=suppressed)
    thr_sups = [s for s in suppressed if s.reason in ("threshold_vs_measurement", "historical_progression")]
    assert len(thr_sups) == 0
    assert len(contradictions) >= 1


def test_plain_values_large_difference_contradiction():
    """Two plain current measurements with large difference → still a contradiction."""
    a = _ev("The rack requires 40 kW of power.", scope="rack")
    b = _ev("The rack requires 200 kW of power.", scope="rack")
    suppressed: list[SuppressedComparison] = []
    contradictions = detect_contradictions([a, b], out_suppressed=suppressed)
    thr_sups = [s for s in suppressed if s.reason in ("threshold_vs_measurement", "historical_progression")]
    assert len(thr_sups) == 0
    assert len(contradictions) >= 1


# ---------------------------------------------------------------------------
# Full false-positive batch — all 5 suppressed
# ---------------------------------------------------------------------------

def test_all_five_false_positives_suppressed():
    """Reproduce the exact 5 contradictions from the AI power-density research and verify 0 survive."""
    items = [
        _ev(
            "AI computing racks can reach power densities of 30–100+ kW per rack, "
            "compared to traditional server racks that typically operate at 7–10 kW per rack.",
            scope="rack",
        ),
        _ev(
            "AI data centers have average rack power density of more than 60 kW in dedicated AI facilities, "
            "compared to 7–10 kW in standard racks.",
            scope="rack",
        ),
        _ev(
            "Air cooling becomes inadequate for AI clusters with rack power densities above 20 kW; "
            "liquid cooling solutions are required for higher densities.",
            scope="rack",
        ),
        _ev(
            "The shift in rack density—from 20–40 kW (cloud era) to 500–600 kW (current AI training) "
            "and toward 1 MW (Rubin Ultra)—means each conversion step wastes energy.",
            scope="rack",
        ),
    ]
    suppressed: list[SuppressedComparison] = []
    contradictions = detect_contradictions(items, out_suppressed=suppressed)
    assert len(contradictions) == 0, (
        f"Expected 0 contradictions, got {len(contradictions)}: "
        + "; ".join(c.explanation for c in contradictions)
    )


# ---------------------------------------------------------------------------
# compute_suppression_metrics — new semantic flags
# ---------------------------------------------------------------------------

def test_metrics_threshold_filtering_present():
    suppressed = [_sup("threshold_vs_measurement")]
    metrics = compute_suppression_metrics(suppressed, final_count=1)
    assert metrics["threshold_filtering_present"] is True
    assert metrics["numeric_semantics"]["threshold_vs_measurement"] == 1


def test_metrics_historical_filtering_present():
    suppressed = [_sup("historical_progression")]
    metrics = compute_suppression_metrics(suppressed, final_count=0)
    assert metrics["historical_filtering_present"] is True
    assert metrics["numeric_semantics"]["historical_progression"] == 1


def test_metrics_all_semantic_absent():
    suppressed = [_sup("scope_mismatch")]
    metrics = compute_suppression_metrics(suppressed, final_count=2)
    assert metrics["threshold_filtering_present"] is False
    assert metrics["historical_filtering_present"] is False
    ns = metrics["numeric_semantics"]
    assert ns["threshold_vs_measurement"] == 0
    assert ns["historical_progression"] == 0


# ---------------------------------------------------------------------------
# QA numeric_semantic_validation
# ---------------------------------------------------------------------------

def test_qa_numeric_semantic_validation():
    from functional_agents.qa_agent import _validate_numeric_semantics
    from functional_agents.context import AgentContext

    ctx = AgentContext(goal="test")
    ctx.contradiction_metrics = {
        "by_reason": {
            "threshold_vs_measurement": 2,
            "historical_progression": 3,
            "range_average_compatible": 1,
            "temporal_progression": 1,
        },
        "numeric_semantics": {
            "threshold_vs_measurement": 2,
            "historical_progression": 3,
            "range_contains_value": 1,
        },
    }
    result = _validate_numeric_semantics(ctx)
    assert result["threshold_logic_present"] is True
    assert result["historical_logic_present"] is True
    assert result["range_logic_present"] is True
    assert result["projection_logic_present"] is True
    assert result["issues"] == []


def test_qa_numeric_semantic_validation_empty():
    from functional_agents.qa_agent import _validate_numeric_semantics
    from functional_agents.context import AgentContext

    ctx = AgentContext(goal="test")
    result = _validate_numeric_semantics(ctx)
    assert result["threshold_logic_present"] is False
    assert result["historical_logic_present"] is False
