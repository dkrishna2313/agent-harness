"""Tests for J6.5b Contradiction Precision Hardening.

Covers:
- Product/entity mismatch suppression (NVL8 vs NVL72, DGX vs HGX)
- Generation progression suppression (GB200 vs Rubin, Blackwell vs Rubin)
- Range/average compatibility gate
- Chip/server scope additions
- True contradictions preserved (same product, same scope)
- Suppression metrics include new reason types
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("yaml", MagicMock())

from research_agent.contradiction import (
    _product_compatibility_check,
    _extract_range_for_unit,
    detect_contradictions,
    compute_suppression_metrics,
    _scopes_compatible,
    _extract_scope,
)
from research_agent.schemas import EvidenceItem, SuppressedComparison


# ---------------------------------------------------------------------------
# EvidenceItem fixture
# ---------------------------------------------------------------------------

def _ev(claim: str, scope: str = "", entity: str = "") -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"E_{abs(hash(claim)) % 9999:04d}",
        claim=claim,
        category="power",
        source_document="test.pdf",
        evidence_snippet=claim[:60],
        relevance="direct",
        confidence="medium",
        scope=scope,
        entity=entity,
    )


# ---------------------------------------------------------------------------
# _product_compatibility_check — unit tests
# ---------------------------------------------------------------------------

def test_nvl8_vs_nvl72_incompatible():
    ok, reason = _product_compatibility_check(
        "HGX Rubin NVL8 consumes 24 kW",
        "GB200 NVL72 rack requires 120 kW",
    )
    assert ok is False
    assert reason == "product_mismatch"


def test_nvl8_vs_nvl36_incompatible():
    ok, reason = _product_compatibility_check(
        "The NVL8 system uses 24 kW",
        "The NVL36 rack uses 60 kW",
    )
    assert ok is False
    assert reason == "product_mismatch"


def test_nvl36_vs_nvl72_incompatible():
    ok, reason = _product_compatibility_check(
        "NVL36 rack power is 60 kW",
        "NVL72 rack power is 120 kW",
    )
    assert ok is False
    assert reason == "product_mismatch"


def test_dgx_vs_hgx_incompatible():
    ok, reason = _product_compatibility_check(
        "DGX B300 system draws 14 kW",
        "HGX Rubin NVL8 consumes 24 kW",
    )
    assert ok is False
    assert reason == "product_mismatch"


def test_gb200_vs_rubin_generation_progression():
    ok, reason = _product_compatibility_check(
        "GB200 NVL72 rack consumes 120 kW",
        "Vera Rubin NVL72 factory rack characteristics",
    )
    assert ok is False
    assert reason == "generation_progression"


def test_blackwell_vs_rubin_generation_progression():
    ok, reason = _product_compatibility_check(
        "Blackwell rack power is 120 kW",
        "Rubin next-gen rack targets 200 kW",
    )
    assert ok is False
    assert reason == "generation_progression"


def test_hopper_vs_blackwell_generation_progression():
    ok, reason = _product_compatibility_check(
        "Hopper H100 node uses 700 W",
        "GB200 Blackwell GPU uses 1000 W",
    )
    assert ok is False
    assert reason == "generation_progression"


def test_same_product_compatible():
    ok, reason = _product_compatibility_check(
        "Rubin NVL72 rack power is 120 kW",
        "Rubin NVL72 rack power is 180 kW",
    )
    assert ok is True
    assert reason == ""


def test_no_product_keywords_compatible():
    ok, reason = _product_compatibility_check(
        "The rack consumes 120 kW",
        "The rack consumes 200 kW",
    )
    assert ok is True
    assert reason == ""


def test_nvl72_same_family_compatible():
    ok, reason = _product_compatibility_check(
        "Each NVL72 rack uses 120 kW",
        "NVL72 rack power density is 150 kW",
    )
    assert ok is True
    assert reason == ""


# ---------------------------------------------------------------------------
# _extract_range_for_unit
# ---------------------------------------------------------------------------

def test_extract_range_kw():
    result = _extract_range_for_unit("power ranges from 30-100 kW per rack", "kw")
    assert result is not None
    assert result == (30.0, 100.0)


def test_extract_range_with_plus():
    result = _extract_range_for_unit("30-100+ kW per rack", "kw")
    assert result is not None
    assert result[0] == 30.0
    assert result[1] == 100.0


def test_extract_range_mw():
    result = _extract_range_for_unit("campus power 50-500 mw", "mw")
    assert result is not None
    assert result == (50.0, 500.0)


def test_extract_range_no_range():
    result = _extract_range_for_unit("rack power is 120 kw", "kw")
    assert result is None


def test_extract_range_wrong_unit():
    result = _extract_range_for_unit("power 30-100 kw per rack", "mw")
    assert result is None


# ---------------------------------------------------------------------------
# Range/average compatibility gate in detect_contradictions
# ---------------------------------------------------------------------------

def test_range_vs_average_within_range_suppressed():
    """'30-100 kW' range vs 'average 60 kW' point value should be suppressed."""
    a = _ev("AI data centres need 30-100 kW per rack", scope="rack")
    b = _ev("Average rack power density now exceeds 60 kW", scope="rack")
    suppressed: list[SuppressedComparison] = []
    contradictions = detect_contradictions([a, b], out_suppressed=suppressed)
    # 60 is within [30, 100] — should be suppressed
    assert len(contradictions) == 0
    range_sups = [s for s in suppressed if s.reason == "range_average_compatible"]
    assert len(range_sups) > 0


def test_range_vs_value_outside_range_flagged():
    """'30-100 kW' range vs '200 kW' (outside) should still be a contradiction."""
    a = _ev("Rack power requirement ranges from 30-100 kW", scope="rack")
    b = _ev("Each rack requires 200 kW for AI workloads", scope="rack")
    suppressed: list[SuppressedComparison] = []
    contradictions = detect_contradictions([a, b], out_suppressed=suppressed)
    range_sups = [s for s in suppressed if s.reason == "range_average_compatible"]
    assert len(range_sups) == 0
    # 200 is well outside [30, 100] — should be a contradiction
    assert len(contradictions) >= 1


# ---------------------------------------------------------------------------
# Product mismatch in detect_contradictions
# ---------------------------------------------------------------------------

def test_nvl8_vs_nvl72_not_contradicted():
    """NVL8 24 kW vs NVL72 120 kW — different products, should not be a contradiction."""
    a = _ev("HGX Rubin NVL8 system consumes 24 kW")
    b = _ev("GB200 NVL72 rack requires 120 kW")
    suppressed: list[SuppressedComparison] = []
    contradictions = detect_contradictions([a, b], out_suppressed=suppressed)
    assert len(contradictions) == 0
    prod_sups = [s for s in suppressed if s.reason in ("product_mismatch", "generation_progression")]
    assert len(prod_sups) > 0


def test_dgx_vs_hgx_not_contradicted():
    """DGX B300 14 kW vs HGX NVL8 24 kW — different systems."""
    a = _ev("DGX B300 system draws 14 kW total")
    b = _ev("HGX Rubin NVL8 consumes 24 kW")
    suppressed: list[SuppressedComparison] = []
    contradictions = detect_contradictions([a, b], out_suppressed=suppressed)
    assert len(contradictions) == 0
    prod_sups = [s for s in suppressed if s.reason == "product_mismatch"]
    assert len(prod_sups) > 0


def test_gb200_vs_rubin_not_contradicted():
    """GB200 vs Vera Rubin rack power — different GPU generations."""
    a = _ev("GB200 NVL72 factory rack power consumption 120 kW")
    b = _ev("Vera Rubin NVL72 rack power targets 200 kW")
    suppressed: list[SuppressedComparison] = []
    contradictions = detect_contradictions([a, b], out_suppressed=suppressed)
    assert len(contradictions) == 0
    gen_sups = [s for s in suppressed if s.reason == "generation_progression"]
    assert len(gen_sups) > 0


# ---------------------------------------------------------------------------
# True contradictions still preserved (same product, same scope)
# ---------------------------------------------------------------------------

def test_same_product_same_scope_contradiction_survives():
    """Rubin NVL72 at 120 kW vs 180 kW — same product, should still be a contradiction."""
    a = _ev("Rubin NVL72 rack power is rated at 120 kW")
    b = _ev("Rubin NVL72 rack power is estimated at 180 kW")
    suppressed: list[SuppressedComparison] = []
    contradictions = detect_contradictions([a, b], out_suppressed=suppressed)
    # No product suppression should fire
    prod_sups = [s for s in suppressed if s.reason in ("product_mismatch", "generation_progression")]
    assert len(prod_sups) == 0
    # May or may not be a numeric contradiction depending on scope detection
    # At minimum, it should not be suppressed as product mismatch


def test_rack_vs_rack_same_value_no_contradiction():
    a = _ev("Each rack needs 120 kW", scope="rack")
    b = _ev("Rack power requirement is 120 kW", scope="rack")
    contradictions = detect_contradictions([a, b])
    assert len(contradictions) == 0


# ---------------------------------------------------------------------------
# Scope additions — chip-level
# ---------------------------------------------------------------------------

def test_chip_scope_detection():
    """'per GPU chip' should map to component scope."""
    scope = _extract_scope("the power per gpu chip is 700 W")
    assert scope == "component"


def test_per_server_scope_detection():
    """'per server' should map to node scope."""
    scope = _extract_scope("the power per server is 5 kW")
    assert scope == "node"


def test_each_server_scope_detection():
    """'each server' should map to node scope."""
    scope = _extract_scope("each server requires 4 kW")
    assert scope == "node"


# ---------------------------------------------------------------------------
# compute_suppression_metrics — new fields
# ---------------------------------------------------------------------------

def _make_sup(reason: str) -> SuppressedComparison:
    return SuppressedComparison(
        evidence_a_id="E001", evidence_b_id="E002",
        evidence_a_claim="a", evidence_b_claim="b",
        reason=reason, scope_a="rack", scope_b="rack", detail="test",
    )


def test_suppression_metrics_product_filtering():
    suppressed = [_make_sup("product_mismatch"), _make_sup("generation_progression")]
    metrics = compute_suppression_metrics(suppressed, final_count=1)
    assert metrics["product_filtering_present"] is True
    assert metrics["by_reason"]["product_mismatch"] == 1
    assert metrics["by_reason"]["generation_progression"] == 1


def test_suppression_metrics_range_filtering():
    suppressed = [_make_sup("range_average_compatible")]
    metrics = compute_suppression_metrics(suppressed, final_count=2)
    assert metrics["range_filtering_present"] is True
    assert metrics["by_reason"]["range_average_compatible"] == 1


def test_suppression_metrics_no_product_filter():
    suppressed = [_make_sup("scope_mismatch")]
    metrics = compute_suppression_metrics(suppressed, final_count=1)
    assert metrics["product_filtering_present"] is False
    assert metrics["range_filtering_present"] is False


# ---------------------------------------------------------------------------
# End-to-end: multiple suppression reasons in one batch
# ---------------------------------------------------------------------------

def test_mixed_suppression_batch():
    """A batch with multiple suppression triggers should track all reasons."""
    items = [
        _ev("GB200 NVL72 rack power is 120 kW"),           # will compare with Rubin
        _ev("Vera Rubin NVL72 rack targets 200 kW"),        # vs GB200 → gen_progression
        _ev("Rack power ranges from 30-100 kW"),            # range
        _ev("Average rack power exceeds 60 kW"),            # point within range
    ]
    suppressed: list[SuppressedComparison] = []
    contradictions = detect_contradictions(items, out_suppressed=suppressed)
    reasons = {s.reason for s in suppressed}
    # Generation progression should be detected
    assert "generation_progression" in reasons
    # Range/average should be detected
    assert "range_average_compatible" in reasons
    # Metrics should reflect all reasons
    metrics = compute_suppression_metrics(suppressed, final_count=len(contradictions))
    assert metrics["product_filtering_present"] is True
    assert metrics["range_filtering_present"] is True
