"""Tests for J6.5c Contradiction Decision Logic (Eligibility Engine).

Covers:
- _classify_comparison_context() returning correct labels
- _INCOMPATIBLE_CONTEXT_PAIRS driving Gate 6.5 suppression
- example_deployment vs reference_architecture suppressed
- example_deployment vs current_deployment suppressed
- vendor_claim vs industry_average suppressed
- current_deployment vs future_projection backstop
- same-context same-scope true contradictions preserved
- compute_suppression_metrics includes context_filtering_present + eligibility_engine
- QA contradiction_decision_validation block populated
- ChallengeAgent uses validated_contradictions (not raw contradictions)
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

sys.modules.setdefault("yaml", MagicMock())

from research_agent.contradiction import (
    _classify_comparison_context,
    _INCOMPATIBLE_CONTEXT_PAIRS,
    detect_contradictions,
    compute_suppression_metrics,
)
from research_agent.schemas import EvidenceItem, SuppressedComparison


# ---------------------------------------------------------------------------
# Helpers
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


def _sup(reason: str) -> SuppressedComparison:
    return SuppressedComparison(
        evidence_a_id="E001", evidence_b_id="E002",
        evidence_a_claim="a", evidence_b_claim="b",
        reason=reason, scope_a="rack", scope_b="rack", detail="test",
    )


# ---------------------------------------------------------------------------
# _classify_comparison_context — label accuracy
# ---------------------------------------------------------------------------

def test_ctx_example_deployment_for_example():
    ctx = _classify_comparison_context("For example, a 5 MW deployment could serve 500 racks.")
    assert ctx == "example_deployment"


def test_ctx_example_deployment_eg():
    ctx = _classify_comparison_context("e.g. an illustrative 10 MW data center")
    assert ctx == "example_deployment"


def test_ctx_example_deployment_hypothetical():
    ctx = _classify_comparison_context("A hypothetical 100 kW rack scenario")
    assert ctx == "example_deployment"


def test_ctx_reference_architecture():
    ctx = _classify_comparison_context("A typical rack configuration draws 30 kW.")
    assert ctx == "reference_architecture"


def test_ctx_reference_architecture_standard():
    ctx = _classify_comparison_context("The standard configuration calls for 120 kW per rack.")
    assert ctx == "reference_architecture"


def test_ctx_industry_average():
    ctx = _classify_comparison_context("Industry average rack power has risen to 60 kW.")
    assert ctx == "industry_average"


def test_ctx_industry_average_typically():
    ctx = _classify_comparison_context("Data centers typically consume 40-60 kW per rack.")
    assert ctx == "industry_average"


def test_ctx_vendor_claim_data_sheet():
    ctx = _classify_comparison_context("NVIDIA's data sheet shows 120 kW for NVL72.")
    assert ctx == "vendor_claim"


def test_ctx_future_projection():
    ctx = _classify_comparison_context("Projected rack power will reach 250 kW by 2030.")
    assert ctx == "future_projection"


def test_ctx_current_deployment():
    ctx = _classify_comparison_context("Current data centers are operating at 40 kW per rack.")
    assert ctx == "current_deployment"


def test_ctx_unknown():
    ctx = _classify_comparison_context("Rack power is 120 kW.")
    assert ctx == "unknown"


# ---------------------------------------------------------------------------
# _INCOMPATIBLE_CONTEXT_PAIRS membership
# ---------------------------------------------------------------------------

def test_example_vs_reference_incompatible():
    assert frozenset({"example_deployment", "reference_architecture"}) in _INCOMPATIBLE_CONTEXT_PAIRS


def test_example_vs_current_incompatible():
    assert frozenset({"example_deployment", "current_deployment"}) in _INCOMPATIBLE_CONTEXT_PAIRS


def test_example_vs_future_incompatible():
    assert frozenset({"example_deployment", "future_projection"}) in _INCOMPATIBLE_CONTEXT_PAIRS


def test_vendor_vs_industry_incompatible():
    assert frozenset({"vendor_claim", "industry_average"}) in _INCOMPATIBLE_CONTEXT_PAIRS


def test_current_vs_future_incompatible():
    assert frozenset({"current_deployment", "future_projection"}) in _INCOMPATIBLE_CONTEXT_PAIRS


def test_example_vs_unknown_not_incompatible():
    # unknown on one side → gate doesn't fire
    assert frozenset({"example_deployment", "unknown"}) not in _INCOMPATIBLE_CONTEXT_PAIRS


# ---------------------------------------------------------------------------
# detect_contradictions — context suppression
# ---------------------------------------------------------------------------

def test_example_vs_reference_arch_suppressed():
    """'For example, 5 MW' vs '100 MW AI factory' should be context-suppressed."""
    a = _ev("For example, a 5 MW deployment could serve a small campus.", scope="facility")
    b = _ev("AI factory deployments require at least 100 MW on campus.", scope="facility")
    suppressed: list[SuppressedComparison] = []
    contradictions = detect_contradictions([a, b], out_suppressed=suppressed)
    assert len(contradictions) == 0
    ctx_sups = [s for s in suppressed if s.reason == "context_mismatch"]
    assert len(ctx_sups) > 0


def test_example_vs_current_deployment_suppressed():
    """Hypothetical rack vs current rack — context mismatch, not contradiction."""
    a = _ev("A hypothetical rack could draw 40 kW in this example.")
    b = _ev("Current rack installations are operating at 120 kW today.", scope="rack")
    suppressed: list[SuppressedComparison] = []
    contradictions = detect_contradictions([a, b], out_suppressed=suppressed)
    assert len(contradictions) == 0
    ctx_sups = [s for s in suppressed if s.reason == "context_mismatch"]
    assert len(ctx_sups) > 0


def test_vendor_vs_industry_average_suppressed():
    """NVIDIA data sheet spec vs industry average — not a contradiction."""
    a = _ev("NVIDIA's data sheet shows 120 kW for the NVL72 rack.", scope="rack")
    b = _ev("Industry average rack power has risen to 60 kW.", scope="rack")
    suppressed: list[SuppressedComparison] = []
    contradictions = detect_contradictions([a, b], out_suppressed=suppressed)
    assert len(contradictions) == 0
    ctx_sups = [s for s in suppressed if s.reason == "context_mismatch"]
    assert len(ctx_sups) > 0


def test_unknown_context_both_proceeds_to_value_check():
    """When both contexts are unknown, the context gate doesn't fire."""
    a = _ev("Rack power is 40 kW.", scope="rack")
    b = _ev("Rack power is 200 kW.", scope="rack")
    suppressed: list[SuppressedComparison] = []
    contradictions = detect_contradictions([a, b], out_suppressed=suppressed)
    ctx_sups = [s for s in suppressed if s.reason == "context_mismatch"]
    assert len(ctx_sups) == 0
    # 40 vs 200 is an 80% difference — should be a real contradiction
    assert len(contradictions) >= 1


def test_same_context_true_contradiction_survives():
    """Two current-deployment claims with same scope and conflicting values survive."""
    a = _ev("Current data centers operate at 40 kW per rack today.", scope="rack")
    b = _ev("Current facilities are now running at 150 kW per rack today.", scope="rack")
    suppressed: list[SuppressedComparison] = []
    contradictions = detect_contradictions([a, b], out_suppressed=suppressed)
    ctx_sups = [s for s in suppressed if s.reason == "context_mismatch"]
    assert len(ctx_sups) == 0
    # 40 vs 150 is a 73% difference — should be a contradiction (not suppressed by context)


def test_example_vs_example_suppressed():
    """Two example claims are both suppressed — illustrative figures never contradict."""
    a = _ev("For example, a rack might draw 40 kW.", scope="rack")
    b = _ev("For example, a rack might draw 200 kW.", scope="rack")
    suppressed: list[SuppressedComparison] = []
    contradictions = detect_contradictions([a, b], out_suppressed=suppressed)
    # example_deployment fires unconditionally — even example vs example is suppressed
    ctx_sups = [s for s in suppressed if s.reason == "context_mismatch"]
    assert len(ctx_sups) > 0
    assert len(contradictions) == 0


# ---------------------------------------------------------------------------
# compute_suppression_metrics — new fields
# ---------------------------------------------------------------------------

def test_metrics_context_filtering_present():
    suppressed = [_sup("context_mismatch")]
    metrics = compute_suppression_metrics(suppressed, final_count=1)
    assert metrics["context_filtering_present"] is True


def test_metrics_context_filtering_absent():
    suppressed = [_sup("scope_mismatch")]
    metrics = compute_suppression_metrics(suppressed, final_count=1)
    assert metrics["context_filtering_present"] is False


def test_metrics_eligibility_engine_block():
    suppressed = [_sup("context_mismatch"), _sup("scope_mismatch")]
    metrics = compute_suppression_metrics(suppressed, final_count=3)
    ee = metrics["eligibility_engine"]
    assert ee["candidate_pairs"] == 5   # 3 final + 2 suppressed
    assert ee["eligible_pairs"] == 3
    assert ee["suppressed_pairs"] == 2


def test_metrics_eligibility_engine_zero_suppressed():
    metrics = compute_suppression_metrics([], final_count=4)
    ee = metrics["eligibility_engine"]
    assert ee["candidate_pairs"] == 4
    assert ee["eligible_pairs"] == 4
    assert ee["suppressed_pairs"] == 0


# ---------------------------------------------------------------------------
# QA contradiction_decision_validation (unit-level)
# ---------------------------------------------------------------------------

def test_qa_contradiction_decision_validation_populated():
    from functional_agents.qa_agent import _validate_contradiction_decision_logic
    from functional_agents.context import AgentContext

    ctx = AgentContext(goal="test")
    ctx.contradiction_metrics = {
        "candidate_count": 10,
        "suppressed_count": 7,
        "final_count": 3,
        "by_reason": {"context_mismatch": 2, "scope_mismatch": 5},
        "scope_filtering_present": True,
        "entity_filtering_present": False,
        "context_filtering_present": True,
        "eligibility_engine": {
            "candidate_pairs": 10,
            "eligible_pairs": 3,
            "suppressed_pairs": 7,
        },
    }
    result = _validate_contradiction_decision_logic(ctx)
    assert result["eligibility_engine_present"] is True
    assert result["scope_filtering_present"] is True
    assert result["context_filtering_present"] is True
    assert result["candidate_pairs"] == 10
    assert result["suppressed_pairs"] == 7
    assert result["eligible_pairs"] == 3
    assert result["issues"] == []


def test_qa_contradiction_decision_validation_missing_metrics():
    from functional_agents.qa_agent import _validate_contradiction_decision_logic
    from functional_agents.context import AgentContext

    ctx = AgentContext(goal="test")
    # No contradiction_metrics set
    result = _validate_contradiction_decision_logic(ctx)
    assert result["eligibility_engine_present"] is False
    assert "eligibility_engine not present" in result["issues"][0]


# ---------------------------------------------------------------------------
# ChallengeAgent uses validated_contradictions
# ---------------------------------------------------------------------------

def test_challenge_agent_prefers_validated_contradictions():
    """ChallengeAgent should consume validated_contradictions, not raw contradictions."""
    from functional_agents.challenge_agent import ChallengeAgent
    from functional_agents.context import AgentContext

    validated = [{"contradiction_id": "C001", "topic": "power", "severity": "high"}]
    raw = [
        {"contradiction_id": "C001", "topic": "power", "severity": "high"},
        {"contradiction_id": "SUPPRESSED_C002", "topic": "scope mismatch", "severity": "high"},
    ]

    ctx = AgentContext(goal="test power density")
    ctx.hypotheses = [{"hypothesis_id": "H001", "statement": "Racks need 120 kW"}]
    ctx.evidence_notes = [{"evidence_items": []}]
    ctx.research_object = {
        "contradictions": raw,
        "validated_contradictions": validated,
        "gaps": [],
    }
    ctx.validated_contradictions = validated

    captured: list[list[dict]] = []

    class CapturingChallengeAgent(ChallengeAgent):
        def _generate_challenges(self, hypotheses, evidence_items, contradictions, gaps, profile_coverage):
            captured.append(contradictions)
            # Return a minimal valid payload
            from types import SimpleNamespace
            mock_payload = SimpleNamespace(
                hypothesis_challenges=[], surviving_hypotheses=[], challenge_synthesis=""
            )
            return mock_payload

    agent = CapturingChallengeAgent()
    agent._execute(ctx)

    assert len(captured) == 1
    # Should have used validated_contradictions (1 item), not raw (2 items)
    assert captured[0] == validated
    assert len(captured[0]) == 1
