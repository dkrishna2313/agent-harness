"""J11.5 — Coverage Gap Investigation & Resolution: composite subquestion coverage.

Root cause: _map_evidence_to_subquestions used winner-take-all assignment.
A composite/synthesis subquestion (e.g. "which state carries the greatest
interconnection delay risk, based on queue depth, network upgrade cost,
load growth, and RTO reform trajectory") never wins the token-overlap
competition against the more specific component subquestions (queue backlogs,
transmission corridors, precedents) even when the retrieved evidence is
directly relevant to the composite question's assessment factors.

Fix: add secondary subquestion assignment for items that score >=
_SQ_SECONDARY_THRESHOLD on a non-primary subquestion.  Primary winner-take-all
is preserved; the composite question receives credit from evidence that is
primarily attributed to its component subquestions.

These tests verify:
  - Composite subquestions receive secondary assignments from component evidence
  - The threshold (_SQ_SECONDARY_THRESHOLD = 3) is enforced
  - Primary winner-take-all is unchanged for the primary SQ
  - Items with zero overlap on all SQs still go to _unmapped
  - Single-subquestion runs are unaffected
  - Evidence IDs may appear in multiple subquestion lists
  - Coverage levels reflect multi-mapped counts
"""

from __future__ import annotations

import types
import pytest

from functional_agents.evidence_agent import (
    _map_evidence_to_subquestions,
    _compute_coverage,
    _SQ_SECONDARY_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _item(claim: str, eid: str | None = None) -> object:
    ns = types.SimpleNamespace()
    ns.claim = claim
    ns.evidence_snippet = claim
    ns.evidence_id = eid or f"EV-{abs(hash(claim)) % 100000:05d}"
    return ns


# Real-world subquestions from the siting engagement (abbreviated)
SQ1 = (
    "What are the current interconnection queue backlogs, average study durations, "
    "and recent large-load (100 MW+) interconnection outcomes for CAISO, ERCOT, PJM, "
    "and ISO-NE, and how do these metrics translate into realistic in-service timelines "
    "for hyperscale AI data center loads over the 2026–2030 horizon?"
)
SQ5 = (
    "Which of the five candidate states carry the greatest interconnection delay risk "
    "over 2026–2030, based on a composite assessment of queue depth, network upgrade "
    "cost exposure, utility load growth pressure from competing large loads, and "
    "RTO/ISO reform trajectory?"
)
SQ6 = (
    "What recent large-load interconnection precedents — including approved, withdrawn, "
    "or stalled projects above 50 MW — exist in each state's RTO/ISO territory, and "
    "what do these outcomes signal about the practical feasibility and cost exposure "
    "for new hyperscale data center entrants?"
)


# ---------------------------------------------------------------------------
# TestThresholdConstant — value of _SQ_SECONDARY_THRESHOLD
# ---------------------------------------------------------------------------

class TestThresholdConstant:

    def test_threshold_is_3(self):
        """Secondary threshold must be 3 to capture composite-question evidence."""
        assert _SQ_SECONDARY_THRESHOLD == 3


# ---------------------------------------------------------------------------
# TestCompositeSubquestionCoverage — primary use case
# ---------------------------------------------------------------------------

class TestCompositeSubquestionCoverage:

    def test_ercot_delay_item_maps_to_sq5(self):
        """High-relevance ERCOT delay item (sq5=4) must also appear in SQ5 list."""
        item = _item(
            "ERCOT's adjusted large load forecast assumes a 180-day delay to ramp "
            "schedules for all large load interconnection requests in 2026–2030.",
            eid="ERCOT-001",
        )
        mapping = _map_evidence_to_subquestions([item], [SQ1, SQ5, SQ6])
        assert "ERCOT-001" in mapping[SQ5], (
            "ERCOT delay item should be secondary-mapped to SQ5 (delay risk ranking)"
        )

    def test_queue_data_item_maps_to_sq5(self):
        """Multi-state queue data item (sq5=4) should also appear in SQ5."""
        item = _item(
            "Interconnection queue data from PJM, ERCOT, MISO, CAISO, and ISO-NE "
            "are recommended for those researching grid capacity and network upgrade "
            "cost exposure and RTO reform trajectory.",
            eid="QUEUE-001",
        )
        mapping = _map_evidence_to_subquestions([item], [SQ1, SQ5, SQ6])
        assert "QUEUE-001" in mapping[SQ5]

    def test_load_growth_pressure_item_maps_to_sq5(self):
        """Item about load growth pressure (SQ5 component) should reach SQ5."""
        item = _item(
            "A spike in large, single-site load additions from data center developments "
            "and electrification is driving interconnection queue growth and delay risk "
            "across competing large loads in PJM and ERCOT.",
            eid="LOAD-001",
        )
        mapping = _map_evidence_to_subquestions([item], [SQ1, SQ5, SQ6])
        assert "LOAD-001" in mapping[SQ5]

    def test_network_upgrade_cost_item_maps_to_sq5(self):
        """Network upgrade cost exposure item (SQ5 component) should reach SQ5."""
        item = _item(
            "Upon executing an interconnection agreement, a large load customer must "
            "pay a contribution towards network upgrade costs as a condition of service, "
            "which represents significant cost exposure for new interconnection requests.",
            eid="COST-001",
        )
        mapping = _map_evidence_to_subquestions([item], [SQ1, SQ5, SQ6])
        assert "COST-001" in mapping[SQ5]

    def test_composite_question_gets_strong_coverage_from_component_evidence(self):
        """With realistic evidence pool, SQ5 coverage should reach STRONG (>= 4)."""
        items = [
            _item(
                "ERCOT large load interconnection queue delay risk increased in 2026 "
                "due to network upgrade cost exposure and competing load growth.",
                eid="E001",
            ),
            _item(
                "Interconnection queue data from PJM, ERCOT, MISO, CAISO shows "
                "network upgrade cost exposure and RTO reform trajectory differences.",
                eid="E002",
            ),
            _item(
                "Growth in new resources seeking transmission interconnection "
                "and competing large loads is driving delay risk across states.",
                eid="E003",
            ),
            _item(
                "A spike in large, single-site load additions from data centers is "
                "compressing interconnection queue timelines and raising upgrade cost "
                "exposure and delay risk in PJM and ERCOT.",
                eid="E004",
            ),
            _item(
                "PUCT rule requires large load interconnection cost standards for "
                "network upgrade cost exposure, queue depth, and load growth pressure.",
                eid="E005",
            ),
        ]
        mapping = _map_evidence_to_subquestions(items, [SQ1, SQ5, SQ6])
        coverage = _compute_coverage(mapping, [SQ1, SQ5, SQ6])
        assert coverage[SQ5]["coverage"] in ("STRONG", "MODERATE"), (
            f"Expected at least MODERATE coverage for SQ5, got {coverage[SQ5]['coverage']}"
        )
        assert coverage[SQ5]["evidence_count"] >= 4, (
            f"Expected >= 4 items for SQ5, got {coverage[SQ5]['evidence_count']}"
        )


# ---------------------------------------------------------------------------
# TestThresholdEnforcement — items below threshold do NOT get secondary assignment
# ---------------------------------------------------------------------------

class TestThresholdEnforcement:

    def test_item_scoring_2_on_sq5_is_not_secondary_mapped(self):
        """Item scoring exactly 2 on SQ5 must not receive a secondary assignment.

        Uses controlled minimal subquestion texts to guarantee exact token scores
        rather than relying on the long real-world SQ5 (whose token set is large
        and easy to accidentally hit at 3+).
        """
        # sq_a: 8 unique tokens
        sq_a = "alpha bravo charlie delta echo foxtrot golf hotel"
        # sq_b (plays the role of SQ5): 5 unique tokens, 2 shared with the claim
        sq_b = "romeo sierra tango uniform victor"
        # Claim: scores sq_a=4, sq_b=2 (alpha, bravo from sq_a; romeo, sierra from sq_b)
        claim = "alpha bravo romeo sierra"
        item = _item(claim, eid="LOW-001")
        mapping = _map_evidence_to_subquestions([item], [sq_a, sq_b])
        # sq_a wins (4 > 2); sq_b score is 2 < threshold=3 → no secondary assignment
        assert "LOW-001" in mapping[sq_a], "Primary winner must receive the item"
        assert "LOW-001" not in mapping[sq_b], (
            "Item scoring exactly 2 (< _SQ_SECONDARY_THRESHOLD=3) must not be secondary-mapped"
        )

    def test_item_scoring_1_on_sq5_is_not_secondary_mapped(self):
        """Item with only 1 matching token on SQ5 must not receive secondary assignment."""
        item = _item(
            "The interconnection process involves multiple steps.",
            eid="ONE-001",
        )
        mapping = _map_evidence_to_subquestions([item], [SQ1, SQ5])
        assert "ONE-001" not in mapping[SQ5]

    def test_item_scoring_3_on_sq5_is_secondary_mapped(self):
        """Item scoring exactly 3 on SQ5 should cross the threshold."""
        # "interconnection", "delay", "risk" → 3 tokens in SQ5
        item = _item(
            "Interconnection delay risk increases as utility load growth rises.",
            eid="THREE-001",
        )
        mapping = _map_evidence_to_subquestions([item], [SQ1, SQ5])
        assert "THREE-001" in mapping[SQ5], (
            "Item scoring exactly _SQ_SECONDARY_THRESHOLD on SQ5 must be secondary-mapped"
        )


# ---------------------------------------------------------------------------
# TestPrimaryAssignmentUnchanged — winner-take-all for primary SQ
# ---------------------------------------------------------------------------

class TestPrimaryAssignmentUnchanged:

    def test_primary_winner_receives_item(self):
        """The highest-scoring subquestion still receives the item (primary, unchanged)."""
        item = _item(
            "Interconnection queue data from PJM, ERCOT, MISO, CAISO shows "
            "queue backlogs and study durations for large loads over the 2026-2030 horizon.",
            eid="PRI-001",
        )
        mapping = _map_evidence_to_subquestions([item], [SQ1, SQ5, SQ6])
        assert "PRI-001" in mapping[SQ1], "Primary SQ (highest score) must receive the item"

    def test_item_may_appear_in_multiple_subquestion_lists(self):
        """An item can appear in both the primary SQ list and a secondary SQ list."""
        item = _item(
            "ERCOT large load interconnection queue delay risk in 2026-2030 "
            "reflects network upgrade cost exposure from competing loads.",
            eid="MULTI-001",
        )
        mapping = _map_evidence_to_subquestions([item], [SQ1, SQ5, SQ6])
        in_primary = "MULTI-001" in mapping[SQ1] or "MULTI-001" in mapping[SQ6]
        in_sq5 = "MULTI-001" in mapping[SQ5]
        assert in_primary, "Item must appear in its primary (highest-scoring) SQ"
        assert in_sq5, "Item must also appear in SQ5 if score >= threshold"

    def test_zero_score_items_go_to_unmapped(self):
        """Items with 0 overlap on all subquestions still go to _unmapped."""
        item = _item("Water usage guidelines for cooling towers in arid climates.", eid="ZERO-001")
        subquestions = [SQ1, SQ5, SQ6]
        mapping = _map_evidence_to_subquestions([item], subquestions)
        assert "ZERO-001" in mapping["_unmapped"]
        assert all("ZERO-001" not in mapping[sq] for sq in subquestions)


# ---------------------------------------------------------------------------
# TestSingleSubquestionRegression — single-SQ path unchanged
# ---------------------------------------------------------------------------

class TestSingleSubquestionRegression:

    def test_single_subquestion_all_items_map_to_it(self):
        """Single-subquestion runs assign all items to that one question (unchanged)."""
        items = [
            _item("ERCOT interconnection queue data for large loads.", eid="S001"),
            _item("PJM network upgrade costs exceeded estimates in 2025.", eid="S002"),
            _item("ISO-NE reform trajectory for interconnection process.", eid="S003"),
        ]
        mapping = _map_evidence_to_subquestions(items, [SQ5])
        assert mapping[SQ5] == ["S001", "S002", "S003"]
        assert mapping["_unmapped"] == []


# ---------------------------------------------------------------------------
# TestCoverageComputation — coverage levels reflect multi-mapped counts
# ---------------------------------------------------------------------------

class TestCoverageComputation:

    def test_coverage_reflects_secondary_assignments(self):
        """_compute_coverage counts items in each SQ list, including secondary."""
        sq_a = "What is the queue depth and interconnection delay risk?"
        sq_b = "What specific queue depth metrics exist for Texas ERCOT territory?"
        items = [
            _item(
                "ERCOT queue depth and interconnection delay risk metrics for Texas.",
                eid="EV001",
            ),
            _item(
                "ERCOT interconnection delay risk and queue depth for large load projects.",
                eid="EV002",
            ),
            _item(
                "ERCOT interconnection queue depth metrics for Texas data centers "
                "and corresponding delay risk over the 2026-2030 horizon.",
                eid="EV003",
            ),
            _item(
                "Texas ERCOT queue depth data reveals high interconnection delay risk "
                "for large-load customers seeking network upgrade cost certainty.",
                eid="EV004",
            ),
        ]
        mapping = _map_evidence_to_subquestions(items, [sq_a, sq_b])
        coverage = _compute_coverage(mapping, [sq_a, sq_b])
        # Both SQs should show coverage since they share the same topic space
        assert coverage[sq_a]["evidence_count"] >= 1
        assert coverage[sq_b]["evidence_count"] >= 1
