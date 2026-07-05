"""J11.3 — Investigation Area Coverage & Evidence Mapping Quality.

Root cause: _map_evidence_to_areas used only `category + topics` as signal text.
In the KB path, topics is always []. Investigation areas are multi-word phrases
that don't appear in short category strings. Only one area with a matching
category keyword ("Power Availability") got mapped; all others were empty.

Fix: include claim text in signal_texts, mirroring _map_evidence_to_subquestions.

These tests directly verify:
  - KB path items (topics=[]) now map via claim text
  - Category-based mapping regression (still works)
  - No false positives for unrelated claims
  - One-to-many mapping (one item can cover multiple areas)
  - Legacy path (topics populated) regression
"""

from __future__ import annotations

import types

import pytest

from functional_agents.evidence_agent import _map_evidence_to_areas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _proxy(claim: str, category: str = "other", topics: list[str] | None = None, eid: str | None = None) -> object:
    """Build a SimpleNamespace proxy matching what _execute_kb produces."""
    ns = types.SimpleNamespace()
    ns.claim = claim
    ns.evidence_snippet = claim
    ns.evidence_id = eid or f"EV-{hash(claim) % 100000:05d}"
    ns.category = category
    ns.topics = topics if topics is not None else []
    return ns


# The investigation areas from a real us_data_center_siting_strategy run
_AREAS = [
    "Interconnection Queue Depth and Backlog by RTO/ISO",
    "Large-Load Study Timelines and Fast-Track Pathways",
    "Transmission Congestion Risk and Constrained Corridors",
    "Utility Willingness-to-Serve and Large-Load Tariff Terms",
    "Behind-the-Meter and Co-Located Generation Viability",
    "Energization Timeline Feasibility for 100+ MW by 2028",
    "Queue Withdrawal Rates and Study Milestone Completion",
    "Power Availability and Grid Capacity by Candidate State",
]


# ---------------------------------------------------------------------------
# TestKBPathMapping — root-cause scenario
# ---------------------------------------------------------------------------

class TestKBPathMapping:
    """KB path: topics=[], only claim text available for area mapping."""

    def test_interconnection_claim_maps_to_interconnection_area(self):
        item = _proxy(
            "PJM interconnection queue contains over 2600 active requests creating "
            "multi-year study backlogs for new large load applicants.",
            category="other",
            topics=[],
        )
        mapping = _map_evidence_to_areas([item], _AREAS)
        assert item.evidence_id in mapping["Interconnection Queue Depth and Backlog by RTO/ISO"]

    def test_transmission_congestion_claim_maps_to_congestion_area(self):
        item = _proxy(
            "Transmission congestion in PJM results in annual costs exceeding "
            "3 billion dollars along constrained corridors in the Mid-Atlantic.",
            category="other",
            topics=[],
        )
        mapping = _map_evidence_to_areas([item], _AREAS)
        assert item.evidence_id in mapping["Transmission Congestion Risk and Constrained Corridors"]

    def test_utility_willingness_claim_maps_to_utility_area(self):
        item = _proxy(
            "Utility willingness-to-serve letters are required for large hyperscale "
            "loads over 100 MW before the interconnection study process begins.",
            category="other",
            topics=[],
        )
        mapping = _map_evidence_to_areas([item], _AREAS)
        assert item.evidence_id in mapping["Utility Willingness-to-Serve and Large-Load Tariff Terms"]

    def test_queue_withdrawal_claim_maps_to_queue_withdrawal_area(self):
        item = _proxy(
            "Queue withdrawal rates in MISO exceeded 70 percent of submitted projects, "
            "indicating study milestone completion is a key project risk.",
            category="other",
            topics=[],
        )
        mapping = _map_evidence_to_areas([item], _AREAS)
        assert item.evidence_id in mapping["Queue Withdrawal Rates and Study Milestone Completion"]

    def test_energization_timeline_claim_maps_correctly(self):
        item = _proxy(
            "Energization timelines for 100 MW facilities range from 3 to 7 years "
            "depending on RTO and proximity to existing substations.",
            category="other",
            topics=[],
        )
        mapping = _map_evidence_to_areas([item], _AREAS)
        assert item.evidence_id in mapping["Energization Timeline Feasibility for 100+ MW by 2028"]

    def test_behind_meter_generation_claim_maps_correctly(self):
        item = _proxy(
            "Behind-the-meter generation via co-located gas peaker plants avoids "
            "the interconnection queue entirely and provides firm capacity.",
            category="other",
            topics=[],
        )
        mapping = _map_evidence_to_areas([item], _AREAS)
        assert item.evidence_id in mapping["Behind-the-Meter and Co-Located Generation Viability"]

    def test_all_8_areas_covered_with_realistic_claims(self):
        """Each investigation area gets at least one mapped item with realistic claims."""
        items = [
            _proxy("PJM interconnection queue depth exceeds 2600 active requests.", category="other"),
            _proxy("ERCOT fast-track large load study timelines for interconnection.", category="power"),
            _proxy("Transmission congestion along constrained corridors in PJM.", category="other"),
            _proxy("Utility willingness-to-serve commitments required for large loads.", category="other"),
            _proxy("Behind-the-meter generation viability for co-located facilities.", category="other"),
            _proxy("Energization timeline feasibility for 100 MW facilities by 2028.", category="other"),
            _proxy("Queue withdrawal rates and study milestone completion risk in MISO.", category="other"),
            _proxy("Power availability and grid capacity by candidate state Texas.", category="power"),
        ]
        mapping = _map_evidence_to_areas(items, _AREAS)
        covered = sum(1 for ids in mapping.values() if ids)
        assert covered == 8, (
            f"Expected all 8 areas covered; got {covered}. Unmapped: "
            + ", ".join(a for a, ids in mapping.items() if not ids)
        )


# ---------------------------------------------------------------------------
# TestNoFalsePositives — unrelated claims stay unmapped
# ---------------------------------------------------------------------------

class TestNoFalsePositives:

    def test_unrelated_claim_not_mapped_to_any_area(self):
        """A claim about nuclear reactor design has no overlap with siting areas."""
        item = _proxy(
            "BWR-X300 reactor design uses passive safety systems requiring no operator action.",
            category="reactor design",
        )
        mapping = _map_evidence_to_areas([item], _AREAS)
        mapped = [area for area, ids in mapping.items() if item.evidence_id in ids]
        assert mapped == [], f"Unexpected false-positive mappings: {mapped}"

    def test_empty_claim_not_mapped(self):
        item = _proxy("", category="other")
        mapping = _map_evidence_to_areas([item], _AREAS)
        mapped = [area for area, ids in mapping.items() if item.evidence_id in ids]
        assert mapped == []

    def test_items_with_no_token_overlap_produce_empty_areas(self):
        items = [_proxy("Cooling tower water consumption rate.", category="cooling")]
        areas = ["Grid Integration Pathway", "Interconnection Queue Depth"]
        mapping = _map_evidence_to_areas(items, areas)
        # cooling tower / water consumption has no token overlap with grid/interconnection
        assert mapping["Grid Integration Pathway"] == []


# ---------------------------------------------------------------------------
# TestCategoryRegressions — category-based mapping still works
# ---------------------------------------------------------------------------

class TestCategoryRegressions:

    def test_power_category_still_maps_to_power_area(self):
        """Power category → Power area via _CATEGORY_TO_AREA fallback still works."""
        item = _proxy("Substation upgrade required.", category="power", topics=[])
        mapping = _map_evidence_to_areas([item], ["Power"])
        assert item.evidence_id in mapping["Power"]

    def test_economics_category_maps_to_economics_area(self):
        item = _proxy("Cost analysis shows 50M capital requirement.", category="economics", topics=[])
        mapping = _map_evidence_to_areas([item], ["Economics"])
        assert item.evidence_id in mapping["Economics"]

    def test_category_token_overlap_works_without_claim(self):
        """Even when claim is empty, category overlap still maps correctly."""
        item = _proxy("", category="power", topics=[])
        # "Power Availability" contains the token "power"
        mapping = _map_evidence_to_areas([item], ["Power Availability"])
        assert item.evidence_id in mapping["Power Availability"]


# ---------------------------------------------------------------------------
# TestOneToManyMapping — a claim can appear in multiple areas
# ---------------------------------------------------------------------------

class TestOneToManyMapping:

    def test_interconnection_claim_appears_in_multiple_areas(self):
        """An interconnection claim is legitimately relevant to multiple areas."""
        item = _proxy(
            "PJM interconnection queue large load study timelines average 4 years.",
            category="other",
            topics=[],
        )
        mapping = _map_evidence_to_areas([item], _AREAS)
        # Must appear in both interconnection queue AND large-load study timelines
        assert item.evidence_id in mapping["Interconnection Queue Depth and Backlog by RTO/ISO"]
        assert item.evidence_id in mapping["Large-Load Study Timelines and Fast-Track Pathways"]

    def test_multiple_items_each_counted_independently(self):
        items = [
            _proxy("Interconnection queue depth issue in PJM.", category="other", eid="EV-001"),
            _proxy("Large-load study timelines are too long.", category="other", eid="EV-002"),
        ]
        mapping = _map_evidence_to_areas(items, _AREAS)
        assert "EV-001" in mapping["Interconnection Queue Depth and Backlog by RTO/ISO"]
        assert "EV-002" in mapping["Large-Load Study Timelines and Fast-Track Pathways"]


# ---------------------------------------------------------------------------
# TestLegacyPathRegression — topics-populated path unchanged
# ---------------------------------------------------------------------------

class TestLegacyPathRegression:

    def test_topics_populated_still_maps_correctly(self):
        """Legacy path items with topics populated continue to map."""
        item = _proxy(
            "Some unrelated claim text.",
            category="power",
            topics=["interconnection", "queue"],
        )
        # Topics contain "interconnection" — should map to interconnection area
        mapping = _map_evidence_to_areas([item], _AREAS)
        assert item.evidence_id in mapping["Interconnection Queue Depth and Backlog by RTO/ISO"]

    def test_empty_items_list_returns_empty_mapping(self):
        mapping = _map_evidence_to_areas([], _AREAS)
        for ids in mapping.values():
            assert ids == []

    def test_empty_areas_list_returns_empty_mapping(self):
        item = _proxy("Grid interconnection queue delay.", category="other")
        mapping = _map_evidence_to_areas([item], [])
        assert mapping == {}


# ---------------------------------------------------------------------------
# TestBeforeAfterQuantitative — the reported production numbers
# ---------------------------------------------------------------------------

class TestBeforeAfterQuantitative:
    """Verify the before/after improvement on the reported production scenario."""

    _REALISTIC_ITEMS = [
        _proxy("Interconnection queue depth in PJM exceeds 2600 requests.", "other", eid="EV-001"),
        _proxy("Large-load fast-track study timelines require pre-application review.", "other", eid="EV-002"),
        _proxy("Transmission congestion along constrained corridors in PJM.", "other", eid="EV-003"),
        _proxy("Utility willingness-to-serve letters required for large loads.", "other", eid="EV-004"),
        _proxy("Behind-the-meter co-located generation avoids queue delays.", "other", eid="EV-005"),
        _proxy("Energization timelines for 100 MW facilities can exceed 5 years.", "other", eid="EV-006"),
        _proxy("Queue withdrawal rates in MISO indicate high study risk.", "other", eid="EV-007"),
        _proxy("Power availability by state: Texas ERCOT has near-term capacity.", "power", eid="EV-008"),
        # Additional evidence for subquestion coverage (also gets area-mapped)
        _proxy("RTO interconnection backlog growing due to large load applications.", "other", eid="EV-009"),
        _proxy("Fast-track pathway allows 100 MW facilities to bypass standard queue.", "power", eid="EV-010"),
    ]

    def test_all_8_areas_now_covered(self):
        mapping = _map_evidence_to_areas(self._REALISTIC_ITEMS, _AREAS)
        covered = sum(1 for ids in mapping.values() if ids)
        assert covered == 8, (
            f"Expected 8/8 areas covered; got {covered}/8. "
            "Unmapped: " + ", ".join(a for a, ids in mapping.items() if not ids)
        )

    def test_power_area_still_gets_items(self):
        mapping = _map_evidence_to_areas(self._REALISTIC_ITEMS, _AREAS)
        power_area = "Power Availability and Grid Capacity by Candidate State"
        assert len(mapping[power_area]) >= 1
