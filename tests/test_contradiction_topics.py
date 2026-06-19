"""Tests for J1.5: Profile-Aware Contradiction Topics.

Verifies that:
1. ``detect_contradictions(evidence, profile=<profile>)`` replaces hard-coded
   topic labels with profile-driven ones.
2. The ``topic_source`` field is set to ``"profile:<name>"`` on a match or
   ``"profile:<name>:fallback"`` when no keyword matched.
3. AI data center terms ("rack power", "cooling type", etc.) never appear in
   SMR-profile output, and vice versa.
4. When no profile is supplied the existing hard-coded topics are preserved.
"""

from __future__ import annotations

import pytest

from research_agent.contradiction import detect_contradictions
from research_agent.profile import DomainProfile, load_profile
from research_agent.schemas import EvidenceItem, assign_evidence_ids

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AI_DC_HARD_CODED_TOPICS = {"rack power", "cooling type", "cooling temperature",
                              "cooling phase", "gpu count", "rack type",
                              "numeric specification", "specification"}


def _ev(claim: str, source: str = "doc.pdf") -> EvidenceItem:
    return EvidenceItem(
        claim=claim,
        source_document=source,
        evidence_snippet=claim,
        category="other",
        relevance="Relevant.",
        confidence="high",
        relevance_score=4,
        source_quality_score=4,
        specificity_score=4,
        overall_score=4.0,
    )


@pytest.fixture(scope="module")
def smr_profile() -> DomainProfile:
    return load_profile("smr")


@pytest.fixture(scope="module")
def dc_profile() -> DomainProfile:
    return load_profile("ai_data_centers")


# ---------------------------------------------------------------------------
# 1. classify_contradiction_topic unit tests on the profile model
# ---------------------------------------------------------------------------


class TestClassifyContradictionTopic:

    def test_smr_construction_duration_matched(self, smr_profile):
        topic, source = smr_profile.classify_contradiction_topic(
            "Reactor Alpha construction duration is 24-36 months.",
            "Reactor Alpha construction duration is 8-12 years.",
        )
        assert topic == "construction"
        assert source == "profile:smr"

    def test_smr_licensing_matched(self, smr_profile):
        topic, source = smr_profile.classify_contradiction_topic(
            "The NRC design certification review takes 3 years.",
            "NRC approval for this design took only 18 months.",
        )
        assert topic == "licensing"
        assert source == "profile:smr"

    def test_smr_economics_matched(self, smr_profile):
        topic, source = smr_profile.classify_contradiction_topic(
            "The projected LCOE is $65/MWh for the first plant.",
            "Industry analysis estimates LCOE at $120/MWh.",
        )
        assert topic == "economics"
        assert source == "profile:smr"

    def test_smr_fuel_cycle_matched(self, smr_profile):
        topic, source = smr_profile.classify_contradiction_topic(
            "HALEU fuel is not commercially available from OECD suppliers.",
            "HALEU can be sourced from Rosatom.",
        )
        assert topic == "fuel_cycle"
        assert source == "profile:smr"

    def test_smr_grid_integration_matched(self, smr_profile):
        topic, source = smr_profile.classify_contradiction_topic(
            "Grid interconnection requires 200 MW transmission upgrade.",
            "No transmission upgrade needed for grid connection.",
        )
        assert topic == "grid_integration"
        assert source == "profile:smr"

    def test_smr_no_match_returns_other_fallback(self, smr_profile):
        topic, source = smr_profile.classify_contradiction_topic(
            "The reactor owner is a private company.",
            "The developer is a public utility.",
        )
        assert topic == "other"
        assert source == "profile:smr:fallback"

    def test_dc_power_matched(self, dc_profile):
        topic, source = dc_profile.classify_contradiction_topic(
            "Rack power draw is 120 kW per rack.",
            "Power consumption per rack is 180 kW.",
        )
        assert topic == "power"
        assert source == "profile:ai_data_centers"

    def test_dc_cooling_matched(self, dc_profile):
        topic, source = dc_profile.classify_contradiction_topic(
            "The system uses air cooling.",
            "The system requires liquid cooling.",
        )
        assert topic == "cooling"
        assert source == "profile:ai_data_centers"

    def test_dc_timeline_matched(self, dc_profile):
        # Use claims without rack-architecture terms so the timeline topic
        # wins before rack_architecture can match.
        topic, source = dc_profile.classify_contradiction_topic(
            "Product platform launches in 2026.",
            "Platform product shipping starts in 2027.",
        )
        assert topic == "timeline"
        assert source == "profile:ai_data_centers"


# ---------------------------------------------------------------------------
# 2. detect_contradictions with profile — end-to-end topic assignment
# ---------------------------------------------------------------------------


class TestDetectContradictionsWithProfile:

    # ---- SMR profile ---------------------------------------------------------

    def test_smr_reactor_alpha_topic_is_construction(self, smr_profile):
        """
        PRIMARY TEST (J1.5 success criterion):
        Reactor Alpha duration contradiction must get topic="construction"
        and topic_source="profile:smr" when the SMR profile is active.
        """
        items = assign_evidence_ids([
            _ev("Reactor Alpha construction duration is 24-36 months.", "vendor.pdf"),
            _ev("Reactor Alpha construction duration is 8-12 years.", "review.pdf"),
        ])
        result = detect_contradictions(items, profile=smr_profile)
        assert result, "Expected 1 contradiction — none detected."
        c = result[0]
        assert c.topic == "construction", (
            f"Expected topic='construction', got '{c.topic}'"
        )
        assert c.topic_source == "profile:smr", (
            f"Expected topic_source='profile:smr', got '{c.topic_source}'"
        )

    def test_smr_no_ai_dc_topic_terms_in_output(self, smr_profile):
        """No AI data center hard-coded topic names may appear in SMR output."""
        items = assign_evidence_ids([
            _ev("Reactor Alpha construction duration is 24-36 months.", "vendor.pdf"),
            _ev("Reactor Alpha construction duration is 8-12 years.", "review.pdf"),
        ])
        result = detect_contradictions(items, profile=smr_profile)
        for c in result:
            assert c.topic not in _AI_DC_HARD_CODED_TOPICS, (
                f"AI data center topic '{c.topic}' appeared in SMR profile output"
            )

    def test_smr_topic_source_present_on_all_contradictions(self, smr_profile):
        """Every contradiction emitted under a profile must have a non-empty topic_source."""
        items = assign_evidence_ids([
            _ev("Reactor Alpha construction duration is 24-36 months.", "vendor.pdf"),
            _ev("Reactor Alpha construction duration is 8-12 years.", "review.pdf"),
        ])
        result = detect_contradictions(items, profile=smr_profile)
        for c in result:
            assert c.topic_source != "", (
                f"Contradiction {c.contradiction_id} has empty topic_source"
            )

    # ---- AI DC profile -------------------------------------------------------

    def test_dc_kw_conflict_topic_is_power(self, dc_profile):
        """GW/kW numeric conflict under AI DC profile → topic='power'."""
        items = assign_evidence_ids([
            _ev("Rack power draw is 120 kW per rack.", "vendor_a.pdf"),
            _ev("Racks require 180 kW of power each.", "vendor_b.pdf"),
        ])
        result = detect_contradictions(items, profile=dc_profile)
        assert result, "Expected a kW conflict — none detected."
        assert result[0].topic == "power", (
            f"Expected topic='power', got '{result[0].topic}'"
        )
        assert result[0].topic_source == "profile:ai_data_centers"

    def test_dc_cooling_type_conflict_topic_is_cooling(self, dc_profile):
        """Air vs liquid cooling conflict under AI DC profile → topic='cooling'."""
        items = assign_evidence_ids([
            _ev("The GB200 NVL72 uses air cooling.", "vendor_a.pdf"),
            _ev("The GB200 NVL72 requires liquid cooling.", "vendor_b.pdf"),
        ])
        result = detect_contradictions(items, profile=dc_profile)
        assert result, "Expected a cooling-type conflict — none detected."
        assert result[0].topic == "cooling", (
            f"Expected topic='cooling', got '{result[0].topic}'"
        )
        assert result[0].topic_source == "profile:ai_data_centers"

    def test_dc_timeline_conflict_topic_is_timeline(self, dc_profile):
        """Year conflict under AI DC profile → topic='timeline'.

        Both claims use product-launch language so both years are classified
        as year_product (compatible) and the categorical conflict fires.
        Claims avoid NVL72/rack terms so the timeline topic wins.
        """
        items = assign_evidence_ids([
            _ev("Product platform launches in 2026.", "vendor_a.pdf"),
            _ev("Platform product shipping starts in 2027.", "vendor_b.pdf"),
        ])
        result = detect_contradictions(items, profile=dc_profile)
        assert result, "Expected a year conflict (2026 vs 2027, year_product) — none detected."
        assert result[0].topic == "timeline", (
            f"Expected topic='timeline', got '{result[0].topic}'"
        )

    def test_dc_no_smr_topic_terms_in_output(self, dc_profile):
        """No SMR-domain topic names may appear in AI DC profile output."""
        _smr_topics = {"construction", "licensing", "economics",
                       "fuel_cycle", "grid_integration", "reactor_design"}
        items = assign_evidence_ids([
            _ev("Rack power draw is 120 kW per rack.", "vendor_a.pdf"),
            _ev("Racks require 180 kW of power each.", "vendor_b.pdf"),
        ])
        result = detect_contradictions(items, profile=dc_profile)
        for c in result:
            assert c.topic not in _smr_topics, (
                f"SMR topic '{c.topic}' appeared in AI DC profile output"
            )

    # ---- No profile: backward compatibility ---------------------------------

    def test_no_profile_uses_hard_coded_topic(self):
        """When no profile is given, hard-coded topic labels are preserved."""
        items = assign_evidence_ids([
            _ev("Reactor Alpha construction duration is 24-36 months.", "vendor.pdf"),
            _ev("Reactor Alpha construction duration is 8-12 years.", "review.pdf"),
        ])
        result = detect_contradictions(items)  # no profile
        assert result, "Expected 1 contradiction — none detected."
        c = result[0]
        assert c.topic == "construction duration", (
            f"Without a profile the hard-coded topic should be preserved. Got '{c.topic}'"
        )

    def test_no_profile_topic_source_is_empty(self):
        """Without a profile, topic_source must remain the empty-string default."""
        items = assign_evidence_ids([
            _ev("Reactor Alpha construction duration is 24-36 months.", "vendor.pdf"),
            _ev("Reactor Alpha construction duration is 8-12 years.", "review.pdf"),
        ])
        result = detect_contradictions(items)  # no profile
        assert result
        assert result[0].topic_source == "", (
            f"Expected empty topic_source without a profile, got '{result[0].topic_source}'"
        )
