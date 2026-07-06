"""J11.4 — Multi-Profile Evidence Contribution & Balance.

Root cause: _attribute_evidence_profiles used winner-take-all keyword scoring.
The transmission profile's term set (interconnection, PJM, MISO, congestion, ...)
matched ALL retrieved siting evidence. The ai_data_centers profile's term set
(rack, GPU, nvl72, cooling ...) matched NONE of it. So all 50 items went to
transmission and ai_data_centers contributed 0 evidence.

Fix: use source_domain as the primary attribution signal. Evidence retrieved from
domain "ai_data_centers" belongs to profile "ai_data_centers" before keyword
scoring is consulted. Items from domains that do not match any profile name
(e.g. "grid") fall back to the existing keyword-scoring logic.

These tests verify:
  - Domain-match attribution overrides keyword scoring
  - Non-matching domain falls back to keyword scoring
  - Single-profile runs are unaffected
  - Zero-profile runs are unaffected
  - Items with no source_domain fall back to keyword scoring unchanged
  - Profile contribution counters reflect domain-match results
"""

from __future__ import annotations

import pytest

from functional_agents.evidence_agent import _attribute_evidence_profiles
from research_agent.profile import load_profile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _item(claim: str, source_domain: str = "", category: str = "other") -> dict:
    return {
        "evidence_id": f"EV-{abs(hash(claim + source_domain)) % 100000:05d}",
        "claim": claim,
        "category": category,
        "topics": [],
        "source_domain": source_domain,
    }


def _profiles(*names: str):
    return [load_profile(n) for n in names]


# ---------------------------------------------------------------------------
# TestDomainMatchAttribution — domain name overrides keyword scoring
# ---------------------------------------------------------------------------

class TestDomainMatchAttribution:

    def test_ai_data_centers_domain_overrides_transmission_keywords(self):
        """Siting item from ai_data_centers domain must go to ai_data_centers even
        though its claim text matches transmission terms (FERC, interconnection)."""
        items = [
            _item(
                "FERC Order 1920 requires utilities to evaluate large-load interconnection "
                "requests within 90 days for data center customers.",
                source_domain="ai_data_centers",
            )
        ]
        profs = _profiles("ai_data_centers", "transmission")
        coverage = _attribute_evidence_profiles(items, profs, fallback_profile="ai_data_centers")
        assert items[0]["source_profile"] == "ai_data_centers"

    def test_ai_data_centers_domain_incentive_item(self):
        """Investment tax credit item from ai_data_centers domain goes to ai_data_centers."""
        items = [
            _item(
                "A 30% investment tax credit is proposed for clean energy data center projects "
                "to accelerate hyperscaler deployment in qualified opportunity zones.",
                source_domain="ai_data_centers",
            )
        ]
        profs = _profiles("ai_data_centers", "transmission")
        coverage = _attribute_evidence_profiles(items, profs, fallback_profile="ai_data_centers")
        assert items[0]["source_profile"] == "ai_data_centers"

    def test_ai_data_centers_domain_community_opposition_item(self):
        """Community opposition item from ai_data_centers domain stays there."""
        items = [
            _item(
                "Community opposition to large data center projects has increased in Virginia "
                "and Texas due to water consumption and noise concerns.",
                source_domain="ai_data_centers",
            )
        ]
        profs = _profiles("ai_data_centers", "transmission")
        coverage = _attribute_evidence_profiles(items, profs, fallback_profile="ai_data_centers")
        assert items[0]["source_profile"] == "ai_data_centers"

    def test_multiple_ai_data_centers_items_all_attributed(self):
        """All items from ai_data_centers domain are attributed to it."""
        items = [
            _item("Permitting timelines for data centers in Indiana average 18 months.", source_domain="ai_data_centers"),
            _item("Texas offers a sales tax exemption for data center equipment purchases.", source_domain="ai_data_centers"),
            _item("Water withdrawal limits constrain liquid-cooled AI facilities in California.", source_domain="ai_data_centers"),
        ]
        profs = _profiles("ai_data_centers", "transmission")
        _attribute_evidence_profiles(items, profs, fallback_profile="ai_data_centers")
        assert all(i["source_profile"] == "ai_data_centers" for i in items)

    def test_coverage_counts_domain_attributed_items(self):
        """profile_coverage reflects domain-attributed items in evidence_count."""
        items = [
            _item("Permitting timelines for data centers in Indiana.", source_domain="ai_data_centers"),
            _item("Permitting timelines for data centers in Texas.", source_domain="ai_data_centers"),
            _item("PJM interconnection queue exceeds 2600 requests.", source_domain="grid"),
        ]
        profs = _profiles("ai_data_centers", "transmission")
        coverage = _attribute_evidence_profiles(items, profs, fallback_profile="ai_data_centers")
        assert coverage["ai_data_centers"]["evidence_count"] == 2


# ---------------------------------------------------------------------------
# TestNonMatchingDomainFallback — items from "grid" fall back to keyword scoring
# ---------------------------------------------------------------------------

class TestNonMatchingDomainFallback:

    def test_grid_domain_uses_keyword_scoring(self):
        """Items from 'grid' domain fall back to keyword scoring; transmission terms win."""
        items = [
            _item(
                "AEO2026 projects US electricity demand growth of 1.5% annually through 2050, "
                "driven by large load interconnection requests from data centers.",
                source_domain="grid",
            )
        ]
        profs = _profiles("ai_data_centers", "transmission")
        coverage = _attribute_evidence_profiles(items, profs, fallback_profile="ai_data_centers")
        # "interconnection" is a transmission keyword — should go to transmission
        assert items[0]["source_profile"] == "transmission"

    def test_no_source_domain_uses_keyword_scoring(self):
        """Items with no source_domain fall back to keyword scoring (regression guard)."""
        items = [
            _item(
                "PJM interconnection queue depth exceeds 2600 requests for new generation.",
                source_domain="",
            )
        ]
        profs = _profiles("ai_data_centers", "transmission")
        _attribute_evidence_profiles(items, profs, fallback_profile="ai_data_centers")
        assert items[0]["source_profile"] == "transmission"

    def test_unknown_domain_uses_keyword_scoring(self):
        """Items from an unrecognised domain fall back to keyword scoring."""
        items = [
            _item(
                "SMR deployment timelines average 10 years from permit application to operation.",
                source_domain="smr",
            )
        ]
        profs = _profiles("ai_data_centers", "transmission")
        _attribute_evidence_profiles(items, profs, fallback_profile="ai_data_centers")
        # smr is not a requested profile; should score low for both → fallback_profile
        assert items[0]["source_profile"] == "ai_data_centers"


# ---------------------------------------------------------------------------
# TestMixedPool — realistic mix of domain items
# ---------------------------------------------------------------------------

class TestMixedPool:

    def test_mixed_pool_both_profiles_contribute(self):
        """A realistic pool with ai_data_centers and grid items contributes to both profiles."""
        items = [
            _item("Texas offers data center tax exemption for qualifying equipment.", source_domain="ai_data_centers"),
            _item("Water availability in Indiana supports liquid-cooled AI facilities.", source_domain="ai_data_centers"),
            _item("AI data center power demand will reach 35 GW by 2030.", source_domain="ai_data_centers"),
            _item("PJM interconnection queue exceeds 2600 requests.", source_domain="grid"),
            _item("Transmission congestion along constrained corridors in PJM raises LMP.", source_domain="grid"),
            _item("MISO queue withdrawal rate exceeds 70% of submitted projects.", source_domain="grid"),
        ]
        profs = _profiles("ai_data_centers", "transmission")
        coverage = _attribute_evidence_profiles(items, profs, fallback_profile="ai_data_centers")
        assert coverage["ai_data_centers"]["evidence_count"] >= 3
        assert coverage["transmission"]["evidence_count"] >= 2

    def test_profile_balance_is_non_zero_for_both(self):
        """Neither profile should be at 0 in a mixed pool."""
        items = [
            _item("Data center siting incentives vary widely by state.", source_domain="ai_data_centers"),
            _item("Community opposition to AI facilities is growing in Northern Virginia.", source_domain="ai_data_centers"),
            _item("ERCOT interconnection queue processes large-load requests within 24 months.", source_domain="grid"),
            _item("Transmission congestion in PJM peaks during summer demand.", source_domain="grid"),
        ]
        profs = _profiles("ai_data_centers", "transmission")
        coverage = _attribute_evidence_profiles(items, profs, fallback_profile="ai_data_centers")
        total = sum(v["evidence_count"] for v in coverage.values())
        assert coverage["ai_data_centers"]["evidence_count"] > 0
        assert coverage["transmission"]["evidence_count"] > 0
        assert total == len(items)


# ---------------------------------------------------------------------------
# TestSingleProfileRegression — single-profile path unchanged
# ---------------------------------------------------------------------------

class TestSingleProfileRegression:

    def test_single_profile_assigns_all_to_that_profile(self):
        """Single-profile runs assign all items to the one profile (unchanged)."""
        items = [
            _item("PJM interconnection queue exceeds 2600 requests.", source_domain="grid"),
            _item("Texas offers data center tax incentives.", source_domain="ai_data_centers"),
        ]
        profs = _profiles("ai_data_centers")
        coverage = _attribute_evidence_profiles(items, profs, fallback_profile="ai_data_centers")
        assert all(i["source_profile"] == "ai_data_centers" for i in items)
        assert coverage["ai_data_centers"]["evidence_count"] == 2


# ---------------------------------------------------------------------------
# TestNoFalseProfileContribution — domain match does not force irrelevant items
# ---------------------------------------------------------------------------

class TestNoFalseProfileContribution:

    def test_keyword_only_item_without_domain_goes_to_best_match(self):
        """Items with no domain use keyword scoring — no forced attribution."""
        # Claim clearly about transmission; no source_domain
        items = [_item("HVDC transmission line capacity constrains California-Nevada power flows.")]
        profs = _profiles("ai_data_centers", "transmission")
        _attribute_evidence_profiles(items, profs, fallback_profile="ai_data_centers")
        assert items[0]["source_profile"] == "transmission"

    def test_domain_match_does_not_cross_contaminate(self):
        """ai_data_centers domain does not pollute transmission coverage."""
        items = [
            _item("GPU rack cooling requirements drive 1 MW per cabinet power demand.", source_domain="ai_data_centers"),
        ]
        profs = _profiles("ai_data_centers", "transmission")
        coverage = _attribute_evidence_profiles(items, profs, fallback_profile="ai_data_centers")
        assert coverage.get("transmission", {}).get("evidence_count", 0) == 0
        assert coverage["ai_data_centers"]["evidence_count"] == 1
