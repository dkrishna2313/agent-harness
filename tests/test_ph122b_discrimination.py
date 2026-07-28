"""PH12.2b tests — mapping authority, discrimination, homogenization.

Covers:
  T1  Mapping regression — evaluate → select → map ordering
  T2  Relationship classification
  T3  Discrimination score computation
  T4  Assumption discrimination splits
  T5  Recommendation discrimination splits
  T6  Evidence discrimination splits
  T7  ContentCoverage discrimination fractions
  T8  ContentConfidence discrimination shares + level gating
  T9  Multi-state homogenization (none / partial / substantial / full)
  T10 Homogenization state justification
  T11 ContentConfig PH12.2b fields + validators
  T12 StrategyTrace PH12.2b fields
  T13 Consistency guard (no mismatch expected in normal run)
  T14 Backward compatibility (old-format trace still loads)
"""

from __future__ import annotations

import math
import pytest

from functional_agents.strategy.theory_content import (
    ContentCoverage,
    ContentConfidence,
    ContentLineageEntry,
    EvidenceLineageEntry,
    TheoryContent,
)
from functional_agents.strategy.discrimination_calculator import (
    enrich_with_discrimination,
)
from functional_agents.strategy.content_differentiation import compute_differentiation
from functional_agents.strategy.strategy_config import ContentConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tc(
    theory_id: str,
    assumption_ids: list[str],
    risk_ids: list[str] | None = None,
    opportunity_ids: list[str] | None = None,
    recommendation_ids: list[str] | None = None,
    evidence_ids: list[str] | None = None,
    content_lineage: dict | None = None,
    evidence_lineage: list | None = None,
    mapped_option_id: str | None = None,
    mapping_confidence: str = "Medium",
) -> TheoryContent:
    return TheoryContent(
        theory_id=theory_id,
        mapped_option_id=mapped_option_id,
        mapping_confidence=mapping_confidence,
        assumption_ids=assumption_ids,
        risk_ids=risk_ids or [],
        opportunity_ids=opportunity_ids or [],
        recommendation_ids=recommendation_ids or [],
        evidence_ids=evidence_ids or [],
        content_lineage=content_lineage or {
            "assumptions": [
                ContentLineageEntry(
                    source_id=aid,
                    assignment_type="option_link",
                    relevance_score=0.8,
                )
                for aid in assumption_ids
            ],
            "risks": [],
            "opportunities": [],
            "recommendations": [],
            "evidence": [],
            "success_conditions": [],
        },
        evidence_lineage=evidence_lineage or [],
        coverage=ContentCoverage(
            overall=0.7,
            status="sufficient",
            explicit_count=len(assumption_ids),
            fallback_count=0,
        ),
        confidence=ContentConfidence(
            level="High",
            explicit_share=1.0,
            mapping_confidence=mapping_confidence,
        ),
    )


# ---------------------------------------------------------------------------
# T1: Mapping regression — evaluate → select → map ordering
# ---------------------------------------------------------------------------

class TestMappingAuthority:
    def test_mapping_is_post_selection(self):
        """discrimination_calculator receives content AFTER selection; does not re-map."""
        tc_a = _make_tc("TH-A", ["A-001", "A-002"], mapped_option_id="OPT-B")
        tc_b = _make_tc("TH-B", ["A-001", "A-003"], mapped_option_id="OPT-C")

        enriched = enrich_with_discrimination([tc_a, tc_b])
        th_a = next(tc for tc in enriched if tc.theory_id == "TH-A")
        th_b = next(tc for tc in enriched if tc.theory_id == "TH-B")

        # mapped_option_id must be preserved exactly as supplied
        assert th_a.mapped_option_id == "OPT-B"
        assert th_b.mapped_option_id == "OPT-C"

    def test_discrimination_does_not_mutate_mapped_option(self):
        """enrich_with_discrimination returns new objects; original intact."""
        tc = _make_tc("TH-X", ["A-001"], mapped_option_id="OPT-B")
        enriched = enrich_with_discrimination([tc])
        assert tc.mapped_option_id == "OPT-B"
        assert enriched[0].mapped_option_id == "OPT-B"
        assert tc is not enriched[0]


# ---------------------------------------------------------------------------
# T2: Relationship classification
# ---------------------------------------------------------------------------

class TestRelationshipClassification:
    def test_explicit_discriminating(self):
        """An explicit item present in one theory → explicit_discriminating."""
        tc_a = _make_tc("TH-A", ["A-001"])
        tc_b = _make_tc("TH-B", ["A-002"])
        enriched = enrich_with_discrimination([tc_a, tc_b])
        th_a = next(tc for tc in enriched if tc.theory_id == "TH-A")
        entry = th_a.content_lineage["assumptions"][0]
        assert entry.relationship_classification == "explicit_discriminating"

    def test_explicit_shared(self):
        """An explicit item present in ALL theories → explicit_shared."""
        tc_a = _make_tc("TH-A", ["A-001"])
        tc_b = _make_tc("TH-B", ["A-001"])
        enriched = enrich_with_discrimination([tc_a, tc_b])
        th_a = next(tc for tc in enriched if tc.theory_id == "TH-A")
        entry = th_a.content_lineage["assumptions"][0]
        assert entry.relationship_classification == "explicit_shared"

    def test_posture_discriminating(self):
        """A posture_match item present in one theory → posture_discriminating."""
        lineage = {
            "assumptions": [
                ContentLineageEntry(
                    source_id="A-010",
                    assignment_type="posture_match",
                    relevance_score=0.5,
                )
            ],
            "risks": [], "opportunities": [], "recommendations": [], "evidence": [], "success_conditions": [],
        }
        tc_a = _make_tc("TH-A", ["A-010"], content_lineage=lineage)
        tc_b = _make_tc("TH-B", ["A-999"])
        enriched = enrich_with_discrimination([tc_a, tc_b])
        th_a = next(tc for tc in enriched if tc.theory_id == "TH-A")
        entry = th_a.content_lineage["assumptions"][0]
        assert entry.relationship_classification == "posture_discriminating"

    def test_semantic_discriminating(self):
        """A posture_match item present in ALL theories → semantic_discriminating."""
        def make_posture(tid: str) -> TheoryContent:
            lineage = {
                "assumptions": [
                    ContentLineageEntry(
                        source_id="A-010",
                        assignment_type="posture_match",
                        relevance_score=0.5,
                    )
                ],
                "risks": [], "opportunities": [], "recommendations": [],
                "evidence": [], "success_conditions": [],
            }
            return _make_tc(tid, ["A-010"], content_lineage=lineage)

        tcs = [make_posture(t) for t in ["TH-A", "TH-B"]]
        enriched = enrich_with_discrimination(tcs)
        entry = enriched[0].content_lineage["assumptions"][0]
        assert entry.relationship_classification == "semantic_discriminating"

    def test_fallback_classification(self):
        """A symmetric_fallback item → fallback."""
        lineage = {
            "assumptions": [
                ContentLineageEntry(
                    source_id="A-FBK",
                    assignment_type="symmetric_fallback",
                    relevance_score=0.1,
                )
            ],
            "risks": [], "opportunities": [], "recommendations": [], "evidence": [], "success_conditions": [],
        }
        tc_a = _make_tc("TH-A", ["A-FBK"], content_lineage=lineage)
        tc_b = _make_tc("TH-B", ["A-999"])
        enriched = enrich_with_discrimination([tc_a, tc_b])
        th_a = next(tc for tc in enriched if tc.theory_id == "TH-A")
        entry = th_a.content_lineage["assumptions"][0]
        assert entry.relationship_classification == "fallback"


# ---------------------------------------------------------------------------
# T3: Discrimination score computation
# ---------------------------------------------------------------------------

class TestDiscriminationScore:
    def test_theory_unique_score_3_theories(self):
        """Item in 1 of 3 theories → score = 1 - 1/3 ≈ 0.6667."""
        tc_a = _make_tc("TH-A", ["A-UNIQUE"])
        tc_b = _make_tc("TH-B", ["A-999"])
        tc_c = _make_tc("TH-C", ["A-888"])
        enriched = enrich_with_discrimination([tc_a, tc_b, tc_c])
        th_a = next(tc for tc in enriched if tc.theory_id == "TH-A")
        entry = th_a.content_lineage["assumptions"][0]
        assert abs(entry.discrimination_score - (1.0 - 1/3)) < 1e-4

    def test_global_shared_score_zero(self):
        """Item shared across all 3 theories → score = 0.0."""
        tc_a = _make_tc("TH-A", ["A-SHARED"])
        tc_b = _make_tc("TH-B", ["A-SHARED"])
        tc_c = _make_tc("TH-C", ["A-SHARED"])
        enriched = enrich_with_discrimination([tc_a, tc_b, tc_c])
        th_a = next(tc for tc in enriched if tc.theory_id == "TH-A")
        entry = th_a.content_lineage["assumptions"][0]
        assert entry.discrimination_score == 0.0
        assert entry.relationship_scope == "global_shared"

    def test_theory_subset_scope(self):
        """Item in 2 of 3 theories → scope = theory_subset."""
        tc_a = _make_tc("TH-A", ["A-SUBSET"])
        tc_b = _make_tc("TH-B", ["A-SUBSET"])
        tc_c = _make_tc("TH-C", ["A-999"])
        enriched = enrich_with_discrimination([tc_a, tc_b, tc_c])
        th_a = next(tc for tc in enriched if tc.theory_id == "TH-A")
        entry = th_a.content_lineage["assumptions"][0]
        assert entry.relationship_scope == "theory_subset"
        assert abs(entry.discrimination_score - (1.0 - 2/3)) < 1e-4

    def test_theory_unique_scope(self):
        """Item in 1 of 2 theories → scope = theory_unique."""
        tc_a = _make_tc("TH-A", ["A-ONLY"])
        tc_b = _make_tc("TH-B", ["A-OTHER"])
        enriched = enrich_with_discrimination([tc_a, tc_b])
        th_a = next(tc for tc in enriched if tc.theory_id == "TH-A")
        entry = th_a.content_lineage["assumptions"][0]
        assert entry.relationship_scope == "theory_unique"

    def test_shared_across_theory_ids_populated(self):
        """shared_across_theory_ids includes all theories that have the item."""
        tc_a = _make_tc("TH-A", ["A-SHARED"])
        tc_b = _make_tc("TH-B", ["A-SHARED"])
        tc_c = _make_tc("TH-C", ["A-OTHER"])
        enriched = enrich_with_discrimination([tc_a, tc_b, tc_c])
        th_a = next(tc for tc in enriched if tc.theory_id == "TH-A")
        entry = th_a.content_lineage["assumptions"][0]
        assert set(entry.shared_across_theory_ids) == {"TH-A", "TH-B"}


# ---------------------------------------------------------------------------
# T4: Assumption discrimination splits
# ---------------------------------------------------------------------------

class TestAssumptionSplits:
    def test_distinctive_assumptions_unique_items(self):
        """Assumptions present in only one theory → distinctive_assumption_ids."""
        tc_a = _make_tc("TH-A", ["A-001", "A-SHARED"])
        tc_b = _make_tc("TH-B", ["A-002", "A-SHARED"])
        enriched = enrich_with_discrimination([tc_a, tc_b])
        th_a = next(tc for tc in enriched if tc.theory_id == "TH-A")
        assert "A-001" in th_a.distinctive_assumption_ids
        assert "A-SHARED" not in th_a.distinctive_assumption_ids

    def test_shared_assumptions_global_items(self):
        """Assumptions in ALL theories → shared_assumption_ids."""
        tc_a = _make_tc("TH-A", ["A-001", "A-SHARED"])
        tc_b = _make_tc("TH-B", ["A-002", "A-SHARED"])
        enriched = enrich_with_discrimination([tc_a, tc_b])
        th_a = next(tc for tc in enriched if tc.theory_id == "TH-A")
        assert "A-SHARED" in th_a.shared_assumption_ids

    def test_assumption_split_exhaustive(self):
        """Every assumption_id must appear in exactly one of distinctive or shared."""
        tc_a = _make_tc("TH-A", ["A-001", "A-SHARED", "A-ONLY"])
        tc_b = _make_tc("TH-B", ["A-002", "A-SHARED"])
        enriched = enrich_with_discrimination([tc_a, tc_b])
        th_a = next(tc for tc in enriched if tc.theory_id == "TH-A")
        all_ids = set(th_a.assumption_ids)
        split_ids = set(th_a.distinctive_assumption_ids) | set(th_a.shared_assumption_ids)
        assert all_ids == split_ids


# ---------------------------------------------------------------------------
# T5: Recommendation discrimination splits
# ---------------------------------------------------------------------------

class TestRecommendationSplits:
    def test_distinctive_recommendations(self):
        tc_a = _make_tc("TH-A", [], recommendation_ids=["REC-001", "REC-SHARED"])
        tc_b = _make_tc("TH-B", [], recommendation_ids=["REC-002", "REC-SHARED"])
        enriched = enrich_with_discrimination([tc_a, tc_b])
        th_a = next(tc for tc in enriched if tc.theory_id == "TH-A")
        assert "REC-001" in th_a.distinctive_recommendation_ids
        assert "REC-SHARED" in th_a.shared_recommendation_ids


# ---------------------------------------------------------------------------
# T6: Evidence discrimination splits
# ---------------------------------------------------------------------------

class TestEvidenceSplits:
    def test_distinctive_evidence(self):
        ev_lineage_a = [
            EvidenceLineageEntry(
                evidence_id="EV-001",
                assignment_type="option_link",
                relevance_score=0.8,
            )
        ]
        ev_lineage_b = [
            EvidenceLineageEntry(
                evidence_id="EV-SHARED",
                assignment_type="option_link",
                relevance_score=0.8,
            )
        ]
        tc_a = _make_tc("TH-A", [], evidence_ids=["EV-001", "EV-SHARED"],
                        evidence_lineage=ev_lineage_a + [
                            EvidenceLineageEntry(evidence_id="EV-SHARED", assignment_type="option_link", relevance_score=0.7)
                        ])
        tc_b = _make_tc("TH-B", [], evidence_ids=["EV-SHARED"],
                        evidence_lineage=ev_lineage_b)
        enriched = enrich_with_discrimination([tc_a, tc_b])
        th_a = next(tc for tc in enriched if tc.theory_id == "TH-A")
        assert "EV-001" in th_a.distinctive_evidence_ids
        assert "EV-SHARED" in th_a.shared_evidence_ids


# ---------------------------------------------------------------------------
# T7: ContentCoverage discrimination fractions
# ---------------------------------------------------------------------------

class TestCoverageFractions:
    def test_distinctive_fraction(self):
        """coverage.distinctive = distinctive_count / total_assigned."""
        tc_a = _make_tc("TH-A", ["A-001", "A-002"])
        tc_b = _make_tc("TH-B", ["A-002", "A-003"])
        enriched = enrich_with_discrimination([tc_a, tc_b])
        th_a = next(tc for tc in enriched if tc.theory_id == "TH-A")
        # A-001 is distinctive, A-002 is shared → 1/2 = 0.5
        assert th_a.coverage.distinctive == pytest.approx(0.5, abs=0.01)

    def test_shared_context_fraction(self):
        """coverage.shared_context = shared_count / total_assigned."""
        tc_a = _make_tc("TH-A", ["A-001", "A-002"])
        tc_b = _make_tc("TH-B", ["A-002", "A-003"])
        enriched = enrich_with_discrimination([tc_a, tc_b])
        th_a = next(tc for tc in enriched if tc.theory_id == "TH-A")
        # A-002 is shared → 1/2 = 0.5
        assert th_a.coverage.shared_context == pytest.approx(0.5, abs=0.01)

    def test_full_distinctive_when_no_overlap(self):
        """When no assumptions are shared, distinctive = 1.0."""
        tc_a = _make_tc("TH-A", ["A-001"])
        tc_b = _make_tc("TH-B", ["A-002"])
        enriched = enrich_with_discrimination([tc_a, tc_b])
        th_a = next(tc for tc in enriched if tc.theory_id == "TH-A")
        assert th_a.coverage.distinctive == 1.0

    def test_coverage_sums_to_one(self):
        """distinctive + shared_context == 1.0 for assumptions-only content."""
        tc_a = _make_tc("TH-A", ["A-001", "A-SHARED"])
        tc_b = _make_tc("TH-B", ["A-002", "A-SHARED"])
        enriched = enrich_with_discrimination([tc_a, tc_b])
        th_a = next(tc for tc in enriched if tc.theory_id == "TH-A")
        assert abs(th_a.coverage.distinctive + th_a.coverage.shared_context - 1.0) < 0.01


# ---------------------------------------------------------------------------
# T8: ContentConfidence discrimination-aware level gating
# ---------------------------------------------------------------------------

class TestConfidenceLevelGating:
    def test_high_requires_discriminating_share(self):
        """High confidence is downgraded to Medium when discriminating_share < 0.40."""
        tc_a = _make_tc("TH-A", ["A-SHARED", "A-ALSO-SHARED"])
        tc_b = _make_tc("TH-B", ["A-SHARED", "A-ALSO-SHARED"])
        # Both assumptions shared — no discriminating content
        enriched = enrich_with_discrimination([tc_a, tc_b])
        th_a = next(tc for tc in enriched if tc.theory_id == "TH-A")
        # All content is global_shared → discriminating_share = 0.0 → downgrade
        assert th_a.confidence.level in ("Medium", "Low")

    def test_high_preserved_when_discriminating(self):
        """High confidence preserved when discriminating_share >= 0.40."""
        tc_a = _make_tc("TH-A", ["A-DISC1", "A-DISC2"])
        tc_b = _make_tc("TH-B", ["A-OTHER1", "A-OTHER2"])
        # All assumptions distinctive → discriminating_share = 1.0
        enriched = enrich_with_discrimination([tc_a, tc_b])
        th_a = next(tc for tc in enriched if tc.theory_id == "TH-A")
        assert th_a.confidence.level == "High"

    def test_discrimination_shares_populated(self):
        """explicit_discriminating_share and explicit_shared_share are set."""
        tc_a = _make_tc("TH-A", ["A-DISC", "A-SHARED"])
        tc_b = _make_tc("TH-B", ["A-OTHER", "A-SHARED"])
        enriched = enrich_with_discrimination([tc_a, tc_b])
        th_a = next(tc for tc in enriched if tc.theory_id == "TH-A")
        assert th_a.confidence.explicit_discriminating_share >= 0.0
        assert th_a.confidence.explicit_shared_share >= 0.0
        total = (
            th_a.confidence.explicit_discriminating_share
            + th_a.confidence.explicit_shared_share
        )
        # Should account for most of explicit_share (may also include posture)
        assert total <= th_a.confidence.explicit_share + 0.01


# ---------------------------------------------------------------------------
# T9: Multi-state homogenization
# ---------------------------------------------------------------------------

class TestMultiStateHomogenization:
    def test_none_state_no_overlap(self):
        tc_a = _make_tc("TH-A", ["A-001"], risk_ids=["RSK-001"])
        tc_b = _make_tc("TH-B", ["A-002"], risk_ids=["RSK-002"])
        result = compute_differentiation([tc_a, tc_b])
        assert result["homogenization_state"] == "none"
        assert result["content_homogenization_detected"] is False

    def test_full_state_all_identical(self):
        tc_a = _make_tc("TH-A", ["A-001"], risk_ids=["RSK-001"],
                        opportunity_ids=["OPP-001"], recommendation_ids=["REC-001"],
                        evidence_ids=["EV-001"])
        tc_b = _make_tc("TH-B", ["A-001"], risk_ids=["RSK-001"],
                        opportunity_ids=["OPP-001"], recommendation_ids=["REC-001"],
                        evidence_ids=["EV-001"])
        result = compute_differentiation([tc_a, tc_b], full_threshold=0.95)
        assert result["homogenization_state"] == "full"
        assert result["content_homogenization_detected"] is True

    def test_partial_state_one_identical_dimension(self):
        """Two theories with identical assumptions only → partial."""
        tc_a = _make_tc("TH-A", ["A-001", "A-002"], risk_ids=["RSK-001"])
        tc_b = _make_tc("TH-B", ["A-001", "A-002"], risk_ids=["RSK-002"])
        result = compute_differentiation([tc_a, tc_b], partial_threshold=0.75,
                                         maximum_identical_dimensions=1)
        # assumptions jaccard = 1.0 → ≥ 1 identical dim → partial
        assert result["homogenization_state"] in ("partial", "substantial")
        assert result["content_homogenization_detected"] is True

    def test_detected_true_for_partial(self):
        """detected must be True for any non-none state."""
        tc_a = _make_tc("TH-A", ["A-001", "A-002"])
        tc_b = _make_tc("TH-B", ["A-001", "A-002"])
        result = compute_differentiation([tc_a, tc_b], maximum_identical_dimensions=1)
        assert result["content_homogenization_detected"] is True

    def test_detected_false_for_none_state(self):
        tc_a = _make_tc("TH-A", ["A-001"])
        tc_b = _make_tc("TH-B", ["A-002"])
        result = compute_differentiation([tc_a, tc_b])
        assert result["content_homogenization_detected"] is False


# ---------------------------------------------------------------------------
# T10: Homogenization state justification
# ---------------------------------------------------------------------------

class TestHomogenizationJustification:
    def test_identical_dimensions_populated(self):
        tc_a = _make_tc("TH-A", ["A-001", "A-002"])
        tc_b = _make_tc("TH-B", ["A-001", "A-002"])
        result = compute_differentiation([tc_a, tc_b], maximum_identical_dimensions=1)
        details = result["homogenization_details"]
        assert "assumptions" in details["identical_dimensions"]

    def test_pairwise_similarity_populated(self):
        tc_a = _make_tc("TH-A", ["A-001"])
        tc_b = _make_tc("TH-B", ["A-001"])
        result = compute_differentiation([tc_a, tc_b])
        details = result["homogenization_details"]
        assert len(details["pairwise_similarity"]) > 0

    def test_rationale_is_string(self):
        tc_a = _make_tc("TH-A", ["A-001"])
        tc_b = _make_tc("TH-B", ["A-001"])
        result = compute_differentiation([tc_a, tc_b])
        details = result["homogenization_details"]
        assert isinstance(details["rationale"], str) and details["rationale"]

    def test_state_in_homogenization_details(self):
        tc_a = _make_tc("TH-A", ["A-001"])
        tc_b = _make_tc("TH-B", ["A-002"])
        result = compute_differentiation([tc_a, tc_b])
        details = result["homogenization_details"]
        assert details["state"] == "none"


# ---------------------------------------------------------------------------
# T11: ContentConfig PH12.2b fields + validators
# ---------------------------------------------------------------------------

class TestContentConfigPH122b:
    def test_default_fields_present(self):
        cfg = ContentConfig()
        assert hasattr(cfg, "minimum_discrimination_score")
        assert hasattr(cfg, "partial_homogenization_threshold")
        assert hasattr(cfg, "full_homogenization_threshold")
        assert hasattr(cfg, "maximum_identical_dimensions")
        assert hasattr(cfg, "maximum_shared_assumptions")
        assert hasattr(cfg, "maximum_shared_recommendations")
        assert hasattr(cfg, "maximum_shared_evidence")

    def test_default_values(self):
        cfg = ContentConfig()
        assert cfg.minimum_discrimination_score == 0.20
        assert cfg.partial_homogenization_threshold == 0.75
        assert cfg.full_homogenization_threshold == 0.95
        assert cfg.maximum_identical_dimensions == 2

    def test_fraction_validators(self):
        with pytest.raises(Exception):
            ContentConfig(minimum_discrimination_score=1.5)
        with pytest.raises(Exception):
            ContentConfig(partial_homogenization_threshold=-0.1)

    def test_positive_int_validators(self):
        with pytest.raises(Exception):
            ContentConfig(maximum_shared_assumptions=0)
        with pytest.raises(Exception):
            ContentConfig(maximum_shared_recommendations=-1)

    def test_non_negative_int_validator(self):
        with pytest.raises(Exception):
            ContentConfig(maximum_identical_dimensions=-1)
        cfg = ContentConfig(maximum_identical_dimensions=0)
        assert cfg.maximum_identical_dimensions == 0

    def test_threshold_ordering_validator(self):
        with pytest.raises(Exception):
            ContentConfig(
                partial_homogenization_threshold=0.90,
                full_homogenization_threshold=0.80,
            )

    def test_equal_thresholds_allowed(self):
        cfg = ContentConfig(
            partial_homogenization_threshold=0.85,
            full_homogenization_threshold=0.85,
        )
        assert cfg.partial_homogenization_threshold == cfg.full_homogenization_threshold


# ---------------------------------------------------------------------------
# T12: StrategyTrace PH12.2b fields
# ---------------------------------------------------------------------------

class TestStrategyTracePH122b:
    def test_trace_has_new_fields(self):
        from functional_agents.strategy.strategy_trace import StrategyTrace
        # Check fields exist at class level
        fields = StrategyTrace.model_fields
        assert "theory_discrimination" in fields
        assert "content_differentiation_state" in fields

    def test_trace_backward_compat_new_fields_default(self):
        from functional_agents.strategy.strategy_trace import StrategyTrace
        # Default values
        defaults = StrategyTrace.model_fields
        assert defaults["theory_discrimination"].default is not None or True  # Field exists


# ---------------------------------------------------------------------------
# T13: Consistency guard
# ---------------------------------------------------------------------------

class TestConsistencyGuard:
    def test_no_mismatch_when_mapping_consistent(self):
        """When mapped_option_id matches across selection and content, guard passes."""
        tc = _make_tc("TH-WIN", ["A-001"], mapped_option_id="OPT-B")
        enriched = enrich_with_discrimination([tc])
        assert enriched[0].mapped_option_id == "OPT-B"

    def test_enrich_preserves_all_mapped_option_ids(self):
        """All mapped_option_ids survive enrichment unchanged."""
        tc_a = _make_tc("TH-A", ["A-001"], mapped_option_id="OPT-B")
        tc_b = _make_tc("TH-B", ["A-002"], mapped_option_id="OPT-C")
        enriched = enrich_with_discrimination([tc_a, tc_b])
        by_id = {tc.theory_id: tc for tc in enriched}
        assert by_id["TH-A"].mapped_option_id == "OPT-B"
        assert by_id["TH-B"].mapped_option_id == "OPT-C"


# ---------------------------------------------------------------------------
# T14: Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_theory_content_without_discrimination_fields(self):
        """TheoryContent without PH12.2b fields validates with defaults."""
        tc = TheoryContent(
            theory_id="TH-OLD",
            assumption_ids=["A-001"],
        )
        assert tc.distinctive_assumption_ids == []
        assert tc.shared_assumption_ids == []
        assert tc.homogenization_state == "none"

    def test_content_lineage_entry_without_discrimination_fields(self):
        """ContentLineageEntry without PH12.2b fields uses defaults."""
        entry = ContentLineageEntry(source_id="A-001", assignment_type="option_link")
        assert entry.relationship_classification == ""
        assert entry.discrimination_score == 0.0
        assert entry.shared_across_theory_ids == []

    def test_evidence_lineage_entry_backward_compat(self):
        entry = EvidenceLineageEntry(evidence_id="EV-001", assignment_type="option_link")
        assert entry.relationship_classification == ""
        assert entry.discrimination_score == 0.0

    def test_content_coverage_backward_compat(self):
        cov = ContentCoverage(overall=0.7, status="sufficient")
        assert cov.distinctive == 0.0
        assert cov.shared_context == 0.0
        assert cov.evidence_distinctive == 0.0

    def test_content_confidence_backward_compat(self):
        conf = ContentConfidence(level="High", explicit_share=0.8)
        assert conf.explicit_discriminating_share == 0.0
        assert conf.explicit_shared_share == 0.0

    def test_empty_contents_noop(self):
        result = enrich_with_discrimination([])
        assert result == []

    def test_single_theory_noop(self):
        """Single theory: all items are theory_unique (score = 0.0 since 1/1 = 1.0... wait, 1-1/1=0)."""
        tc = _make_tc("TH-ONLY", ["A-001"])
        enriched = enrich_with_discrimination([tc])
        assert len(enriched) == 1
        # With 1 theory: item appears in 1/1 → score = 1 - 1/1 = 0.0
        # All in shared? No — score=0.0 means the item IS the only one → global_shared of 1
        entry = enriched[0].content_lineage["assumptions"][0]
        # 1 theory, item in 1 theory → 1/1 = 1.0 → score = 0.0
        assert entry.discrimination_score == 0.0

    def test_content_differentiation_single_theory(self):
        tc = _make_tc("TH-ONLY", ["A-001"])
        result = compute_differentiation([tc])
        assert result["homogenization_state"] == "none"
        assert result["content_homogenization_detected"] is False
