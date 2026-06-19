"""Tests for research_agent.perspectives (J3.2).

Covers:
  - Domain detection from source document filenames (J3.2.1)
  - Perspective classification (J3.2.2)
  - Diversity-aware evidence selection (J3.2.3-4)
  - Perspective coverage and diversity metrics (J3.2.5-6)
  - EvidenceItem.perspective field populated via enricher (J3.2.2)
  - Scorer integration: retrieval_diversity in QAScore (J3.2.8)
"""

from __future__ import annotations

import pytest

from research_agent.perspectives import (
    PERSPECTIVES_AI_DC,
    PERSPECTIVES_SMR,
    build_diversity_metrics,
    classify_perspective,
    compute_diversity_score,
    compute_perspective_coverage,
    detect_domain,
    select_diverse_evidence,
)


# ---------------------------------------------------------------------------
# Domain detection
# ---------------------------------------------------------------------------

class TestDetectDomain:
    def test_nvidia_doc_is_ai_dc(self):
        assert detect_domain("nvidia_gb200_nvl72_overview.txt") == "ai_dc"

    def test_smr_doc_detected(self):
        assert detect_domain("bwrx_300_technical_overview.txt") == "smr"

    def test_nuclear_doc_detected(self):
        assert detect_domain("nuclear_power_economics_2024.txt") == "smr"

    def test_nrc_doc_detected(self):
        assert detect_domain("nrc_licensing_timeline.txt") == "smr"

    def test_haleu_doc_detected(self):
        assert detect_domain("haleu_supply_chain_analysis.txt") == "smr"

    def test_unknown_doc_defaults_ai_dc(self):
        assert detect_domain("generic_source.txt") == "ai_dc"

    def test_empty_doc_defaults_ai_dc(self):
        assert detect_domain("") == "ai_dc"


# ---------------------------------------------------------------------------
# Perspective classification (J3.2.2)
# ---------------------------------------------------------------------------

class TestClassifyPerspective:
    # AI DC perspectives
    def test_power_claim(self):
        p = classify_perspective(
            "The NVL72 rack draws 120 kW of power at full utilisation.",
            "nvidia_gb200_specs.txt",
        )
        assert p == "power"

    def test_cooling_claim(self):
        p = classify_perspective(
            "Direct liquid cooling via CDU units is required for the GB200.",
            "nvidia_thermal_design.txt",
        )
        assert p == "cooling"

    def test_networking_claim(self):
        p = classify_perspective(
            "NVLink provides 900 GB/s chip-to-chip bandwidth between Grace and Blackwell.",
            "nvidia_nvlink_architecture.txt",
        )
        assert p == "networking"

    def test_economics_claim(self):
        p = classify_perspective(
            "Total cost of ownership (TCO) is $1.2M; return on investment exceeds capex by year 3.",
            "ai_dc_tco_analysis.txt",
        )
        assert p == "economics"

    def test_deployment_claim(self):
        p = classify_perspective(
            "GB200 NVL72 availability is planned for Q2 2025 with a 16-week lead time.",
            "nvidia_roadmap.txt",
        )
        assert p == "deployment"

    # SMR perspectives
    def test_licensing_claim(self):
        p = classify_perspective(
            "NRC design certification for BWRX-300 is expected by 2027 after regulatory review.",
            "bwrx_300_licensing.txt",
        )
        assert p == "licensing"

    def test_fuel_claim(self):
        p = classify_perspective(
            "BWRX-300 requires HALEU fuel enriched to 19.75% U-235 for its fuel cycle.",
            "smr_fuel_requirements.txt",
        )
        assert p == "fuel"

    def test_construction_claim(self):
        p = classify_perspective(
            "Modular construction enables BWRX-300 build time of 3–4 years from first concrete.",
            "smr_construction_timeline.txt",
        )
        assert p == "construction"

    def test_grid_integration_claim(self):
        p = classify_perspective(
            "The BWRX-300 can load follow between 40% and 100% output for grid integration.",
            "smr_grid_services.txt",
        )
        assert p == "grid_integration"

    def test_public_acceptance_claim(self):
        p = classify_perspective(
            "Community consultation and public acceptance are critical for siting approval.",
            "smr_public_engagement.txt",
        )
        assert p == "public_acceptance"

    def test_returns_general_for_no_match(self):
        p = classify_perspective(
            "The document describes various aspects of the project.",
            "generic_report.txt",
        )
        assert p == "general"

    def test_case_insensitive(self):
        p = classify_perspective(
            "COOLING via CDU UNITS requires DIRECT LIQUID COOLING infrastructure.",
            "nvidia_cooling.txt",
        )
        assert p == "cooling"


# ---------------------------------------------------------------------------
# Perspective coverage (J3.2.5)
# ---------------------------------------------------------------------------

class TestComputePerspectiveCoverage:
    def _make_item(self, perspective: str, eid: str = ""):
        """Minimal dict-like object for testing."""
        from research_agent.schemas import EvidenceItem
        return EvidenceItem(
            evidence_id=eid or perspective,
            claim=f"Test claim for {perspective}",
            source_document="test.txt",
            evidence_snippet="snippet",
            category="architecture",
            relevance="relevant",
            confidence="high",
            perspective=perspective,
        )

    def test_counts_per_perspective(self):
        items = [
            self._make_item("power", "e1"),
            self._make_item("power", "e2"),
            self._make_item("cooling", "e3"),
            self._make_item("economics", "e4"),
        ]
        coverage = compute_perspective_coverage(items)
        assert coverage["power"] == 2
        assert coverage["cooling"] == 1
        assert coverage["economics"] == 1

    def test_empty_items_returns_empty(self):
        assert compute_perspective_coverage([]) == {}

    def test_general_fallback_counted(self):
        from research_agent.schemas import EvidenceItem
        item = EvidenceItem(
            evidence_id="e1",
            claim="generic claim",
            source_document="test.txt",
            evidence_snippet="snippet",
            category="architecture",
            relevance="relevant",
            confidence="medium",
            perspective="",  # no perspective set
        )
        coverage = compute_perspective_coverage([item])
        assert coverage.get("general", 0) == 1


# ---------------------------------------------------------------------------
# Diversity score (J3.2.6)
# ---------------------------------------------------------------------------

class TestComputeDiversityScore:
    def _make_item(self, perspective: str):
        from research_agent.schemas import EvidenceItem
        return EvidenceItem(
            evidence_id=perspective,
            claim=f"claim for {perspective}",
            source_document="nvidia_test.txt",
            evidence_snippet="snippet",
            category="architecture",
            relevance="relevant",
            confidence="high",
            perspective=perspective,
        )

    def test_zero_for_empty_list(self):
        assert compute_diversity_score([], "ai_dc") == 0.0

    def test_full_coverage_scores_one(self):
        all_perspectives = list(PERSPECTIVES_AI_DC.keys())
        items = [self._make_item(p) for p in all_perspectives]
        score = compute_diversity_score(items, "ai_dc")
        assert score == 1.0

    def test_partial_coverage_scores_fraction(self):
        # 3 of 7 AI DC perspectives covered
        items = [self._make_item(p) for p in ["power", "cooling", "networking"]]
        score = compute_diversity_score(items, "ai_dc")
        expected = round(3 / len(PERSPECTIVES_AI_DC), 4)
        assert score == expected

    def test_general_excluded_from_numerator(self):
        # All items are "general" — no real perspectives → score = 0
        from research_agent.schemas import EvidenceItem
        items = [
            EvidenceItem(
                evidence_id="e1",
                claim="generic",
                source_document="test.txt",
                evidence_snippet="s",
                category="architecture",
                relevance="r",
                confidence="low",
                perspective="general",
            )
        ]
        assert compute_diversity_score(items, "ai_dc") == 0.0


# ---------------------------------------------------------------------------
# Diversity metrics dict (J3.2.6)
# ---------------------------------------------------------------------------

class TestBuildDiversityMetrics:
    def _make_items(self, perspectives: list[str]):
        from research_agent.schemas import EvidenceItem
        return [
            EvidenceItem(
                evidence_id=f"e{i}",
                claim=f"claim {p}",
                source_document="nvidia_test.txt",
                evidence_snippet="snippet",
                category="architecture",
                relevance="relevant",
                confidence="high",
                perspective=p,
            )
            for i, p in enumerate(perspectives)
        ]

    def test_structure(self):
        items = self._make_items(["power", "cooling", "economics", "power"])
        metrics = build_diversity_metrics(items, "ai_dc")
        assert "unique_perspectives" in metrics
        assert "evidence_items" in metrics
        assert "diversity_score" in metrics
        assert "perspective_coverage" in metrics
        assert "perspectives_found" in metrics

    def test_unique_count(self):
        items = self._make_items(["power", "cooling", "economics", "power"])
        metrics = build_diversity_metrics(items, "ai_dc")
        assert metrics["unique_perspectives"] == 3  # power, cooling, economics
        assert metrics["evidence_items"] == 4

    def test_perspectives_found_sorted(self):
        items = self._make_items(["networking", "cooling", "power"])
        metrics = build_diversity_metrics(items, "ai_dc")
        assert metrics["perspectives_found"] == sorted(["cooling", "networking", "power"])


# ---------------------------------------------------------------------------
# Diversity-aware evidence selection (J3.2.3-4)
# ---------------------------------------------------------------------------

class TestSelectDiverseEvidence:
    def _make_item(self, perspective: str, eid: str, score: float = 3.0):
        from research_agent.schemas import EvidenceItem
        return EvidenceItem(
            evidence_id=eid,
            claim=f"claim {eid}",
            source_document="test.txt",
            evidence_snippet="snippet",
            category="architecture",
            relevance="relevant",
            confidence="high",
            perspective=perspective,
            overall_score=score,
        )

    def test_no_truncation_when_under_limit(self):
        items = [self._make_item("power", f"e{i}", max(1.0, 5.0 - i)) for i in range(5)]
        selected = select_diverse_evidence(items, top_n=10)
        assert len(selected) == 5

    def test_caps_overrepresented_perspective_when_alternatives_exist(self):
        # 8 'power' + 2 'cooling'; cap=3, top_n=5
        # Pass 1: 1 power, 1 cooling (seed both)
        # Pass 2: 2 more power (reaches cap 3), 1 more cooling
        # Result: 3 power + 2 cooling = 5 total; power capped at 3
        items = (
            [self._make_item("power", f"pw{i}", max(1.0, 5.0 - i * 0.1)) for i in range(8)]
            + [self._make_item("cooling", f"cl{i}", max(1.0, 3.0 - i * 0.1)) for i in range(2)]
        )
        selected = select_diverse_evidence(items, top_n=5, max_per_perspective=3)
        power_count = sum(1 for item in selected if item.perspective == "power")
        cooling_count = sum(1 for item in selected if item.perspective == "cooling")
        assert power_count == 3
        assert cooling_count == 2
        assert len(selected) == 5

    def test_diverse_items_all_represented(self):
        perspectives = ["power", "cooling", "networking", "economics", "deployment"]
        items = []
        for p in perspectives:
            for j in range(5):
                items.append(self._make_item(p, f"{p}_{j}", 3.0))
        selected = select_diverse_evidence(items, top_n=20, max_per_perspective=4)
        seen_perspectives = {item.perspective for item in selected}
        assert seen_perspectives == set(perspectives)

    def test_no_duplicates_in_selection(self):
        items = [self._make_item("power", f"e{i}", max(1.0, 5.0 - i)) for i in range(20)]
        selected = select_diverse_evidence(items, top_n=10, max_per_perspective=10)
        ids = [item.evidence_id for item in selected]
        assert len(ids) == len(set(ids))

    def test_overflow_fills_remaining_slots(self):
        # 3 perspectives × 2 items each = 6, cap=2, top_n=8 → all 6 + 2 overflow
        items = []
        for p in ["power", "cooling", "networking"]:
            for j in range(4):
                items.append(self._make_item(p, f"{p}_{j}", 3.0))
        selected = select_diverse_evidence(items, top_n=10, max_per_perspective=2)
        assert len(selected) == 10

    def test_single_perspective_falls_back_to_overflow(self):
        items = [self._make_item("power", f"e{i}", max(1.0, 5.0 - i)) for i in range(10)]
        selected = select_diverse_evidence(items, top_n=5, max_per_perspective=2)
        # Pass 1: 1 item. Pass 2: 1 more. Pass 3: 3 more (overflow ignores cap).
        assert len(selected) == 5


# ---------------------------------------------------------------------------
# EvidenceItem.perspective via enricher (J3.2.2 integration)
# ---------------------------------------------------------------------------

class TestEnricherIntegration:
    def test_enricher_sets_perspective(self):
        from research_agent.evidence_enricher import enrich_evidence_with_metadata
        from research_agent.schemas import EvidenceItem

        item = EvidenceItem(
            evidence_id="E001",
            claim="NVLink provides 900 GB/s chip-to-chip bandwidth.",
            source_document="nvidia_nvlink.txt",
            evidence_snippet="NVLink provides 900 GB/s bandwidth between Grace and Blackwell.",
            category="networking",
            relevance="relevant",
            confidence="high",
        )
        enriched = enrich_evidence_with_metadata([item])
        assert len(enriched) == 1
        assert enriched[0].perspective == "networking"

    def test_enricher_sets_smr_perspective(self):
        from research_agent.evidence_enricher import enrich_evidence_with_metadata
        from research_agent.schemas import EvidenceItem

        item = EvidenceItem(
            evidence_id="E002",
            claim="NRC design certification for BWRX-300 requires regulatory review.",
            source_document="bwrx_300_nrc_filing.txt",
            evidence_snippet="The NRC will complete design certification review by 2027.",
            category="operations",
            relevance="relevant",
            confidence="high",
        )
        enriched = enrich_evidence_with_metadata([item])
        assert enriched[0].perspective == "licensing"

    def test_enricher_preserves_existing_perspective(self):
        from research_agent.evidence_enricher import enrich_evidence_with_metadata
        from research_agent.schemas import EvidenceItem

        item = EvidenceItem(
            evidence_id="E003",
            claim="This is a pre-classified claim.",
            source_document="nvidia_test.txt",
            evidence_snippet="snippet",
            category="power",
            relevance="relevant",
            confidence="high",
            perspective="already_set",
        )
        enriched = enrich_evidence_with_metadata([item])
        assert enriched[0].perspective == "already_set"


# ---------------------------------------------------------------------------
# Scorer integration: retrieval_diversity in QAScore (J3.2.8)
# ---------------------------------------------------------------------------

class TestScorerDiversityIntegration:
    def test_retrieval_diversity_in_score(self):
        from research_agent.evaluation.scorer import score_qa_response
        from research_agent.evaluation.loader import QAQuestion
        from research_agent.schemas import ResearchMemo

        question = QAQuestion(
            question_id="DIV_001",
            domain="nvidia",
            difficulty="easy",
            question="What is the power draw of the NVL72?",
            must_include=["120 kW"],
            must_not_include=[],
            acceptable_alternatives=[],
        )
        memo = ResearchMemo(
            title="Power",
            question=question.question,
            executive_summary="The GB200 NVL72 rack draws 120 kW at full load.",
            metadata={
                "retrieval_diversity": {
                    "unique_perspectives": 4,
                    "evidence_items": 12,
                    "diversity_score": 0.5714,
                    "perspective_coverage": {"power": 5, "cooling": 3, "networking": 2, "economics": 2},
                    "perspectives_found": ["cooling", "economics", "networking", "power"],
                }
            },
        )
        score = score_qa_response(question, memo)
        assert score.retrieval_diversity["unique_perspectives"] == 4
        assert score.retrieval_diversity["diversity_score"] == pytest.approx(0.5714)
        assert "perspective_coverage" in score.retrieval_diversity

    def test_missing_diversity_metadata_returns_empty_dict(self):
        from research_agent.evaluation.scorer import score_qa_response
        from research_agent.evaluation.loader import QAQuestion
        from research_agent.schemas import ResearchMemo

        question = QAQuestion(
            question_id="DIV_002",
            domain="nvidia",
            difficulty="easy",
            question="GPU count?",
            must_include=["72"],
            must_not_include=[],
            acceptable_alternatives=[],
        )
        memo = ResearchMemo(
            title="GPUs",
            question=question.question,
            executive_summary="The NVL72 contains 72 B200 GPUs.",
            metadata={},  # no retrieval_diversity key
        )
        score = score_qa_response(question, memo)
        assert score.retrieval_diversity == {}
