"""Tests for H1: Source Quality Weighting."""

from __future__ import annotations

from pathlib import Path

import pytest

from dc_power_agent.source_quality import classify_source_quality, build_source_quality_map, classify_source_quality_with_profile
from dc_power_agent.schemas import EvidenceItem, SourceQuality, assign_evidence_ids
from dc_power_agent.retrieval import score_retrieval, select_top_chunks, classify_document_priority
from dc_power_agent.schemas import Chunk
from dc_power_agent.contradiction import detect_contradictions
from dc_power_agent.agent import DcPowerAgent, score_evidence_items, rank_evidence_items
from dc_power_agent.claude_client import MockClaudeClient
from dc_power_agent.trace import build_trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(text: str, name: str):
    from dc_power_agent.schemas import SourceDocument
    return SourceDocument(
        path=Path(f"sources/{name}"),
        title=name,
        extension=Path(name).suffix,
        text=text,
    )


def _chunk(text: str, doc_name: str, chunk_id: str = "c001") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_name=doc_name,
        chunk_number=1,
        text=text,
        start_offset=0,
        end_offset=len(text),
    )


def _evidence(claim: str, source: str, eid: str = "E001") -> EvidenceItem:
    return EvidenceItem(
        evidence_id=eid,
        claim=claim,
        source_document=source,
        evidence_snippet=claim,
        category="power",
        relevance="relevant",
        confidence="medium",
    )


# ---------------------------------------------------------------------------
# 1. Source classification
# ---------------------------------------------------------------------------

class TestSourceClassification:

    def test_nvidia_technical_blog_score_5(self):
        sq = classify_source_quality("Inside the NVIDIA Vera Rubin Platform.pdf")
        assert sq.source_quality_score == 5
        assert sq.source_type == "nvidia_technical"

    def test_nvidia_blackwell_architecture_score_5(self):
        sq = classify_source_quality("NVIDIA Blackwell Architecture Technical Overview.pdf")
        assert sq.source_quality_score == 5

    def test_nvidia_nvl_score_5(self):
        sq = classify_source_quality("NVIDIA NVL72 Platform Specification.pdf")
        assert sq.source_quality_score == 5

    def test_nvidia_platform_score_5(self):
        sq = classify_source_quality("NVIDIA Vera Rubin Platform Deep Dive.pdf")
        assert sq.source_quality_score == 5

    def test_nvidia_marketing_score_4(self):
        sq = classify_source_quality("NVIDIA Press Release Q3 2025.pdf")
        assert sq.source_quality_score == 4
        assert sq.source_type == "nvidia_marketing"

    def test_whitepaper_score_4(self):
        sq = classify_source_quality("Dell EMC Cooling Whitepaper.pdf")
        assert sq.source_quality_score == 4
        assert sq.source_type == "vendor_whitepaper"

    def test_solution_brief_score_4(self):
        sq = classify_source_quality("HPE Solution Brief AI Infrastructure.pdf")
        assert sq.source_quality_score == 4
        assert sq.source_type == "vendor_brief"

    def test_storagereview_score_3(self):
        sq = classify_source_quality("StorageReview.com NVIDIA GB200 Analysis.pdf")
        assert sq.source_quality_score == 3
        assert sq.source_type == "independent_technical"

    def test_independent_analysis_score_3(self):
        sq = classify_source_quality("Dissecting the Rubin NVL72 Power Architecture.pdf")
        assert sq.source_quality_score == 3

    def test_review_score_3(self):
        sq = classify_source_quality("AI Data Center Infrastructure Review 2025.pdf")
        assert sq.source_quality_score == 3

    def test_blog_score_2(self):
        sq = classify_source_quality("datacenter_blog_cooling_trends.pdf")
        assert sq.source_quality_score == 2
        assert sq.source_type == "blog"

    def test_synthetic_test_file_score_1(self):
        sq = classify_source_quality("test_power_a.txt")
        assert sq.source_quality_score == 1
        assert sq.source_type == "synthetic"

    def test_bare_txt_file_score_1(self):
        sq = classify_source_quality("random_notes.txt")
        assert sq.source_quality_score == 1
        assert sq.source_type == "synthetic"

    def test_unknown_pdf_score_2(self):
        sq = classify_source_quality("general_infrastructure_doc.pdf")
        assert sq.source_quality_score == 2
        assert sq.source_type == "unknown"

    def test_rationale_populated(self):
        sq = classify_source_quality("Inside the NVIDIA Vera Rubin Platform.pdf")
        assert sq.rationale.strip() != ""

    def test_source_document_field_preserved(self):
        name = "NVIDIA NVL72 Architecture.pdf"
        sq = classify_source_quality(name)
        assert sq.source_document == name

    def test_case_insensitive(self):
        lower = classify_source_quality("inside the nvidia vera rubin platform.pdf")
        upper = classify_source_quality("INSIDE THE NVIDIA VERA RUBIN PLATFORM.PDF")
        assert lower.source_quality_score == upper.source_quality_score


class TestBuildSourceQualityMap:

    def test_returns_all_docs(self):
        names = ["Inside the NVIDIA Vera Rubin Platform.pdf", "test_power.txt"]
        result = build_source_quality_map(names)
        assert set(result.keys()) == set(names)

    def test_values_are_source_quality(self):
        names = ["NVIDIA NVL72.pdf"]
        result = build_source_quality_map(names)
        assert isinstance(result["NVIDIA NVL72.pdf"], SourceQuality)

    def test_empty_list(self):
        assert build_source_quality_map([]) == {}


# ---------------------------------------------------------------------------
# 2. Retrieval weighting
# ---------------------------------------------------------------------------

class TestRetrievalWeighting:

    def test_nvidia_doc_has_higher_doc_priority_than_txt(self):
        nvidia_prio = classify_document_priority("Inside the NVIDIA Vera Rubin Platform.pdf")
        test_prio = classify_document_priority("test_power_a.txt")
        assert nvidia_prio > test_prio

    def test_score_retrieval_uses_quality_map(self):
        """Pre-built map should give same result as on-the-fly classification."""
        chunk = _chunk("The rack draws 132 kW at peak power.", "NVIDIA NVL72 Platform.pdf")
        q = "What is the power consumption of NVL72 racks?"
        question_terms = {"rack", "power", "kw", "nvl72"}
        detected_topics = {"power"}
        sq_map = build_source_quality_map(["NVIDIA NVL72 Platform.pdf"])

        with_map = score_retrieval(chunk, q, question_terms, detected_topics, sq_map)
        without_map = score_retrieval(chunk, q, question_terms, detected_topics, None)
        # Both should produce same score since map and on-the-fly use same classifier
        assert with_map.overall_retrieval_score == without_map.overall_retrieval_score
        assert with_map.source_quality_score == 5
        assert with_map.document_priority_score == 1.0

    def test_high_quality_source_ranks_above_low_quality_on_similar_relevance(self):
        """Two equally relevant chunks: NVIDIA doc should outrank test file."""
        text = "The rack power draw is 132 kW at peak load."
        nvidia_chunk = _chunk(text, "Inside the NVIDIA Vera Rubin Platform.pdf", "c001")
        test_chunk = _chunk(text, "test_power_a.txt", "c002")
        q = "What is the rack power draw?"
        question_terms = {"rack", "power", "kw"}
        detected_topics = {"power"}
        sq_map = build_source_quality_map([
            "Inside the NVIDIA Vera Rubin Platform.pdf",
            "test_power_a.txt",
        ])
        rs_nvidia = score_retrieval(nvidia_chunk, q, question_terms, detected_topics, sq_map)
        rs_test = score_retrieval(test_chunk, q, question_terms, detected_topics, sq_map)
        assert rs_nvidia.overall_retrieval_score > rs_test.overall_retrieval_score

    def test_source_quality_score_field_populated_in_retrieval_score(self):
        chunk = _chunk("GPU power data.", "test_data.txt")
        rs = score_retrieval(chunk, "power", {"power"}, {"power"})
        assert rs.source_quality_score == 1  # .txt → synthetic → 1

    def test_select_top_chunks_accepts_quality_map(self):
        """select_top_chunks must not raise when quality map is provided."""
        chunks = [
            _chunk("NVL72 rack draws 132 kW.", "NVIDIA NVL72.pdf", "c001"),
            _chunk("Test power data.", "test_power.txt", "c002"),
        ]
        sq_map = build_source_quality_map(["NVIDIA NVL72.pdf", "test_power.txt"])
        selected, scores = select_top_chunks(
            chunks, "What is rack power?", top_n=2, source_quality_map=sq_map
        )
        assert len(selected) == 2


# ---------------------------------------------------------------------------
# 3. Evidence ranking with quality weighting
# ---------------------------------------------------------------------------

class TestEvidenceRankingQuality:

    def test_nvidia_source_scores_higher_than_test_file(self):
        from dc_power_agent.source_quality import build_source_quality_map
        sq_map = build_source_quality_map([
            "Inside the NVIDIA Vera Rubin Platform.pdf",
            "test_power_a.txt",
        ])
        items = assign_evidence_ids([
            EvidenceItem(
                claim="The NVL72 rack draws 132 kW of power.",
                source_document="Inside the NVIDIA Vera Rubin Platform.pdf",
                evidence_snippet="The NVL72 rack draws 132 kW of power.",
                category="power", relevance="relevant", confidence="high",
            ),
            EvidenceItem(
                claim="The NVL72 rack draws 132 kW of power.",
                source_document="test_power_a.txt",
                evidence_snippet="The NVL72 rack draws 132 kW of power.",
                category="power", relevance="relevant", confidence="high",
            ),
        ])
        scored = score_evidence_items("rack power", items, sq_map)
        nvidia_item = next(i for i in scored if "NVIDIA" in i.source_document)
        test_item = next(i for i in scored if "test_power" in i.source_document)
        assert nvidia_item.source_quality_score == 5
        assert test_item.source_quality_score == 1
        assert nvidia_item.overall_score > test_item.overall_score

    def test_source_quality_class_populated(self):
        sq_map = build_source_quality_map(["NVIDIA NVL72 Architecture.pdf"])
        items = assign_evidence_ids([
            EvidenceItem(
                claim="NVL72 rack.",
                source_document="NVIDIA NVL72 Architecture.pdf",
                evidence_snippet="NVL72 rack.",
                category="rack architecture", relevance="relevant", confidence="high",
            ),
        ])
        scored = score_evidence_items("rack architecture", items, sq_map)
        assert scored[0].source_quality_class == "nvidia_technical"

    def test_source_quality_class_synthetic(self):
        items = assign_evidence_ids([
            EvidenceItem(
                claim="Test claim.",
                source_document="test_power_a.txt",
                evidence_snippet="Test claim.",
                category="power", relevance="relevant", confidence="low",
            ),
        ])
        scored = score_evidence_items("power", items)
        assert scored[0].source_quality_class == "synthetic"

    def test_rank_prefers_higher_quality_with_equal_relevance(self):
        from dc_power_agent.source_quality import build_source_quality_map
        sq_map = build_source_quality_map([
            "Inside the NVIDIA Vera Rubin Platform.pdf",
            "test_power_a.txt",
        ])
        items = assign_evidence_ids([
            EvidenceItem(
                claim="Rack power is 132 kW.",
                source_document="test_power_a.txt",
                evidence_snippet="Rack power is 132 kW.",
                category="power", relevance="relevant", confidence="medium",
            ),
            EvidenceItem(
                claim="Rack power is 132 kW.",
                source_document="Inside the NVIDIA Vera Rubin Platform.pdf",
                evidence_snippet="Rack power is 132 kW.",
                category="power", relevance="relevant", confidence="medium",
            ),
        ])
        ranked = rank_evidence_items(score_evidence_items("rack power", items, sq_map))
        assert "NVIDIA" in ranked[0].source_document


# ---------------------------------------------------------------------------
# 4. Contradiction confidence assessment
# ---------------------------------------------------------------------------

class TestContradictionConfidence:

    def _make_evidence(self, claim_a: str, source_a: str, claim_b: str, source_b: str):
        return assign_evidence_ids([
            EvidenceItem(
                claim=claim_a, source_document=source_a,
                evidence_snippet=claim_a, category="power",
                relevance="relevant", confidence="high",
            ),
            EvidenceItem(
                claim=claim_b, source_document=source_b,
                evidence_snippet=claim_b, category="power",
                relevance="relevant", confidence="high",
            ),
        ])

    def test_both_high_quality_gives_high_confidence(self):
        """Two NVIDIA docs contradicting each other → high confidence."""
        sq_map = build_source_quality_map([
            "Inside the NVIDIA Vera Rubin Platform.pdf",
            "NVIDIA NVL72 Architecture.pdf",
        ])
        items = self._make_evidence(
            "The NVL72 rack draws 120 kW.",
            "Inside the NVIDIA Vera Rubin Platform.pdf",
            "The NVL72 rack draws 200 kW.",
            "NVIDIA NVL72 Architecture.pdf",
        )
        contradictions = detect_contradictions(items, sq_map)
        assert contradictions
        c = contradictions[0]
        assert c.source_quality_a == 5
        assert c.source_quality_b == 5
        assert c.confidence == "high"

    def test_high_vs_low_quality_gives_low_confidence(self):
        """NVIDIA doc (5) vs test file (1) → low confidence contradiction."""
        sq_map = build_source_quality_map([
            "Inside the NVIDIA Vera Rubin Platform.pdf",
            "test_power_a.txt",
        ])
        items = self._make_evidence(
            "The NVL72 rack draws 120 kW.",
            "Inside the NVIDIA Vera Rubin Platform.pdf",
            "The NVL72 rack draws 200 kW.",
            "test_power_a.txt",
        )
        contradictions = detect_contradictions(items, sq_map)
        assert contradictions
        c = contradictions[0]
        assert c.confidence == "low"

    def test_medium_quality_sources_give_medium_confidence(self):
        """Two independent analyses (score 3) → medium confidence."""
        sq_map = build_source_quality_map([
            "StorageReview NVL72 Analysis.pdf",
            "AI Data Center Infrastructure Review.pdf",
        ])
        items = self._make_evidence(
            "The NVL72 rack draws 120 kW.",
            "StorageReview NVL72 Analysis.pdf",
            "The NVL72 rack draws 200 kW.",
            "AI Data Center Infrastructure Review.pdf",
        )
        contradictions = detect_contradictions(items, sq_map)
        assert contradictions
        assert contradictions[0].confidence == "medium"

    def test_no_quality_map_defaults_medium(self):
        """Without quality map, confidence defaults to medium."""
        items = self._make_evidence(
            "The NVL72 rack draws 120 kW.", "source_a.pdf",
            "The NVL72 rack draws 200 kW.", "source_b.pdf",
        )
        contradictions = detect_contradictions(items, None)
        assert contradictions
        assert contradictions[0].confidence == "medium"

    def test_quality_scores_recorded_on_contradiction(self):
        sq_map = build_source_quality_map([
            "Inside the NVIDIA Vera Rubin Platform.pdf",
            "test_power_a.txt",
        ])
        items = self._make_evidence(
            "The NVL72 rack draws 120 kW.",
            "Inside the NVIDIA Vera Rubin Platform.pdf",
            "The NVL72 rack draws 200 kW.",
            "test_power_a.txt",
        )
        contradictions = detect_contradictions(items, sq_map)
        assert contradictions
        c = contradictions[0]
        assert c.source_quality_a in (5, 1)  # order may vary
        assert c.source_quality_b in (5, 1)
        assert c.source_quality_a != c.source_quality_b


# ---------------------------------------------------------------------------
# 5. Trace output
# ---------------------------------------------------------------------------

class TestTraceSourceQuality:

    def test_trace_includes_source_quality_map(self):
        doc = _doc("The NVL72 rack draws 132 kW.", "Inside the NVIDIA Vera Rubin Platform.pdf")
        memo = DcPowerAgent(client=MockClaudeClient()).analyze(
            "What is the rack power for NVL72?", [doc]
        )
        trace = build_trace(
            question="What is the rack power for NVL72?",
            source_directory=Path("sources"),
            output_path=Path("outputs/memo.md"),
            documents=[doc],
            memo=memo,
            mock_mode=True,
        )
        assert "source_quality_map" in trace
        assert isinstance(trace["source_quality_map"], dict)

    def test_source_quality_map_values_are_ints(self):
        doc = _doc("Rack power data.", "NVIDIA NVL72 Architecture.pdf")
        memo = DcPowerAgent(client=MockClaudeClient()).analyze("rack power", [doc])
        trace = build_trace(
            question="rack power",
            source_directory=Path("sources"),
            output_path=Path("outputs/memo.md"),
            documents=[doc],
            memo=memo,
            mock_mode=True,
        )
        for name, score in trace["source_quality_map"].items():
            assert isinstance(score, int), f"{name}: expected int score, got {type(score)}"

    def test_source_quality_map_correct_score_nvidia(self):
        doc = _doc("NVL72 specs.", "Inside the NVIDIA Vera Rubin Platform.pdf")
        memo = DcPowerAgent(client=MockClaudeClient()).analyze("rack power", [doc])
        trace = build_trace(
            question="rack power",
            source_directory=Path("sources"),
            output_path=Path("outputs/memo.md"),
            documents=[doc],
            memo=memo,
            mock_mode=True,
        )
        assert trace["source_quality_map"].get("Inside the NVIDIA Vera Rubin Platform.pdf") == 5

    def test_evidence_items_have_source_quality_class(self):
        doc = _doc("NVL72 cooling.", "Inside the NVIDIA Vera Rubin Platform.pdf")
        memo = DcPowerAgent(client=MockClaudeClient()).analyze("cooling", [doc])
        evidence_items = memo.source_notes or memo.evidence
        for item in evidence_items:
            assert hasattr(item, "source_quality_class")
            assert item.source_quality_class != ""


# ---------------------------------------------------------------------------
# 6. Profile-aware source quality classification (J1.3)
# ---------------------------------------------------------------------------

class TestProfileAwareSourceQuality:
    """classify_source_quality_with_profile / build_source_quality_map(profile=...)."""

    @pytest.fixture()
    def smr_profile(self):
        from dc_power_agent.profile import load_profile
        return load_profile("smr")

    def test_doe_liftoff_scores_5(self, smr_profile):
        from dc_power_agent.source_quality import classify_source_quality_with_profile
        sq = classify_source_quality_with_profile(
            "DOE Liftoff Report Advanced Nuclear.pdf", smr_profile
        )
        assert sq.source_quality_score == 5
        assert sq.source_type == "authoritative_primary"

    def test_iaea_scores_5(self, smr_profile):
        from dc_power_agent.source_quality import classify_source_quality_with_profile
        sq = classify_source_quality_with_profile(
            "IAEA SMR Catalogue 2024.pdf", smr_profile
        )
        assert sq.source_quality_score == 5
        assert sq.source_type == "authoritative_primary"

    def test_nrc_scores_5(self, smr_profile):
        from dc_power_agent.source_quality import classify_source_quality_with_profile
        sq = classify_source_quality_with_profile(
            "NRC Licensing Guidance.pdf", smr_profile
        )
        assert sq.source_quality_score == 5
        assert sq.source_type == "authoritative_primary"

    def test_nea_scores_5(self, smr_profile):
        from dc_power_agent.source_quality import classify_source_quality_with_profile
        sq = classify_source_quality_with_profile(
            "NEA Small Modular Reactor Dashboard.pdf", smr_profile
        )
        assert sq.source_quality_score == 5
        assert sq.source_type == "authoritative_primary"

    def test_inl_scores_5(self, smr_profile):
        from dc_power_agent.source_quality import classify_source_quality_with_profile
        sq = classify_source_quality_with_profile(
            "INL Small Reactors in Microgrids.pdf", smr_profile
        )
        assert sq.source_quality_score == 5
        assert sq.source_type == "authoritative_primary"

    def test_nuscale_scores_4(self, smr_profile):
        from dc_power_agent.source_quality import classify_source_quality_with_profile
        sq = classify_source_quality_with_profile(
            "NuScale Power Module Design Overview.pdf", smr_profile
        )
        assert sq.source_quality_score == 4
        assert sq.source_type == "industry_vendor"

    def test_terrapower_scores_4(self, smr_profile):
        from dc_power_agent.source_quality import classify_source_quality_with_profile
        sq = classify_source_quality_with_profile(
            "TerraPower Natrium Reactor Overview.pdf", smr_profile
        )
        assert sq.source_quality_score == 4
        assert sq.source_type == "industry_vendor"

    def test_wna_scores_4(self, smr_profile):
        from dc_power_agent.source_quality import classify_source_quality_with_profile
        sq = classify_source_quality_with_profile(
            "World Nuclear Association SMR Report.pdf", smr_profile
        )
        assert sq.source_quality_score == 4
        assert sq.source_type == "industry_vendor"

    def test_analysis_scores_3(self, smr_profile):
        from dc_power_agent.source_quality import classify_source_quality_with_profile
        sq = classify_source_quality_with_profile(
            "Nuclear Economics Analysis 2024.pdf", smr_profile
        )
        assert sq.source_quality_score == 3
        assert sq.source_type == "independent_technical"

    def test_synthetic_test_file_scores_1(self, smr_profile):
        from dc_power_agent.source_quality import classify_source_quality_with_profile
        sq = classify_source_quality_with_profile("test_smr_data.txt", smr_profile)
        assert sq.source_quality_score == 1
        assert sq.source_type == "synthetic"

    def test_unknown_smr_doc_falls_back_to_base(self, smr_profile):
        from dc_power_agent.source_quality import classify_source_quality_with_profile
        sq = classify_source_quality_with_profile(
            "random_document_no_signals.pdf", smr_profile
        )
        # Falls back to base classifier → score 2, unknown
        assert sq.source_quality_score == 2
        assert sq.source_type == "unknown"

    def test_build_map_with_smr_profile(self, smr_profile):
        names = [
            "DOE Liftoff Report Advanced Nuclear.pdf",
            "IAEA SMR Catalogue 2024.pdf",
            "NRC Licensing Guidance.pdf",
            "NEA Small Modular Reactor Dashboard.pdf",
            "INL Small Reactors in Microgrids.pdf",
            "NuScale Power Module Design Overview.pdf",
            "Nuclear Economics Analysis 2024.pdf",
            "test_smr.txt",
        ]
        sq_map = build_source_quality_map(names, profile=smr_profile)
        assert sq_map["DOE Liftoff Report Advanced Nuclear.pdf"].source_quality_score == 5
        assert sq_map["IAEA SMR Catalogue 2024.pdf"].source_quality_score == 5
        assert sq_map["NRC Licensing Guidance.pdf"].source_quality_score == 5
        assert sq_map["NEA Small Modular Reactor Dashboard.pdf"].source_quality_score == 5
        assert sq_map["INL Small Reactors in Microgrids.pdf"].source_quality_score == 5
        assert sq_map["NuScale Power Module Design Overview.pdf"].source_quality_score == 4
        assert sq_map["Nuclear Economics Analysis 2024.pdf"].source_quality_score == 3
        assert sq_map["test_smr.txt"].source_quality_score == 1

    def test_build_map_without_profile_unchanged(self):
        """Without profile, NVIDIA docs still score correctly (backward compat)."""
        sq_map = build_source_quality_map(["Inside the NVIDIA Vera Rubin Platform.pdf"])
        assert sq_map["Inside the NVIDIA Vera Rubin Platform.pdf"].source_quality_score == 5

    def test_rationale_includes_profile_name(self, smr_profile):
        from dc_power_agent.source_quality import classify_source_quality_with_profile
        sq = classify_source_quality_with_profile(
            "DOE Liftoff Report Advanced Nuclear.pdf", smr_profile
        )
        assert "smr" in sq.rationale.lower()

    def test_case_insensitive_matching(self, smr_profile):
        from dc_power_agent.source_quality import classify_source_quality_with_profile
        lower = classify_source_quality_with_profile(
            "doe liftoff report.pdf", smr_profile
        )
        upper = classify_source_quality_with_profile(
            "DOE LIFTOFF REPORT.PDF", smr_profile
        )
        assert lower.source_quality_score == 5
        assert upper.source_quality_score == 5

    def test_synthetic_takes_priority_over_primary(self, smr_profile):
        """A test file named with 'nrc' should be synthetic, not authoritative."""
        from dc_power_agent.source_quality import classify_source_quality_with_profile
        sq = classify_source_quality_with_profile("test_nrc_fixture.txt", smr_profile)
        assert sq.source_quality_score == 1
        assert sq.source_type == "synthetic"
