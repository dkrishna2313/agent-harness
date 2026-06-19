"""Tests for dc_power_agent.evidence_enricher (J3.1)."""

from __future__ import annotations

import pytest

from dc_power_agent.evidence_enricher import (
    classify_evidence_type,
    tag_evidence_topics,
    enrich_evidence_with_metadata,
    build_evidence_density_stats,
)
from dc_power_agent.schemas import EvidenceItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _item(claim: str, snippet: str = "", **kwargs) -> EvidenceItem:
    return EvidenceItem(
        claim=claim,
        source_document="test.pdf",
        evidence_snippet=snippet or claim,
        category="other",
        relevance="relevant",
        confidence="medium",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# J3.1.1: classify_evidence_type
# ---------------------------------------------------------------------------

class TestClassifyEvidenceType:
    def test_metric_kw(self):
        assert classify_evidence_type("Total rack power is 120 kW.") == "metric"

    def test_metric_mw(self):
        assert classify_evidence_type("BWRX-300 outputs 300 MW of electrical power.") == "metric"

    def test_metric_percent(self):
        assert classify_evidence_type("Efficiency improved by 15%.") == "metric"

    def test_metric_gpus(self):
        assert classify_evidence_type("The rack contains 72 GPUs.") == "metric"

    def test_metric_years(self):
        # "3-4 years" is a timeline, but years as a unit → metric
        assert classify_evidence_type("Construction takes 3-4 years.") in ("metric", "timeline")

    def test_comparison(self):
        assert classify_evidence_type("SMRs have higher cost per kW compared to large reactors.") == "comparison"

    def test_comparison_whereas(self):
        assert classify_evidence_type("Large reactors benefit from scale whereas SMRs use factory production.") == "comparison"

    def test_causal_due_to(self):
        assert classify_evidence_type("HALEU shortages delay deployment due to limited domestic production.") == "causal"

    def test_causal_leads_to(self):
        assert classify_evidence_type("Factory fabrication leads to cost reduction.") == "causal"

    def test_forecast_expected(self):
        assert classify_evidence_type("SMR deployment is expected to accelerate by 2030.") == "forecast"

    def test_forecast_will(self):
        assert classify_evidence_type("Construction will begin in 2026.") == "forecast"

    def test_risk_barrier(self):
        assert classify_evidence_type("Regulatory uncertainty is a key barrier to SMR deployment.") == "risk"

    def test_risk_challenge(self):
        assert classify_evidence_type("Financing remains a major challenge.") == "risk"

    def test_constraint_depends_on(self):
        assert classify_evidence_type("Deployment depends on HALEU supply availability.") == "constraint"

    def test_constraint_limited_by(self):
        assert classify_evidence_type("Output is limited by grid capacity.") == "constraint"

    def test_timeline_schedule(self):
        # "targets" is a forecast signal; "36 months" is a metric signal — either classification is valid
        assert classify_evidence_type("The construction schedule targets 36 months.") in ("metric", "timeline", "forecast")

    def test_fact_default(self):
        # No strong signal → fact
        assert classify_evidence_type("SMR designs use passive safety systems.") == "fact"

    def test_comparison_beats_metric(self):
        # "more than" is a comparison signal even if there's also a number
        result = classify_evidence_type("SMR cost is more than 50% higher than large reactor cost.")
        assert result == "comparison"


# ---------------------------------------------------------------------------
# J3.1.3: tag_evidence_topics
# ---------------------------------------------------------------------------

class TestTagEvidenceTopics:
    def test_no_profile_returns_empty(self):
        assert tag_evidence_topics("Any text about nuclear power", None) == []

    def test_smr_economics_tagged(self):
        from dc_power_agent.profile import load_profile
        profile = load_profile("smr")
        topics = tag_evidence_topics("SMR LCOE is driven by construction cost and financing.", profile)
        assert "economics" in topics

    def test_smr_construction_tagged(self):
        from dc_power_agent.profile import load_profile
        profile = load_profile("smr")
        topics = tag_evidence_topics("Construction duration is 3-4 years for BWRX-300.", profile)
        assert "construction" in topics

    def test_smr_grid_integration_tagged(self):
        from dc_power_agent.profile import load_profile
        profile = load_profile("smr")
        topics = tag_evidence_topics("SMRs can perform load following on the grid.", profile)
        assert "grid integration" in topics

    def test_ai_dc_power_tagged(self):
        from dc_power_agent.profile import load_profile
        profile = load_profile("ai_data_centers")
        topics = tag_evidence_topics("Rack power consumption reaches 120 kW.", profile)
        assert "power" in topics

    def test_ai_dc_cooling_tagged(self):
        from dc_power_agent.profile import load_profile
        profile = load_profile("ai_data_centers")
        topics = tag_evidence_topics("Liquid cooling via CDU is required for NVL72.", profile)
        assert "cooling" in topics

    def test_multiple_topics_tagged(self):
        from dc_power_agent.profile import load_profile
        profile = load_profile("smr")
        topics = tag_evidence_topics(
            "Construction costs and licensing delays affect LCOE.", profile
        )
        # should tag at least two topics
        assert len(topics) >= 2


# ---------------------------------------------------------------------------
# J3.1 enrichment pass
# ---------------------------------------------------------------------------

class TestEnrichEvidenceWithMetadata:
    def test_sets_evidence_type(self):
        items = [_item("The NVL72 rack consumes 120 kW.")]
        enriched = enrich_evidence_with_metadata(items)
        assert enriched[0].evidence_type == "metric"

    def test_preserves_existing_evidence_type(self):
        items = [_item("Text", evidence_type="comparison")]
        enriched = enrich_evidence_with_metadata(items)
        assert enriched[0].evidence_type == "comparison"

    def test_sets_topics_with_profile(self):
        from dc_power_agent.profile import load_profile
        profile = load_profile("smr")
        items = [_item("SMR construction takes 3-4 years from first concrete to commercial operation.")]
        enriched = enrich_evidence_with_metadata(items, profile)
        assert "construction" in enriched[0].topics

    def test_empty_topics_without_profile(self):
        items = [_item("Some SMR fact.")]
        enriched = enrich_evidence_with_metadata(items, profile=None)
        assert enriched[0].topics == []

    def test_preserves_existing_topics(self):
        items = [_item("Text", topics=["pre-existing"])]
        enriched = enrich_evidence_with_metadata(items, profile=None)
        assert enriched[0].topics == ["pre-existing"]

    def test_idempotent_on_already_enriched(self):
        items = [_item("Text", evidence_type="fact", topics=["economics"])]
        once = enrich_evidence_with_metadata(items)
        twice = enrich_evidence_with_metadata(once)
        assert twice[0].evidence_type == "fact"
        assert twice[0].topics == ["economics"]

    def test_returns_same_count(self):
        items = [_item(f"Claim {i}") for i in range(5)]
        enriched = enrich_evidence_with_metadata(items)
        assert len(enriched) == 5


# ---------------------------------------------------------------------------
# J3.1.6: density metrics
# ---------------------------------------------------------------------------

class TestBuildEvidenceDensityStats:
    def _make_items(self, n: int, type_: str = "fact") -> list[EvidenceItem]:
        return [_item(f"Claim {i}", evidence_type=type_) for i in range(n)]

    def test_basic_structure(self):
        stats = build_evidence_density_stats(self._make_items(10), chunks_processed=4)
        assert "chunks_processed" in stats
        assert "evidence_items" in stats
        assert "evidence_per_chunk" in stats
        assert "evidence_type_distribution" in stats
        assert "topic_distribution" in stats

    def test_evidence_per_chunk(self):
        stats = build_evidence_density_stats(self._make_items(20), chunks_processed=5)
        assert stats["evidence_per_chunk"] == pytest.approx(4.0)

    def test_zero_chunks(self):
        stats = build_evidence_density_stats(self._make_items(5), chunks_processed=0)
        assert stats["evidence_per_chunk"] == 0.0

    def test_type_distribution(self):
        items = (
            self._make_items(3, "metric")
            + self._make_items(2, "comparison")
            + self._make_items(1, "causal")
        )
        stats = build_evidence_density_stats(items, chunks_processed=5)
        dist = stats["evidence_type_distribution"]
        assert dist["metric"] == 3
        assert dist["comparison"] == 2
        assert dist["causal"] == 1

    def test_topic_distribution(self):
        items = [
            _item("A", topics=["economics", "construction"]),
            _item("B", topics=["economics"]),
            _item("C", topics=[]),
        ]
        stats = build_evidence_density_stats(items, chunks_processed=3)
        dist = stats["topic_distribution"]
        assert dist.get("economics") == 2
        assert dist.get("construction") == 1


# ---------------------------------------------------------------------------
# J3.1 agent integration: SMR profile produces richer extraction
# ---------------------------------------------------------------------------

class TestAgentJ31Integration:
    def test_smr_extraction_produces_typed_items(self, tmp_path):
        from dc_power_agent.agent import DcPowerAgent
        from dc_power_agent.claude_client import MockClaudeClient
        from dc_power_agent.profile import load_profile
        from dc_power_agent.schemas import SourceDocument

        doc_text = (
            "HALEU supply is limited due to insufficient domestic production. "
            "Russia is no longer a viable supplier. "
            "SMR LCOE depends on construction cost and financing rates. "
            "Factory fabrication reduces on-site labour and improves quality. "
            "Economy of scale favours large reactors. "
            "Load following capability allows grid flexibility. "
            "Construction duration is expected to be 3-4 years for BWRX-300. "
            "Regulatory approval requires NRC design certification."
        )
        source_file = tmp_path / "smr_doc.txt"
        source_file.write_text(doc_text)
        profile = load_profile("smr")

        agent = DcPowerAgent(client=MockClaudeClient(), profile=profile)
        docs = [SourceDocument(path=source_file, title="SMR Doc", extension=".txt", text=doc_text)]
        memo = agent.analyze("How do SMRs compare in economics of scale?", docs)

        # All evidence items should have evidence_type set
        assert all(it.evidence_type for it in memo.evidence), (
            "Some evidence items missing evidence_type"
        )

        # Should have density stats in trace
        assert "evidence_density" in memo.metadata
        density = memo.metadata["evidence_density"]
        assert density["evidence_items"] > 0
        assert "evidence_type_distribution" in density

    def test_smr_economy_of_scale_extracted(self, tmp_path):
        from dc_power_agent.agent import DcPowerAgent
        from dc_power_agent.claude_client import MockClaudeClient
        from dc_power_agent.profile import load_profile
        from dc_power_agent.schemas import SourceDocument

        doc_text = (
            "SMRs suffer a diseconomy of scale relative to large reactors. "
            "The overnight cost per kW is higher for smaller units. "
            "However factory fabrication and serial production offer a learning rate "
            "of 10-20% cost reduction per doubling of units. "
            "Economy of scale advantages of large reactors may be offset by "
            "modular construction efficiencies in NOAK builds."
        )
        source_file = tmp_path / "economics.txt"
        source_file.write_text(doc_text)
        profile = load_profile("smr")

        agent = DcPowerAgent(client=MockClaudeClient(), profile=profile)
        docs = [SourceDocument(path=source_file, title="Economics", extension=".txt", text=doc_text)]
        memo = agent.analyze(
            "How do SMRs compare to large nuclear reactors in terms of economics of scale?",
            docs,
        )

        all_claims = " ".join(it.claim.lower() + " " + it.evidence_snippet.lower() for it in memo.evidence)
        assert "economy" in all_claims or "scale" in all_claims or "factory" in all_claims, (
            f"Expected economy/scale/factory in extracted evidence. Claims: {[it.claim for it in memo.evidence[:5]]}"
        )

    def test_smr_load_following_extracted(self, tmp_path):
        from dc_power_agent.agent import DcPowerAgent
        from dc_power_agent.claude_client import MockClaudeClient
        from dc_power_agent.profile import load_profile
        from dc_power_agent.schemas import SourceDocument

        doc_text = (
            "SMR designs include load following capability to match grid demand. "
            "Many designs can ramp output from 100% to 20% power over 30 minutes. "
            "This grid flexibility advantage is important for decarbonisation. "
            "No western commercial SMR has yet demonstrated this at scale."
        )
        source_file = tmp_path / "grid.txt"
        source_file.write_text(doc_text)
        profile = load_profile("smr")

        agent = DcPowerAgent(client=MockClaudeClient(), profile=profile)
        docs = [SourceDocument(path=source_file, title="Grid", extension=".txt", text=doc_text)]
        memo = agent.analyze(
            "What are the grid flexibility benefits of SMRs?",
            docs,
        )

        all_text = " ".join(it.claim.lower() + " " + it.evidence_snippet.lower() for it in memo.evidence)
        assert "load" in all_text or "grid" in all_text or "flexibility" in all_text, (
            "Expected load/grid/flexibility in extracted evidence"
        )
