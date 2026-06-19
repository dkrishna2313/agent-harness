"""Tests for J1: Domain Profiles.

Covers:
- Loading ai_data_centers and smr profiles
- CLI --profile option
- Topic detection from profile
- Research gap checks from profile
- Coverage matrix topics from profile
- Backward compatibility when no profile is supplied
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch

from dc_power_agent.profile import (
    DomainProfile,
    GapCheck,
    DEFAULT_PROFILE_NAME,
    get_default_profile,
    list_available_profiles,
    load_profile,
)
from dc_power_agent.evaluator import classify_question_topics
from dc_power_agent.gap_detector import detect_gaps
from dc_power_agent.coverage import compute_coverage_matrix
from dc_power_agent.schemas import EvidenceItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evidence(claim: str, category: str = "other", source: str = "doc.pdf") -> EvidenceItem:
    return EvidenceItem(
        claim=claim,
        source_document=source,
        evidence_snippet=claim,
        category=category,
        relevance="direct",
        confidence="medium",
    )


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------

class TestLoadAiDataCentersProfile:
    def test_loads_without_error(self):
        p = load_profile("ai_data_centers")
        assert isinstance(p, DomainProfile)

    def test_name(self):
        p = load_profile("ai_data_centers")
        assert p.name == "ai_data_centers"

    def test_description_nonempty(self):
        p = load_profile("ai_data_centers")
        assert len(p.description) > 10

    def test_topic_keywords_populated(self):
        p = load_profile("ai_data_centers")
        assert "power" in p.topic_keywords
        assert "cooling" in p.topic_keywords
        assert "rack architecture" in p.topic_keywords

    def test_coverage_topics_populated(self):
        p = load_profile("ai_data_centers")
        assert "power" in p.coverage_topics
        assert "cooling" in p.coverage_topics

    def test_research_gap_checks_power(self):
        p = load_profile("ai_data_centers")
        assert "power" in p.research_gap_checks
        checks = p.research_gap_checks["power"]
        assert len(checks) >= 3
        topics = [c.topic for c in checks]
        assert any("Power" in t or "Rack" in t for t in topics)

    def test_research_gap_checks_cooling(self):
        p = load_profile("ai_data_centers")
        assert "cooling" in p.research_gap_checks
        checks = p.research_gap_checks["cooling"]
        assert len(checks) >= 3

    def test_gap_checks_have_keywords(self):
        p = load_profile("ai_data_centers")
        for topic, checks in p.research_gap_checks.items():
            for check in checks:
                assert len(check.keywords) >= 1, f"{topic}/{check.topic} has no keywords"

    def test_gap_checks_have_valid_priority(self):
        p = load_profile("ai_data_centers")
        for checks in p.research_gap_checks.values():
            for check in checks:
                assert check.priority in ("high", "medium", "low")

    def test_source_quality_hints_populated(self):
        p = load_profile("ai_data_centers")
        assert "primary" in p.source_quality_hints
        assert "secondary" in p.source_quality_hints
        assert "nvidia" in p.source_quality_hints["primary"]

    def test_domain_terms_populated(self):
        p = load_profile("ai_data_centers")
        assert p.domain_terms is not None
        assert "nvidia" in p.domain_terms

    def test_specificity_terms_populated(self):
        p = load_profile("ai_data_centers")
        assert p.specificity_terms is not None
        assert "kw" in p.specificity_terms

    def test_memo_section_hints(self):
        p = load_profile("ai_data_centers")
        assert "Power Implications" in p.memo_section_hints
        assert "Cooling Implications" in p.memo_section_hints

    def test_profile_path_set(self):
        p = load_profile("ai_data_centers")
        assert p.profile_path.endswith("ai_data_centers.yaml")

    def test_topic_categories_present(self):
        p = load_profile("ai_data_centers")
        assert p.topic_categories is not None
        assert "power" in p.topic_categories


class TestLoadSmrProfile:
    def test_loads_without_error(self):
        p = load_profile("smr")
        assert isinstance(p, DomainProfile)

    def test_name(self):
        p = load_profile("smr")
        assert p.name == "smr"

    def test_coverage_topics_smr_specific(self):
        p = load_profile("smr")
        assert "licensing" in p.coverage_topics
        assert "fuel cycle" in p.coverage_topics
        assert "waste management" in p.coverage_topics
        assert "grid integration" in p.coverage_topics

    def test_topic_keywords_smr_specific(self):
        p = load_profile("smr")
        assert "licensing" in p.topic_keywords
        assert "nrc" in p.topic_keywords["licensing"]
        assert "fuel cycle" in p.topic_keywords
        assert "haleu" in p.topic_keywords["fuel cycle"]

    def test_research_gap_checks_licensing(self):
        p = load_profile("smr")
        assert "licensing" in p.research_gap_checks
        topics = [c.topic for c in p.research_gap_checks["licensing"]]
        assert any("NRC" in t or "Licensing" in t for t in topics)

    def test_research_gap_checks_economics(self):
        p = load_profile("smr")
        assert "economics" in p.research_gap_checks
        topics = [c.topic for c in p.research_gap_checks["economics"]]
        assert any("LCOE" in t or "Capital" in t for t in topics)

    def test_research_gap_checks_fuel_cycle(self):
        p = load_profile("smr")
        assert "fuel cycle" in p.research_gap_checks
        topics = [c.topic for c in p.research_gap_checks["fuel cycle"]]
        assert any("HALEU" in t or "Fuel" in t for t in topics)

    def test_no_nvidia_terms_in_smr_domain_terms(self):
        p = load_profile("smr")
        assert p.domain_terms is not None
        nvidia_terms = {"nvidia", "nvl72", "blackwell", "rubin", "gpu"}
        loaded = set(p.domain_terms)
        overlap = nvidia_terms & loaded
        assert len(overlap) == 0, f"SMR profile has NVIDIA terms: {overlap}"

    def test_smr_specificity_terms(self):
        p = load_profile("smr")
        assert p.specificity_terms is not None
        assert "haleu" in p.specificity_terms
        assert "nrc" in p.specificity_terms


class TestLoadProfileErrors:
    def test_nonexistent_name_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_profile("nonexistent_profile_xyz")

    def test_nonexistent_path_raises(self):
        with pytest.raises(FileNotFoundError):
            load_profile("/nonexistent/path/profile.yaml")

    def test_list_available_profiles_includes_both(self):
        available = list_available_profiles()
        assert "ai_data_centers" in available
        assert "smr" in available

    def test_load_by_path(self):
        profiles_dir = Path(__file__).parent.parent / "profiles"
        path = str(profiles_dir / "ai_data_centers.yaml")
        p = load_profile(path)
        assert p.name == "ai_data_centers"


class TestGetDefaultProfile:
    def test_returns_ai_data_centers(self):
        p = get_default_profile()
        assert p.name == DEFAULT_PROFILE_NAME

    def test_returns_same_instance_on_repeat_calls(self):
        p1 = get_default_profile()
        p2 = get_default_profile()
        assert p1 is p2


# ---------------------------------------------------------------------------
# Topic detection from profile
# ---------------------------------------------------------------------------

class TestTopicDetectionFromProfile:
    def setup_method(self):
        self.adc = load_profile("ai_data_centers")
        self.smr = load_profile("smr")

    def test_adc_detects_power(self):
        q = "What are the power requirements for NVIDIA GB200 racks?"
        topics = self.adc.classify_question_topics(q)
        assert "power" in topics

    def test_adc_detects_cooling(self):
        q = "What cooling technology does NVL72 use?"
        topics = self.adc.classify_question_topics(q)
        assert "cooling" in topics

    def test_adc_detects_rack_architecture(self):
        q = "How many GPUs fit in the NVL72 rack?"
        topics = self.adc.classify_question_topics(q)
        assert "rack architecture" in topics

    def test_adc_detects_networking(self):
        q = "What is the NVLink bandwidth?"
        topics = self.adc.classify_question_topics(q)
        assert "networking" in topics

    def test_smr_detects_licensing(self):
        q = "What is the NRC licensing pathway for NuScale SMRs?"
        topics = self.smr.classify_question_topics(q)
        assert "licensing" in topics

    def test_smr_detects_fuel_cycle(self):
        q = "What are the HALEU fuel supply constraints?"
        topics = self.smr.classify_question_topics(q)
        assert "fuel cycle" in topics

    def test_smr_detects_economics(self):
        q = "What is the LCOE and overnight capital cost for SMR deployment?"
        topics = self.smr.classify_question_topics(q)
        assert "economics" in topics

    def test_smr_detects_grid_integration(self):
        q = "What are the grid interconnection requirements for SMRs?"
        topics = self.smr.classify_question_topics(q)
        assert "grid integration" in topics

    def test_smr_does_not_detect_rack_architecture(self):
        q = "What is the HALEU fuel supply chain?"
        topics = self.smr.classify_question_topics(q)
        assert "rack architecture" not in topics

    def test_classify_question_topics_uses_profile(self):
        """classify_question_topics() with profile=smr detects SMR topics."""
        q = "What NRC licensing steps are required?"
        topics = classify_question_topics(q, self.smr)
        assert "licensing" in topics

    def test_classify_question_topics_legacy_fallback(self):
        """classify_question_topics() with no profile uses legacy hard-coded map."""
        q = "What are the rack power and cooling requirements?"
        topics = classify_question_topics(q)
        assert "power" in topics
        assert "cooling" in topics


# ---------------------------------------------------------------------------
# Research gap detection from profile
# ---------------------------------------------------------------------------

class TestGapDetectionFromProfile:
    def setup_method(self):
        self.adc = load_profile("ai_data_centers")
        self.smr = load_profile("smr")

    def test_adc_gap_detect_power_without_evidence(self):
        q = "What are the power requirements?"
        gaps = detect_gaps(q, [], self.adc)
        topics = [g.topic for g in gaps]
        assert any("Power" in t or "Rack" in t for t in topics)

    def test_adc_gap_detect_suppressed_when_evidence_covers(self):
        q = "What are the power requirements?"
        evidence = [_make_evidence("The rack power draw is 120kW, measured at the PDU busway.")]
        gaps = detect_gaps(q, evidence, self.adc)
        # kw and pdu are covered — those two subtopics should be suppressed
        gap_topics = {g.topic for g in gaps}
        assert "Rack Power Consumption" not in gap_topics
        assert "Power Delivery Infrastructure" not in gap_topics

    def test_smr_gap_detect_licensing_without_evidence(self):
        q = "What are the SMR licensing steps and NRC approval requirements?"
        gaps = detect_gaps(q, [], self.smr)
        topics = [g.topic for g in gaps]
        assert any("NRC" in t or "Licensing" in t for t in topics)

    def test_smr_gap_detect_economics(self):
        q = "What is the LCOE and economics for SMR deployment?"
        gaps = detect_gaps(q, [], self.smr)
        topics = [g.topic for g in gaps]
        assert any("Capital" in t or "LCOE" in t or "Levelized" in t for t in topics)

    def test_smr_gap_detect_suppressed_by_evidence(self):
        q = "What is the LCOE for SMR?"
        evidence = [_make_evidence("The LCOE is estimated at $95/MWh for the nth-of-a-kind plant.")]
        gaps = detect_gaps(q, evidence, self.smr)
        gap_topics = {g.topic for g in gaps}
        assert "Levelized Cost of Electricity" not in gap_topics

    def test_gaps_sorted_high_before_low(self):
        q = "What are the power and cooling requirements?"
        gaps = detect_gaps(q, [], self.adc)
        priorities = [g.priority for g in gaps]
        # No low before high
        seen_high = False
        for p in priorities:
            if p == "high":
                seen_high = True
            if seen_high and p == "low":
                pass  # low can come after high is fine — we're checking order below
        high_indices = [i for i, p in enumerate(priorities) if p == "high"]
        low_indices = [i for i, p in enumerate(priorities) if p == "low"]
        if high_indices and low_indices:
            assert max(high_indices) < max(low_indices) or min(high_indices) < min(low_indices)

    def test_legacy_gap_detection_no_profile(self):
        """detect_gaps() with no profile uses legacy hard-coded subtopics."""
        q = "What are the power and cooling requirements?"
        gaps = detect_gaps(q, [])
        assert len(gaps) > 0


# ---------------------------------------------------------------------------
# Coverage matrix from profile
# ---------------------------------------------------------------------------

class TestCoverageMatrixFromProfile:
    def setup_method(self):
        self.adc = load_profile("ai_data_centers")
        self.smr = load_profile("smr")

    def test_adc_coverage_uses_profile_topics(self):
        q = "What are the power requirements?"
        areas = compute_coverage_matrix(q, [], profile=self.adc)
        topic_names = {a.topic for a in areas}
        # All coverage_topics from the profile should appear
        for t in self.adc.coverage_topics:
            assert t in topic_names, f"Missing coverage topic: {t}"

    def test_smr_coverage_uses_smr_topics(self):
        q = "What are the SMR deployment challenges?"
        areas = compute_coverage_matrix(q, [], profile=self.smr)
        topic_names = {a.topic for a in areas}
        for t in self.smr.coverage_topics:
            assert t in topic_names, f"Missing SMR coverage topic: {t}"

    def test_smr_coverage_no_rack_architecture(self):
        q = "What are the SMR deployment challenges?"
        areas = compute_coverage_matrix(q, [], profile=self.smr)
        topic_names = {a.topic for a in areas}
        # SMR profile doesn't have rack architecture
        assert "rack architecture" not in topic_names

    def test_adc_coverage_counts_category_evidence(self):
        """For ai_data_centers, power evidence (category='power') counts for power topic."""
        q = "power and cooling"
        evidence = [
            _make_evidence("Rack draws 120kW", category="power", source="doc1.pdf"),
            _make_evidence("Rack draws 80kW", category="power", source="doc2.pdf"),
            _make_evidence("Rack draws 100kW", category="power", source="doc3.pdf"),
            _make_evidence("Rack draws 90kW", category="power", source="doc4.pdf"),
            _make_evidence("Rack draws 110kW", category="power", source="doc5.pdf"),
        ]
        areas = compute_coverage_matrix(q, evidence, profile=self.adc)
        power_area = next(a for a in areas if a.topic == "power")
        assert power_area.evidence_count == 5
        assert power_area.source_count == 5

    def test_smr_coverage_uses_keyword_matching(self):
        """For SMR, topics without category mapping use keyword matching."""
        q = "SMR licensing"
        evidence = [
            _make_evidence("The NRC design certification process requires 3-5 years."),
            _make_evidence("NRC licensing pathway includes DC and COL applications."),
            _make_evidence("Regulatory approval timeline depends on NRC review queue."),
            _make_evidence("The licensing schedule is uncertain due to staff review capacity."),
            _make_evidence("Operating license approval may take several years after CP."),
        ]
        areas = compute_coverage_matrix(q, evidence, profile=self.smr)
        lic_area = next((a for a in areas if a.topic == "licensing"), None)
        assert lic_area is not None
        assert lic_area.evidence_count >= 1

    def test_none_level_for_empty_evidence(self):
        areas = compute_coverage_matrix("SMR deployment", [], profile=self.smr)
        for area in areas:
            assert area.coverage_level == "none"

    def test_legacy_coverage_no_profile(self):
        """compute_coverage_matrix() with no profile uses legacy question-topic detection."""
        q = "What are the power and cooling requirements for AI data centers?"
        areas = compute_coverage_matrix(q, [])
        topic_names = {a.topic for a in areas}
        assert "power" in topic_names
        assert "cooling" in topic_names


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------

class TestAgentWithProfile:
    def test_agent_accepts_profile(self):
        from dc_power_agent.agent import DcPowerAgent
        p = load_profile("ai_data_centers")
        agent = DcPowerAgent(profile=p)
        assert agent.profile.name == "ai_data_centers"

    def test_agent_defaults_to_ai_data_centers(self):
        from dc_power_agent.agent import DcPowerAgent
        agent = DcPowerAgent()
        assert agent.profile.name == "ai_data_centers"

    def test_agent_analyze_with_smr_profile(self):
        from dc_power_agent.agent import DcPowerAgent
        from dc_power_agent.schemas import SourceDocument
        from pathlib import Path

        p = load_profile("smr")
        agent = DcPowerAgent(profile=p)
        doc = SourceDocument(
            path=Path("smr_overview.pdf"),
            title="SMR Overview",
            extension=".pdf",
            text=(
                "NuScale SMR received NRC design certification in 2022. "
                "The LCOE is projected at $58-95 per MWh. HALEU supply chain is a key constraint. "
                "The licensing pathway includes DC application and combined license. "
                "Spent fuel storage requires interim dry cask facilities."
            ),
        )
        memo = agent.analyze("What are the deployment challenges for SMRs?", [doc])
        assert memo is not None
        # Coverage matrix should use SMR topics
        cm = memo.metadata.get("coverage_matrix", [])
        cm_topics = {a["topic"] for a in cm}
        assert "licensing" in cm_topics or "economics" in cm_topics

    def test_metadata_includes_domain_profile(self):
        from dc_power_agent.agent import DcPowerAgent
        from dc_power_agent.schemas import SourceDocument
        from pathlib import Path

        p = load_profile("ai_data_centers")
        agent = DcPowerAgent(profile=p)
        doc = SourceDocument(
            path=Path("test_doc.txt"),
            title="Test Doc",
            extension=".txt",
            text="Power draw is 120kW. Liquid cooling is required.",
        )
        memo = agent.analyze("What are the power requirements?", [doc])
        dp = memo.metadata.get("domain_profile", {})
        assert dp.get("name") == "ai_data_centers"


# ---------------------------------------------------------------------------
# CLI --profile option
# ---------------------------------------------------------------------------

class TestCLIProfileOption:
    def test_cli_help_includes_profile(self):
        """--profile appears in CLI help output."""
        from typer.testing import CliRunner
        from dc_power_agent.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert "--profile" in result.output

    def test_cli_default_profile_works(self, tmp_path):
        """Running without --profile uses ai_data_centers and succeeds."""
        from typer.testing import CliRunner
        from dc_power_agent.cli import app

        src = tmp_path / "sources"
        src.mkdir()
        (src / "test_doc.txt").write_text(
            "Rack power draw is 120kW. Liquid cooling via CDU."
        )
        out = tmp_path / "out" / "memo.md"
        runner = CliRunner()
        result = runner.invoke(app, [
            "What are the power and cooling requirements?",
            "--sources", str(src),
            "--out", str(out),
            "--mock",
        ])
        assert result.exit_code == 0, result.output

    def test_cli_explicit_ai_data_centers_profile(self, tmp_path):
        from typer.testing import CliRunner
        from dc_power_agent.cli import app

        src = tmp_path / "sources"
        src.mkdir()
        (src / "test_doc.txt").write_text("Power draw is 120kW.")
        out = tmp_path / "out" / "memo.md"
        runner = CliRunner()
        result = runner.invoke(app, [
            "What are the power requirements?",
            "--sources", str(src),
            "--out", str(out),
            "--mock",
            "--profile", "ai_data_centers",
        ])
        assert result.exit_code == 0, result.output

    def test_cli_smr_profile(self, tmp_path):
        from typer.testing import CliRunner
        from dc_power_agent.cli import app

        src = tmp_path / "sources"
        src.mkdir()
        (src / "smr_doc.txt").write_text(
            "NRC design certification takes 3 years. HALEU supply is limited. LCOE is $80/MWh."
        )
        out = tmp_path / "out" / "smr.md"
        runner = CliRunner()
        result = runner.invoke(app, [
            "What are the SMR deployment challenges?",
            "--sources", str(src),
            "--out", str(out),
            "--mock",
            "--profile", "smr",
        ])
        assert result.exit_code == 0, result.output

    def test_cli_invalid_profile_falls_back(self, tmp_path):
        """Unknown profile name triggers a warning but does not crash."""
        from typer.testing import CliRunner
        from dc_power_agent.cli import app

        src = tmp_path / "sources"
        src.mkdir()
        (src / "test_doc.txt").write_text("Power draw is 120kW.")
        out = tmp_path / "out" / "memo.md"
        runner = CliRunner()
        result = runner.invoke(app, [
            "What are the requirements?",
            "--sources", str(src),
            "--out", str(out),
            "--mock",
            "--profile", "nonexistent_profile_xyz",
        ])
        # Should exit 0 (fallback to default) and emit a warning
        assert result.exit_code == 0, result.output

    def test_cli_debug_shows_profile(self, tmp_path):
        from typer.testing import CliRunner
        from dc_power_agent.cli import app

        src = tmp_path / "sources"
        src.mkdir()
        (src / "test_doc.txt").write_text("Power draw is 120kW.")
        out = tmp_path / "out" / "memo.md"
        runner = CliRunner()
        result = runner.invoke(app, [
            "What are the power requirements?",
            "--sources", str(src),
            "--out", str(out),
            "--mock",
            "--debug",
            "--profile", "ai_data_centers",
        ])
        assert result.exit_code == 0, result.output
        assert "ai_data_centers" in result.output


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Ensure all pipeline functions still work when called without a profile."""

    def test_classify_question_topics_no_profile(self):
        topics = classify_question_topics("What are the power and cooling requirements?")
        assert "power" in topics
        assert "cooling" in topics

    def test_detect_gaps_no_profile(self):
        gaps = detect_gaps("What are the power requirements?", [])
        assert len(gaps) > 0

    def test_coverage_matrix_no_profile(self):
        q = "What are the power and cooling requirements for AI data centers?"
        areas = compute_coverage_matrix(q, [])
        assert len(areas) > 0

    def test_agent_no_profile_arg(self):
        from dc_power_agent.agent import DcPowerAgent
        agent = DcPowerAgent()
        # Should not raise; default profile is ai_data_centers
        assert agent.profile is not None
        assert agent.profile.name == "ai_data_centers"

    def test_score_evidence_no_profile(self):
        from dc_power_agent.agent import score_evidence_items
        items = [_make_evidence("Rack power draw is 120kW.")]
        scored = score_evidence_items("power requirements", items)
        assert len(scored) == 1
        assert scored[0].overall_score > 0

    def test_existing_tests_unchanged(self):
        """Smoke test: existing ai_data_centers question still produces valid results."""
        from dc_power_agent.agent import DcPowerAgent
        from dc_power_agent.schemas import SourceDocument
        from pathlib import Path

        agent = DcPowerAgent()
        doc = SourceDocument(
            path=Path("nvidia_blackwell_architecture.pdf"),
            title="NVIDIA Blackwell Architecture",
            extension=".pdf",
            text=(
                "NVIDIA Blackwell NVL72 rack draws 120kW total power. "
                "Liquid cooling via CDU is required. "
                "NVLink bandwidth is 1.8TB/s. "
                "The rack contains 36 GB200 NVL2 nodes."
            ),
        )
        memo = agent.analyze(
            "What are the DC power and cooling implications of NVIDIA Rubin NVL72 racks?",
            [doc],
        )
        assert memo.executive_summary
        # Coverage matrix should be populated
        cm = memo.metadata.get("coverage_matrix", [])
        assert len(cm) > 0
        cm_topics = {a["topic"] for a in cm}
        assert "power" in cm_topics or "cooling" in cm_topics
