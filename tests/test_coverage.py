"""Tests for H2: Coverage Matrix."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_agent.coverage import (
    compute_coverage_matrix,
    _coverage_level,
    _build_rationale,
    _avg_quality,
    STRONG_MIN_EVIDENCE,
    STRONG_MIN_SOURCES,
)
from research_agent.schemas import (
    CoverageArea,
    EvidenceItem,
    ResearchGap,
    SourceQuality,
    assign_evidence_ids,
)
from research_agent.agent import DcPowerAgent
from research_agent.claude_client import MockClaudeClient
from research_agent.markdown import memo_to_markdown
from research_agent.trace import build_trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(text: str, name: str):
    from research_agent.schemas import SourceDocument
    return SourceDocument(
        path=Path(f"sources/{name}"),
        title=name,
        extension=Path(name).suffix,
        text=text,
    )


def _ev(
    claim: str,
    source: str,
    category: str = "power",
    eid: str = "E001",
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=eid,
        claim=claim,
        source_document=source,
        evidence_snippet=claim,
        category=category,
        relevance="relevant",
        confidence="medium",
    )


def _gap(topic: str, priority: str, gid: str = "G001") -> ResearchGap:
    return ResearchGap(
        gap_id=gid,
        topic=topic,
        priority=priority,
        description=f"Missing {topic} info.",
        rationale="Needed for planning.",
    )


def _sq(name: str, score: int) -> SourceQuality:
    return SourceQuality(
        source_document=name,
        source_type="unknown",
        source_quality_score=score,
        rationale="test",
    )


# ---------------------------------------------------------------------------
# 1. Coverage level scoring rules
# ---------------------------------------------------------------------------

class TestCoverageScoringRules:
    """Tests use _coverage_level (pure level) and _build_rationale (text)."""

    # ---- level thresholds — evidence_count + source_count only -------------

    def test_no_evidence_returns_none(self):
        assert _coverage_level(evidence_count=0, source_count=0) == "none"

    def test_one_evidence_item_returns_weak(self):
        assert _coverage_level(evidence_count=1, source_count=1) == "weak"

    def test_strong_threshold_met(self):
        assert _coverage_level(
            evidence_count=STRONG_MIN_EVIDENCE,
            source_count=STRONG_MIN_SOURCES,
        ) == "strong"

    def test_strong_9_items_4_sources(self):
        """Rubin corpus scenario: 9 items / 4 sources → strong."""
        assert _coverage_level(evidence_count=9, source_count=4) == "strong"

    def test_strong_regardless_of_quality(self):
        """Quality (any value) must NOT change the level."""
        # Even with implausible low quality, strong evidence wins
        assert _coverage_level(evidence_count=STRONG_MIN_EVIDENCE, source_count=STRONG_MIN_SOURCES) == "strong"

    def test_below_strong_evidence_threshold_is_moderate(self):
        assert _coverage_level(
            evidence_count=STRONG_MIN_EVIDENCE - 1,
            source_count=STRONG_MIN_SOURCES,
        ) == "moderate"

    def test_below_strong_source_threshold_is_moderate(self):
        assert _coverage_level(
            evidence_count=STRONG_MIN_EVIDENCE,
            source_count=STRONG_MIN_SOURCES - 1,
        ) == "moderate"

    def test_two_items_single_source_moderate(self):
        assert _coverage_level(evidence_count=2, source_count=1) == "moderate"

    def test_two_items_two_sources_moderate(self):
        assert _coverage_level(evidence_count=2, source_count=2) == "moderate"

    # ---- rationale text ----------------------------------------------------

    def test_rationale_none_mentions_no_evidence(self):
        r = _build_rationale(
            topic="power", evidence_count=0, source_count=0,
            avg_quality=3.0, level="none", high_gaps=[],
        )
        assert "no evidence" in r.lower()

    def test_rationale_weak_leads_with_weak(self):
        r = _build_rationale(
            topic="cooling", evidence_count=1, source_count=1,
            avg_quality=3.0, level="weak", high_gaps=[],
        )
        assert r.startswith("Weak")

    def test_rationale_moderate_leads_with_moderate(self):
        r = _build_rationale(
            topic="networking", evidence_count=3, source_count=2,
            avg_quality=3.0, level="moderate", high_gaps=[],
        )
        assert r.startswith("Moderate")

    def test_rationale_strong_leads_with_strong(self):
        r = _build_rationale(
            topic="power", evidence_count=9, source_count=4,
            avg_quality=5.0, level="strong", high_gaps=[],
        )
        assert r.startswith("Strong")

    def test_rationale_strong_includes_counts(self):
        r = _build_rationale(
            topic="power", evidence_count=9, source_count=4,
            avg_quality=5.0, level="strong", high_gaps=[],
        )
        assert "9 evidence items" in r
        assert "4 sources" in r

    def test_rationale_mentions_high_quality(self):
        r = _build_rationale(
            topic="power", evidence_count=9, source_count=4,
            avg_quality=5.0, level="strong", high_gaps=[],
        )
        assert "high-quality" in r.lower()

    def test_rationale_gap_note_appended_for_high_gaps(self):
        from research_agent.schemas import ResearchGap
        gaps = [ResearchGap(
            gap_id="G001", topic="PDU topology", priority="high",
            description="Missing PDU info.", rationale="Needed.",
        )]
        r = _build_rationale(
            topic="power", evidence_count=9, source_count=4,
            avg_quality=5.0, level="strong", high_gaps=gaps,
        )
        assert "research gaps remain" in r.lower()
        assert "PDU topology" in r

    def test_rationale_strong_with_gap_still_starts_with_strong(self):
        """Gap note is appended; the lead sentence stays Strong."""
        from research_agent.schemas import ResearchGap
        gaps = [ResearchGap(
            gap_id="G001", topic="PDU topology", priority="high",
            description="x", rationale="y",
        )]
        r = _build_rationale(
            topic="power", evidence_count=9, source_count=4,
            avg_quality=5.0, level="strong", high_gaps=gaps,
        )
        assert r.startswith("Strong")

    def test_rationale_no_gap_note_when_no_high_gaps(self):
        r = _build_rationale(
            topic="power", evidence_count=9, source_count=4,
            avg_quality=5.0, level="strong", high_gaps=[],
        )
        assert "gap" not in r.lower()

    def test_level_never_influenced_by_quality(self):
        """_coverage_level has no quality parameter — quality cannot gate any level."""
        import inspect
        sig = inspect.signature(_coverage_level)
        assert "avg_quality" not in sig.parameters
        assert "quality" not in sig.parameters


# ---------------------------------------------------------------------------
# 2. compute_coverage_matrix
# ---------------------------------------------------------------------------

class TestComputeCoverageMatrix:

    def test_returns_one_area_per_detected_topic(self):
        items = assign_evidence_ids([
            _ev("Rack draws 132 kW.", "nvidia.pdf", "power"),
            _ev("Liquid cooling required.", "nvidia.pdf", "cooling"),
        ])
        result = compute_coverage_matrix(
            "What are the power and cooling requirements for NVL72 racks?",
            items,
        )
        topics = {a.topic for a in result}
        assert "power" in topics
        assert "cooling" in topics

    def test_no_topics_detected_returns_empty(self):
        result = compute_coverage_matrix("Tell me about abstract things.", [])
        assert result == []

    def test_strong_coverage_when_many_items_diverse_sources(self):
        # >= STRONG_MIN_EVIDENCE items from >= STRONG_MIN_SOURCES docs → strong
        items = assign_evidence_ids([
            _ev(f"Power claim {i}.", f"nvidia_{i}.pdf", "power", f"E{i:03d}")
            for i in range(1, STRONG_MIN_EVIDENCE + 1)
        ])
        result = compute_coverage_matrix("What is the rack power draw?", items)
        power_area = next(a for a in result if a.topic == "power")
        assert power_area.coverage_level == "strong"
        assert power_area.evidence_count == STRONG_MIN_EVIDENCE

    def test_none_coverage_for_uncovered_topic(self):
        # Question asks about cooling but evidence is all power
        items = assign_evidence_ids([
            _ev("Power draw is 132 kW.", "nvidia.pdf", "power"),
        ])
        result = compute_coverage_matrix(
            "What are the power and cooling requirements?", items
        )
        cooling_area = next((a for a in result if a.topic == "cooling"), None)
        assert cooling_area is not None
        assert cooling_area.coverage_level == "none"
        assert cooling_area.evidence_count == 0

    def test_high_priority_gap_does_not_downgrade_strong_coverage(self):
        """9 items / 4 sources is strong regardless of gaps."""
        items = assign_evidence_ids([
            _ev(f"Power draw {i}.", f"nvidia_{i % 4}.pdf", "power", f"E{i:03d}")
            for i in range(1, 10)
        ])
        gaps = [_gap("rack power", "high", "G001")]
        result = compute_coverage_matrix("What is rack power?", items, research_gaps=gaps)
        power_area = next(a for a in result if a.topic == "power")
        assert power_area.coverage_level == "strong"

    def test_source_quality_reflected_in_rationale(self):
        # Quality enriches rationale even though it doesn't gate the level
        sq_map = {f"nvidia{i}.pdf": _sq(f"nvidia{i}.pdf", 5) for i in range(1, 6)}
        items = assign_evidence_ids([
            _ev(f"Power claim {i}.", f"nvidia{i}.pdf", "power", f"E{i:03d}")
            for i in range(1, 6)
        ])
        result = compute_coverage_matrix(
            "What is rack power?", items, source_quality_map=sq_map
        )
        power_area = next(a for a in result if a.topic == "power")
        # Level is strong from counts alone; rationale mentions quality
        assert power_area.coverage_level == "strong"
        assert "high-quality" in power_area.rationale.lower()

    def test_areas_sorted_alphabetically(self):
        items = assign_evidence_ids([
            _ev("Cooling note.", "a.pdf", "cooling"),
            _ev("Power note.", "b.pdf", "power"),
        ])
        result = compute_coverage_matrix(
            "What are the power and cooling requirements?", items
        )
        topics = [a.topic for a in result]
        assert topics == sorted(topics)

    def test_coverage_area_fields_present(self):
        items = assign_evidence_ids([_ev("Power.", "x.pdf", "power")])
        result = compute_coverage_matrix("What is rack power?", items)
        assert result
        area = result[0]
        assert isinstance(area, CoverageArea)
        assert area.topic
        assert isinstance(area.evidence_count, int)
        assert isinstance(area.source_count, int)
        assert area.coverage_level in ("strong", "moderate", "weak", "none")
        assert area.rationale.strip()

    def test_rack_architecture_category_contributes_to_rack_architecture_topic(self):
        items = assign_evidence_ids([
            _ev("NVL72 rack form factor.", "a.pdf", "rack architecture", "E001"),
            _ev("Rack density.", "b.pdf", "rack architecture", "E002"),
            _ev("Architecture overview.", "c.pdf", "architecture", "E003"),
        ])
        result = compute_coverage_matrix(
            "What is the rack architecture of NVL72?", items
        )
        ra = next((a for a in result if a.topic == "rack architecture"), None)
        assert ra is not None
        assert ra.evidence_count == 3  # both "rack architecture" and "architecture" count

    def test_operations_topic_moderate_with_three_items(self):
        # 3 items < STRONG_MIN_EVIDENCE (5), so moderate
        items = assign_evidence_ids([
            _ev("Commissioning steps.", "ops.pdf", "operations", "E001"),
            _ev("Monitoring strategy.", "ops2.pdf", "operations", "E002"),
            _ev("Maintenance window.", "ops3.pdf", "operations", "E003"),
        ])
        result = compute_coverage_matrix(
            "What are the operations and maintenance considerations?", items
        )
        ops_area = next((a for a in result if a.topic == "operations"), None)
        assert ops_area is not None
        assert ops_area.coverage_level == "moderate"

    def test_operations_topic_strong_with_five_items(self):
        items = assign_evidence_ids([
            _ev(f"Operations fact {i}.", f"ops{i}.pdf", "operations", f"E{i:03d}")
            for i in range(1, 6)
        ])
        result = compute_coverage_matrix(
            "What are the operations and maintenance considerations?", items,
        )
        ops_area = next((a for a in result if a.topic == "operations"), None)
        assert ops_area is not None
        assert ops_area.coverage_level == "strong"


# ---------------------------------------------------------------------------
# 3. Integration with agent (mock path)
# ---------------------------------------------------------------------------

class TestAgentCoverageIntegration:

    def test_memo_has_coverage_matrix_field(self):
        doc = _doc(
            "The NVL72 rack draws 132 kW and uses liquid cooling.",
            "Inside the NVIDIA Vera Rubin Platform.pdf",
        )
        memo = DcPowerAgent(client=MockClaudeClient()).analyze(
            "What are the power and cooling requirements for NVL72 racks?", [doc]
        )
        assert isinstance(memo.coverage_matrix, list)

    def test_coverage_matrix_in_metadata(self):
        doc = _doc("Rack power draw is 132 kW.", "nvidia.pdf")
        memo = DcPowerAgent(client=MockClaudeClient()).analyze(
            "What is the rack power draw?", [doc]
        )
        assert "coverage_matrix" in memo.metadata
        assert isinstance(memo.metadata["coverage_matrix"], list)

    def test_coverage_areas_match_detected_topics(self):
        doc = _doc(
            "The rack draws 132 kW. Liquid cooling is required.",
            "Inside the NVIDIA Vera Rubin Platform.pdf",
        )
        question = "What are the power and cooling requirements for NVL72?"
        memo = DcPowerAgent(client=MockClaudeClient()).analyze(question, [doc])
        topics_in_matrix = {a.topic for a in memo.coverage_matrix}
        # power and cooling should both be covered
        assert "power" in topics_in_matrix
        assert "cooling" in topics_in_matrix


# ---------------------------------------------------------------------------
# 4. Trace output
# ---------------------------------------------------------------------------

class TestCoverageTrace:

    def test_trace_includes_coverage_matrix(self):
        doc = _doc("Rack power is 132 kW.", "NVIDIA NVL72.pdf")
        question = "What is rack power?"
        memo = DcPowerAgent(client=MockClaudeClient()).analyze(question, [doc])
        trace = build_trace(
            question=question,
            source_directory=Path("sources"),
            output_path=Path("outputs/memo.md"),
            documents=[doc],
            memo=memo,
            mock_mode=True,
        )
        assert "coverage_matrix" in trace
        assert isinstance(trace["coverage_matrix"], list)

    def test_trace_coverage_matrix_has_required_fields(self):
        doc = _doc("Rack power is 132 kW.", "NVIDIA NVL72.pdf")
        question = "What is rack power?"
        memo = DcPowerAgent(client=MockClaudeClient()).analyze(question, [doc])
        trace = build_trace(
            question=question,
            source_directory=Path("sources"),
            output_path=Path("outputs/memo.md"),
            documents=[doc],
            memo=memo,
            mock_mode=True,
        )
        for area in trace["coverage_matrix"]:
            assert "topic" in area
            assert "coverage_level" in area
            assert "evidence_count" in area
            assert "source_count" in area


# ---------------------------------------------------------------------------
# 5. Markdown rendering
# ---------------------------------------------------------------------------

class TestCoverageMarkdown:

    def test_coverage_matrix_section_in_markdown(self):
        doc = _doc("Rack power is 132 kW.", "Inside the NVIDIA Vera Rubin Platform.pdf")
        memo = DcPowerAgent(client=MockClaudeClient()).analyze(
            "What is rack power?", [doc]
        )
        md = memo_to_markdown(memo)
        assert "## Coverage Matrix" in md

    def test_coverage_level_appears_in_section(self):
        doc = _doc("Rack power is 132 kW.", "Inside the NVIDIA Vera Rubin Platform.pdf")
        memo = DcPowerAgent(client=MockClaudeClient()).analyze(
            "What is rack power?", [doc]
        )
        md = memo_to_markdown(memo)
        # At least one coverage level label must appear
        assert any(label in md for label in ("Strong", "Moderate", "Weak", "None"))

    def test_empty_coverage_matrix_renders_gracefully(self):
        from research_agent.schemas import ResearchMemo
        memo = ResearchMemo(
            title="Test",
            question="What?",
            executive_summary="Summary.",
            coverage_matrix=[],
        )
        md = memo_to_markdown(memo)
        assert "## Coverage Matrix" in md
        assert "No topic coverage data available." in md

    def test_coverage_section_before_open_questions(self):
        doc = _doc("Rack power is 132 kW.", "Inside the NVIDIA Vera Rubin Platform.pdf")
        memo = DcPowerAgent(client=MockClaudeClient()).analyze(
            "What is rack power?", [doc]
        )
        md = memo_to_markdown(memo)
        coverage_pos = md.index("## Coverage Matrix")
        open_q_pos = md.index("## Open Questions")
        assert coverage_pos < open_q_pos

    def test_topic_name_appears_in_section(self):
        doc = _doc("Rack power is 132 kW.", "Inside the NVIDIA Vera Rubin Platform.pdf")
        memo = DcPowerAgent(client=MockClaudeClient()).analyze(
            "What is rack power?", [doc]
        )
        md = memo_to_markdown(memo)
        # "Power" (title-cased) should appear in coverage section
        coverage_section = md[md.index("## Coverage Matrix"):]
        assert "Power" in coverage_section


# ---------------------------------------------------------------------------
# 6. Evaluator metrics
# ---------------------------------------------------------------------------

class TestCoverageEvaluatorMetrics:

    def test_evaluator_emits_coverage_matrix_metrics_info(self):
        from research_agent.evaluator import evaluate_memo
        doc = _doc("Rack power is 132 kW.", "Inside the NVIDIA Vera Rubin Platform.pdf")
        memo = DcPowerAgent(client=MockClaudeClient()).analyze(
            "What is rack power?", [doc]
        )
        warnings = evaluate_memo(memo, [doc], mock_llm=True)
        codes = [w.code for w in warnings]
        assert "coverage_matrix_metrics" in codes

    def test_coverage_metrics_warning_is_info_severity(self):
        from research_agent.evaluator import evaluate_memo
        doc = _doc("Rack power is 132 kW.", "Inside the NVIDIA Vera Rubin Platform.pdf")
        memo = DcPowerAgent(client=MockClaudeClient()).analyze(
            "What is rack power?", [doc]
        )
        warnings = evaluate_memo(memo, [doc], mock_llm=True)
        coverage_w = next(w for w in warnings if w.code == "coverage_matrix_metrics")
        assert coverage_w.severity == "info"

    def test_no_warning_for_weak_coverage(self):
        """Weak coverage must NOT generate a blocking warning (only informational)."""
        from research_agent.evaluator import evaluate_memo
        doc = _doc("Single power note.", "test.txt")
        memo = DcPowerAgent(client=MockClaudeClient()).analyze("What is rack power?", [doc])
        warnings = evaluate_memo(memo, [doc], mock_llm=True)
        blocking = [w for w in warnings if w.severity == "warning" and "coverage" in w.code]
        assert not blocking


# ---------------------------------------------------------------------------
# 7. Interaction with research gaps
# ---------------------------------------------------------------------------

class TestCoverageGapInteraction:
    """Coverage and gaps are independent signals: gaps never change the level."""

    def _strong_items(self, n: int = 5) -> list[EvidenceItem]:
        """Return n power items spread across STRONG_MIN_SOURCES sources."""
        return assign_evidence_ids([
            _ev(f"Power fact {i}.", f"src_{i % STRONG_MIN_SOURCES}.pdf", "power", f"E{i:03d}")
            for i in range(1, n + 1)
        ])

    def test_high_priority_gaps_do_not_downgrade_strong_coverage(self):
        """9 items across 4 sources → strong even with high-priority gaps."""
        items = assign_evidence_ids([
            _ev(f"Power draw {i}.", f"nvidia_{i % 4}.pdf", "power", f"E{i:03d}")
            for i in range(1, 10)
        ])
        gaps = [_gap("rack power", "high", "G001")]
        result = compute_coverage_matrix(
            "What is rack power delivery?", items, research_gaps=gaps,
        )
        power = next(a for a in result if a.topic == "power")
        assert power.coverage_level == "strong"

    def test_high_priority_gap_appears_in_rationale(self):
        """Level stays strong; matching gap topic is appended to rationale."""
        items = self._strong_items(8)
        gaps = [_gap("power delivery", "high", "G001")]
        result = compute_coverage_matrix(
            "What is rack power delivery?", items, research_gaps=gaps,
        )
        power = next(a for a in result if a.topic == "power")
        assert power.coverage_level == "strong"
        assert "research gaps remain" in power.rationale.lower()
        assert "power delivery" in power.rationale

    def test_many_gaps_do_not_downgrade_coverage(self):
        items = self._strong_items(8)
        gaps = [
            _gap("rack power", "high", "G001"),
            _gap("power delivery", "high", "G002"),
            _gap("ups", "medium", "G003"),
        ]
        result = compute_coverage_matrix("What is rack power?", items, research_gaps=gaps)
        power = next(a for a in result if a.topic == "power")
        assert power.coverage_level == "strong"
        assert power.evidence_count == 8

    def test_gaps_in_rationale_but_not_in_level(self):
        """Gaps affect the rationale text but not the coverage level."""
        items = self._strong_items(5)
        gaps = [_gap("rack power", "high", "G001")]
        with_gaps = compute_coverage_matrix("What is rack power?", items, research_gaps=gaps)
        without_gaps = compute_coverage_matrix("What is rack power?", items, research_gaps=[])
        levels_with = {a.topic: a.coverage_level for a in with_gaps}
        levels_without = {a.topic: a.coverage_level for a in without_gaps}
        assert levels_with == levels_without
        pw = next(a for a in with_gaps if a.topic == "power")
        assert "research gaps remain" in pw.rationale.lower()

    def test_low_priority_gaps_not_appended_to_rationale(self):
        items = self._strong_items(5)
        gaps = [_gap("power quality", "low", "G001")]
        result = compute_coverage_matrix("What is rack power?", items, research_gaps=gaps)
        power = next(a for a in result if a.topic == "power")
        assert "research gaps remain" not in power.rationale.lower()
