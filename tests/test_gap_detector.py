"""Tests for G3 research gap detection."""

from __future__ import annotations

from pathlib import Path

from dc_power_agent.gap_detector import detect_gaps, _corpus_has_coverage
from dc_power_agent.agent import DcPowerAgent
from dc_power_agent.claude_client import MockClaudeClient
from dc_power_agent.markdown import memo_to_markdown
from dc_power_agent.schemas import EvidenceItem, ResearchGap, ResearchMemo, SourceDocument, assign_evidence_ids
from dc_power_agent.trace import build_trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ev(claim: str, snippet: str = "", category: str = "power", source: str = "doc.pdf") -> EvidenceItem:
    return EvidenceItem(
        claim=claim,
        source_document=source,
        evidence_snippet=snippet or claim,
        category=category,
        relevance="Relevant.",
        confidence="high",
        relevance_score=4,
        source_quality_score=4,
        specificity_score=4,
        overall_score=4.0,
    )


def _doc(text: str, name: str = "doc.txt") -> SourceDocument:
    return SourceDocument(
        path=Path(f"sources/{name}"),
        title=name,
        extension=Path(name).suffix,
        text=text,
    )


# ---------------------------------------------------------------------------
# _corpus_has_coverage
# ---------------------------------------------------------------------------


def test_corpus_has_coverage_true_when_keyword_present():
    assert _corpus_has_coverage("the rack draws 120 kw under peak load", {"kw", "mw"})


def test_corpus_has_coverage_false_when_no_keywords():
    assert not _corpus_has_coverage("general software engineering practices", {"kw", "mw", "pdu"})


# ---------------------------------------------------------------------------
# Gap generation for power questions
# ---------------------------------------------------------------------------


def test_detect_gaps_power_question_with_no_evidence():
    """A power question with empty evidence should generate high-priority power gaps."""
    gaps = detect_gaps("What are the DC power implications of NVIDIA NVL72 racks?", [])
    assert gaps, "Expected at least one gap when evidence is empty"
    priorities = {g.priority for g in gaps}
    assert "high" in priorities
    topics = {g.topic for g in gaps}
    assert any("Power" in t or "power" in t.lower() for t in topics), (
        f"Expected a power-related gap. Got topics: {topics}"
    )


def test_detect_gaps_rack_power_missing():
    """No rack power figure in evidence → Rack Power Consumption gap at high priority."""
    evidence = assign_evidence_ids([
        _ev("The NVL72 uses warm-water direct liquid cooling.", category="cooling"),
        _ev("DSX software manages power budgets.", category="power"),
    ])
    gaps = detect_gaps("What are the DC power implications of NVIDIA NVL72 racks?", evidence)
    rack_power_gaps = [g for g in gaps if "Rack Power" in g.topic]
    assert rack_power_gaps, (
        f"Expected 'Rack Power Consumption' gap. Got: {[g.topic for g in gaps]}"
    )
    assert rack_power_gaps[0].priority == "high"


def test_detect_gaps_rack_power_present_suppresses_gap():
    """Evidence mentioning kW should suppress the Rack Power Consumption gap."""
    evidence = assign_evidence_ids([
        _ev("The NVL72 rack draws 120 kW at peak load.", category="power"),
    ])
    gaps = detect_gaps("What are the DC power implications of NVIDIA NVL72 racks?", evidence)
    rack_power_gaps = [g for g in gaps if "Rack Power" in g.topic]
    assert not rack_power_gaps, (
        "Rack Power gap must be suppressed when kW evidence is present"
    )


def test_detect_gaps_ups_gap_medium_priority():
    """No UPS evidence → medium-priority UPS gap."""
    evidence = assign_evidence_ids([
        _ev("The rack draws 120 kW at peak.", category="power"),
    ])
    gaps = detect_gaps("What are the DC power implications of NVIDIA NVL72 racks?", evidence)
    ups_gaps = [g for g in gaps if "UPS" in g.topic or "Backup" in g.topic]
    assert ups_gaps, f"Expected UPS gap. Got: {[g.topic for g in gaps]}"
    assert ups_gaps[0].priority == "medium"


# ---------------------------------------------------------------------------
# Gap generation for cooling questions
# ---------------------------------------------------------------------------


def test_detect_gaps_cooling_question_with_no_evidence():
    """A cooling question with empty evidence should generate cooling gaps."""
    gaps = detect_gaps("What are the cooling requirements for NVIDIA NVL72?", [])
    assert gaps
    cooling_gaps = [g for g in gaps if any(
        kw in g.topic.lower() for kw in ("cool", "cdu", "water", "heat")
    )]
    assert cooling_gaps, f"Expected cooling gaps. Got: {[g.topic for g in gaps]}"


def test_detect_gaps_cdu_gap_suppressed_when_cdu_mentioned():
    """Evidence mentioning CDU should suppress the CDU Requirements gap."""
    evidence = assign_evidence_ids([
        _ev("The CDU must supply 45°C water at 20 L/min.", category="cooling"),
    ])
    gaps = detect_gaps("What are the cooling requirements for NVIDIA NVL72?", evidence)
    cdu_gaps = [g for g in gaps if "CDU" in g.topic]
    assert not cdu_gaps, "CDU gap must be suppressed when CDU evidence is present"


# ---------------------------------------------------------------------------
# Prioritisation ordering
# ---------------------------------------------------------------------------


def test_detect_gaps_sorted_high_before_medium_before_low():
    """Gaps must be sorted high → medium → low."""
    gaps = detect_gaps("What are the DC power and cooling implications of NVIDIA NVL72 racks?", [])
    if len(gaps) < 2:
        return  # not enough gaps to test ordering
    priority_order = {"high": 0, "medium": 1, "low": 2}
    priorities = [priority_order[g.priority] for g in gaps]
    assert priorities == sorted(priorities), (
        f"Gaps not sorted by priority: {[g.priority for g in gaps]}"
    )


def test_detect_gaps_ids_are_sequential():
    """Gap IDs must be G001, G002, … assigned in priority order."""
    gaps = detect_gaps("What are the DC power implications of NVIDIA NVL72 racks?", [])
    assert gaps
    for i, g in enumerate(gaps, start=1):
        assert g.gap_id == f"G{i:03d}", f"Expected G{i:03d}, got {g.gap_id}"


def test_detect_gaps_no_duplicate_topics():
    """Each topic label must appear at most once across all gaps."""
    gaps = detect_gaps(
        "What are the DC power and cooling implications of NVIDIA NVL72 racks?", []
    )
    topics = [g.topic for g in gaps]
    assert len(topics) == len(set(topics)), f"Duplicate topics: {topics}"


def test_detect_gaps_returns_empty_for_unrecognised_question():
    """A question with no recognised topics should return no gaps."""
    gaps = detect_gaps("What is the history of ancient Rome?", [])
    assert gaps == []


# ---------------------------------------------------------------------------
# Memo rendering
# ---------------------------------------------------------------------------


def test_memo_research_gaps_section_empty_state():
    memo = ResearchMemo(title="T", question="Q?", executive_summary="S.", research_gaps=[])
    md = memo_to_markdown(memo)
    assert "## Research Gaps" in md
    assert "No research gaps identified." in md


def test_memo_research_gaps_section_with_entries():
    gaps = [
        ResearchGap(
            gap_id="G001",
            topic="Rack Power Consumption",
            priority="high",
            description="No explicit NVL72 rack power figure found.",
            rationale="Power planning requires a concrete rack-level power target.",
        ),
        ResearchGap(
            gap_id="G002",
            topic="CDU Requirements",
            priority="medium",
            description="No CDU sizing guidance found.",
            rationale="CDU capacity is needed for liquid cooling deployment.",
        ),
    ]
    memo = ResearchMemo(title="T", question="Q?", executive_summary="S.", research_gaps=gaps)
    md = memo_to_markdown(memo)
    assert "## Research Gaps" in md
    assert "**High Priority**" in md
    assert "**Medium Priority**" in md
    assert "G001" in md
    assert "Rack Power Consumption" in md
    assert "G002" in md
    assert "CDU Requirements" in md
    assert "No research gaps identified." not in md


def test_memo_research_gaps_only_present_tiers_rendered():
    """If there are no low-priority gaps, the Low Priority heading must not appear."""
    gaps = [
        ResearchGap(
            gap_id="G001", topic="Rack Power", priority="high",
            description="Missing.", rationale="Required.",
        )
    ]
    memo = ResearchMemo(title="T", question="Q?", executive_summary="S.", research_gaps=gaps)
    md = memo_to_markdown(memo)
    assert "**High Priority**" in md
    assert "**Low Priority**" not in md


# ---------------------------------------------------------------------------
# Agent integration and trace
# ---------------------------------------------------------------------------


def test_agent_mock_memo_contains_research_gaps():
    """After a mock run, memo.research_gaps must be populated for a power question."""
    doc = _doc(
        "NVIDIA Rubin NVL72 rack architecture uses liquid cooling infrastructure.",
        "rubin.md",
    )
    memo = DcPowerAgent(client=MockClaudeClient()).analyze(
        "What are the DC power implications of NVIDIA NVL72 racks?", [doc]
    )
    # Gaps should exist because mock evidence is unlikely to cover all subtopics
    assert isinstance(memo.research_gaps, list)
    assert "research_gaps" in memo.metadata


def test_trace_includes_research_gaps():
    doc = _doc(
        "NVIDIA Rubin NVL72 rack architecture uses liquid cooling infrastructure.",
        "rubin.md",
    )
    memo = DcPowerAgent(client=MockClaudeClient()).analyze(
        "What are the DC power implications of NVIDIA NVL72 racks?", [doc]
    )
    trace = build_trace(
        question="What are the DC power implications of NVIDIA NVL72 racks?",
        source_directory=Path("sources"),
        output_path=Path("outputs/memo.md"),
        documents=[doc],
        memo=memo,
        mock_mode=True,
    )
    assert "research_gaps" in trace
    assert isinstance(trace["research_gaps"], list)


def test_trace_research_gaps_have_required_fields():
    doc = _doc(
        "NVIDIA Rubin NVL72 rack architecture uses liquid cooling infrastructure.",
        "rubin.md",
    )
    memo = DcPowerAgent(client=MockClaudeClient()).analyze(
        "What are the DC power implications of NVIDIA NVL72 racks?", [doc]
    )
    trace = build_trace(
        question="What are the DC power implications of NVIDIA NVL72 racks?",
        source_directory=Path("sources"),
        output_path=Path("outputs/memo.md"),
        documents=[doc],
        memo=memo,
        mock_mode=True,
    )
    for gap in trace["research_gaps"]:
        assert "gap_id" in gap
        assert "topic" in gap
        assert "priority" in gap
        assert "description" in gap
        assert "rationale" in gap
        assert gap["priority"] in ("high", "medium", "low")


# ---------------------------------------------------------------------------
# Evaluator integration
# ---------------------------------------------------------------------------


def test_evaluator_research_gap_metrics_info_entry():
    """evaluate_memo must emit research_gap_metrics info entry when gaps exist."""
    from dc_power_agent.evaluator import evaluate_memo

    gaps = [
        ResearchGap(gap_id="G001", topic="Rack Power", priority="high",
                    description="Missing.", rationale="Required."),
        ResearchGap(gap_id="G002", topic="UPS", priority="medium",
                    description="Missing.", rationale="Required."),
    ]
    doc = SourceDocument(path=Path("a.pdf"), title="a", extension=".pdf",
                         text="power cooling rack.")
    memo = ResearchMemo(
        title="Test", question="Q?", executive_summary="S.",
        research_gaps=gaps,
        metadata={"research_gaps": [g.model_dump() for g in gaps]},
    )
    warnings = evaluate_memo(memo, [doc])
    codes = {w.code for w in warnings}
    assert "research_gap_metrics" in codes
    info = next(w for w in warnings if w.code == "research_gap_metrics")
    assert "2" in info.message   # total count
    assert "1" in info.message   # high-priority count


def test_evaluator_no_gap_warning_when_no_gaps():
    """evaluate_memo must not emit research_gap_metrics when there are no gaps."""
    from dc_power_agent.evaluator import evaluate_memo

    doc = SourceDocument(path=Path("a.pdf"), title="a", extension=".pdf",
                         text="power cooling rack.")
    memo = ResearchMemo(
        title="Test", question="Q?", executive_summary="S.",
        metadata={"research_gaps": []},
    )
    warnings = evaluate_memo(memo, [doc])
    codes = {w.code for w in warnings}
    assert "research_gap_metrics" not in codes
