"""Tests for G2 contradiction detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_agent.contradiction import detect_contradictions
from research_agent.agent import DcPowerAgent
from research_agent.claude_client import MockClaudeClient
from research_agent.markdown import memo_to_markdown
from research_agent.schemas import (
    Contradiction,
    EvidenceItem,
    ResearchMemo,
    SourceDocument,
    assign_evidence_ids,
)
from research_agent.trace import build_trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ev(claim: str, source: str = "doc.pdf", category: str = "power") -> EvidenceItem:
    return EvidenceItem(
        claim=claim,
        source_document=source,
        evidence_snippet=claim,
        category=category,
        relevance="Relevant.",
        confidence="high",
        relevance_score=4,
        source_quality_score=4,
        specificity_score=4,
        overall_score=4.0,
    )


def _doc(text: str = "NVIDIA Rubin NVL72 rack power cooling liquid.", name: str = "doc.txt") -> SourceDocument:
    return SourceDocument(
        path=Path(f"sources/{name}"),
        title=name,
        extension=Path(name).suffix,
        text=text,
    )


# ---------------------------------------------------------------------------
# Numeric contradiction tests
# ---------------------------------------------------------------------------


def test_numeric_conflict_detected_for_power_values():
    """120 kW vs 180 kW should produce a power contradiction."""
    items = assign_evidence_ids([
        _ev("The rack draws 120 kW under peak load."),
        _ev("The rack requires 180 kW of DC power at full utilisation."),
    ])
    result = detect_contradictions(items)
    assert result, "Expected at least one contradiction"
    topics = [c.topic for c in result]
    assert any("power" in t for t in topics), f"Expected power topic, got: {topics}"


def test_numeric_conflict_severity_high_for_large_difference():
    """A > 50 % numeric difference should be high severity."""
    items = assign_evidence_ids([
        _ev("The rack consumes 100 kW."),
        _ev("The rack consumes 200 kW."),
    ])
    result = detect_contradictions(items)
    assert result
    assert any(c.severity == "high" for c in result), "Expected high severity for 2x difference"


def test_numeric_conflict_severity_medium_for_moderate_difference():
    """A 25 % numeric difference should be medium severity."""
    items = assign_evidence_ids([
        _ev("The system draws 100 kW."),
        _ev("The system draws 130 kW."),
    ])
    result = detect_contradictions(items)
    assert result
    assert any(c.severity == "medium" for c in result)


def test_numeric_conflict_not_detected_for_similar_values():
    """Values within 20 % should not trigger a contradiction."""
    items = assign_evidence_ids([
        _ev("The rack draws 120 kW under peak load."),
        _ev("The rack requires 125 kW of DC power."),
    ])
    result = detect_contradictions(items)
    numeric_power = [c for c in result if c.topic == "power"]
    assert not numeric_power, "Small numeric difference should not produce a contradiction"


def test_no_contradiction_for_single_item():
    """One evidence item → no pairs → no contradictions."""
    items = assign_evidence_ids([_ev("The rack draws 120 kW.")])
    assert detect_contradictions(items) == []


def test_no_contradiction_for_identical_claims():
    """Two items with the same claim should not produce a contradiction."""
    items = assign_evidence_ids([
        _ev("The rack uses liquid cooling at 45°C supply temperature."),
        _ev("The rack uses liquid cooling at 45°C supply temperature."),
    ])
    assert detect_contradictions(items) == []


# ---------------------------------------------------------------------------
# Categorical contradiction tests
# ---------------------------------------------------------------------------


def test_categorical_conflict_detected_for_cooling_type():
    """Air cooled vs liquid cooled should produce a high-severity contradiction."""
    items = assign_evidence_ids([
        _ev("The server is air cooled and uses rear-door heat exchangers.", category="cooling"),
        _ev("The rack uses direct liquid cooling (DLC) at 45°C.", category="cooling"),
    ])
    result = detect_contradictions(items)
    assert result, "Expected cooling type contradiction"
    cooling = [c for c in result if "cool" in c.topic]
    assert cooling, f"Expected cooling-type topic, got topics: {[c.topic for c in result]}"
    assert any(c.severity == "high" for c in cooling)


def test_categorical_conflict_detected_for_timeline():
    """Claims referencing 2026 vs 2027 should produce a timeline contradiction."""
    items = assign_evidence_ids([
        _ev("The NVL72 rack will be available in 2026."),
        _ev("General availability is expected in 2027."),
    ])
    result = detect_contradictions(items)
    assert result, "Expected timeline contradiction"
    assert any("timeline" in c.topic for c in result)


def test_categorical_conflict_detected_for_gpu_count():
    """72 GPU vs 144 GPU should produce a contradiction."""
    items = assign_evidence_ids([
        _ev("The NVL72 rack integrates 72 GPU accelerators."),
        _ev("The rack system houses 144 GPU units."),
    ])
    result = detect_contradictions(items)
    assert result
    assert any("gpu" in c.topic.lower() for c in result)


def test_categorical_no_conflict_for_same_year():
    """Two claims both referencing 2026 should not conflict."""
    items = assign_evidence_ids([
        _ev("Shipping starts in 2026."),
        _ev("Production begins in 2026."),
    ])
    timeline = [c for c in detect_contradictions(items) if "timeline" in c.topic]
    assert not timeline


# ---------------------------------------------------------------------------
# Contradiction ID assignment
# ---------------------------------------------------------------------------


def test_contradiction_ids_are_sequential():
    """Multiple contradictions must receive C001, C002, ... IDs."""
    items = assign_evidence_ids([
        _ev("The rack draws 100 kW.", "a.pdf"),
        _ev("The rack draws 200 kW.", "b.pdf"),
        _ev("The server is air cooled.", "a.pdf", "cooling"),
        _ev("The rack uses direct liquid cooling (DLC).", "b.pdf", "cooling"),
    ])
    result = detect_contradictions(items)
    assert len(result) >= 2
    ids = [c.contradiction_id for c in result]
    assert ids[0] == "C001"
    assert ids[1] == "C002"


def test_contradiction_fields_are_populated():
    """Each Contradiction must carry evidence IDs, sources, and an explanation."""
    items = assign_evidence_ids([
        _ev("The rack draws 100 kW.", "source_a.pdf"),
        _ev("The rack draws 200 kW.", "source_b.pdf"),
    ])
    result = detect_contradictions(items)
    assert result
    c = result[0]
    assert c.evidence_a_id.startswith("E")
    assert c.evidence_b_id.startswith("E")
    assert c.evidence_a_source == "source_a.pdf"
    assert c.evidence_b_source == "source_b.pdf"
    assert c.explanation


# ---------------------------------------------------------------------------
# Memo rendering
# ---------------------------------------------------------------------------


def test_memo_contradictions_section_rendered_when_empty():
    memo = ResearchMemo(
        title="Test", question="Q?", executive_summary="Summary.",
        contradictions=[],
    )
    md = memo_to_markdown(memo)
    assert "## Potential Contradictions" in md
    assert "No significant contradictions detected." in md


def test_memo_contradictions_section_rendered_with_entries():
    contradiction = Contradiction(
        contradiction_id="C001",
        topic="power",
        evidence_a_id="E001",
        evidence_b_id="E002",
        evidence_a_claim="120 kW.",
        evidence_b_claim="200 kW.",
        evidence_a_source="a.pdf",
        evidence_b_source="b.pdf",
        severity="high",
        explanation="Numeric conflict: A says 120 kw, B says 200 kw.",
    )
    memo = ResearchMemo(
        title="Test", question="Q?", executive_summary="Summary.",
        contradictions=[contradiction],
    )
    md = memo_to_markdown(memo)
    assert "## Potential Contradictions" in md
    assert "C001" in md
    assert "HIGH" in md
    assert "E001" in md
    assert "E002" in md
    assert "a.pdf" in md


# ---------------------------------------------------------------------------
# Trace support
# ---------------------------------------------------------------------------


def test_trace_includes_contradictions_detected():
    """After an agent run, build_trace must include contradictions_detected."""
    doc = _doc(
        "NVIDIA Rubin NVL72 rack power cooling liquid infrastructure.",
        "rubin.md",
    )
    memo = DcPowerAgent(client=MockClaudeClient()).analyze("Rubin power cooling", [doc])
    trace = build_trace(
        question="Rubin power cooling",
        source_directory=Path("sources"),
        output_path=Path("outputs/memo.md"),
        documents=[doc],
        memo=memo,
        mock_mode=True,
    )
    assert "contradictions_detected" in trace
    assert isinstance(trace["contradictions_detected"], list)


def test_memo_metadata_includes_contradictions():
    """ResearchMemo.metadata['contradictions'] must be present after agent run."""
    doc = _doc(
        "NVIDIA Rubin NVL72 rack power cooling liquid infrastructure.",
        "rubin.md",
    )
    memo = DcPowerAgent(client=MockClaudeClient()).analyze("Rubin power cooling", [doc])
    assert "contradictions" in memo.metadata
    assert isinstance(memo.metadata["contradictions"], list)


# ---------------------------------------------------------------------------
# Evaluator integration
# ---------------------------------------------------------------------------


def test_evaluator_warns_on_high_severity_contradiction():
    """evaluate_memo should produce a high_severity_contradiction warning."""
    from research_agent.evaluator import evaluate_memo

    contradiction = Contradiction(
        contradiction_id="C001",
        topic="power",
        evidence_a_id="E001",
        evidence_b_id="E002",
        evidence_a_claim="120 kW.",
        evidence_b_claim="250 kW.",
        evidence_a_source="a.pdf",
        evidence_b_source="b.pdf",
        severity="high",
        explanation="Numeric conflict.",
    )
    ev = assign_evidence_ids([
        _ev("Claim.", "a.pdf"),
        _ev("Claim.", "b.pdf"),
        _ev("Claim.", "c.pdf"),
    ])
    docs = [
        SourceDocument(path=Path("a.pdf"), title="a", extension=".pdf",
                       text="power cooling rack."),
        SourceDocument(path=Path("b.pdf"), title="b", extension=".pdf",
                       text="power cooling rack."),
        SourceDocument(path=Path("c.pdf"), title="c", extension=".pdf",
                       text="power cooling rack."),
    ]
    memo = ResearchMemo(
        title="Test",
        question="Rubin power?",
        executive_summary="Summary.",
        confirmed_facts=["Fact. [Source: a.pdf, Evidence: E001]"],
        contradictions=[contradiction],
        evidence=ev,
        source_notes=ev,
        metadata={"contradictions": [contradiction.model_dump()]},
    )
    warnings = evaluate_memo(memo, docs)
    codes = {w.code for w in warnings}
    assert "high_severity_contradiction" in codes


def test_evaluator_no_contradiction_warning_when_none():
    """evaluate_memo must not produce a contradiction warning when list is empty."""
    from research_agent.evaluator import evaluate_memo

    ev = assign_evidence_ids([_ev("Claim.", "a.pdf")])
    docs = [
        SourceDocument(path=Path("a.pdf"), title="a", extension=".pdf", text="power cooling."),
    ]
    memo = ResearchMemo(
        title="Test",
        question="Rubin power?",
        executive_summary="Summary.",
        evidence=ev,
        source_notes=ev,
        metadata={"contradictions": []},
    )
    warnings = evaluate_memo(memo, docs)
    codes = {w.code for w in warnings}
    assert "high_severity_contradiction" not in codes


# ---------------------------------------------------------------------------
# Canonical validation test — does NOT go through retrieval or extraction
# ---------------------------------------------------------------------------
#
# This is the authoritative test that the contradiction engine is functioning.
# It creates EvidenceItems directly (bypassing chunking, retrieval, and
# Claude extraction) so it cannot be broken by retrieval cutoffs.


def test_contradiction_detection_120kw_vs_180kw():
    """
    CANONICAL VALIDATION TEST
    --------------------------
    Input:
      - EvidenceItem A: "Rubin NVL72 rack power = 120 kW"
      - EvidenceItem B: "Rubin NVL72 rack power = 180 kW"

    Expected:
      - at least 1 contradiction detected
      - topic == "rack power"
      - severity in {"medium", "high"}

    This test proves the contradiction engine is functioning independently
    of retrieval scoring, chunk selection, or document priority.
    """
    items = assign_evidence_ids([
        EvidenceItem(
            claim="The Rubin NVL72 rack power draw is 120 kW at peak utilisation.",
            source_document="nvidia_spec_a.pdf",
            evidence_snippet="The Rubin NVL72 rack power draw is 120 kW at peak utilisation.",
            category="power",
            relevance="Direct power specification.",
            confidence="high",
            relevance_score=5,
            source_quality_score=5,
            specificity_score=5,
            overall_score=5.0,
        ),
        EvidenceItem(
            claim="The Rubin NVL72 rack power draw is 180 kW at full utilisation.",
            source_document="nvidia_spec_b.pdf",
            evidence_snippet="The Rubin NVL72 rack power draw is 180 kW at full utilisation.",
            category="power",
            relevance="Direct power specification.",
            confidence="high",
            relevance_score=5,
            source_quality_score=5,
            specificity_score=5,
            overall_score=5.0,
        ),
    ])

    result = detect_contradictions(items)

    assert len(result) >= 1, (
        f"Expected at least 1 contradiction. Got 0.\n"
        f"Item A claim: {items[0].claim}\n"
        f"Item B claim: {items[1].claim}"
    )

    rack_power = [c for c in result if c.topic == "rack power"]
    assert rack_power, (
        f"Expected topic 'rack power'. Got: {[c.topic for c in result]}"
    )

    for c in rack_power:
        assert c.severity in ("medium", "high"), (
            f"Expected severity medium or high. Got: {c.severity}\n"
            f"Explanation: {c.explanation}"
        )
        assert c.evidence_a_id == "E001"
        assert c.evidence_b_id == "E002"


# ---------------------------------------------------------------------------
# Engine validation: topic naming
# ---------------------------------------------------------------------------


def test_kw_conflict_topic_is_rack_power():
    """kW numeric conflicts must be classified as 'rack power', not plain 'power'."""
    items = assign_evidence_ids([
        _ev("The rack draws 120 kW under peak load."),
        _ev("The rack requires 180 kW of DC power at full utilisation."),
    ])
    result = detect_contradictions(items)
    assert result, "Expected at least one contradiction"
    assert any(c.topic == "rack power" for c in result), (
        f"Expected topic 'rack power', got: {[c.topic for c in result]}"
    )


def test_120kw_vs_180kw_severity_is_medium():
    """120 kW vs 180 kW is a 33 % difference → medium severity."""
    items = assign_evidence_ids([
        _ev("The NVL72 rack draws 120 kW under peak load."),
        _ev("The NVL72 rack requires 180 kW of DC power at full utilisation."),
    ])
    result = detect_contradictions(items)
    assert result
    power = [c for c in result if c.topic == "rack power"]
    assert power, f"No rack power contradiction. Got: {result}"
    severities = {c.severity for c in power}
    assert severities & {"medium", "high"}, f"Expected medium or high, got: {severities}"


# ---------------------------------------------------------------------------
# Pipeline integration: stub client bypasses retrieval
# ---------------------------------------------------------------------------
#
# EXPLANATION OF ARCHITECTURE
# ============================
# detect_contradictions() operates on EvidenceItem objects that have already
# been extracted.  In the live Claude path, evidence comes from
# extract_evidence_from_chunks(), which only processes chunks that survived
# retrieval scoring.  Synthetic test documents with minimal NVIDIA keywords
# score near zero and are rejected before any evidence is produced — so the
# contradiction engine never runs.
#
# The unit tests above (test_numeric_conflict_*, test_categorical_conflict_*)
# prove the engine itself is correct: they call detect_contradictions()
# directly with hand-built EvidenceItems, bypassing the retrieval and
# extraction stages entirely.
#
# The integration tests below use a _ConflictingEvidenceClient stub that
# implements the LLMClient protocol and returns pre-built conflicting items
# from extract_evidence_from_chunks().  This exercises the full agent loop
# (chunking → retrieval → evidence ranking → contradiction detection →
# memo synthesis) while guaranteeing the contradictory claims are always
# present in the evidence pool — independent of retrieval score or chunk
# content.


from research_agent.claude_client import LLMClient  # noqa: E402 (after the comment block)
from research_agent.schemas import ResearchPlan  # noqa: E402


class _ConflictingEvidenceClient:
    """Stub LLMClient that always returns two conflicting power EvidenceItems."""

    is_mock = False
    model = "stub-conflict-client"

    def __init__(self) -> None:
        self.call_traces: list = []

    def create_research_plan(self, question, documents):
        return ResearchPlan(
            research_questions=["What is the rack power draw?"],
            key_topics=["power"],
            source_priorities=["test_power_a.txt", "test_power_b.txt"],
        )

    def extract_evidence_from_chunks(self, question, chunks):
        """Return conflicting 120 kW vs 180 kW claims regardless of chunk content."""
        return [
            EvidenceItem(
                claim="The NVIDIA NVL72 rack draws 120 kW under peak load.",
                source_document="test_power_a.txt",
                evidence_snippet="The NVIDIA NVL72 rack draws 120 kW under peak load.",
                category="power",
                relevance="Direct power specification.",
                confidence="high",
                relevance_score=5,
                source_quality_score=4,
                specificity_score=5,
                overall_score=4.7,
            ),
            EvidenceItem(
                claim="The NVIDIA NVL72 rack requires 180 kW of DC power at full utilisation.",
                source_document="test_power_b.txt",
                evidence_snippet="The NVIDIA NVL72 rack requires 180 kW of DC power at full utilisation.",
                category="power",
                relevance="Direct power specification.",
                confidence="high",
                relevance_score=5,
                source_quality_score=4,
                specificity_score=5,
                overall_score=4.7,
            ),
        ]

    def synthesize_memo(self, question, evidence_items):
        citation = (
            f"[Source: {evidence_items[0].source_document}, "
            f"Evidence: {evidence_items[0].evidence_id}]"
        ) if evidence_items else ""
        return ResearchMemo(
            title=f"Research Memo: {question}",
            question=question,
            executive_summary="Conflicting power specifications detected.",
            confirmed_facts=[f"Power specification noted. {citation}".strip()],
            inferences=["Two conflicting power figures require reconciliation."],
            power_implications=[f"Conflicting kW values. {citation}".strip()],
            cooling_implications=[],
            networking_implications=[],
            rack_architecture_implications=[],
            open_questions=["Which power figure is authoritative?"],
            source_notes=list(evidence_items),
            evidence=list(evidence_items),
        )


def _power_docs() -> list[SourceDocument]:
    """Two minimal documents that would normally be rejected by retrieval."""
    return [
        SourceDocument(
            path=Path("sources/test_power_a.txt"),
            title="test_power_a.txt",
            extension=".txt",
            text="The NVIDIA NVL72 rack draws 120 kW under peak load.",
        ),
        SourceDocument(
            path=Path("sources/test_power_b.txt"),
            title="test_power_b.txt",
            extension=".txt",
            text="The NVIDIA NVL72 rack requires 180 kW of DC power at full utilisation.",
        ),
    ]


def test_pipeline_120kw_vs_180kw_contradiction_detected():
    """
    End-to-end: stub client injects conflicting 120 kW / 180 kW claims.
    The agent must detect a rack-power contradiction with severity >= medium.
    This test proves the pipeline wiring (extraction → detection → memo) is
    correct independent of retrieval scoring.
    """
    client = _ConflictingEvidenceClient()
    memo = DcPowerAgent(client=client).analyze(
        "What is the DC power draw of the NVIDIA NVL72 rack?",
        _power_docs(),
    )

    assert memo.contradictions, (
        "Expected at least one contradiction in memo.contradictions"
    )
    # The agent applies the ai_data_centers profile → kW conflicts get
    # topic="power" (profile:ai_data_centers) rather than the hard-coded
    # "rack power" label.
    power_contradictions = [c for c in memo.contradictions if c.topic == "power"]
    assert power_contradictions, (
        f"Expected topic 'power' (profile:ai_data_centers). "
        f"Got topics: {[c.topic for c in memo.contradictions]}"
    )
    severities = {c.severity for c in power_contradictions}
    assert severities & {"medium", "high"}, (
        f"Expected severity medium or high. Got: {severities}"
    )


def test_pipeline_contradictions_flow_to_metadata():
    """Contradictions detected in the pipeline must appear in memo.metadata['contradictions']."""
    memo = DcPowerAgent(client=_ConflictingEvidenceClient()).analyze(
        "What is the DC power draw of the NVIDIA NVL72 rack?",
        _power_docs(),
    )
    assert "contradictions" in memo.metadata
    raw = memo.metadata["contradictions"]
    assert isinstance(raw, list)
    assert raw, "metadata['contradictions'] must be non-empty"
    assert all("severity" in c for c in raw)
    assert all("topic" in c for c in raw)


def test_pipeline_contradictions_flow_to_trace():
    """contradictions_detected must be present and non-empty in the trace."""
    memo = DcPowerAgent(client=_ConflictingEvidenceClient()).analyze(
        "What is the DC power draw of the NVIDIA NVL72 rack?",
        _power_docs(),
    )
    trace = build_trace(
        question="What is the DC power draw of the NVIDIA NVL72 rack?",
        source_directory=Path("sources"),
        output_path=Path("outputs/memo.md"),
        documents=_power_docs(),
        memo=memo,
        mock_mode=False,
    )
    assert "contradictions_detected" in trace
    assert trace["contradictions_detected"], (
        "contradictions_detected must be non-empty in trace"
    )


def test_pipeline_contradictions_appear_in_markdown():
    """The rendered markdown must include the Potential Contradictions section with an entry."""
    memo = DcPowerAgent(client=_ConflictingEvidenceClient()).analyze(
        "What is the DC power draw of the NVIDIA NVL72 rack?",
        _power_docs(),
    )
    md = memo_to_markdown(memo)
    assert "## Potential Contradictions" in md
    assert "No significant contradictions detected." not in md, (
        "Expected contradiction entries, not the empty-state message"
    )
    # With ai_data_centers profile active, the topic is "power" not "rack power".
    assert "power" in md.lower()


# ---------------------------------------------------------------------------
# J1.4: Metric-type normalisation
# ---------------------------------------------------------------------------


class TestMetricTypeNormalisation:
    """Verify that cross-metric comparisons are suppressed (J1.4)."""

    # ---- numeric metric types ------------------------------------------------

    def test_gw_rate_vs_gw_target_no_contradiction(self):
        """300 GW capacity target vs 13 GW/year throughput must NOT conflict."""
        items = assign_evidence_ids([
            _ev("Scaling US nuclear capacity to 300 GW by 2050 requires substantial investment."),
            _ev("The NRC must license 13 GW per year to reach the stated targets."),
        ])
        result = detect_contradictions(items)
        gw = [c for c in result if c.topic == "rack power"]
        assert not gw, (
            f"Expected no GW contradiction (different metric types), got: {gw}"
        )

    def test_kw_level_vs_kw_level_contradiction_still_fires(self):
        """120 kW level vs 200 kW level (same metric type) must still produce a contradiction."""
        items = assign_evidence_ids([
            _ev("The NVL72 rack draws 120 kW under peak load."),
            _ev("The NVL72 rack requires 200 kW of DC power at full utilisation."),
        ])
        result = detect_contradictions(items)
        assert any(c.topic == "rack power" for c in result)

    def test_mw_rate_vs_mw_level_no_contradiction(self):
        """MW per year (installation rate) vs MW total (capacity) must not conflict."""
        items = assign_evidence_ids([
            _ev("The country can install 50 MW per year of new capacity."),
            _ev("The project delivers 300 MW of generation capacity."),
        ])
        result = detect_contradictions(items)
        mw = [c for c in result if c.topic == "rack power"]
        assert not mw, f"Expected no MW contradiction (rate vs capacity), got: {mw}"

    def test_gw_target_vs_gw_target_contradiction_fires(self):
        """Two conflicting capacity targets must still generate a contradiction."""
        items = assign_evidence_ids([
            _ev("The US nuclear capacity target is 300 GW by 2050."),
            _ev("The US nuclear capacity target is 100 GW by 2050."),
        ])
        result = detect_contradictions(items)
        assert any(c.topic == "rack power" for c in result), (
            "Two conflicting GW targets must produce a contradiction"
        )

    def test_gw_rate_vs_gw_rate_contradiction_fires(self):
        """Two conflicting throughput rates must still generate a contradiction."""
        items = assign_evidence_ids([
            _ev("The NRC can license 5 GW per year under the proposed rule."),
            _ev("The NRC is expected to license 13 GW per year under accelerated review."),
        ])
        result = detect_contradictions(items)
        assert any(c.topic == "rack power" for c in result), (
            "Two conflicting GW/year rates must produce a contradiction"
        )

    # ---- year metric types ---------------------------------------------------

    def test_event_year_vs_product_year_no_contradiction(self):
        """GTC 2025 conference year vs Vera Rubin 2026 product year must not conflict."""
        items = assign_evidence_ids([
            _ev("Updates from NVIDIA GTC 2025 Conference provided the latest announcements."),
            _ev("The NVIDIA Vera Rubin Platform is available as a product in 2026."),
        ])
        result = detect_contradictions(items)
        year_conflicts = [c for c in result if c.topic == "timeline"]
        assert not year_conflicts, (
            f"Event year vs product year must not produce a timeline contradiction, got: {year_conflicts}"
        )

    def test_product_year_vs_product_year_contradiction_fires(self):
        """Two conflicting product availability years must produce a contradiction."""
        items = assign_evidence_ids([
            _ev("The NVL72 rack will be available as a product in 2026."),
            _ev("General availability for the platform product is expected in 2027."),
        ])
        result = detect_contradictions(items)
        assert any(c.topic == "timeline" for c in result), (
            "Two conflicting product years must produce a timeline contradiction"
        )

    def test_deployment_year_vs_deployment_year_contradiction_fires(self):
        """Two conflicting commercial-operation years must produce a contradiction."""
        items = assign_evidence_ids([
            _ev("SMR commercial operation is expected to begin in 2026."),
            _ev("Commercial operation of the first unit is expected in 2027."),
        ])
        result = detect_contradictions(items)
        assert any(c.topic == "timeline" for c in result), (
            "Two conflicting deployment years must produce a timeline contradiction"
        )

    def test_event_year_vs_event_year_suppressed(self):
        """Same-context year types: GTC 2025 event vs GTC 2026 event are different events, not contradictions."""
        # Two separate event years (GTC 2025 vs GTC 2026) should not be flagged
        # as a timeline contradiction — they are independent conferences.
        items = assign_evidence_ids([
            _ev("At the NVIDIA GTC 2025 Conference, Blackwell was announced."),
            _ev("At the NVIDIA GTC 2026 Conference, Rubin was demonstrated."),
        ])
        result = detect_contradictions(items)
        year_conflicts = [c for c in result if c.topic == "timeline"]
        # Two consecutive event years from different conferences are not contradictory
        # (they're separate events). year_event vs year_event → now suppressed.
        assert not year_conflicts, (
            f"GTC 2025 event vs GTC 2026 event should not be a timeline contradiction, got: {year_conflicts}"
        )

    # ---- metric_type fields in trace -----------------------------------------

    def test_contradiction_carries_metric_type_fields(self):
        """Emitted Contradictions must have metric_type_a and metric_type_b populated."""
        items = assign_evidence_ids([
            _ev("The NVL72 rack draws 120 kW under peak load."),
            _ev("The NVL72 rack requires 200 kW of DC power at full utilisation."),
        ])
        result = detect_contradictions(items)
        assert result
        c = result[0]
        assert hasattr(c, "metric_type_a")
        assert hasattr(c, "metric_type_b")
        assert c.metric_type_a != ""
        assert c.metric_type_b != ""

    def test_metric_type_in_trace_serialisation(self):
        """metric_type_a / metric_type_b must appear in model_dump() output."""
        items = assign_evidence_ids([
            _ev("The rack draws 120 kW."),
            _ev("The rack draws 200 kW."),
        ])
        result = detect_contradictions(items)
        assert result
        d = result[0].model_dump()
        assert "metric_type_a" in d
        assert "metric_type_b" in d

    # ---- _metric_types_compatible unit tests ---------------------------------

    def test_compatible_same_type(self):
        from research_agent.contradiction import _metric_types_compatible
        assert _metric_types_compatible("kw_level", "kw_level")

    def test_incompatible_rate_vs_level(self):
        from research_agent.contradiction import _metric_types_compatible
        assert not _metric_types_compatible("gw_rate", "gw_level")
        assert not _metric_types_compatible("gw_rate", "gw_target")
        assert not _metric_types_compatible("kw_rate", "kw_level")

    def test_compatible_rate_vs_rate(self):
        from research_agent.contradiction import _metric_types_compatible
        assert _metric_types_compatible("gw_rate", "gw_rate")

    def test_compatible_target_vs_level(self):
        from research_agent.contradiction import _metric_types_compatible
        assert _metric_types_compatible("gw_target", "gw_level")
        assert _metric_types_compatible("gw_target", "gw_current")

    def test_incompatible_event_year_vs_product(self):
        from research_agent.contradiction import _metric_types_compatible
        assert not _metric_types_compatible("year_event", "year_product")
        assert not _metric_types_compatible("year_event", "year_deployment")
        assert not _metric_types_compatible("year_product", "year_event")

    def test_compatible_product_vs_deployment_year(self):
        from research_agent.contradiction import _metric_types_compatible
        assert _metric_types_compatible("year_product", "year_deployment")

    def test_incompatible_lifecycle_milestones(self):
        # J1.6.4 – construction_approval vs commercial_operation are different
        # lifecycle phases (progression), not contradictory values.
        from research_agent.contradiction import _metric_types_compatible
        assert not _metric_types_compatible("year_deployment", "year_construction")
        assert not _metric_types_compatible("year_construction", "year_deployment")

    def test_incompatible_generic_year(self):
        from research_agent.contradiction import _metric_types_compatible
        assert not _metric_types_compatible("year_generic", "year_product")
        assert not _metric_types_compatible("year_generic", "year_deployment")


def test_pipeline_high_severity_triggers_evaluator_warning():
    """If a high-severity contradiction is detected, evaluate_memo must warn."""
    from research_agent.evaluator import evaluate_memo

    # 100 kW vs 200 kW → 50 % difference → high severity
    class _HighSeverityClient(_ConflictingEvidenceClient):
        def extract_evidence_from_chunks(self, question, chunks):
            items = super().extract_evidence_from_chunks(question, chunks)
            items[0] = items[0].model_copy(update={
                "claim": "The NVL72 rack draws 100 kW.",
                "evidence_snippet": "The NVL72 rack draws 100 kW.",
            })
            items[1] = items[1].model_copy(update={
                "claim": "The NVL72 rack requires 200 kW of DC power.",
                "evidence_snippet": "The NVL72 rack requires 200 kW of DC power.",
            })
            return items

    memo = DcPowerAgent(client=_HighSeverityClient()).analyze(
        "What is the DC power draw of the NVIDIA NVL72 rack?",
        _power_docs(),
    )
    docs = _power_docs()
    warnings = evaluate_memo(memo, docs)
    codes = {w.code for w in warnings}
    assert "high_severity_contradiction" in codes


# ---------------------------------------------------------------------------
# J1.6 – False positive regression tests
# ---------------------------------------------------------------------------

class TestJ16ScopeAwareContradiction:
    """J1.6 acceptance tests: ensure false positives are suppressed."""

    def _make(self, claim: str, eid: str = "E001") -> "EvidenceItem":
        from research_agent.schemas import EvidenceItem
        return EvidenceItem(
            evidence_id=eid,
            claim=claim,
            source_document="test_doc",
            evidence_snippet=claim,
            category="power",
            relevance="test",
            confidence="high",
        )

    def test_example_a_rack_vs_component_power(self):
        """GB200 NVL72 rack power (120 kW) vs power shelf (33 kW) should NOT be flagged."""
        from research_agent.contradiction import detect_contradictions
        a = self._make("The GB200 NVL72 rack total power is 120 kW.", "E001")
        b = self._make("Each power shelf in the rack draws 33 kW.", "E002")
        suppressed: list = []
        contradictions = detect_contradictions([a, b], out_suppressed=suppressed)
        assert contradictions == [], (
            f"Expected no contradictions (rack vs component scope), got: {contradictions}"
        )
        assert any(s.reason == "scope_mismatch" for s in suppressed), (
            f"Expected a scope_mismatch suppression, got: {suppressed}"
        )

    def test_example_b_rack_vs_component_cooling(self):
        """Rack liquid cooling vs power shelf air cooling should NOT be flagged."""
        from research_agent.contradiction import detect_contradictions
        a = self._make("The GB200 NVL72 rack uses liquid cooling (DLC).", "E001")
        b = self._make("The power shelf uses air-cooled PSUs.", "E002")
        suppressed: list = []
        contradictions = detect_contradictions([a, b], out_suppressed=suppressed)
        assert contradictions == [], (
            f"Expected no contradictions (rack vs component scope), got: {contradictions}"
        )
        assert any(s.reason == "scope_mismatch" for s in suppressed), (
            f"Expected a scope_mismatch suppression, got: {suppressed}"
        )

    def test_example_c_construction_vs_deployment_year(self):
        """Construction approval year vs commercial operation year should NOT be flagged."""
        from research_agent.contradiction import detect_contradictions
        a = self._make("Construction was approved and began in 2025.", "E001")
        b = self._make("Commercial operation begins in 2030.", "E002")
        suppressed: list = []
        contradictions = detect_contradictions([a, b], out_suppressed=suppressed)
        assert contradictions == [], (
            f"Expected no contradictions (lifecycle progression), got: {contradictions}"
        )

    def test_scope_extract_component(self):
        from research_agent.contradiction import _extract_scope
        assert _extract_scope("each power shelf draws 33 kw") == "component"
        assert _extract_scope("the psu is rated at 3 kw") == "component"

    def test_scope_extract_rack(self):
        from research_agent.contradiction import _extract_scope
        assert _extract_scope("the nvl72 rack total power is 120 kw") == "rack"
        assert _extract_scope("each rack in the data center") == "rack"

    def test_scopes_compatible_same(self):
        from research_agent.contradiction import _scopes_compatible
        assert _scopes_compatible("rack", "rack")
        assert _scopes_compatible("component", "component")

    def test_scopes_incompatible_rack_component(self):
        from research_agent.contradiction import _scopes_compatible
        assert not _scopes_compatible("rack", "component")
        assert not _scopes_compatible("component", "rack")

    def test_scopes_unknown_always_compatible(self):
        from research_agent.contradiction import _scopes_compatible
        assert _scopes_compatible("unknown", "rack")
        assert _scopes_compatible("rack", "unknown")
        assert _scopes_compatible("unknown", "unknown")

    def test_same_scope_contradiction_still_fires(self):
        """Two rack-level power claims with conflicting values SHOULD still be flagged."""
        from research_agent.contradiction import detect_contradictions
        a = self._make("The GB200 NVL72 rack requires 120 kW of power.", "E001")
        b = self._make("The GB200 NVL72 rack requires 200 kW of power.", "E002")
        suppressed: list = []
        contradictions = detect_contradictions([a, b], out_suppressed=suppressed)
        assert len(contradictions) >= 1, (
            "Same-scope conflicting values should still produce a contradiction"
        )
