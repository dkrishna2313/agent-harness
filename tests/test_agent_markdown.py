from pathlib import Path

from dc_power_agent.agent import (
    DcPowerAgent,
    extract_evidence,
    rank_evidence_items,
    score_evidence_items,
    select_top_evidence,
)
from dc_power_agent.claude_client import MockClaudeClient
from dc_power_agent.markdown import memo_to_markdown
from dc_power_agent.schemas import EvidenceItem, ResearchMemo, ResearchPlan, SourceDocument


def test_agent_generates_required_memo_sections():
    document = SourceDocument(
        path=Path("sources/rubin.txt"),
        title="rubin",
        extension=".txt",
        text=(
            "NVIDIA Rubin rack systems imply higher power density. "
            "Liquid cooling and thermal design are important for AI factories."
        ),
    )

    memo = DcPowerAgent(client=MockClaudeClient()).analyze("Explain Rubin power", [document])
    rendered = memo_to_markdown(memo)

    for heading in [
        "## Executive Summary",
        "## Confirmed Facts",
        "## Inferences",
        "## Power Implications",
        "## Cooling Implications",
        "## Open Questions",
        "## Source Notes",
        "## Evaluation Warnings",
    ]:
        assert heading in rendered

    assert "mock Claude client" in rendered
    assert "rubin" in rendered
    assert "[Source: rubin.txt, Evidence: E001]" in rendered


def test_source_notes_render_structured_evidence():
    memo = ResearchMemo(
        title="Research Memo",
        question="Explain Rubin power",
        executive_summary="Summary.",
        source_notes=[
            EvidenceItem(
                evidence_id="E001",
                claim="Power note: Rubin racks increase power density.",
                source_document="rubin.md",
                evidence_snippet="Rubin racks increase power density for AI factory deployments.",
                category="power",
                relevance="Directly relevant to the question.",
                confidence="high",
            )
        ],
    )

    rendered = memo_to_markdown(memo)

    assert "### rubin.md" in rendered
    assert "**Evidence ID:** E001" in rendered
    assert "Rubin racks increase power density" in rendered
    assert "**Category:** power" in rendered
    assert "**Relevance:** Directly relevant to the question." in rendered
    assert "**Confidence:** high" in rendered


def test_mock_evidence_extraction_creates_three_notes_for_short_sources():
    document = SourceDocument(
        path=Path("sources/rubin.md"),
        title="rubin",
        extension=".md",
        text="NVIDIA Rubin sources mention rack-scale power and cooling.",
    )

    evidence = extract_evidence("Explain Rubin power", [document])

    assert 3 <= len(evidence) <= 8
    assert {item.source_document for item in evidence} == {"rubin.md"}
    assert all(item.evidence_snippet for item in evidence)
    assert [item.evidence_id for item in evidence[:3]] == ["E001", "E002", "E003"]
    assert {"power", "cooling", "rack architecture"}.issubset(
        {item.category for item in evidence}
    )


def test_mock_evidence_extraction_prefers_required_topic_coverage():
    document = SourceDocument(
        path=Path("sources/rubin.md"),
        title="rubin",
        extension=".md",
        text=(
            "Rubin architecture uses GPU and CPU components for AI compute. "
            "Power distribution and energy buffering affect the data center. "
            "Liquid cooling and thermal design remove heat. "
            "Networking uses NVLink, Ethernet, and InfiniBand switches. "
            "Rack architecture uses rack-scale NVL72 systems."
        ),
    )

    evidence = extract_evidence("Explain Rubin architecture power cooling networking rack", [document])
    categories = {item.category for item in evidence}

    assert {
        "architecture",
        "power",
        "cooling",
        "networking",
        "rack architecture",
    }.issubset(categories)


def test_evidence_scoring_scores_specific_relevant_vendor_evidence_higher():
    evidence = [
        EvidenceItem(
            evidence_id="E001",
            claim="NVIDIA Rubin NVL72 rack power reaches concrete facility limits.",
            source_document="nvidia_rubin_spec.pdf",
            evidence_snippet="NVIDIA Rubin NVL72 rack architecture uses 120kW rack power with liquid cooling.",
            category="power",
            relevance="Directly relevant to the question.",
            confidence="high",
        ),
        EvidenceItem(
            evidence_id="E002",
            claim="Rubin is important.",
            source_document="commentary_blog.md",
            evidence_snippet="Rubin could be a major next-generation AI platform.",
            category="other",
            relevance="General context.",
            confidence="low",
        ),
    ]

    scored = score_evidence_items("Explain Rubin rack power", evidence)

    assert scored[0].relevance_score == 5
    assert scored[0].source_quality_score == 5
    assert scored[0].specificity_score >= 4
    assert scored[0].overall_score > scored[1].overall_score


def test_evidence_ranking_sorts_by_overall_score():
    evidence = [
        _scored_item("E001", 2.1),
        _scored_item("E002", 4.8),
        _scored_item("E003", 3.5),
    ]

    ranked = rank_evidence_items(evidence)

    assert [item.evidence_id for item in ranked] == ["E002", "E003", "E001"]


def test_top_evidence_filtering_uses_ranked_items():
    evidence = [
        _scored_item("E001", 2.1),
        _scored_item("E002", 4.8),
        _scored_item("E003", 3.5),
    ]

    selected = select_top_evidence(evidence, top_n=2)

    assert [item.evidence_id for item in selected] == ["E002", "E003"]


def test_agent_limits_ranked_evidence_before_synthesis():
    client = _RecordingClient()
    document = SourceDocument(
        path=Path("sources/rubin.md"),
        title="rubin",
        extension=".md",
        text="Rubin rack architecture affects power, cooling, and networking.",
    )

    memo = DcPowerAgent(client=client, top_evidence=2).analyze(
        "Explain Rubin rack power",
        [document],
    )

    assert client.synthesis_evidence_count == 2
    assert memo.metadata["evidence_items_used_for_synthesis"] == 2
    assert len(memo.evidence) == 3


def test_mock_agent_small_top_evidence_still_populates_all_implication_sections():
    """Regression: top_evidence=2 must not silence categories absent from the top-2 items."""
    document = SourceDocument(
        path=Path("sources/rubin.md"),
        title="rubin",
        extension=".md",
        text=(
            "Rubin architecture uses GPU and CPU accelerators for AI compute. "
            "Power distribution and energy buffering affect the data center. "
            "Liquid cooling and thermal design remove heat efficiently. "
            "Networking uses NVLink, Ethernet, and InfiniBand switches. "
            "Rack architecture uses rack-scale NVL72 systems with high power density."
        ),
    )

    memo = DcPowerAgent(client=MockClaudeClient(), top_evidence=2).analyze(
        "Explain Rubin power cooling networking rack architecture",
        [document],
    )

    fallback_power = "No direct source excerpt about power distribution was selected in the mock pass."
    fallback_cooling = "No direct source excerpt about cooling infrastructure was selected in the mock pass."
    fallback_networking = "No direct source excerpt about networking infrastructure was selected in the mock pass."
    fallback_rack = "No direct source excerpt about rack architecture was selected in the mock pass."

    assert memo.power_implications != [fallback_power], "power_implications should not fall back to placeholder"
    assert memo.cooling_implications != [fallback_cooling], "cooling_implications should not fall back to placeholder"
    assert memo.networking_implications != [fallback_networking], "networking_implications should not fall back to placeholder"
    assert memo.rack_architecture_implications != [fallback_rack], "rack_architecture_implications should not fall back to placeholder"

    # Synthesis window is respected
    assert memo.metadata["evidence_items_used_for_synthesis"] == 2

    # Source notes expose all evidence, not just the synthesis window
    assert len(memo.source_notes) > 2


def test_mock_agent_small_top_evidence_citation_ids_survive_filtering():
    """Regression: citation IDs in implication sections must appear in source_notes."""
    document = SourceDocument(
        path=Path("sources/rubin.md"),
        title="rubin",
        extension=".md",
        text=(
            "Power distribution and energy buffering affect the data center at 120kW rack power. "
            "Liquid cooling removes heat from 72-GPU NVL72 rack-scale systems. "
            "Networking uses NVLink and InfiniBand switches for cluster connectivity. "
            "Rack architecture uses cabinet-level power distribution and CDU cooling."
        ),
    )

    memo = DcPowerAgent(client=MockClaudeClient(), top_evidence=2).analyze(
        "Explain Rubin power cooling networking rack",
        [document],
    )

    import re
    citation_pattern = re.compile(r"\[Source:\s*[^,\]]+,\s*Evidence:\s*(E\d{3})\]")
    known_ids = {item.evidence_id for item in memo.source_notes if item.evidence_id}

    for field_name in ["confirmed_facts", "power_implications", "cooling_implications",
                       "networking_implications", "rack_architecture_implications"]:
        for entry in getattr(memo, field_name):
            for cited_id in citation_pattern.findall(entry):
                assert cited_id in known_ids, (
                    f"{field_name} cites {cited_id} which is not in source_notes "
                    f"(known: {sorted(known_ids)})"
                )


def test_claude_path_citation_ids_survive_top_evidence_filtering():
    """Regression: citations in Claude-synthesized memo must be resolvable against full source_notes."""
    client = _RecordingClient()
    document = SourceDocument(
        path=Path("sources/rubin.md"),
        title="rubin",
        extension=".md",
        text="Rubin rack architecture affects power, cooling, and networking.",
    )

    memo = DcPowerAgent(client=client, top_evidence=2).analyze(
        "Explain Rubin rack power",
        [document],
    )

    import re
    citation_pattern = re.compile(r"\[Source:\s*[^,\]]+,\s*Evidence:\s*(E\d{3})\]")
    known_ids = {item.evidence_id for item in memo.source_notes if item.evidence_id}

    # The recording client emits citations using the first synthesis item's ID.
    # That ID must be present in the full source_notes set (not just the 2-item window).
    for field_name in ["confirmed_facts", "power_implications", "rack_architecture_implications"]:
        for entry in getattr(memo, field_name):
            for cited_id in citation_pattern.findall(entry):
                assert cited_id in known_ids, (
                    f"{field_name} cites {cited_id} not found in source_notes "
                    f"(known: {sorted(known_ids)})"
                )


def _scored_item(evidence_id: str, overall_score: float) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        claim=f"Claim {evidence_id}.",
        source_document="rubin.md",
        evidence_snippet="Rubin rack architecture affects power and cooling.",
        category="power",
        relevance="Relevant.",
        confidence="medium",
        relevance_score=4,
        source_quality_score=3,
        specificity_score=3,
        overall_score=overall_score,
    )


class _RecordingClient:
    is_mock = False
    model = "recording-client"

    def __init__(self) -> None:
        self.call_traces = []
        self.synthesis_evidence_count = 0

    def create_research_plan(self, _question, _documents):
        return ResearchPlan(
            research_questions=["What matters?"],
            key_topics=["power"],
            source_priorities=["rubin.md"],
        )

    def extract_evidence_from_chunks(self, question, chunks):
        self.chunks_seen = list(chunks)
        return self.extract_evidence(question, chunks)

    def extract_evidence(self, _question, _documents):
        return [
            EvidenceItem(
                claim="NVIDIA Rubin NVL72 rack power reaches 120kW.",
                source_document="nvidia_rubin_spec.pdf",
                evidence_snippet="NVIDIA Rubin NVL72 rack power reaches 120kW with liquid cooling.",
                category="power",
                relevance="Directly relevant.",
                confidence="high",
            ),
            EvidenceItem(
                claim="NVIDIA Rubin networking uses NVLink.",
                source_document="nvidia_networking_spec.pdf",
                evidence_snippet="NVIDIA Rubin networking uses NVLink and Ethernet switching.",
                category="networking",
                relevance="Relevant infrastructure context.",
                confidence="high",
            ),
            EvidenceItem(
                claim="Rubin may matter.",
                source_document="commentary_blog.md",
                evidence_snippet="Rubin may be an important AI platform.",
                category="other",
                relevance="General context.",
                confidence="low",
            ),
        ]

    def synthesize_memo(self, question, evidence_items):
        self.synthesis_evidence_count = len(evidence_items)
        citation = f"[Source: {evidence_items[0].source_document}, Evidence: {evidence_items[0].evidence_id}]"
        return ResearchMemo(
            title=f"Research Memo: {question}",
            question=question,
            executive_summary="Summary.",
            confirmed_facts=[f"Fact. {citation}"],
            inferences=["Inference."],
            power_implications=[f"Power. {citation}"],
            cooling_implications=[],
            networking_implications=[],
            rack_architecture_implications=[f"Rack. {citation}"],
            open_questions=["Question."],
            source_notes=list(evidence_items),
            evidence=list(evidence_items),
        )
