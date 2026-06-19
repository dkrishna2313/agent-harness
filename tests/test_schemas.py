import pytest
from pydantic import ValidationError

from dc_power_agent.schemas import EvidenceItem


def test_evidence_item_schema_accepts_expected_fields():
    item = EvidenceItem(
        claim="Rubin systems affect rack power planning.",
        evidence_id="E001",
        source_document="rubin.md",
        evidence_snippet="Rubin rack systems imply higher power density.",
        category="power",
        relevance="Directly relevant to the user question.",
        confidence="high",
    )

    assert item.source_document == "rubin.md"
    assert item.confidence == "high"
    assert item.overall_score == 3.0


def test_evidence_item_rejects_unknown_confidence():
    with pytest.raises(ValidationError):
        EvidenceItem(
            claim="Claim.",
            source_document="rubin.md",
            evidence_snippet="Evidence.",
            category="power",
            relevance="Relevant.",
            confidence="certain",
        )


def test_evidence_item_rejects_unknown_category():
    with pytest.raises(ValidationError):
        EvidenceItem(
            claim="Claim.",
            source_document="rubin.md",
            evidence_snippet="Evidence.",
            category="finance",
            relevance="Relevant.",
            confidence="medium",
        )


def test_evidence_item_rejects_out_of_range_score():
    with pytest.raises(ValidationError):
        EvidenceItem(
            claim="Claim.",
            source_document="rubin.md",
            evidence_snippet="Evidence.",
            category="power",
            relevance="Relevant.",
            confidence="medium",
            relevance_score=6,
        )
