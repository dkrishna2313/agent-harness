"""Tests for frozen J8.0 ontology models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from knowledge.extractor import _classify_evidence_type
from knowledge.models import (
    Contradiction,
    Evidence,
    ExtractionRun,
    KnowledgeMetadata,
    Source,
    SourceManifestEntry,
)


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


def _make_source(**kwargs) -> Source:
    text = kwargs.pop("canonical_text", "The reactor achieves 300 MWe output.")
    fingerprint = Source.compute_fingerprint(text)
    defaults = dict(
        source_id=Source.compute_source_id(fingerprint),
        uri="smr_sources/test.pdf",
        title="Test Source",
        retrieved_date="2026-06-28",
        fingerprint=fingerprint,
        document_type="PDF",
        domain="smr",
        canonical_text=text,
    )
    defaults.update(kwargs)
    return Source(**defaults)


def test_source_is_frozen():
    s = _make_source()
    with pytest.raises((ValidationError, TypeError)):
        s.title = "changed"


def test_source_fingerprint_deterministic():
    text = "The reactor achieves 300 MWe output."
    fp1 = Source.compute_fingerprint(text)
    fp2 = Source.compute_fingerprint(text)
    assert fp1 == fp2
    assert len(fp1) == 64


def test_source_id_from_fingerprint():
    text = "Consistent text for ID derivation."
    fp = Source.compute_fingerprint(text)
    sid = Source.compute_source_id(fp)
    assert sid == fp[:32]
    assert len(sid) == 32


def test_source_char_count():
    text = "hello world"
    s = _make_source(canonical_text=text)
    assert s.char_count == len(text)


def test_source_content_addressed_dedup():
    # Two files with same content → same source_id
    text = "Identical content."
    s1 = _make_source(canonical_text=text, uri="path/a.pdf")
    s2 = _make_source(canonical_text=text, uri="path/b.pdf")
    assert s1.source_id == s2.source_id
    assert s1.fingerprint == s2.fingerprint


def test_source_different_content_different_id():
    s1 = _make_source(canonical_text="Content A")
    s2 = _make_source(canonical_text="Content B")
    assert s1.source_id != s2.source_id


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def _make_evidence(**kwargs) -> Evidence:
    defaults = dict(
        statement="BWRX-300 has a thermal output of 300 MWt.",
        supporting_source_ids=["abc123"],
        extraction_run_id="run-001",
        category="reactor design",
    )
    defaults.update(kwargs)
    return Evidence(**defaults)


def test_evidence_is_frozen():
    ev = _make_evidence()
    with pytest.raises((ValidationError, TypeError)):
        ev.statement = "changed"


def test_evidence_has_unique_id():
    ev1 = _make_evidence()
    ev2 = _make_evidence()
    assert ev1.evidence_id != ev2.evidence_id


def test_evidence_statement_fingerprint_deterministic():
    ev = _make_evidence(statement="The reactor outputs 300 MWe.")
    fp1 = ev.statement_fingerprint
    ev2 = _make_evidence(statement="The reactor outputs 300 MWe.")
    assert fp1 == ev2.statement_fingerprint


def test_evidence_fingerprint_case_insensitive():
    ev1 = _make_evidence(statement="BWRX-300 has 300 MWt output.")
    ev2 = _make_evidence(statement="bwrx-300 has 300 mwt output.")
    assert ev1.statement_fingerprint == ev2.statement_fingerprint


def test_evidence_supersedes_chain():
    ev_old = _make_evidence(statement="Old claim.")
    ev_new = _make_evidence(
        statement="Improved claim.",
        supersedes=[ev_old.evidence_id],
    )
    assert ev_old.evidence_id in ev_new.supersedes
    assert ev_new.superseded_by is None
    assert ev_old.superseded_by is None  # immutable; superseded_by set on creation


def test_evidence_supports_multiple_sources():
    ev = _make_evidence(supporting_source_ids=["src1", "src2", "src3"])
    assert len(ev.supporting_source_ids) == 3


def test_evidence_supports_multiple_profiles():
    ev = _make_evidence(profile_ids=["smr", "economics", "nuclear_policy"])
    assert "economics" in ev.profile_ids


def test_evidence_type_default():
    ev = _make_evidence()
    assert ev.evidence_type == "STRATEGIC"


def test_evidence_type_explicit():
    for etype in ("STRATEGIC", "TECHNICAL", "PROVENANCE", "ADMINISTRATIVE"):
        ev = _make_evidence(evidence_type=etype)
        assert ev.evidence_type == etype


# ---------------------------------------------------------------------------
# Evidence type classifier
# ---------------------------------------------------------------------------


def test_classify_administrative_revision():
    assert _classify_evidence_type("The document is revision H, dated October 2025.", "") == "ADMINISTRATIVE"


def test_classify_administrative_copyright():
    assert _classify_evidence_type("Copyright 2025 GE Vernova. All rights reserved.", "") == "ADMINISTRATIVE"


def test_classify_administrative_trademark():
    assert _classify_evidence_type("GE is a trademark of General Electric Company.", "") == "ADMINISTRATIVE"


def test_classify_provenance_authored():
    assert _classify_evidence_type("This report was authored by the Idaho National Laboratory.", "") == "PROVENANCE"


def test_classify_technical_by_units():
    assert _classify_evidence_type("The BWRX-300 produces 300 MWe of electricity.", "") == "TECHNICAL"


def test_classify_technical_by_category():
    assert _classify_evidence_type("The reactor uses natural circulation cooling.", "reactor design") == "TECHNICAL"


def test_classify_strategic_deployment_risk():
    t = "HALEU fuel availability is a critical deployment risk for advanced SMRs."
    assert _classify_evidence_type(t, "") == "STRATEGIC"


def test_classify_strategic_market():
    t = "Several utilities have announced interest in SMR procurement for the 2030s."
    assert _classify_evidence_type(t, "") == "STRATEGIC"


# ---------------------------------------------------------------------------
# KnowledgeMetadata
# ---------------------------------------------------------------------------


def test_metadata_is_mutable():
    meta = KnowledgeMetadata(evidence_id="ev-001")
    meta.state = "SUPERSEDED"  # should not raise
    assert meta.state == "SUPERSEDED"


def test_metadata_compute_overall_score():
    meta = KnowledgeMetadata(
        evidence_id="ev-001",
        relevance_score=4.0,
        source_quality_score=3.0,
        specificity_score=5.0,
    )
    assert meta.compute_overall_score() == 4.0


def test_metadata_defaults():
    meta = KnowledgeMetadata(evidence_id="ev-001")
    assert meta.state == "ACTIVE"
    assert meta.review_status == "UNREVIEWED"
    assert meta.version == 1
    assert meta.confidence == 0.5
    assert meta.retrieval_enabled is True
    assert meta.retrieval_priority == 3
    assert meta.strategic_value == 0.5


def test_metadata_retrieval_fields_explicit():
    meta = KnowledgeMetadata(
        evidence_id="ev-001",
        retrieval_enabled=False,
        retrieval_priority=1,
        strategic_value=0.05,
    )
    assert meta.retrieval_enabled is False
    assert meta.retrieval_priority == 1
    assert meta.strategic_value == 0.05


def test_metadata_valid_states():
    for state in ("ACTIVE", "SUPERSEDED", "LOW_CONFIDENCE", "RETRACTED", "ARCHIVED"):
        meta = KnowledgeMetadata(evidence_id="ev", state=state)
        assert meta.state == state


# ---------------------------------------------------------------------------
# ExtractionRun
# ---------------------------------------------------------------------------


def test_extraction_run_defaults():
    run = ExtractionRun(model_version="claude-sonnet-4-6", prompt_version="kb-v1.0")
    assert run.status == "RUNNING"
    assert run.run_id != ""
    assert run.evidence_ids_produced == []


def test_extraction_run_narrow_scope():
    # ExtractionRun only tracks evidence construction — no general audit fields
    run = ExtractionRun(model_version="claude-sonnet-4-6", prompt_version="kb-v1.0")
    assert hasattr(run, "model_version")
    assert hasattr(run, "prompt_version")
    assert hasattr(run, "evidence_ids_produced")
    assert not hasattr(run, "event_type")  # not a general audit log


# ---------------------------------------------------------------------------
# Contradiction
# ---------------------------------------------------------------------------


def test_contradiction_structure():
    c = Contradiction(
        evidence_id_a="ev-a",
        evidence_id_b="ev-b",
        contradiction_type="DIRECT",
    )
    assert c.resolution_status == "OPEN"
    assert c.severity == "MEDIUM"
    assert c.contradiction_id != ""


# ---------------------------------------------------------------------------
# SourceManifestEntry
# ---------------------------------------------------------------------------


def test_manifest_entry_roundtrip():
    entry = SourceManifestEntry(
        source_id="abc123",
        fingerprint="deadbeef" * 8,
        domain="smr",
        uri="/path/to/file.pdf",
        evidence_ids=["ev-1", "ev-2"],
        extraction_run_id="run-42",
    )
    data = entry.model_dump()
    restored = SourceManifestEntry.model_validate(data)
    assert restored.source_id == entry.source_id
    assert restored.evidence_ids == ["ev-1", "ev-2"]
