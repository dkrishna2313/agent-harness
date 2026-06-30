"""Tests for KnowledgeStore persistence layer."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest

from knowledge.models import (
    Evidence,
    ExtractionRun,
    KnowledgeMetadata,
    Source,
    SourceManifestEntry,
)
from knowledge.store import KnowledgeStore


@pytest.fixture()
def store(tmp_path: Path) -> KnowledgeStore:
    return KnowledgeStore(root=tmp_path / "ks")


def _make_source(domain: str = "smr", text: str = "Test text.") -> Source:
    fp = Source.compute_fingerprint(text)
    return Source(
        source_id=Source.compute_source_id(fp),
        uri="test.pdf",
        title="Test",
        retrieved_date=date.today(),
        fingerprint=fp,
        document_type="PDF",
        domain=domain,
        canonical_text=text,
    )


def _make_evidence(run_id: str = "r1", statement: str = "A claim.") -> Evidence:
    return Evidence(statement=statement, supporting_source_ids=["src1"], extraction_run_id=run_id)


def _make_metadata(evidence_id: str) -> KnowledgeMetadata:
    return KnowledgeMetadata(evidence_id=evidence_id)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_store_creates_directories(store: KnowledgeStore):
    assert (store.root / "sources").exists()
    assert (store.root / "evidence").exists()
    assert (store.root / "metadata").exists()
    assert (store.root / "extraction_runs").exists()
    assert (store.root / "manifests").exists()
    assert (store.root / "embeddings" / "evidence").exists()
    assert (store.root / "_meta" / "schema_version.json").exists()


def test_schema_version_written(store: KnowledgeStore):
    data = json.loads((store.root / "_meta" / "schema_version.json").read_text())
    assert data["schema_version"] == "1.0.0"


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


def test_write_and_read_source(store: KnowledgeStore):
    s = _make_source(text="Unique text for source read test.")
    store.write_source(s)
    recovered = store.read_source(s.domain, s.source_id)
    assert recovered is not None
    assert recovered.source_id == s.source_id
    assert recovered.fingerprint == s.fingerprint


def test_has_source(store: KnowledgeStore):
    s = _make_source(text="Has source test.")
    assert not store.has_source(s.domain, s.source_id)
    store.write_source(s)
    assert store.has_source(s.domain, s.source_id)


def test_iter_sources(store: KnowledgeStore):
    sources = [_make_source(text=f"Source {i}.") for i in range(3)]
    for s in sources:
        store.write_source(s)
    found = list(store.iter_sources("smr"))
    assert len(found) == 3


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def test_write_and_read_evidence(store: KnowledgeStore):
    ev = _make_evidence()
    store.write_evidence(ev, "smr")
    recovered = store.read_evidence("smr", ev.evidence_id)
    assert recovered is not None
    assert recovered.evidence_id == ev.evidence_id
    assert recovered.statement == ev.statement


def test_evidence_batch_write(store: KnowledgeStore):
    items = [_make_evidence(statement=f"Claim {i}.") for i in range(5)]
    store.write_evidence_batch(items, "smr")
    assert store.evidence_count("smr") == 5


def test_evidence_count(store: KnowledgeStore):
    assert store.evidence_count("smr") == 0
    store.write_evidence(_make_evidence(statement="First."), "smr")
    store.write_evidence(_make_evidence(statement="Second."), "smr")
    assert store.evidence_count("smr") == 2


def test_iter_evidence(store: KnowledgeStore):
    items = [_make_evidence(statement=f"Statement {i}.") for i in range(4)]
    store.write_evidence_batch(items, "smr")
    recovered = list(store.iter_evidence("smr"))
    assert len(recovered) == 4


def test_statement_fingerprints(store: KnowledgeStore):
    ev1 = _make_evidence(statement="Claim alpha.")
    ev2 = _make_evidence(statement="Claim beta.")
    store.write_evidence_batch([ev1, ev2], "smr")
    fps = store.get_statement_fingerprints("smr")
    assert ev1.statement_fingerprint in fps
    assert ev2.statement_fingerprint in fps


def test_evidence_cross_domain_isolation(store: KnowledgeStore):
    ev_smr = _make_evidence(statement="SMR claim.")
    ev_ai = _make_evidence(statement="AI claim.")
    store.write_evidence(ev_smr, "smr")
    store.write_evidence(ev_ai, "ai_data_centers")
    assert store.evidence_count("smr") == 1
    assert store.evidence_count("ai_data_centers") == 1


# ---------------------------------------------------------------------------
# KnowledgeMetadata
# ---------------------------------------------------------------------------


def test_write_and_iter_metadata(store: KnowledgeStore):
    ev = _make_evidence()
    meta = _make_metadata(ev.evidence_id)
    store.write_metadata(meta, "smr")
    found = list(store.iter_metadata("smr"))
    assert len(found) == 1
    assert found[0].evidence_id == ev.evidence_id


def test_metadata_batch_write(store: KnowledgeStore):
    metas = [_make_metadata(f"ev-{i}") for i in range(5)]
    store.write_metadata_batch(metas, "smr")
    found = list(store.iter_metadata("smr"))
    assert len(found) == 5


# ---------------------------------------------------------------------------
# ExtractionRun
# ---------------------------------------------------------------------------


def test_write_and_iter_extraction_runs(store: KnowledgeStore):
    run = ExtractionRun(model_version="test-model", prompt_version="v1")
    store.write_extraction_run(run)
    runs = list(store.iter_extraction_runs())
    assert len(runs) == 1
    assert runs[0].run_id == run.run_id


def test_latest_extraction_run(store: KnowledgeStore):
    run1 = ExtractionRun(model_version="m", prompt_version="v1")
    run2 = ExtractionRun(model_version="m", prompt_version="v2")
    store.write_extraction_run(run1)
    store.write_extraction_run(run2)
    latest = store.latest_extraction_run()
    assert latest is not None
    assert latest.run_id == run2.run_id


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_roundtrip(store: KnowledgeStore):
    entry = SourceManifestEntry(
        source_id="src1",
        fingerprint="fp1",
        domain="smr",
        uri="/path/file.pdf",
        evidence_ids=["ev1"],
        extraction_run_id="run1",
    )
    store.update_manifest_entry(entry)
    manifest = store.load_manifest()
    assert "src1" in manifest
    assert manifest["src1"].fingerprint == "fp1"
    assert manifest["src1"].evidence_ids == ["ev1"]


def test_manifest_empty_initially(store: KnowledgeStore):
    assert store.load_manifest() == {}


def test_manifest_update_preserves_others(store: KnowledgeStore):
    e1 = SourceManifestEntry(source_id="s1", fingerprint="f1", domain="smr", uri="a.pdf")
    e2 = SourceManifestEntry(source_id="s2", fingerprint="f2", domain="smr", uri="b.pdf")
    store.update_manifest_entry(e1)
    store.update_manifest_entry(e2)
    manifest = store.load_manifest()
    assert len(manifest) == 2
    assert "s1" in manifest
    assert "s2" in manifest


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def test_has_embedding_false_initially(store: KnowledgeStore):
    assert not store.has_embedding("nonexistent-id")


def test_write_and_read_embedding(store: KnowledgeStore):
    vector = [0.1, 0.2, 0.3, 0.4]
    store.write_embedding("ev-test", vector)
    assert store.has_embedding("ev-test")
    recovered = store.read_embedding("ev-test")
    assert recovered is not None
    assert len(recovered) == 4
    assert abs(recovered[0] - 0.1) < 1e-5


def test_read_missing_embedding_returns_none(store: KnowledgeStore):
    assert store.read_embedding("does-not-exist") is None


# ---------------------------------------------------------------------------
# Atomic write integrity
# ---------------------------------------------------------------------------


def test_write_source_is_idempotent(store: KnowledgeStore):
    s = _make_source(text="Idempotent source.")
    store.write_source(s)
    store.write_source(s)  # second write should overwrite cleanly
    assert store.has_source(s.domain, s.source_id)
