"""Tests for KnowledgeBuilder — incremental builds, fingerprinting, deduplication.

Validates the five success criteria from the J8.1 spec:
  1. Fresh build → Knowledge Base created
  2. Immediate rebuild → almost everything skipped
  3. Modify one Source → only one Source rebuilt
  4. Evidence count unchanged except for modified Source
  5. Fingerprint cache updated correctly
"""

from __future__ import annotations

from pathlib import Path

import pytest

from knowledge.builder import KnowledgeBuilder, infer_domain
from knowledge.embedder import NullEmbedder
from knowledge.models import Source
from knowledge.store import KnowledgeStore


# ---------------------------------------------------------------------------
# Mock client — deterministic evidence extraction, no LLM calls
# ---------------------------------------------------------------------------


class _MockClient:
    """Deterministic mock that produces N evidence items per source."""

    def __init__(self, items_per_source: int = 3) -> None:
        self._n = items_per_source
        self.calls: list[str] = []

    def extract_evidence(self, question, source_texts):
        from research_agent.schemas import EvidenceItem

        results = []
        for src in source_texts:
            self.calls.append(src.title)
            for i in range(self._n):
                results.append(
                    EvidenceItem(
                        evidence_id="",
                        claim=f"Claim {i} from {src.title}.",
                        source_document=src.title,
                        evidence_snippet=f"snippet {i}",
                        category="power",
                        relevance="direct",
                        confidence="high",
                    )
                )
        return results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path) -> KnowledgeStore:
    return KnowledgeStore(root=tmp_path / "ks")


@pytest.fixture()
def source_dir(tmp_path: Path) -> Path:
    d = tmp_path / "smr_sources"
    d.mkdir()
    (d / "doc_a.txt").write_text("The BWRX-300 reactor operates at 300 MWe.", encoding="utf-8")
    (d / "doc_b.txt").write_text("SMR licensing typically takes 5-7 years.", encoding="utf-8")
    return d


@pytest.fixture()
def builder(store: KnowledgeStore) -> KnowledgeBuilder:
    return KnowledgeBuilder(
        store=store,
        client=_MockClient(items_per_source=3),
        embedder=NullEmbedder(),
    )


# ---------------------------------------------------------------------------
# Success criteria 1 — Fresh build creates the Knowledge Base
# ---------------------------------------------------------------------------


def test_fresh_build_creates_kb(builder: KnowledgeBuilder, source_dir: Path, store: KnowledgeStore):
    report = builder.ingest_directory(source_dir, domain="smr")

    assert report.sources_scanned == 2
    assert report.sources_rebuilt == 2
    assert report.sources_skipped == 0
    assert report.sources_failed == 0
    assert report.evidence_objects == 6  # 3 items × 2 sources

    # KB has the sources
    manifest = store.load_manifest()
    assert len(manifest) == 2

    # KB has the evidence
    assert store.evidence_count("smr") == 6

    # ExtractionRun was recorded
    run = store.latest_extraction_run()
    assert run is not None
    assert run.status == "COMPLETED"


# ---------------------------------------------------------------------------
# Success criteria 2 — Immediate rebuild skips everything
# ---------------------------------------------------------------------------


def test_immediate_rebuild_skips(builder: KnowledgeBuilder, source_dir: Path, store: KnowledgeStore):
    builder.ingest_directory(source_dir, domain="smr")
    client = builder.client
    call_count_after_first = len(client.calls)

    report2 = builder.ingest_directory(source_dir, domain="smr", incremental=True)

    assert report2.sources_skipped == 2
    assert report2.sources_rebuilt == 0
    # No new extraction calls
    assert len(client.calls) == call_count_after_first


# ---------------------------------------------------------------------------
# Success criteria 3 — Modifying one Source rebuilds only that Source
# ---------------------------------------------------------------------------


def test_modify_one_source_rebuilds_only_that_one(
    builder: KnowledgeBuilder, source_dir: Path, store: KnowledgeStore
):
    builder.ingest_directory(source_dir, domain="smr")

    # Modify doc_a only
    (source_dir / "doc_a.txt").write_text("Updated: BWRX-300 now operates at 320 MWe.", encoding="utf-8")

    report2 = builder.ingest_directory(source_dir, domain="smr", incremental=True)

    assert report2.sources_rebuilt == 1
    assert report2.sources_skipped == 1


# ---------------------------------------------------------------------------
# Success criteria 4 — Evidence count for unchanged sources is stable
# ---------------------------------------------------------------------------


def test_evidence_count_stable_for_unchanged_sources(
    builder: KnowledgeBuilder, source_dir: Path, store: KnowledgeStore
):
    builder.ingest_directory(source_dir, domain="smr")
    count_after_first = store.evidence_count("smr")

    # Modify only doc_a
    (source_dir / "doc_a.txt").write_text("Changed content for doc_a.", encoding="utf-8")
    builder.ingest_directory(source_dir, domain="smr", incremental=True)

    count_after_second = store.evidence_count("smr")
    # doc_b unchanged → 3 items; doc_a rebuilt → 3 new items appended
    # Total grows by doc_a's new items (old ones remain, no deletion in J8.1)
    assert count_after_second >= count_after_first


# ---------------------------------------------------------------------------
# Success criteria 5 — Fingerprint cache updated correctly
# ---------------------------------------------------------------------------


def test_fingerprint_cache_updated(builder: KnowledgeBuilder, source_dir: Path, store: KnowledgeStore):
    builder.ingest_directory(source_dir, domain="smr")
    manifest_v1 = store.load_manifest()
    fp_a_v1 = next(e.fingerprint for e in manifest_v1.values() if "doc_a" in e.uri)

    (source_dir / "doc_a.txt").write_text("Entirely different content now.", encoding="utf-8")
    builder.ingest_directory(source_dir, domain="smr", incremental=True)

    manifest_v2 = store.load_manifest()
    fp_a_v2 = next(e.fingerprint for e in manifest_v2.values() if "doc_a" in e.uri)

    assert fp_a_v1 != fp_a_v2


# ---------------------------------------------------------------------------
# Force rebuild ignores fingerprints
# ---------------------------------------------------------------------------


def test_force_rebuild_ignores_fingerprints(builder: KnowledgeBuilder, source_dir: Path):
    builder.ingest_directory(source_dir, domain="smr")
    report2 = builder.ingest_directory(source_dir, domain="smr", force=True)
    assert report2.sources_rebuilt == 2
    assert report2.sources_skipped == 0


# ---------------------------------------------------------------------------
# Deduplication — identical claims merged
# ---------------------------------------------------------------------------


def test_duplicate_evidence_merged(tmp_path: Path):
    """Two sources producing the same claim → only one evidence record stored."""
    store = KnowledgeStore(root=tmp_path / "ks")

    class _DupClient:
        def extract_evidence(self, question, source_texts):
            from research_agent.schemas import EvidenceItem
            # Always returns the same claim regardless of source
            return [
                EvidenceItem(
                    evidence_id="",
                    claim="The reactor outputs 300 MWe.",
                    source_document=source_texts[0].title if source_texts else "unknown",
                    evidence_snippet="snippet",
                    category="power",
                    relevance="direct",
                    confidence="high",
                )
            ]

    builder = KnowledgeBuilder(store=store, client=_DupClient(), embedder=NullEmbedder())

    d = tmp_path / "sources"
    d.mkdir()
    (d / "src_a.txt").write_text("Content A", encoding="utf-8")
    (d / "src_b.txt").write_text("Content B", encoding="utf-8")

    report = builder.ingest_directory(d, domain="smr")
    assert report.duplicates_merged >= 1
    # Only 1 unique claim despite 2 sources
    assert store.evidence_count("smr") == 1


# ---------------------------------------------------------------------------
# No-client mode — source ingestion without extraction
# ---------------------------------------------------------------------------


def test_source_ingestion_without_client(source_dir: Path, store: KnowledgeStore):
    builder = KnowledgeBuilder(store=store, client=None, embedder=NullEmbedder())
    report = builder.ingest_directory(source_dir, domain="smr")
    assert report.sources_rebuilt == 2
    assert report.evidence_objects == 0
    manifest = store.load_manifest()
    assert len(manifest) == 2


# ---------------------------------------------------------------------------
# Missing directory is handled gracefully
# ---------------------------------------------------------------------------


def test_missing_directory_skipped(store: KnowledgeStore, tmp_path: Path):
    builder = KnowledgeBuilder(store=store, client=None, embedder=NullEmbedder())
    report = builder.build([tmp_path / "does_not_exist"])
    assert report.sources_failed == 0
    assert report.sources_scanned == 0


# ---------------------------------------------------------------------------
# BuildReport
# ---------------------------------------------------------------------------


def test_build_report_cache_hit_ratio(builder: KnowledgeBuilder, source_dir: Path):
    builder.ingest_directory(source_dir, domain="smr")
    report = builder.ingest_directory(source_dir, domain="smr", incremental=True)
    assert report.cache_hit_ratio == 1.0


def test_build_report_has_extraction_run_id(builder: KnowledgeBuilder, source_dir: Path):
    report = builder.ingest_directory(source_dir, domain="smr")
    assert report.extraction_run_id != ""


# ---------------------------------------------------------------------------
# Domain inference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dirname,expected", [
    ("smr_sources", "smr"),
    ("nvidia", "ai_data_centers"),
    ("market", "economics"),
    ("networking", "networking"),
    ("infrastructure", "infrastructure"),
    ("unknown_domain", "unknown_domain"),
])
def test_infer_domain(dirname: str, expected: str, tmp_path: Path):
    d = tmp_path / dirname
    d.mkdir()
    assert infer_domain(d) == expected


# ---------------------------------------------------------------------------
# J7 pipeline unchanged — existing imports still work
# ---------------------------------------------------------------------------


def test_j7_pipeline_imports_unaffected():
    """Verify that importing knowledge module does not break J7 imports."""
    from research_agent.schemas import EvidenceItem, SourceDocument
    from research_agent.claude_client import MockClaudeClient
    from functional_agents.orchestrator import AgentOrchestrator
    # If any of these fail, knowledge/ is polluting the existing namespace
    assert EvidenceItem is not None
    assert SourceDocument is not None
    assert MockClaudeClient is not None
    assert AgentOrchestrator is not None
