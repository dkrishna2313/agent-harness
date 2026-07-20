"""Tests for --profile flag on `knowledge status` and `knowledge list-sources`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from knowledge.cli import app as knowledge_app, _evidence_counts_for_profile
from knowledge.store import KnowledgeStore

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixture: minimal knowledge store with two profiles
# ---------------------------------------------------------------------------

def _write_evidence(store_dir: Path, domain: str, items: list[dict]) -> None:
    ev_dir = store_dir / "evidence" / domain
    ev_dir.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(json.dumps(item) for item in items)
    (ev_dir / "evidence.jsonl").write_text(lines + "\n", encoding="utf-8")

    index = {item["evidence_id"]: i for i, item in enumerate(items)}
    (ev_dir / "evidence_index.json").write_text(json.dumps(index), encoding="utf-8")


def _write_manifest(store_dir: Path, entries: list[dict]) -> None:
    manifests_dir = store_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest = {e["source_id"]: e for e in entries}
    (manifests_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _make_evidence(ev_id: str, source_id: str, profile_ids: list[str]) -> dict:
    return {
        "evidence_id": ev_id,
        "statement": f"Statement {ev_id}",
        "evidence_type": "STRATEGIC",
        "supporting_source_ids": [source_id],
        "profile_ids": profile_ids,
        "extraction_run_id": "run-test-001",
    }


def _make_source(source_id: str, domain: str, uri: str, ev_ids: list[str]) -> dict:
    return {
        "source_id": source_id,
        "domain": domain,
        "uri": uri,
        "title": f"Title {source_id}",
        "evidence_ids": ev_ids,
        "fingerprint": source_id,
    }


@pytest.fixture()
def store_dir(tmp_path: Path) -> Path:
    sd = tmp_path / "knowledge_store"
    sd.mkdir()

    # sports domain: 3 evidence items, 2 sources
    _write_evidence(sd, "sports", [
        _make_evidence("EV-001", "SRC-sports-1", ["sports"]),
        _make_evidence("EV-002", "SRC-sports-1", ["sports"]),
        _make_evidence("EV-003", "SRC-sports-2", ["sports"]),
    ])
    # smr domain: 2 evidence items, 1 source; 1 untagged item
    _write_evidence(sd, "smr", [
        _make_evidence("EV-101", "SRC-smr-1", ["smr"]),
        _make_evidence("EV-102", "SRC-smr-1", ["smr"]),
        _make_evidence("EV-103", "SRC-smr-1", []),
    ])

    _write_manifest(sd, [
        _make_source("SRC-sports-1", "sports", "sports_sources/doc1.pdf", ["EV-001", "EV-002"]),
        _make_source("SRC-sports-2", "sports", "sports_sources/doc2.pdf", ["EV-003"]),
        _make_source("SRC-smr-1",    "smr",    "smr_sources/doc3.pdf",    ["EV-101", "EV-102", "EV-103"]),
    ])
    return sd


# ---------------------------------------------------------------------------
# Unit tests for _evidence_counts_for_profile
# ---------------------------------------------------------------------------

def test_evidence_counts_sports(store_dir):
    store = KnowledgeStore(store_dir)
    count, source_ids = _evidence_counts_for_profile(store, "sports")
    assert count == 3
    assert source_ids == {"SRC-sports-1", "SRC-sports-2"}


def test_evidence_counts_smr(store_dir):
    store = KnowledgeStore(store_dir)
    count, source_ids = _evidence_counts_for_profile(store, "smr")
    assert count == 2  # EV-103 has no profile tag
    assert source_ids == {"SRC-smr-1"}


def test_evidence_counts_unknown_profile_returns_zero(store_dir):
    store = KnowledgeStore(store_dir)
    count, source_ids = _evidence_counts_for_profile(store, "unknown_profile")
    assert count == 0
    assert source_ids == set()


# ---------------------------------------------------------------------------
# CLI: status --profile
# ---------------------------------------------------------------------------

def test_status_profile_sports(store_dir):
    result = runner.invoke(
        knowledge_app,
        ["status", "--store", str(store_dir), "--profile", "sports"],
    )
    assert result.exit_code == 0
    assert "sports" in result.output
    assert "3" in result.output          # 3 evidence items


def test_status_profile_shows_source_count(store_dir):
    result = runner.invoke(
        knowledge_app,
        ["status", "--store", str(store_dir), "--profile", "sports"],
    )
    assert result.exit_code == 0
    assert "2" in result.output          # 2 sources with hits


def test_status_profile_smr(store_dir):
    result = runner.invoke(
        knowledge_app,
        ["status", "--store", str(store_dir), "--profile", "smr"],
    )
    assert result.exit_code == 0
    assert "smr" in result.output
    assert "2" in result.output          # 2 tagged evidence items


def test_status_unknown_profile_shows_zero(store_dir):
    result = runner.invoke(
        knowledge_app,
        ["status", "--store", str(store_dir), "--profile", "does_not_exist"],
    )
    assert result.exit_code == 0
    assert "0" in result.output


def test_status_no_profile_shows_all(store_dir):
    result = runner.invoke(
        knowledge_app,
        ["status", "--store", str(store_dir)],
    )
    assert result.exit_code == 0
    assert "sports" in result.output
    assert "smr" in result.output


# ---------------------------------------------------------------------------
# CLI: list-sources --profile
# ---------------------------------------------------------------------------

def test_list_sources_profile_sports(store_dir):
    result = runner.invoke(
        knowledge_app,
        ["list-sources", "--store", str(store_dir), "--profile", "sports"],
    )
    assert result.exit_code == 0
    assert "SRC-sports-1" in result.output
    assert "SRC-sports-2" in result.output
    assert "SRC-smr-1" not in result.output


def test_list_sources_profile_smr(store_dir):
    result = runner.invoke(
        knowledge_app,
        ["list-sources", "--store", str(store_dir), "--profile", "smr"],
    )
    assert result.exit_code == 0
    assert "SRC-smr-1" in result.output
    assert "SRC-sports-1" not in result.output


def test_list_sources_unknown_profile_prints_none(store_dir):
    result = runner.invoke(
        knowledge_app,
        ["list-sources", "--store", str(store_dir), "--profile", "unknown"],
    )
    assert result.exit_code == 0
    assert "No sources indexed" in result.output


def test_list_sources_domain_and_profile_combined(store_dir):
    result = runner.invoke(
        knowledge_app,
        ["list-sources", "--store", str(store_dir), "--domain", "smr", "--profile", "smr"],
    )
    assert result.exit_code == 0
    assert "SRC-smr-1" in result.output
    assert "SRC-sports-1" not in result.output


def test_list_sources_domain_profile_mismatch_returns_none(store_dir):
    # domain=smr but profile=sports — no overlap
    result = runner.invoke(
        knowledge_app,
        ["list-sources", "--store", str(store_dir), "--domain", "smr", "--profile", "sports"],
    )
    assert result.exit_code == 0
    assert "No sources indexed" in result.output


def test_list_sources_no_profile_shows_all(store_dir):
    result = runner.invoke(
        knowledge_app,
        ["list-sources", "--store", str(store_dir)],
    )
    assert result.exit_code == 0
    assert "SRC-sports-1" in result.output
    assert "SRC-smr-1" in result.output
