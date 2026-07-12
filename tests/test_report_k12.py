"""K1.2 — Reference Integration tests for executive report rendering."""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import MagicMock

import pytest

from functional_agents.report_agent import (
    _build_domain_evidence_for_references,
    _build_references_section,
    _replace_supporting_evidence_uuids,
    _replace_uuid_citation_markers,
)
from functional_agents.reference import Reference, ReferenceBuilder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ref(
    source_id: str = "src-001",
    evidence_ids: list[str] | None = None,
    citation_text: str = "ERCOT. Grid Study. 2024.",
    title: str = "Grid Study",
) -> Reference:
    return Reference(
        reference_id=source_id,
        source_id=source_id,
        evidence_ids=evidence_ids or ["e1"],
        title=title,
        citation_text=citation_text,
    )


def _make_context(domain_evidence: list[dict] | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.domain_evidence = domain_evidence or []
    return ctx


# ---------------------------------------------------------------------------
# _build_domain_evidence_for_references
# ---------------------------------------------------------------------------

class TestBuildDomainEvidenceForReferences:
    def test_empty_domain_evidence(self):
        ctx = _make_context([])
        result = _build_domain_evidence_for_references(ctx)
        assert result == []

    def test_none_domain_evidence(self):
        ctx = _make_context(None)
        result = _build_domain_evidence_for_references(ctx)
        assert result == []

    def test_single_domain_single_evidence(self):
        ctx = _make_context([{
            "decision_domain_id": "d1",
            "evidence": [{
                "evidence_id": "97e4e613-96ce-4d31-b111-513257cfb06e",
                "claim": "Some claim.",
                "source_document": "81efa89092f24557014a592f21761236",
            }],
        }])
        result = _build_domain_evidence_for_references(ctx)
        assert len(result) == 1
        assert result[0]["evidence_id"] == "97e4e613-96ce-4d31-b111-513257cfb06e"
        assert result[0]["supporting_source_ids"] == ["81efa89092f24557014a592f21761236"]

    def test_multiple_domains(self):
        ctx = _make_context([
            {"decision_domain_id": "d1", "evidence": [
                {"evidence_id": "e1", "source_document": "src1"},
                {"evidence_id": "e2", "source_document": "src2"},
            ]},
            {"decision_domain_id": "d2", "evidence": [
                {"evidence_id": "e3", "source_document": "src3"},
            ]},
        ])
        result = _build_domain_evidence_for_references(ctx)
        assert len(result) == 3
        ids = [r["evidence_id"] for r in result]
        assert ids == ["e1", "e2", "e3"]

    def test_missing_source_document_yields_empty_list(self):
        ctx = _make_context([{"decision_domain_id": "d1", "evidence": [
            {"evidence_id": "e1"},
        ]}])
        result = _build_domain_evidence_for_references(ctx)
        assert result[0]["supporting_source_ids"] == []

    def test_empty_evidence_list_in_domain(self):
        ctx = _make_context([{"decision_domain_id": "d1", "evidence": []}])
        result = _build_domain_evidence_for_references(ctx)
        assert result == []

    def test_missing_evidence_key(self):
        ctx = _make_context([{"decision_domain_id": "d1"}])
        result = _build_domain_evidence_for_references(ctx)
        assert result == []


# ---------------------------------------------------------------------------
# _build_references_section
# ---------------------------------------------------------------------------

class TestBuildReferencesSection:
    def test_empty_refs_returns_empty(self):
        assert _build_references_section([]) == []

    def test_single_ref(self):
        ref = _make_ref(citation_text="ERCOT. Grid Study. 2024.")
        lines = _build_references_section([ref])
        text = "\n".join(lines)
        assert "## References" in text
        assert "1. ERCOT. Grid Study. 2024." in text

    def test_multiple_refs_numbered(self):
        refs = [
            _make_ref("s1", citation_text="Alpha Corp. Report A. 2023."),
            _make_ref("s2", citation_text="Beta Inc. Report B. 2024."),
        ]
        lines = _build_references_section(refs)
        text = "\n".join(lines)
        assert "1. Alpha Corp. Report A. 2023." in text
        assert "2. Beta Inc. Report B. 2024." in text

    def test_section_has_separator(self):
        ref = _make_ref()
        lines = _build_references_section([ref])
        assert lines[0] == "---"

    def test_section_has_intro(self):
        ref = _make_ref()
        lines = _build_references_section([ref])
        text = "\n".join(lines)
        assert "*Sources cited in this report.*" in text

    def test_ends_with_blank_line(self):
        ref = _make_ref()
        lines = _build_references_section([ref])
        assert lines[-1] == ""


# ---------------------------------------------------------------------------
# _replace_uuid_citation_markers
# ---------------------------------------------------------------------------

class TestReplaceUuidCitationMarkers:
    _UUID = "97e4e613-96ce-4d31-b111-513257cfb06e"
    _CITATION = "ERCOT. Grid Study. 2024."

    def test_replaces_matching_marker(self):
        text = f"Finding text [Source: abc123, Evidence: {self._UUID}] more text."
        result = _replace_uuid_citation_markers(text, {self._UUID: self._CITATION})
        assert f"[{self._CITATION}]" in result
        assert self._UUID not in result

    def test_leaves_unknown_uuid_unchanged(self):
        text = f"[Source: abc, Evidence: {self._UUID}]"
        result = _replace_uuid_citation_markers(text, {})
        assert text == result

    def test_empty_map_noop(self):
        text = f"[Source: abc, Evidence: {self._UUID}]"
        assert _replace_uuid_citation_markers(text, {}) == text

    def test_multiple_markers_replaced(self):
        uuid2 = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        text = (
            f"[Source: x, Evidence: {self._UUID}] and "
            f"[Source: y, Evidence: {uuid2}]"
        )
        mapping = {self._UUID: "Cite A.", uuid2: "Cite B."}
        result = _replace_uuid_citation_markers(text, mapping)
        assert "[Cite A.]" in result
        assert "[Cite B.]" in result
        assert self._UUID not in result
        assert uuid2 not in result

    def test_no_markers_unchanged(self):
        text = "Plain text with no markers."
        assert _replace_uuid_citation_markers(text, {self._UUID: self._CITATION}) == text

    def test_partial_match_uuid_not_replaced(self):
        # Only the [Source: ..., Evidence: uuid] pattern triggers replacement
        text = f"Evidence UUID: {self._UUID}"
        result = _replace_uuid_citation_markers(text, {self._UUID: self._CITATION})
        assert self._UUID in result  # no change — not in marker format


# ---------------------------------------------------------------------------
# _replace_supporting_evidence_uuids
# ---------------------------------------------------------------------------

class TestReplaceSupportingEvidenceUuids:
    _UUID = "97e4e613-96ce-4d31-b111-513257cfb06e"
    _CITATION = "ERCOT. Grid Study. 2024."

    def test_replaces_uuid_on_supporting_line(self):
        text = f"  - Supporting evidence: {self._UUID}"
        result = _replace_supporting_evidence_uuids(text, {self._UUID: self._CITATION})
        assert self._CITATION in result
        assert self._UUID not in result

    def test_multiple_uuids_on_one_line(self):
        uuid2 = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        text = f"  - Supporting evidence: {self._UUID}, {uuid2}"
        mapping = {self._UUID: "Cite A.", uuid2: "Cite B."}
        result = _replace_supporting_evidence_uuids(text, mapping)
        assert "Cite A." in result
        assert "Cite B." in result
        assert self._UUID not in result

    def test_unknown_uuid_preserved(self):
        text = f"  - Supporting evidence: {self._UUID}"
        result = _replace_supporting_evidence_uuids(text, {})
        assert self._UUID in result

    def test_empty_map_noop(self):
        text = f"  - Supporting evidence: {self._UUID}"
        assert _replace_supporting_evidence_uuids(text, {}) == text

    def test_non_uuid_parts_preserved(self):
        text = "  - Supporting evidence: some-non-uuid-value"
        result = _replace_supporting_evidence_uuids(text, {self._UUID: self._CITATION})
        assert "some-non-uuid-value" in result

    def test_other_lines_untouched(self):
        text = (
            f"  - Supporting evidence: {self._UUID}\n"
            "  - Confidence: HIGH\n"
            "  - Other line\n"
        )
        result = _replace_supporting_evidence_uuids(text, {self._UUID: self._CITATION})
        assert "Confidence: HIGH" in result
        assert "Other line" in result


# ---------------------------------------------------------------------------
# Integration: ReferenceBuilder → references section
# ---------------------------------------------------------------------------

class TestReferenceIntegration:
    def _make_builder(self, sources: dict) -> ReferenceBuilder:
        return ReferenceBuilder(source_resolver=lambda sid: sources.get(sid))

    def test_references_sorted_alphabetically(self):
        sources = {
            "s1": {"source_id": "s1", "title": "Zebra Report", "author": None, "organization": None, "publisher": None, "publication_date": None},
            "s2": {"source_id": "s2", "title": "Alpha Report", "author": None, "organization": None, "publisher": None, "publication_date": None},
        }
        builder = self._make_builder(sources)
        ev = [
            {"evidence_id": "e1", "supporting_source_ids": ["s1"]},
            {"evidence_id": "e2", "supporting_source_ids": ["s2"]},
        ]
        refs = builder.build(ev)
        lines = _build_references_section(refs)
        text = "\n".join(lines)
        assert text.index("Alpha") < text.index("Zebra")

    def test_domain_evidence_normalization_feeds_builder(self):
        ctx = _make_context([{
            "decision_domain_id": "d1",
            "evidence": [
                {"evidence_id": "e1", "source_document": "s1"},
                {"evidence_id": "e2", "source_document": "s1"},
            ],
        }])
        normalized = _build_domain_evidence_for_references(ctx)
        sources = {
            "s1": {"source_id": "s1", "title": "Key Report", "author": "J. Smith", "organization": None, "publisher": None, "publication_date": date(2024, 1, 1)},
        }
        builder = self._make_builder(sources)
        refs = builder.build(normalized)
        assert len(refs) == 1
        assert set(refs[0].evidence_ids) == {"e1", "e2"}
        assert refs[0].citation_text == "J. Smith. Key Report. 2024."

    def test_no_references_when_sources_unresolvable(self):
        ctx = _make_context([{
            "decision_domain_id": "d1",
            "evidence": [{"evidence_id": "e1", "source_document": "s-missing"}],
        }])
        normalized = _build_domain_evidence_for_references(ctx)
        builder = ReferenceBuilder(source_resolver=lambda _: None)
        refs = builder.build(normalized)
        assert refs == []
        assert _build_references_section(refs) == []

    def test_full_pipeline_uuid_replacement(self):
        uuid = "97e4e613-96ce-4d31-b111-513257cfb06e"
        citation = "ERCOT. Grid Study. 2024."
        eid_to_citation = {uuid: citation}

        report_text = (
            f"- Finding one. [Source: src1, Evidence: {uuid}]\n"
            f"  - Supporting evidence: {uuid}\n"
        )
        result = _replace_uuid_citation_markers(report_text, eid_to_citation)
        result = _replace_supporting_evidence_uuids(result, eid_to_citation)

        assert uuid not in result
        assert citation in result
        assert f"[{citation}]" in result
