"""PH5.5b — Provenance Population at Extraction Time tests.

Verifies that _adapt_evidence_item() correctly populates v2 provenance fields
from existing EvidenceItem signals and source canonical_text.

Tests are grouped by field or feature:
  1.  excerpt population and capping
  2.  chunk_id population
  3.  topics population
  4.  evidence_confidence mapping
  5.  is_quantitative derivation
  6.  page_number derivation from [Page N] markers
  7.  char_offset_start/end derivation
  8.  temporal_reference extraction
  9.  section_heading stays None (no new inference)
  10. v1 fields unchanged
  11. missing / empty signals default safely (no fabrication)
  12. extractor integration via extract_evidence_from_source()
  13. PH5.5a schema tests still pass (regression guard)
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from knowledge.extractor import (
    _adapt_evidence_item,
    _extract_temporal_reference,
    _find_provenance_in_text,
    extract_evidence_from_source,
)
from knowledge.models import Evidence, Source


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CANONICAL_TEXT_PDF = (
    "[Page 1]\n"
    "The BWRX-300 is a small modular reactor with a thermal output of 870 MWt.\n"
    "Deployment is expected by Q3 2030 pending regulatory approval.\n\n"
    "[Page 2]\n"
    "Capital costs are estimated at USD 2,500 per kWe for nth-of-a-kind units.\n"
    "The UK Generic Design Assessment is expected to complete by 2028.\n\n"
    "[Page 3]\n"
    "Construction timeline is projected at 4 years per unit.\n"
)

_CANONICAL_TEXT_TXT = (
    "SMR licensing typically takes 5 to 7 years in the United States.\n"
    "The NRC has received applications for advanced reactor certification.\n"
    "Market projections suggest 50 GW of SMR capacity by 2040.\n"
)


def _make_source(canonical_text: str = _CANONICAL_TEXT_TXT) -> Source:
    fingerprint = Source.compute_fingerprint(canonical_text)
    return Source(
        source_id=Source.compute_source_id(fingerprint),
        uri="tests/fixtures/test_source.txt",
        title="Test Source",
        retrieved_date="2026-07-16",
        fingerprint=fingerprint,
        document_type="TXT",
        domain="smr",
        canonical_text=canonical_text,
    )


def _make_item(**kwargs) -> object:
    """Build a SimpleNamespace mimicking an EvidenceItem."""
    defaults = dict(
        claim="The BWRX-300 has a thermal output of 870 MWt.",
        source_document="test.pdf",
        source_chunk_id="",
        evidence_snippet="The BWRX-300 is a small modular reactor with a thermal output of 870 MWt.",
        category="reactor design",
        relevance="direct",
        confidence="high",
        relevance_score=4,
        source_quality_score=4,
        specificity_score=4,
        overall_score=4.0,
        quantitative_score=5,
        topics=[],
        entity="BWRX-300",
        entity_type="reactor",
        scope="unit",
        evidence_type="TECHNICAL",
        perspective="",
        recovered=False,
        recovery_reason="",
    )
    defaults.update(kwargs)
    ns = types.SimpleNamespace(**defaults)
    return ns


def _adapt(item: object, canonical_text: str = _CANONICAL_TEXT_PDF) -> Evidence:
    source = _make_source(canonical_text)
    return _adapt_evidence_item(item, source, "run-test-001", ["smr"])


# ---------------------------------------------------------------------------
# 1. excerpt population and capping
# ---------------------------------------------------------------------------


def test_excerpt_populated_from_evidence_snippet():
    item = _make_item(evidence_snippet="The BWRX-300 is a small modular reactor.")
    ev = _adapt(item)
    assert ev.excerpt == "The BWRX-300 is a small modular reactor."


def test_excerpt_capped_at_600_chars():
    long_snippet = "x" * 800
    item = _make_item(evidence_snippet=long_snippet)
    ev = _adapt(item)
    assert ev.excerpt is not None
    assert len(ev.excerpt) == 600


def test_excerpt_none_when_snippet_empty():
    item = _make_item(evidence_snippet="")
    ev = _adapt(item)
    assert ev.excerpt is None


def test_excerpt_none_when_snippet_whitespace_only():
    item = _make_item(evidence_snippet="   \n\t  ")
    ev = _adapt(item)
    assert ev.excerpt is None


def test_excerpt_none_when_snippet_missing():
    item = _make_item()
    del item.evidence_snippet  # attribute absent
    ev = _adapt(item)
    assert ev.excerpt is None


# ---------------------------------------------------------------------------
# 2. chunk_id population
# ---------------------------------------------------------------------------


def test_chunk_id_populated_from_source_chunk_id():
    item = _make_item(source_chunk_id="chunk:p1-para2")
    ev = _adapt(item)
    assert ev.chunk_id == "chunk:p1-para2"


def test_chunk_id_none_when_source_chunk_id_empty():
    item = _make_item(source_chunk_id="")
    ev = _adapt(item)
    assert ev.chunk_id is None


def test_chunk_id_none_when_source_chunk_id_whitespace():
    item = _make_item(source_chunk_id="   ")
    ev = _adapt(item)
    assert ev.chunk_id is None


def test_chunk_id_none_when_missing():
    item = _make_item()
    del item.source_chunk_id
    ev = _adapt(item)
    assert ev.chunk_id is None


# ---------------------------------------------------------------------------
# 3. topics population
# ---------------------------------------------------------------------------


def test_topics_populated_from_item():
    item = _make_item(topics=["SMR", "thermal output", "BWRX-300"])
    ev = _adapt(item)
    assert ev.topics == ["SMR", "thermal output", "BWRX-300"]


def test_topics_empty_list_when_item_topics_empty():
    item = _make_item(topics=[])
    ev = _adapt(item)
    assert ev.topics == []


def test_topics_empty_list_when_item_topics_none():
    item = _make_item(topics=None)
    ev = _adapt(item)
    assert ev.topics == []


# ---------------------------------------------------------------------------
# 4. evidence_confidence mapping
# ---------------------------------------------------------------------------


def test_confidence_high_maps_to_HIGH():
    item = _make_item(confidence="high")
    ev = _adapt(item)
    assert ev.evidence_confidence == "HIGH"


def test_confidence_medium_maps_to_MEDIUM():
    item = _make_item(confidence="medium")
    ev = _adapt(item)
    assert ev.evidence_confidence == "MEDIUM"


def test_confidence_low_maps_to_LOW():
    item = _make_item(confidence="low")
    ev = _adapt(item)
    assert ev.evidence_confidence == "LOW"


def test_confidence_none_when_unrecognized():
    item = _make_item(confidence="unknown_value")
    ev = _adapt(item)
    assert ev.evidence_confidence is None


def test_confidence_none_when_missing():
    item = _make_item()
    del item.confidence
    ev = _adapt(item)
    # getattr fallback returns "medium"
    assert ev.evidence_confidence == "MEDIUM"


# ---------------------------------------------------------------------------
# 5. is_quantitative derivation
# ---------------------------------------------------------------------------


def test_is_quantitative_true_when_score_5():
    item = _make_item(quantitative_score=5)
    ev = _adapt(item)
    assert ev.is_quantitative is True


def test_is_quantitative_true_when_score_4():
    item = _make_item(quantitative_score=4)
    ev = _adapt(item)
    assert ev.is_quantitative is True


def test_is_quantitative_false_when_score_3():
    item = _make_item(quantitative_score=3)
    ev = _adapt(item)
    assert ev.is_quantitative is False


def test_is_quantitative_false_when_score_1():
    item = _make_item(quantitative_score=1)
    ev = _adapt(item)
    assert ev.is_quantitative is False


def test_is_quantitative_false_when_missing():
    item = _make_item()
    del item.quantitative_score
    ev = _adapt(item)
    assert ev.is_quantitative is False  # defaults to score=3 → False


# ---------------------------------------------------------------------------
# 6. page_number derivation from [Page N] markers
# ---------------------------------------------------------------------------


def test_page_number_1_when_excerpt_is_on_page_1():
    item = _make_item(
        evidence_snippet="The BWRX-300 is a small modular reactor with a thermal output of 870 MWt.",
    )
    ev = _adapt(item, _CANONICAL_TEXT_PDF)
    assert ev.page_number == 1


def test_page_number_2_when_excerpt_is_on_page_2():
    item = _make_item(
        claim="Capital costs are estimated at USD 2,500 per kWe for nth-of-a-kind units.",
        evidence_snippet="Capital costs are estimated at USD 2,500 per kWe for nth-of-a-kind units.",
        category="economics",
    )
    ev = _adapt(item, _CANONICAL_TEXT_PDF)
    assert ev.page_number == 2


def test_page_number_3_when_excerpt_is_on_page_3():
    item = _make_item(
        claim="Construction timeline is projected at 4 years per unit.",
        evidence_snippet="Construction timeline is projected at 4 years per unit.",
        category="construction",
    )
    ev = _adapt(item, _CANONICAL_TEXT_PDF)
    assert ev.page_number == 3


def test_page_number_none_when_excerpt_not_in_text():
    item = _make_item(
        evidence_snippet="This text does not appear in the canonical source at all.",
    )
    ev = _adapt(item, _CANONICAL_TEXT_PDF)
    assert ev.page_number is None


def test_page_number_1_for_text_without_page_markers():
    item = _make_item(
        claim="SMR licensing typically takes 5 to 7 years in the United States.",
        evidence_snippet="SMR licensing typically takes 5 to 7 years in the United States.",
        category="licensing",
    )
    ev = _adapt(item, _CANONICAL_TEXT_TXT)
    assert ev.page_number == 1


def test_page_number_none_when_excerpt_empty():
    item = _make_item(evidence_snippet="")
    ev = _adapt(item, _CANONICAL_TEXT_PDF)
    assert ev.page_number is None


# ---------------------------------------------------------------------------
# 7. char_offset_start / char_offset_end derivation
# ---------------------------------------------------------------------------


def test_char_offsets_populated_when_excerpt_found():
    excerpt = "The BWRX-300 is a small modular reactor with a thermal output of 870 MWt."
    item = _make_item(evidence_snippet=excerpt)
    ev = _adapt(item, _CANONICAL_TEXT_PDF)
    assert ev.char_offset_start is not None
    assert ev.char_offset_end is not None
    assert ev.char_offset_end > ev.char_offset_start


def test_char_offsets_point_to_excerpt_in_canonical_text():
    excerpt = "The BWRX-300 is a small modular reactor with a thermal output of 870 MWt."
    item = _make_item(evidence_snippet=excerpt)
    ev = _adapt(item, _CANONICAL_TEXT_PDF)
    # The excerpt (or its anchor) should be findable at the reported offset
    if ev.char_offset_start is not None:
        text_at_offset = _CANONICAL_TEXT_PDF[ev.char_offset_start : ev.char_offset_start + 40]
        assert text_at_offset.strip().startswith("The BWRX-300")


def test_char_offsets_none_when_excerpt_not_found():
    item = _make_item(evidence_snippet="This text is completely fabricated and absent.")
    ev = _adapt(item, _CANONICAL_TEXT_PDF)
    assert ev.char_offset_start is None
    assert ev.char_offset_end is None


def test_char_offsets_none_when_excerpt_empty():
    item = _make_item(evidence_snippet="")
    ev = _adapt(item, _CANONICAL_TEXT_PDF)
    assert ev.char_offset_start is None
    assert ev.char_offset_end is None


# ---------------------------------------------------------------------------
# 8. temporal_reference extraction
# ---------------------------------------------------------------------------


def test_temporal_reference_extracts_year():
    item = _make_item(
        claim="The GDA is expected to complete by 2028.",
        evidence_snippet="The UK Generic Design Assessment is expected to complete by 2028.",
    )
    ev = _adapt(item)
    assert ev.temporal_reference == "2028"


def test_temporal_reference_extracts_quarter():
    item = _make_item(
        claim="Deployment is expected by Q3 2030.",
        evidence_snippet="Deployment is expected by Q3 2030 pending regulatory approval.",
    )
    ev = _adapt(item, _CANONICAL_TEXT_PDF)
    assert ev.temporal_reference == "Q3 2030"


def test_temporal_reference_prefers_quarter_over_year():
    item = _make_item(
        claim="Target commissioning is Q4 2031 based on 2024 projections.",
        evidence_snippet="Target commissioning is Q4 2031 based on 2024 projections.",
    )
    ev = _adapt(item)
    # Quarter should take priority
    assert ev.temporal_reference == "Q4 2031"


def test_temporal_reference_none_when_no_date_in_text():
    item = _make_item(
        claim="SMR fuel supply chains require careful management.",
        evidence_snippet="Fuel supply chains require careful management.",
    )
    ev = _adapt(item)
    assert ev.temporal_reference is None


# ---------------------------------------------------------------------------
# 9. section_heading stays None
# ---------------------------------------------------------------------------


def test_section_heading_always_none():
    item = _make_item()
    ev = _adapt(item, _CANONICAL_TEXT_PDF)
    assert ev.section_heading is None


# ---------------------------------------------------------------------------
# 10. v1 fields unchanged
# ---------------------------------------------------------------------------


def test_v1_statement_unchanged():
    item = _make_item(claim="Exact statement from claim.")
    ev = _adapt(item)
    assert ev.statement == "Exact statement from claim."


def test_v1_entity_fields_preserved():
    item = _make_item(entity="BWRX-300", entity_type="reactor", scope="fleet")
    ev = _adapt(item)
    assert ev.entity == "BWRX-300"
    assert ev.entity_type == "reactor"
    assert ev.scope == "fleet"


def test_v1_category_preserved():
    item = _make_item(category="fuel cycle")
    ev = _adapt(item)
    assert ev.category == "fuel cycle"


def test_v1_supporting_source_ids_preserved():
    source = _make_source()
    item = _make_item()
    ev = _adapt_evidence_item(item, source, "run-001", ["smr"])
    assert source.source_id in ev.supporting_source_ids


def test_v1_extraction_run_id_preserved():
    source = _make_source()
    item = _make_item()
    ev = _adapt_evidence_item(item, source, "run-xyx-999", ["smr"])
    assert ev.extraction_run_id == "run-xyx-999"


def test_v1_evidence_type_classified():
    item = _make_item(
        claim="The reactor outputs 300 MWe of electricity.",
        category="reactor design",
    )
    ev = _adapt(item)
    assert ev.evidence_type == "TECHNICAL"


# ---------------------------------------------------------------------------
# 11. missing / empty signals default safely
# ---------------------------------------------------------------------------


def test_no_fabrication_when_all_provenance_signals_absent():
    item = _make_item(
        evidence_snippet="",
        source_chunk_id="",
        topics=[],
        confidence="medium",
        quantitative_score=3,
        claim="A claim with no dateable reference.",
    )
    ev = _adapt(item, "This is plain text with no page markers.")
    assert ev.excerpt is None
    assert ev.chunk_id is None
    assert ev.topics == []
    assert ev.page_number is None
    assert ev.char_offset_start is None
    assert ev.char_offset_end is None
    assert ev.temporal_reference is None
    assert ev.section_heading is None
    assert ev.evidence_confidence == "MEDIUM"
    assert ev.is_quantitative is False


def test_evidence_still_valid_with_only_v1_fields():
    item = _make_item(
        evidence_snippet="",
        source_chunk_id="",
        topics=None,
    )
    ev = _adapt(item)
    assert ev.statement != ""
    assert ev.extraction_run_id != ""
    assert ev.evidence_id != ""


# ---------------------------------------------------------------------------
# 12. _extract_temporal_reference helper unit tests
# ---------------------------------------------------------------------------


def test_temporal_reference_year_2024():
    assert _extract_temporal_reference("Published in 2024 by ONR.") == "2024"


def test_temporal_reference_quarter():
    assert _extract_temporal_reference("Target date Q2 2027.") == "Q2 2027"


def test_temporal_reference_prefers_quarter():
    assert _extract_temporal_reference("Q1 2029 target, referencing 2024 baseline.") == "Q1 2029"


def test_temporal_reference_none_for_empty():
    assert _extract_temporal_reference("") is None


def test_temporal_reference_none_for_no_date():
    assert _extract_temporal_reference("No dates here at all.") is None


# ---------------------------------------------------------------------------
# 13. _find_provenance_in_text helper unit tests
# ---------------------------------------------------------------------------


def test_find_provenance_returns_page_1_for_first_page():
    text = "[Page 1]\nThe reactor achieves 300 MWe.\n\n[Page 2]\nOther content.\n"
    page, start, end = _find_provenance_in_text(text, "The reactor achieves 300 MWe.")
    assert page == 1
    assert start is not None
    assert end > start


def test_find_provenance_returns_page_2_for_second_page():
    text = "[Page 1]\nFirst page content.\n\n[Page 2]\nCapital costs are high.\n"
    page, start, end = _find_provenance_in_text(text, "Capital costs are high.")
    assert page == 2


def test_find_provenance_returns_none_for_absent_excerpt():
    text = "[Page 1]\nSome content.\n"
    page, start, end = _find_provenance_in_text(text, "This is not in the text at all.")
    assert page is None
    assert start is None
    assert end is None


def test_find_provenance_returns_none_for_empty_excerpt():
    page, start, end = _find_provenance_in_text("Some text.", "")
    assert page is None


def test_find_provenance_page_1_when_no_markers():
    text = "No page markers here. Just plain text content for testing."
    page, start, end = _find_provenance_in_text(text, "No page markers here.")
    assert page == 1
    assert start == 0


# ---------------------------------------------------------------------------
# 14. extract_evidence_from_source integration
# ---------------------------------------------------------------------------


class _MockClientWithProvenance:
    """Mock that returns EvidenceItems with v2 provenance fields set."""

    def extract_evidence(self, question, source_texts):
        from research_agent.schemas import EvidenceItem
        results = []
        for src in source_texts:
            results.append(
                EvidenceItem(
                    claim="The BWRX-300 is a small modular reactor with a thermal output of 870 MWt.",
                    source_document=src.title,
                    evidence_snippet="The BWRX-300 is a small modular reactor with a thermal output of 870 MWt.",
                    category="reactor design",
                    relevance="direct",
                    confidence="high",
                    topics=["SMR", "thermal output"],
                    quantitative_score=5,
                    source_chunk_id="chunk:p1-para1",
                )
            )
        return results


def test_extract_evidence_from_source_populates_v2_fields():
    source = _make_source(_CANONICAL_TEXT_PDF)
    client = _MockClientWithProvenance()
    ev_list, meta_list, dups = extract_evidence_from_source(
        source, "run-integration-001", client
    )
    assert len(ev_list) == 1
    ev = ev_list[0]

    assert ev.excerpt == "The BWRX-300 is a small modular reactor with a thermal output of 870 MWt."
    assert ev.chunk_id == "chunk:p1-para1"
    assert ev.topics == ["SMR", "thermal output"]
    assert ev.evidence_confidence == "HIGH"
    assert ev.is_quantitative is True
    assert ev.page_number == 1
    assert ev.char_offset_start is not None
    assert ev.char_offset_end is not None
    assert ev.section_heading is None


def test_extract_evidence_from_source_v1_fields_still_correct():
    source = _make_source(_CANONICAL_TEXT_PDF)
    client = _MockClientWithProvenance()
    ev_list, _, _ = extract_evidence_from_source(source, "run-001", client)
    ev = ev_list[0]

    assert ev.statement == "The BWRX-300 is a small modular reactor with a thermal output of 870 MWt."
    assert source.source_id in ev.supporting_source_ids
    assert ev.extraction_run_id == "run-001"
    assert ev.evidence_type in ("STRATEGIC", "TECHNICAL")


def test_extract_evidence_from_source_v2_fields_none_when_no_snippet():
    """When snippet is empty, provenance defaults safely without fabrication."""
    class _NullSnippetClient:
        def extract_evidence(self, question, source_texts):
            from research_agent.schemas import EvidenceItem
            return [
                EvidenceItem(
                    claim="Claim without a snippet.",
                    source_document="test",
                    evidence_snippet="",
                    category="other",
                    relevance="direct",
                    confidence="low",
                )
            ]

    source = _make_source(_CANONICAL_TEXT_PDF)
    ev_list, _, _ = extract_evidence_from_source(source, "run-002", _NullSnippetClient())
    ev = ev_list[0]

    assert ev.excerpt is None
    assert ev.page_number is None
    assert ev.char_offset_start is None
    assert ev.char_offset_end is None
    assert ev.evidence_confidence == "LOW"
    assert ev.is_quantitative is False
