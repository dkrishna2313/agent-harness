"""Tests for knowledge/source_normalizer.py — Tier 2 regex extraction.

Tier 1 (PDF/DOCX metadata) and Tier 3 (LLM) require external deps or network;
those are exercised in the integration build rather than the unit test suite.
"""

from __future__ import annotations

from datetime import date

import pytest

from knowledge.source_normalizer import (
    SourceNormalization,
    _extract_text_metadata,
    _org_from_copyright,
    _parse_pdf_date,
    _is_skip_author,
)


# ---------------------------------------------------------------------------
# SourceNormalization.merge
# ---------------------------------------------------------------------------


def test_merge_prefers_self():
    high = SourceNormalization(title="Good Title", author="Real Author")
    low = SourceNormalization(title="Fallback", author="Fallback Author", organization="Org")
    merged = high.merge(low)
    assert merged.title == "Good Title"
    assert merged.author == "Real Author"
    assert merged.organization == "Org"  # filled from low


def test_merge_fills_none_from_lower():
    high = SourceNormalization(title="Title")
    low = SourceNormalization(document_version="Revision H", page_count=122)
    merged = high.merge(low)
    assert merged.title == "Title"
    assert merged.document_version == "Revision H"
    assert merged.page_count == 122


def test_needs_llm_true_when_all_provenance_empty():
    norm = SourceNormalization(title="Some Title")
    assert norm.needs_llm() is True


def test_needs_llm_false_when_copyright_present():
    norm = SourceNormalization(copyright="© 2025 ACME Corp.")
    assert norm.needs_llm() is False


def test_needs_llm_false_when_author_present():
    norm = SourceNormalization(author="U.S. Department of Energy")
    assert norm.needs_llm() is False


# ---------------------------------------------------------------------------
# PDF date parsing
# ---------------------------------------------------------------------------


def test_parse_pdf_date_basic():
    assert _parse_pdf_date("D:20251014135859") == date(2025, 10, 14)


def test_parse_pdf_date_with_offset():
    assert _parse_pdf_date("D:20241001071228-04'00'") == date(2024, 10, 1)


def test_parse_pdf_date_none():
    assert _parse_pdf_date(None) is None
    assert _parse_pdf_date("") is None


# ---------------------------------------------------------------------------
# Author filtering
# ---------------------------------------------------------------------------


def test_skip_creative_services():
    assert _is_skip_author("Creative Services") is True


def test_skip_microsoft():
    assert _is_skip_author("Microsoft Word") is True


def test_keep_real_author():
    assert _is_skip_author("U.S. Department of Energy") is False
    assert _is_skip_author("Barney C. Hadden") is False
    assert _is_skip_author("IEA, International Energy Agency") is False


# ---------------------------------------------------------------------------
# Copyright → organization extraction
# ---------------------------------------------------------------------------


def test_org_from_copyright_ge_vernova():
    s = "© 2025 GE Vernova Hitachi Nuclear Energy Americas LLC. All rights reserved."
    assert _org_from_copyright(s) == "GE Vernova Hitachi Nuclear Energy Americas LLC"


def test_org_from_copyright_iea():
    s = "Copyright 2022 International Energy Agency"
    assert _org_from_copyright(s) == "International Energy Agency"


def test_org_from_copyright_year_range():
    s = "© 2020–2024 World Nuclear Association"
    assert _org_from_copyright(s) == "World Nuclear Association"


def test_org_from_copyright_returns_none_for_short():
    assert _org_from_copyright("© 2025 AB") is None


# ---------------------------------------------------------------------------
# Tier 2 text extraction
# ---------------------------------------------------------------------------

_BWRX_FIRST_PAGE = """\
[Page 1]
005N9751 REVISION H

BWRX-300 General Description

© 2025 GE Vernova Hitachi Nuclear Energy Americas LLC. All rights reserved.
GE is a trademark of General Electric Company used under trademark license.
HITACHI is a trademark of Hitachi, Ltd. used under trademark license.

October 2025
"""

_DOE_FIRST_PAGE = """\
[Page 1]
DOE/EE-0001

Pathways to Commercial Liftoff: Advanced Nuclear

U.S. Department of Energy

October 2024
"""

_NEA_FIRST_PAGE = """\
[Page 1]
NEA/NDC-2023/3

The NEA Small Modular Reactor Dashboard

Nuclear Energy Agency (NEA)

February 2023
"""


def test_document_number_bwrx():
    norm = _extract_text_metadata(_BWRX_FIRST_PAGE)
    assert norm.document_number == "005N9751"


def test_document_version_revision_h():
    norm = _extract_text_metadata(_BWRX_FIRST_PAGE)
    assert norm.document_version == "Revision H"


def test_copyright_bwrx():
    norm = _extract_text_metadata(_BWRX_FIRST_PAGE)
    assert norm.copyright is not None
    assert "GE Vernova" in norm.copyright


def test_organization_from_copyright():
    norm = _extract_text_metadata(_BWRX_FIRST_PAGE)
    assert norm.organization == "GE Vernova Hitachi Nuclear Energy Americas LLC"
    assert norm.publisher == "GE Vernova Hitachi Nuclear Energy Americas LLC"


def test_publication_date_bwrx():
    norm = _extract_text_metadata(_BWRX_FIRST_PAGE)
    assert norm.publication_date == date(2025, 10, 1)


def test_document_number_doe():
    norm = _extract_text_metadata(_DOE_FIRST_PAGE)
    assert norm.document_number == "DOE/EE-0001"


def test_publication_date_doe():
    norm = _extract_text_metadata(_DOE_FIRST_PAGE)
    assert norm.publication_date == date(2024, 10, 1)


def test_document_number_nea():
    norm = _extract_text_metadata(_NEA_FIRST_PAGE)
    assert norm.document_number == "NEA/NDC-2023/3"


def test_publication_date_nea():
    norm = _extract_text_metadata(_NEA_FIRST_PAGE)
    assert norm.publication_date == date(2023, 2, 1)


def test_version_pattern():
    text = "[Page 1]\nSoftware Package\nVersion 3.2.1\nSome content"
    norm = _extract_text_metadata(text)
    assert norm.document_version == "Version 3.2.1"


def test_no_false_positives_on_clean_text():
    text = "[Page 1]\nSmall Modular Reactors provide clean energy solutions.\n"
    norm = _extract_text_metadata(text)
    assert norm.document_number is None
    assert norm.document_version is None
    assert norm.copyright is None
    assert norm.organization is None


# ---------------------------------------------------------------------------
# New Source model fields
# ---------------------------------------------------------------------------


def test_source_new_fields_have_defaults():
    from knowledge.models import Source
    from datetime import date as _date

    text = "Nuclear power content."
    fp = Source.compute_fingerprint(text)
    s = Source(
        source_id=Source.compute_source_id(fp),
        uri="smr_sources/test.pdf",
        title="Test",
        retrieved_date=_date(2026, 6, 28),
        fingerprint=fp,
        document_type="PDF",
        domain="smr",
        canonical_text=text,
    )
    assert s.subtitle is None
    assert s.organization is None
    assert s.document_version is None
    assert s.document_number is None


def test_source_new_fields_populated():
    from knowledge.models import Source
    from datetime import date as _date

    text = "Nuclear power content."
    fp = Source.compute_fingerprint(text)
    s = Source(
        source_id=Source.compute_source_id(fp),
        uri="smr_sources/test.pdf",
        title="BWRX-300 General Description",
        subtitle="Design Overview",
        author=None,
        organization="GE Vernova Hitachi Nuclear Energy Americas LLC",
        publisher="GE Vernova Hitachi Nuclear Energy Americas LLC",
        publication_date=_date(2025, 10, 1),
        retrieved_date=_date(2026, 6, 28),
        fingerprint=fp,
        document_type="PDF",
        domain="smr",
        copyright="© 2025 GE Vernova Hitachi Nuclear Energy Americas LLC",
        canonical_text=text,
        page_count=122,
        document_version="Revision H",
        document_number="005N9751",
    )
    assert s.organization == "GE Vernova Hitachi Nuclear Energy Americas LLC"
    assert s.document_version == "Revision H"
    assert s.document_number == "005N9751"
    assert s.page_count == 122
    assert s.publication_date == _date(2025, 10, 1)
