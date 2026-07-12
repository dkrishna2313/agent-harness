"""K1.1 — Reference model and ReferenceBuilder tests."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from functional_agents.reference import (
    Reference,
    ReferenceBuilder,
    _build_citation_text,
    _get_field,
    _make_short_title,
    _parse_date,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _source_dict(
    source_id: str = "src-001",
    title: str = "Grid Interconnection Study",
    author: str | None = "J. Smith",
    organization: str | None = None,
    publisher: str | None = "ERCOT",
    publication_date: Any = date(2024, 3, 1),
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "title": title,
        "author": author,
        "organization": organization,
        "publisher": publisher,
        "publication_date": publication_date,
    }


def _evidence(
    evidence_id: str,
    source_ids: list[str],
) -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "statement": f"Claim from {evidence_id}.",
        "supporting_source_ids": source_ids,
    }


# ---------------------------------------------------------------------------
# Reference model
# ---------------------------------------------------------------------------

class TestReference:
    def test_construction_minimal(self):
        ref = Reference(
            reference_id="s1",
            source_id="s1",
            evidence_ids=["e1"],
            title="Test Doc",
            citation_text="Test Doc.",
        )
        assert ref.reference_id == "s1"
        assert ref.evidence_ids == ["e1"]
        assert ref.citation_text == "Test Doc."

    def test_defaults(self):
        ref = Reference(
            reference_id="s1",
            source_id="s1",
            evidence_ids=[],
            title="T",
            citation_text="T.",
        )
        assert ref.author is None
        assert ref.organization is None
        assert ref.publisher is None
        assert ref.publication_date is None
        assert ref.short_title is None
        assert ref.url is None
        assert ref.page_number is None
        assert ref.notes is None

    def test_immutable(self):
        ref = Reference(
            reference_id="s1",
            source_id="s1",
            evidence_ids=["e1"],
            title="T",
            citation_text="T.",
        )
        with pytest.raises(Exception):
            ref.title = "changed"  # type: ignore[misc]

    def test_full_construction(self):
        ref = Reference(
            reference_id="abc",
            source_id="abc",
            evidence_ids=["e1", "e2"],
            title="Long title here",
            short_title="Short",
            author="A. Author",
            organization="ERCOT",
            publisher="DOE",
            publication_date=date(2024, 6, 1),
            url=None,
            page_number=None,
            citation_text="ERCOT. Long title here. 2024.",
            notes="Draft version",
        )
        assert ref.organization == "ERCOT"
        assert ref.publication_date == date(2024, 6, 1)
        assert ref.citation_text == "ERCOT. Long title here. 2024."


# ---------------------------------------------------------------------------
# _parse_date
# ---------------------------------------------------------------------------

class TestParseDate:
    def test_date_object(self):
        d = date(2024, 1, 15)
        assert _parse_date(d) == d

    def test_string_iso(self):
        assert _parse_date("2024-01-15") == date(2024, 1, 15)

    def test_string_with_time(self):
        assert _parse_date("2024-01-15T12:00:00") == date(2024, 1, 15)

    def test_none(self):
        assert _parse_date(None) is None

    def test_invalid_string(self):
        assert _parse_date("not-a-date") is None

    def test_datetime_object(self):
        from datetime import datetime
        dt = datetime(2024, 5, 10, 9, 0)
        assert _parse_date(dt) == date(2024, 5, 10)


# ---------------------------------------------------------------------------
# _make_short_title
# ---------------------------------------------------------------------------

class TestMakeShortTitle:
    def test_short_title_returns_none(self):
        assert _make_short_title("Short") is None

    def test_exactly_60_chars_returns_none(self):
        title = "A" * 60
        assert _make_short_title(title) is None

    def test_61_chars_truncates(self):
        title = "A" * 61
        result = _make_short_title(title)
        assert result is not None
        assert len(result) <= 60
        assert result.endswith("…")

    def test_long_title_truncates(self):
        title = "The Long-Term System Assessment for ERCOT's Transmission Network"
        result = _make_short_title(title)
        assert result is not None
        assert len(result) <= 60
        assert result.endswith("…")


# ---------------------------------------------------------------------------
# _build_citation_text
# ---------------------------------------------------------------------------

class TestBuildCitationText:
    def test_full_fields_org_wins(self):
        result = _build_citation_text(
            title="Grid Study",
            organization="ERCOT",
            author="J. Smith",
            publisher="DOE",
            publication_date=date(2024, 3, 1),
        )
        assert result == "ERCOT. Grid Study. 2024."

    def test_author_fallback_when_no_org(self):
        result = _build_citation_text(
            title="Grid Study",
            organization=None,
            author="J. Smith",
            publisher="DOE",
            publication_date=date(2024, 3, 1),
        )
        assert result == "J. Smith. Grid Study. 2024."

    def test_publisher_fallback_when_no_org_or_author(self):
        result = _build_citation_text(
            title="Grid Study",
            organization=None,
            author=None,
            publisher="DOE",
            publication_date=date(2024, 3, 1),
        )
        assert result == "DOE. Grid Study. 2024."

    def test_title_only_when_no_who(self):
        result = _build_citation_text(
            title="Grid Study",
            organization=None,
            author=None,
            publisher=None,
            publication_date=date(2024, 3, 1),
        )
        assert result == "Grid Study. 2024."

    def test_no_year(self):
        result = _build_citation_text(
            title="Grid Study",
            organization="ERCOT",
            author=None,
            publisher=None,
            publication_date=None,
        )
        assert result == "ERCOT. Grid Study."

    def test_title_only_no_who_no_year(self):
        result = _build_citation_text(
            title="Untitled",
            organization=None,
            author=None,
            publisher=None,
            publication_date=None,
        )
        assert result == "Untitled."

    def test_always_ends_with_period(self):
        for kwargs in [
            dict(title="T", organization="O", author=None, publisher=None, publication_date=None),
            dict(title="T", organization=None, author=None, publisher=None, publication_date=date(2020, 1, 1)),
            dict(title="T", organization=None, author=None, publisher=None, publication_date=None),
        ]:
            assert _build_citation_text(**kwargs).endswith(".")


# ---------------------------------------------------------------------------
# _get_field
# ---------------------------------------------------------------------------

class TestGetField:
    def test_dict(self):
        d = {"title": "My Doc", "author": None}
        assert _get_field(d, "title") == "My Doc"
        assert _get_field(d, "author") is None
        assert _get_field(d, "missing") is None

    def test_object(self):
        class Obj:
            title = "My Doc"
            author = None
        obj = Obj()
        assert _get_field(obj, "title") == "My Doc"
        assert _get_field(obj, "author") is None
        assert _get_field(obj, "missing") is None


# ---------------------------------------------------------------------------
# ReferenceBuilder
# ---------------------------------------------------------------------------

class TestReferenceBuilder:
    def _builder(self, sources: dict[str, dict]) -> ReferenceBuilder:
        return ReferenceBuilder(source_resolver=lambda sid: sources.get(sid))

    def test_empty_input(self):
        builder = self._builder({})
        assert builder.build([]) == []

    def test_single_evidence_single_source(self):
        sources = {"s1": _source_dict(source_id="s1")}
        builder = self._builder(sources)
        refs = builder.build([_evidence("e1", ["s1"])])
        assert len(refs) == 1
        assert refs[0].source_id == "s1"
        assert refs[0].evidence_ids == ["e1"]

    def test_multiple_evidence_same_source_groups_correctly(self):
        sources = {"s1": _source_dict(source_id="s1")}
        builder = self._builder(sources)
        ev = [_evidence("e1", ["s1"]), _evidence("e2", ["s1"]), _evidence("e3", ["s1"])]
        refs = builder.build(ev)
        assert len(refs) == 1
        assert sorted(refs[0].evidence_ids) == ["e1", "e2", "e3"]

    def test_multiple_evidence_different_sources(self):
        sources = {
            "s1": _source_dict(source_id="s1", title="A Doc", organization="Org A"),
            "s2": _source_dict(source_id="s2", title="B Doc", organization="Org B"),
        }
        builder = self._builder(sources)
        ev = [_evidence("e1", ["s1"]), _evidence("e2", ["s2"])]
        refs = builder.build(ev)
        assert len(refs) == 2

    def test_evidence_with_no_source_ids_is_skipped(self):
        sources = {"s1": _source_dict(source_id="s1")}
        builder = self._builder(sources)
        ev = [
            {"evidence_id": "e1", "statement": "claim", "supporting_source_ids": []},
            _evidence("e2", ["s1"]),
        ]
        refs = builder.build(ev)
        assert len(refs) == 1
        assert refs[0].evidence_ids == ["e2"]

    def test_unresolvable_source_is_skipped(self):
        builder = self._builder({})  # resolver returns None for everything
        refs = builder.build([_evidence("e1", ["s-missing"])])
        assert refs == []

    def test_multi_source_evidence_appears_in_each_reference(self):
        sources = {
            "s1": _source_dict(source_id="s1", title="Doc A"),
            "s2": _source_dict(source_id="s2", title="Doc B"),
        }
        builder = self._builder(sources)
        ev = [_evidence("e1", ["s1", "s2"])]
        refs = builder.build(ev)
        assert len(refs) == 2
        src_ids = {r.source_id for r in refs}
        assert src_ids == {"s1", "s2"}
        for ref in refs:
            assert "e1" in ref.evidence_ids

    def test_evidence_ids_deduplicated_and_sorted(self):
        sources = {"s1": _source_dict(source_id="s1")}
        builder = self._builder(sources)
        # Same evidence_id referenced twice (hypothetical duplicate input)
        ev = [_evidence("e1", ["s1"]), _evidence("e1", ["s1"])]
        refs = builder.build(ev)
        assert len(refs) == 1
        assert refs[0].evidence_ids == ["e1"]

    def test_output_sorted_by_citation_text_casefold(self):
        sources = {
            "s1": _source_dict(source_id="s1", title="Zebra Report", organization=None, author=None, publisher=None),
            "s2": _source_dict(source_id="s2", title="Alpha Report", organization=None, author=None, publisher=None),
            "s3": _source_dict(source_id="s3", title="midpoint Report", organization=None, author=None, publisher=None),
        }
        builder = self._builder(sources)
        ev = [_evidence("e1", ["s1"]), _evidence("e2", ["s2"]), _evidence("e3", ["s3"])]
        refs = builder.build(ev)
        citations = [r.citation_text for r in refs]
        assert citations == sorted(citations, key=str.casefold)

    def test_source_dict_access(self):
        sources = {
            "s1": {
                "source_id": "s1",
                "title": "Dict Source",
                "author": "A. Author",
                "organization": None,
                "publisher": None,
                "publication_date": "2023-07-01",
            }
        }
        builder = self._builder(sources)
        refs = builder.build([_evidence("e1", ["s1"])])
        assert len(refs) == 1
        assert refs[0].title == "Dict Source"
        assert refs[0].author == "A. Author"
        assert refs[0].publication_date == date(2023, 7, 1)

    def test_source_object_access(self):
        class FakeSource:
            source_id = "s1"
            title = "Object Source"
            author = "B. Builder"
            organization = "TestOrg"
            publisher = None
            publication_date = date(2022, 1, 1)

        builder = ReferenceBuilder(source_resolver=lambda _: FakeSource())
        refs = builder.build([_evidence("e1", ["s1"])])
        assert len(refs) == 1
        assert refs[0].organization == "TestOrg"
        assert refs[0].citation_text == "TestOrg. Object Source. 2022."

    def test_citation_text_uses_organization_over_author(self):
        sources = {
            "s1": _source_dict(
                source_id="s1",
                title="Grid Study",
                author="J. Smith",
                organization="ERCOT",
                publication_date=date(2024, 1, 1),
            )
        }
        builder = self._builder(sources)
        refs = builder.build([_evidence("e1", ["s1"])])
        assert refs[0].citation_text == "ERCOT. Grid Study. 2024."

    def test_citation_text_degrades_gracefully_no_metadata(self):
        sources = {
            "s1": {
                "source_id": "s1",
                "title": "Anonymous Report",
                "author": None,
                "organization": None,
                "publisher": None,
                "publication_date": None,
            }
        }
        builder = self._builder(sources)
        refs = builder.build([_evidence("e1", ["s1"])])
        assert refs[0].citation_text == "Anonymous Report."

    def test_url_always_none(self):
        sources = {"s1": _source_dict(source_id="s1")}
        builder = self._builder(sources)
        refs = builder.build([_evidence("e1", ["s1"])])
        assert refs[0].url is None

    def test_page_number_always_none(self):
        sources = {"s1": _source_dict(source_id="s1")}
        builder = self._builder(sources)
        refs = builder.build([_evidence("e1", ["s1"])])
        assert refs[0].page_number is None

    def test_short_title_set_for_long_title(self):
        long_title = "The Comprehensive Long-Term System Assessment for Transmission Networks"
        sources = {"s1": _source_dict(source_id="s1", title=long_title)}
        builder = self._builder(sources)
        refs = builder.build([_evidence("e1", ["s1"])])
        assert refs[0].short_title is not None
        assert len(refs[0].short_title) <= 60

    def test_short_title_none_for_short_title(self):
        sources = {"s1": _source_dict(source_id="s1", title="Short Title")}
        builder = self._builder(sources)
        refs = builder.build([_evidence("e1", ["s1"])])
        assert refs[0].short_title is None

    def test_reference_id_equals_source_id(self):
        sources = {"s1": _source_dict(source_id="s1")}
        builder = self._builder(sources)
        refs = builder.build([_evidence("e1", ["s1"])])
        assert refs[0].reference_id == refs[0].source_id

    def test_publication_date_string_parsed(self):
        sources = {
            "s1": {
                "source_id": "s1",
                "title": "Test",
                "author": None,
                "organization": None,
                "publisher": None,
                "publication_date": "2021-11-30",
            }
        }
        builder = self._builder(sources)
        refs = builder.build([_evidence("e1", ["s1"])])
        assert refs[0].publication_date == date(2021, 11, 30)

    def test_missing_evidence_id_key_graceful(self):
        sources = {"s1": _source_dict(source_id="s1")}
        builder = self._builder(sources)
        ev = [{"supporting_source_ids": ["s1"]}]  # no evidence_id key
        refs = builder.build(ev)
        assert len(refs) == 1
        assert refs[0].evidence_ids == [""]  # empty string fallback

    def test_deterministic_output(self):
        sources = {
            "s1": _source_dict(source_id="s1", title="Doc A"),
            "s2": _source_dict(source_id="s2", title="Doc B"),
        }
        builder = self._builder(sources)
        ev = [_evidence("e1", ["s1"]), _evidence("e2", ["s2"]), _evidence("e3", ["s1"])]
        refs1 = builder.build(ev)
        refs2 = builder.build(ev)
        assert [r.reference_id for r in refs1] == [r.reference_id for r in refs2]
        assert [r.evidence_ids for r in refs1] == [r.evidence_ids for r in refs2]
        assert [r.citation_text for r in refs1] == [r.citation_text for r in refs2]
