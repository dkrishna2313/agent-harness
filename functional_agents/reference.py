"""Reference — K1.1.

Communication-layer projection of Knowledge Store objects.

Architecture
------------
The Reference abstraction sits between Evidence (knowledge representation) and
Communication (report/deliverable output).  It is generated deterministically
from existing Source and Evidence records and is never persisted.

    Source  ──►  Evidence  ──►  Reference  ──►  Communication
   (knowledge)  (knowledge)  (communication)     (report)

Responsibilities
----------------
Reference  — human-readable citation metadata; communication-friendly
             representation of one Source and its supporting Evidence records.
             Performs no reasoning.  Owns no knowledge.

ReferenceBuilder — accepts Evidence records, resolves Source records, groups
                   Evidence by Source, and generates Reference objects.
                   Performs no rendering beyond constructing a canonical
                   citation string.

Constraints
-----------
- Reference is immutable (frozen Pydantic model).
- Reference is not persisted.
- Reference exposes no filesystem paths, no Source IDs, no Evidence UUIDs
  in rendered output — those are implementation artifacts, not communication
  artifacts.  The citation_text field is the communication surface.
- The builder never modifies the Knowledge Store.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Reference model
# ---------------------------------------------------------------------------


class Reference(BaseModel):
    """Immutable communication-layer citation object.

    One Reference is produced per unique Source cited by one or more Evidence
    records.  Not persisted — generated deterministically on demand.

    Fields mirror the bibliographic metadata of the originating Source, plus
    the set of Evidence records that reference it.  ``citation_text`` is the
    canonical rendered citation string for use in reports.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    reference_id: str               # equals source_id — content-addressed, stable
    source_id: str                  # FK → knowledge_store Source

    # Evidence grouping
    evidence_ids: list[str]         # Evidence records that cite this Source (sorted)

    # Bibliographic metadata projected from Source
    title: str
    short_title: str | None = None  # truncated title for inline use; None if title ≤ 60 chars
    author: str | None = None
    organization: str | None = None
    publisher: str | None = None
    publication_date: date | None = None
    url: str | None = None          # always None in current implementation (no URLs in store)

    # Page-level provenance — always None in current model (no location data at Evidence level)
    page_number: int | None = None

    # Derived
    citation_text: str              # canonical citation string ready for rendering
    notes: str | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MAX_SHORT_TITLE = 60


def _get_field(source: Any, name: str) -> Any:
    """Get a named field from either a Source object or a dict."""
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def _parse_date(val: Any) -> date | None:
    """Coerce str, datetime, date, or None to date | None."""
    if val is None:
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()
    try:
        return datetime.strptime(str(val)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _make_short_title(title: str) -> str | None:
    """Return a truncated title if it exceeds ``_MAX_SHORT_TITLE`` chars, else None."""
    if len(title) <= _MAX_SHORT_TITLE:
        return None
    return title[: _MAX_SHORT_TITLE - 1].rstrip() + "…"  # …


def _build_citation_text(
    *,
    title: str,
    organization: str | None,
    author: str | None,
    publisher: str | None,
    publication_date: date | None,
) -> str:
    """Build canonical citation string with graceful degradation.

    Priority for the 'who' element: organization > author > publisher.
    Format: ``{who}. {title}. {year}.``
    Missing elements are omitted; title is always included.
    """
    parts: list[str] = []

    who = organization or author or publisher
    if who:
        parts.append(who)

    parts.append(title)

    if publication_date is not None:
        parts.append(str(publication_date.year))

    return ". ".join(parts) + "."


# ---------------------------------------------------------------------------
# ReferenceBuilder
# ---------------------------------------------------------------------------


class ReferenceBuilder:
    """Builds Reference objects from Evidence records and a Source resolver.

    Groups Evidence by ``source_id``.  Produces one Reference per unique
    Source that can be resolved.  Returns references sorted by ``citation_text``
    (case-insensitive).

    Args:
        source_resolver: callable that accepts a ``source_id`` string and
            returns a Source object (``knowledge.models.Source``) or a dict
            with the same field names, or ``None`` if the source cannot be
            resolved.  The builder never mutates the returned object.

    Usage::

        from knowledge.store import KnowledgeStore
        from functional_agents.reference import ReferenceBuilder

        store = KnowledgeStore("knowledge_store/")
        builder = ReferenceBuilder(source_resolver=store.find_source)
        refs = builder.build(context.domain_evidence.get("ai_data_centers", []))
    """

    def __init__(
        self,
        source_resolver: Callable[[str], Any | None],
    ) -> None:
        self._resolver = source_resolver

    def build(self, evidence_records: list[dict[str, Any]]) -> list[Reference]:
        """Build References from a list of evidence dicts.

        Each dict must carry:
        - ``evidence_id`` (or ``id``) — unique identifier
        - ``supporting_source_ids`` — list of source_id strings

        Evidence with no ``supporting_source_ids`` is silently skipped.
        Evidence whose source cannot be resolved is silently skipped.
        A single Evidence record with multiple source IDs contributes to each.

        Returns:
            List of Reference objects sorted by ``citation_text`` (casefold).
        """
        # Group evidence_ids by source_id
        source_to_evids: dict[str, list[str]] = {}
        for ev in evidence_records:
            ev_id = ev.get("evidence_id") or ev.get("id") or ""
            for sid in (ev.get("supporting_source_ids") or []):
                source_to_evids.setdefault(sid, []).append(ev_id)

        references: list[Reference] = []
        for source_id, ev_ids in source_to_evids.items():
            source = self._resolver(source_id)
            if source is None:
                continue
            references.append(self._make_reference(source_id, source, ev_ids))

        return sorted(references, key=lambda r: r.citation_text.casefold())

    def _make_reference(
        self,
        source_id: str,
        source: Any,
        evidence_ids: list[str],
    ) -> Reference:
        title = _get_field(source, "title") or "Untitled"
        author = _get_field(source, "author") or None
        organization = _get_field(source, "organization") or None
        publisher = _get_field(source, "publisher") or None
        publication_date = _parse_date(_get_field(source, "publication_date"))

        citation = _build_citation_text(
            title=title,
            organization=organization,
            author=author,
            publisher=publisher,
            publication_date=publication_date,
        )

        return Reference(
            reference_id=source_id,
            source_id=source_id,
            evidence_ids=sorted(set(evidence_ids)),
            title=title,
            short_title=_make_short_title(title),
            author=author,
            organization=organization,
            publisher=publisher,
            publication_date=publication_date,
            url=None,
            page_number=None,
            citation_text=citation,
            notes=None,
        )
