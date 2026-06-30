"""Source normalization — three-tier extraction of structured document provenance.

Tier 1 — PDF/DOCX embedded metadata (XMP, document properties).
          Fast, deterministic. page_count is always available here.

Tier 2 — Deterministic regex on first-page canonical text.
          copyright → organization, document_version, document_number,
          publication_date from month-year patterns.

Tier 3 — LLM (claude-haiku) via anthropic SDK.
          Only invoked when Tier 1+2 leave all provenance fields empty
          (author, organization, publisher, copyright all None).
          The LLM never overwrites a value set by a higher tier.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, fields as _dc_fields
from datetime import date
from pathlib import Path

LOGGER = logging.getLogger(__name__)

_MONTH_MAP: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# PDF /Creator and /Author values that indicate software or internal teams, not document authors
_SKIP_AUTHOR_VALUES: frozenset[str] = frozenset([
    "creative services", "microsoft", "adobe", "acrobat", "libreoffice",
    "openoffice", "latex", "indesign", "quarkxpress", "mozilla",
    "pdfium", "chrome", "safari", "webkit",
])

# PDF /Title patterns that are internal InDesign/Word filenames, not human-readable titles
_INTERNAL_TITLE_RE = re.compile(r'^[A-Z]{2,4}\d{4,}[A-Z_]')  # e.g. "GEA34989B BWRX-300..."

# PDF creation date format: "D:YYYYMMDDHHmmSS[+offset]"
_PDF_DATE_RE = re.compile(r"D:(\d{4})(\d{2})(\d{2})")

# Document revision patterns
_REVISION_RE = re.compile(
    r"(?:^|\s)(?:revision|rev\.?)\s+([A-Z0-9][A-Z0-9._-]*)",
    re.IGNORECASE | re.MULTILINE,
)
_VERSION_RE = re.compile(
    r"(?:^|\s)(?:version|ver\.?)\s+(\d+(?:\.\d+)+)",
    re.IGNORECASE | re.MULTILINE,
)

# Document number patterns (ordered most-to-least specific)
_DOC_NUM_RES: list[re.Pattern] = [
    re.compile(r"\b\d{3}[A-Z]\d{4}\b"),                              # 005N9751
    re.compile(r"\bDOE[-/][A-Z]{1,4}[-/][A-Z0-9][-\d]{2,}\b"),      # DOE/ID-Number, DOE/EE-0001
    re.compile(r"\bNUREG[-/]\d{4,5}\b"),                              # NUREG-1234
    re.compile(r"\bNEA[-/][A-Z]{2,}[-/]\d{4}(?:/\d+)?\b"),           # NEA/NDC-2024/1
    re.compile(r"\bINL[-/][A-Z]{2,4}[-/]\d{2,4}[-/]\d{4,6}\b"),      # INL/EXT-24-12345
    re.compile(r"\bIAEA[-/][A-Z\-]{3,}[-/]\d{4}\b"),                  # IAEA-TECDOC-2024
]

# Copyright notice
_COPYRIGHT_RE = re.compile(
    r"(?:©|copyright\s*©?)\s*(?:\d{4}[-–]\d{4}|\d{4})[^\n.]{3,150}",
    re.IGNORECASE,
)

# Month + optional day + year
_MONTH_YEAR_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+(?:\d{1,2},?\s+)?(\d{4})\b",
    re.IGNORECASE,
)

# LLM JSON field list
_LLM_FIELDS = (
    "title", "subtitle", "author", "organization", "publisher",
    "publication_date", "document_version", "document_number", "copyright",
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SourceNormalization:
    """Structured provenance extracted from a source document."""

    title: str | None = None
    subtitle: str | None = None
    author: str | None = None
    organization: str | None = None
    publisher: str | None = None
    publication_date: date | None = None
    document_version: str | None = None
    document_number: str | None = None
    copyright: str | None = None
    page_count: int | None = None

    def merge(self, lower: "SourceNormalization") -> "SourceNormalization":
        """Return a new normalization preferring self (higher tier), falling back to lower."""
        out = SourceNormalization()
        for f in _dc_fields(self):
            v = getattr(self, f.name)
            setattr(out, f.name, v if v is not None else getattr(lower, f.name))
        return out

    def needs_llm(self) -> bool:
        """True when all provenance fields are empty — LLM fallback warranted."""
        return (
            self.author is None
            and self.organization is None
            and self.publisher is None
            and self.copyright is None
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def normalize_source(
    path: Path,
    canonical_text: str,
    *,
    use_llm: bool = True,
) -> SourceNormalization:
    """Return structured provenance for a source document.

    Applies Tier 1 → Tier 2 → merge → optional Tier 3.
    Each tier only fills gaps; it never overwrites higher-tier values.

    Parameters
    ----------
    path:
        Filesystem path to the original document (needed for PDF metadata).
    canonical_text:
        Full extracted text (already computed by the builder).
    use_llm:
        If True and Tier 1+2 leave all provenance empty, make a haiku call.
    """
    t1 = _extract_embedded_metadata(path)
    t2 = _extract_text_metadata(canonical_text)
    merged = t1.merge(t2)

    if use_llm and merged.needs_llm():
        LOGGER.debug("normalizer: P1+P2 insufficient for %s — invoking LLM", path.name)
        t3 = _extract_via_llm(_first_pages(canonical_text, max_pages=2))
        merged = merged.merge(t3)

    LOGGER.debug(
        "normalizer: %s → org=%r  version=%r  doc_num=%r  pub_date=%r",
        path.name, merged.organization, merged.document_version,
        merged.document_number, merged.publication_date,
    )
    return merged


# ---------------------------------------------------------------------------
# Tier 1 — embedded metadata
# ---------------------------------------------------------------------------


def _extract_embedded_metadata(path: Path) -> SourceNormalization:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _pdf_metadata(path)
    if suffix == ".docx":
        return _docx_metadata(path)
    return SourceNormalization()


def _pdf_metadata(path: Path) -> SourceNormalization:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        info = reader.metadata or {}
        page_count = len(reader.pages)

        raw_title = _clean(info.get("/Title"))
        title = raw_title if raw_title and not _INTERNAL_TITLE_RE.match(raw_title) else None

        raw_author = _clean(info.get("/Author"))
        author = raw_author if raw_author and not _is_skip_author(raw_author) else None

        # /Subject is often used for a subtitle in technical documents
        subtitle = _clean(info.get("/Subject")) or None

        # /Creator is usually the generating application — skip as publisher
        pub_date = _parse_pdf_date(info.get("/CreationDate") or info.get("/ModDate"))

        return SourceNormalization(
            title=title,
            subtitle=subtitle,
            author=author,
            publication_date=pub_date,
            page_count=page_count,
        )
    except Exception as exc:
        LOGGER.debug("normalizer: PDF metadata extraction failed for %s — %s", path.name, exc)
        return SourceNormalization()


def _docx_metadata(path: Path) -> SourceNormalization:
    try:
        import docx
        doc = docx.Document(str(path))
        props = doc.core_properties
        title = _clean(getattr(props, "title", None)) or None
        author = _clean(getattr(props, "author", None)) or None
        if author and _is_skip_author(author):
            author = None
        pub_date = None
        created = getattr(props, "created", None)
        if created and hasattr(created, "date"):
            pub_date = created.date()
        return SourceNormalization(title=title, author=author, publication_date=pub_date)
    except Exception as exc:
        LOGGER.debug("normalizer: DOCX metadata extraction failed for %s — %s", path.name, exc)
        return SourceNormalization()


# ---------------------------------------------------------------------------
# Tier 2 — deterministic text parsing
# ---------------------------------------------------------------------------


def _extract_text_metadata(canonical_text: str) -> SourceNormalization:
    # Search the first 3 pages for most patterns; use full text only for doc number
    search_text = _first_pages(canonical_text, max_pages=3)
    full_early = canonical_text[:800]  # doc numbers often repeat in page headers

    norm = SourceNormalization()

    # Document version / revision
    m = _REVISION_RE.search(search_text) or _REVISION_RE.search(full_early)
    if m:
        norm.document_version = f"Revision {m.group(1).upper()}"
    elif (m := _VERSION_RE.search(search_text)):
        norm.document_version = f"Version {m.group(1)}"

    # Document number
    for pat in _DOC_NUM_RES:
        m = pat.search(full_early)
        if m:
            norm.document_number = m.group(0)
            break

    # Copyright → copyright text + organization
    m = _COPYRIGHT_RE.search(search_text)
    if m:
        norm.copyright = _clean(m.group(0))
        org = _org_from_copyright(norm.copyright)
        if org:
            norm.organization = org
            norm.publisher = org

    # Publication date from textual month-year (only if no better source)
    m = _MONTH_YEAR_RE.search(search_text)
    if m:
        month_str = m.group(1).lower()
        year = int(m.group(2))
        month = _MONTH_MAP.get(month_str, 1)
        try:
            norm.publication_date = date(year, month, 1)
        except ValueError:
            pass

    return norm


# ---------------------------------------------------------------------------
# Tier 3 — LLM fallback (haiku)
# ---------------------------------------------------------------------------


def _extract_via_llm(first_page_text: str) -> SourceNormalization:
    try:
        import anthropic as _anthropic

        client = _anthropic.Anthropic()
        fields_list = ", ".join(_LLM_FIELDS)
        prompt = (
            f"Extract document metadata. Return ONLY a JSON object with these keys "
            f"({fields_list}). Use null for any field not clearly present. "
            f"publication_date must be YYYY-MM-DD or null.\n\n"
            f"Document text:\n{first_page_text[:2000]}"
        )
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text
        m = re.search(r"\{[\s\S]*?\}", raw)
        if not m:
            return SourceNormalization()
        data = json.loads(m.group())

        pub_date: date | None = None
        if data.get("publication_date"):
            from datetime import datetime
            try:
                pub_date = datetime.strptime(str(data["publication_date"]), "%Y-%m-%d").date()
            except ValueError:
                pass

        return SourceNormalization(
            title=data.get("title") or None,
            subtitle=data.get("subtitle") or None,
            author=data.get("author") or None,
            organization=data.get("organization") or None,
            publisher=data.get("publisher") or None,
            publication_date=pub_date,
            document_version=data.get("document_version") or None,
            document_number=data.get("document_number") or None,
            copyright=data.get("copyright") or None,
        )
    except Exception as exc:
        LOGGER.debug("normalizer: LLM extraction failed — %s", exc)
        return SourceNormalization()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_pages(canonical_text: str, max_pages: int = 3) -> str:
    """Extract up to max_pages pages from canonical_text (uses [Page N] markers)."""
    marker = f"[Page {max_pages + 1}]"
    if marker in canonical_text:
        return canonical_text[:canonical_text.index(marker)]
    return canonical_text[:max_pages * 3000]


def _parse_pdf_date(raw: str | None) -> date | None:
    if not raw:
        return None
    m = _PDF_DATE_RE.search(str(raw))
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _is_skip_author(value: str) -> bool:
    v = value.lower()
    return any(skip in v for skip in _SKIP_AUTHOR_VALUES)


def _org_from_copyright(copyright_str: str) -> str | None:
    """Extract organization name from a copyright string.

    '© 2025 GE Vernova Hitachi Nuclear Energy Americas LLC. All rights reserved.'
    → 'GE Vernova Hitachi Nuclear Energy Americas LLC'
    """
    # Strip leading copyright marker + year(s)
    stripped = re.sub(
        r"^(?:©|copyright\s*©?)\s*(?:\d{4}[-–]\d{4}|\d{4})\s*",
        "",
        copyright_str,
        flags=re.IGNORECASE,
    ).strip()
    # Strip trailing boilerplate — stop at the period after org name
    stripped = re.sub(r"\.\s*(?:all rights reserved|used under).*$", "", stripped, flags=re.IGNORECASE)
    # Remove trailing punctuation
    stripped = stripped.rstrip(".,;").strip()
    return stripped if stripped and len(stripped) > 3 else None


def _clean(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = " ".join(str(value).split()).strip()
    return cleaned if cleaned else None
