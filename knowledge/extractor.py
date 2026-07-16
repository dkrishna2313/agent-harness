"""Evidence extraction from Source text for the Knowledge Builder.

Wraps the existing ClaudeClient extraction infrastructure.
Produces Evidence objects conforming to the J8.0 ontology, v2 schema (PH5.5b).

Design:
- Uses an adapter to translate between the existing EvidenceItem schema
  (used by the J7 pipeline) and the new Evidence schema (Knowledge Base).
- Chunk is an implementation detail of this extractor; it never enters the KB.
- The extraction question is strategic: only claims with reuse value for
  future decision-makers are extracted (J8.2 onwards).
- EvidenceType classification is post-hoc: the LLM returns claims;
  the extractor assigns type via keyword heuristics and category hints.
- ADMINISTRATIVE and PROVENANCE evidence is persisted with
  retrieval_enabled=False, preserving the audit trail without polluting
  Planner retrieval.

PH5.5b — Provenance Population:
- excerpt, chunk_id, topics, evidence_confidence, is_quantitative are
  sourced from existing EvidenceItem fields (no new inference).
- page_number, char_offset_start, char_offset_end are derived from the
  [Page N] markers in canonical_text + substring search for the excerpt.
- temporal_reference is extracted via a deterministic year/quarter regex.
- section_heading is left None — no reliable signal without new inference.
- All provenance fields default to None / [] when not determinable;
  nothing is fabricated.
"""

from __future__ import annotations

import logging
import re as _re
from typing import TYPE_CHECKING

from .models import Evidence, EvidenceType, KnowledgeMetadata

if TYPE_CHECKING:
    from .models import Source

LOGGER = logging.getLogger(__name__)

_KB_EXTRACTION_QUESTION = (
    "Extract atomic claims that would be valuable for answering future strategic research "
    "questions from executives, investors, or policy makers.\n\n"
    "Include:\n"
    "- Technical specifications (power output, fuel type, cycle length, cooling system)\n"
    "- Performance parameters (capacity factor, construction time, operating life)\n"
    "- Cost estimates (capital cost, LCOE, operating cost per MWh)\n"
    "- Deployment timelines, licensing milestones, and commercialisation schedules\n"
    "- Regulatory requirements and current licensing status\n"
    "- Risk factors and deployment challenges\n"
    "- Market projections, demand forecasts, and government commitments\n"
    "- Policy positions and legislation affecting deployment\n"
    "- Competitive comparisons and technology differentiators\n"
    "- Operational requirements (grid integration, fuel supply, workforce, water)\n\n"
    "Exclude:\n"
    "- Document revision numbers, document identifiers, and report numbers\n"
    "- Copyright notices, trademark statements, and boilerplate disclaimers\n"
    "- Table of contents entries and acknowledgements\n"
    "- Administrative metadata and document formatting information\n\n"
    "Every claim extracted must justify its presence by answering a question "
    "a decision-maker would plausibly ask about this technology."
)

_PROMPT_VERSION = "kb-v2.0"

# ---------------------------------------------------------------------------
# Evidence type classification
# ---------------------------------------------------------------------------

_ADMIN_PATTERNS: tuple[str, ...] = (
    "document number",
    "doc number",
    "document id",
    "report number",
    "is revision",
    " rev.",
    "document is revision",
    "document is rev",
    "is copyrighted",
    "copyright",
    "all rights reserved",
    "is a trademark",
    "is a registered trademark",
    "trademark of",
    "trademark license",
    "used under trademark",
    "table of contents",
    "acknowledgement",
    "acknowledgment",
    "for internal use only",
    "proprietary information",
    "document is ",
    "this document is ",
    "document are ",
    "pages long",
    "page document",
    "isbn",
    "issn",
    "catalog number",
    "catalogue number",
)

_PROVENANCE_PATTERNS: tuple[str, ...] = (
    "authored by",
    "written by",
    "published by",
    "prepared by",
    "produced by",
    "prepared for",
    "submitted to",
    "this study was",
    "this report was",
    "this paper was",
)

# Numeric-with-unit heuristic for TECHNICAL classification
_TECHNICAL_UNIT_RE = _re.compile(
    r"\d+(?:\.\d+)?\s*"
    r"(?:mw[eth]?|kw[eth]?|gw[eth]?|°[cf]|psi|bar|mpa|kpa|kg|lb|"
    r"t\b|mt\b|m³|m3|mwh|kwh|gwh|%|hz|rpm|mm\b|cm\b|km\b|m\b|"
    r"years?\b|months?\b|days?\b|hours?\b|usd|\\$)",
    _re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# PH5.5b — provenance derivation helpers
# ---------------------------------------------------------------------------

# Confidence mapping from EvidenceItem.confidence (lowercase) to v2 Literal
_CONFIDENCE_UPCASE: dict[str, str] = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}

# Temporal reference extraction — year (2000–2099) and quarter patterns
_YEAR_RE = _re.compile(r"\b(20\d{2}|19\d{2})\b")
_QUARTER_RE = _re.compile(r"\bQ[1-4]\s+(?:20\d{2}|19\d{2})\b", _re.IGNORECASE)

# [Page N] marker used by builder._extract_pdf_text()
_PAGE_MARKER_RE = _re.compile(r"\[Page (\d+)\]")

# Maximum excerpt length (canonical model: ≤600 chars)
_EXCERPT_MAX = 600


def _extract_temporal_reference(text: str) -> str | None:
    """Return the most specific temporal reference found in *text*, or None.

    Prefers quarter-level precision (e.g. 'Q3 2028') over year-only (e.g. '2028').
    Uses existing text signals only — no new inference.
    """
    if not text:
        return None
    m = _QUARTER_RE.search(text)
    if m:
        return m.group(0)
    m = _YEAR_RE.search(text)
    if m:
        return m.group(0)
    return None


def _find_provenance_in_text(
    canonical_text: str,
    excerpt: str,
) -> tuple[int | None, int | None, int | None]:
    """Locate *excerpt* in *canonical_text* and return (page, start, end).

    Uses the [Page N] markers emitted by _extract_pdf_text() to derive the
    page number.  Searches with progressively shorter prefixes to tolerate
    minor LLM paraphrasing at snippet boundaries.

    Returns (None, None, None) when the excerpt cannot be located — never
    fabricates a position.
    """
    if not excerpt or not canonical_text:
        return None, None, None

    # Try progressively shorter search anchors to tolerate whitespace differences
    for length in (200, 100, 60):
        anchor = excerpt[:length].strip()
        if not anchor:
            continue
        idx = canonical_text.find(anchor)
        if idx != -1:
            # Determine char offsets using full excerpt length
            end = min(idx + len(excerpt), len(canonical_text))
            # Determine page: find the last [Page N] marker before idx
            text_before = canonical_text[:idx]
            page_matches = _PAGE_MARKER_RE.findall(text_before)
            page = int(page_matches[-1]) if page_matches else 1
            return page, idx, end

    return None, None, None


_RETRIEVAL_DEFAULTS: dict[str, dict] = {
    "STRATEGIC":      {"retrieval_enabled": True,  "retrieval_priority": 5, "strategic_value": 0.80},
    "TECHNICAL":      {"retrieval_enabled": True,  "retrieval_priority": 4, "strategic_value": 0.60},
    "PROVENANCE":     {"retrieval_enabled": False, "retrieval_priority": 2, "strategic_value": 0.20},
    "ADMINISTRATIVE": {"retrieval_enabled": False, "retrieval_priority": 1, "strategic_value": 0.05},
}


def _classify_evidence_type(statement: str, category: str) -> EvidenceType:
    """Classify an extracted claim into one of the four EvidenceType values.

    Ordering: ADMINISTRATIVE check first (hard reject), then PROVENANCE,
    then TECHNICAL (specs/units), then STRATEGIC as the default.
    """
    s = statement.lower()

    if any(p in s for p in _ADMIN_PATTERNS):
        return "ADMINISTRATIVE"

    if any(p in s for p in _PROVENANCE_PATTERNS):
        return "PROVENANCE"

    cat = category.lower()
    technical_category_hints = (
        "technical", "specification", "engineering", "performance",
        "design", "safety", "thermal", "nuclear", "reactor", "fuel",
    )
    if any(hint in cat for hint in technical_category_hints):
        return "TECHNICAL"

    if _TECHNICAL_UNIT_RE.search(statement):
        return "TECHNICAL"

    return "STRATEGIC"


# ---------------------------------------------------------------------------
# Public extraction entry point
# ---------------------------------------------------------------------------


def extract_evidence_from_source(
    source: "Source",
    extraction_run_id: str,
    client: object,
    *,
    existing_fingerprints: set[str] | None = None,
    profile_ids: list[str] | None = None,
) -> tuple[list[Evidence], list[KnowledgeMetadata], int]:
    """Extract Evidence objects from a Source using the ClaudeClient.

    Parameters
    ----------
    source:
        The Source to extract from.
    extraction_run_id:
        ID of the current ExtractionRun.
    client:
        A ClaudeClient (real or mock) that implements extract_evidence().
    existing_fingerprints:
        Set of statement_fingerprint values already in the KB for deduplication.
    profile_ids:
        Profile IDs to tag all produced evidence with.

    Returns
    -------
    (evidence_list, metadata_list, duplicates_merged)
    """
    from research_agent.schemas import SourceDocument

    if existing_fingerprints is None:
        existing_fingerprints = set()
    profile_ids = profile_ids or []

    # Adapt our Source to SourceDocument for the existing ClaudeClient
    source_doc = SourceDocument(
        path=source.uri,
        title=source.title,
        extension=f".{source.document_type.lower()}",
        text=source.canonical_text,
    )

    try:
        raw_items = client.extract_evidence(
            _KB_EXTRACTION_QUESTION,
            [source_doc],
        )
    except Exception as exc:
        LOGGER.error("extractor: evidence extraction failed for source %s — %s", source.source_id, exc)
        return [], [], 0

    evidence_list: list[Evidence] = []
    metadata_list: list[KnowledgeMetadata] = []
    duplicates_merged = 0

    for item in raw_items:
        ev = _adapt_evidence_item(item, source, extraction_run_id, profile_ids)

        if ev.statement_fingerprint in existing_fingerprints:
            duplicates_merged += 1
            LOGGER.debug("extractor: duplicate evidence merged (fingerprint=%s)", ev.statement_fingerprint)
            continue

        existing_fingerprints.add(ev.statement_fingerprint)
        meta = _build_metadata(ev, item)
        evidence_list.append(ev)
        metadata_list.append(meta)

    strategic_count = sum(1 for e in evidence_list if e.evidence_type in ("STRATEGIC", "TECHNICAL"))
    admin_count = len(evidence_list) - strategic_count
    LOGGER.info(
        "extractor: source=%s  extracted=%d  strategic/technical=%d  admin/provenance=%d  duplicates_merged=%d",
        source.source_id,
        len(evidence_list),
        strategic_count,
        admin_count,
        duplicates_merged,
    )
    return evidence_list, metadata_list, duplicates_merged


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def _adapt_evidence_item(
    item: object,
    source: "Source",
    extraction_run_id: str,
    profile_ids: list[str],
) -> Evidence:
    """Translate a research_agent EvidenceItem to a KB Evidence v2 record.

    v1 fields (statement, entity, category, etc.) are unchanged.
    v2 provenance fields are populated from existing EvidenceItem signals
    and the source's canonical_text — nothing is fabricated.
    """
    category = getattr(item, "category", "")
    statement = getattr(item, "claim", "")
    evidence_type = _classify_evidence_type(statement, category)

    # --- v2: excerpt (from evidence_snippet, capped at _EXCERPT_MAX chars) ---
    raw_snippet = getattr(item, "evidence_snippet", "") or ""
    excerpt: str | None = raw_snippet[:_EXCERPT_MAX] if raw_snippet.strip() else None

    # --- v2: chunk_id (from source_chunk_id when non-empty) ---
    raw_chunk_id = getattr(item, "source_chunk_id", "") or ""
    chunk_id: str | None = raw_chunk_id.strip() or None

    # --- v2: topics ---
    topics: list[str] = list(getattr(item, "topics", []) or [])

    # --- v2: evidence_confidence ---
    raw_confidence = (getattr(item, "confidence", "medium") or "medium").lower()
    evidence_confidence = _CONFIDENCE_UPCASE.get(raw_confidence) or None

    # --- v2: is_quantitative (quantitative_score >= 4 = HIGH numeric richness) ---
    quantitative_score = int(getattr(item, "quantitative_score", 3) or 3)
    is_quantitative: bool = quantitative_score >= 4

    # --- v2: passage location (page_number, char_offset_start/end) ---
    # Derived from [Page N] markers in canonical_text + excerpt search.
    # Left None when the excerpt cannot be located — never fabricated.
    page_number: int | None = None
    char_offset_start: int | None = None
    char_offset_end: int | None = None
    if excerpt and source.canonical_text:
        page_number, char_offset_start, char_offset_end = _find_provenance_in_text(
            source.canonical_text, excerpt
        )

    # --- v2: temporal_reference ---
    # Search both the excerpt and the statement for year / quarter patterns.
    search_text = f"{excerpt or ''} {statement}"
    temporal_reference = _extract_temporal_reference(search_text)

    return Evidence(
        statement=statement,
        evidence_type=evidence_type,
        supporting_source_ids=[source.source_id],
        profile_ids=list(profile_ids),
        extraction_run_id=extraction_run_id,
        entity=getattr(item, "entity", ""),
        entity_type=getattr(item, "entity_type", ""),
        scope=getattr(item, "scope", ""),
        category=category,
        # v2 provenance fields
        excerpt=excerpt,
        chunk_id=chunk_id,
        topics=topics,
        evidence_confidence=evidence_confidence,
        is_quantitative=is_quantitative,
        page_number=page_number,
        char_offset_start=char_offset_start,
        char_offset_end=char_offset_end,
        temporal_reference=temporal_reference,
        # section_heading: no reliable signal without new inference — left None
    )


def _build_metadata(ev: Evidence, item: object) -> KnowledgeMetadata:
    """Build KnowledgeMetadata from the raw EvidenceItem quality scores."""
    confidence_map = {"high": 0.85, "medium": 0.60, "low": 0.35}
    confidence_str = getattr(item, "confidence", "medium")
    confidence = confidence_map.get(confidence_str, 0.60)

    relevance = float(getattr(item, "relevance_score", 3))
    source_quality = float(getattr(item, "source_quality_score", 3))
    specificity = float(getattr(item, "specificity_score", 3))
    overall = round((relevance + source_quality + specificity) / 3.0, 2)

    credibility = "HIGH" if source_quality >= 4 else ("LOW" if source_quality <= 2 else "MEDIUM")

    retrieval = _RETRIEVAL_DEFAULTS[ev.evidence_type]

    return KnowledgeMetadata(
        evidence_id=ev.evidence_id,
        confidence=confidence,
        credibility=credibility,
        relevance_score=relevance,
        source_quality_score=source_quality,
        specificity_score=specificity,
        overall_score=overall,
        review_status="AUTO_REVIEWED",
        retrieval_enabled=retrieval["retrieval_enabled"],
        retrieval_priority=retrieval["retrieval_priority"],
        strategic_value=retrieval["strategic_value"],
    )
