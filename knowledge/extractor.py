"""Evidence extraction from Source text for the Knowledge Builder.

Wraps the existing ClaudeClient extraction infrastructure.
Produces Evidence objects conforming to the frozen J8.0 ontology.

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
    """Translate a research_agent EvidenceItem to a KB Evidence record."""
    category = getattr(item, "category", "")
    statement = getattr(item, "claim", "")
    evidence_type = _classify_evidence_type(statement, category)
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
