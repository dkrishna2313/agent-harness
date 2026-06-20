"""EvidenceAgent – runs the research engine and organizes evidence (J5.2)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)

# Coverage thresholds (evidence items per subquestion)
_STRONG = 4
_MODERATE = 2
_WEAK = 1


def _coverage_level(count: int) -> str:
    if count >= _STRONG:
        return "STRONG"
    if count >= _MODERATE:
        return "MODERATE"
    if count >= _WEAK:
        return "WEAK"
    return "NONE"


def _tokens(text: str) -> set[str]:
    """Lower-cased word tokens from text, ignoring short stop words."""
    _STOPWORDS = {"the", "a", "an", "of", "in", "is", "are", "to", "for",
                  "and", "or", "what", "how", "why", "does", "do", "can",
                  "with", "that", "this", "it", "be", "by", "at", "as"}
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _overlap_score(text: str, reference: str) -> int:
    """Count shared tokens between *text* and *reference*."""
    return len(_tokens(text) & _tokens(reference))


def _map_evidence_to_subquestions(
    evidence_items: list[Any],
    subquestions: list[str],
) -> dict[str, list[str]]:
    """Map each evidence item to its best-matching subquestion by token overlap.

    An item is assigned to the subquestion with the highest overlap score.
    Items with zero overlap on all subquestions go into "_unmapped".
    """
    mapping: dict[str, list[str]] = {sq: [] for sq in subquestions}
    mapping["_unmapped"] = []

    for item in evidence_items:
        item_text = f"{getattr(item, 'claim', '')} {getattr(item, 'evidence_snippet', '')}"
        best_sq = None
        best_score = 0
        for sq in subquestions:
            score = _overlap_score(item_text, sq)
            if score > best_score:
                best_score = score
                best_sq = sq
        eid = getattr(item, "evidence_id", "") or ""
        if best_sq and best_score > 0:
            mapping[best_sq].append(eid)
        else:
            mapping["_unmapped"].append(eid)

    return mapping


# Category → investigation area keyword mapping
_CATEGORY_TO_AREA: dict[str, str] = {
    "power": "Power",
    "cooling": "Cooling",
    "networking": "Networking",
    "rack architecture": "Rack Architecture",
    "architecture": "Architecture",
    "operations": "Operations",
    "gpu": "GPU",
    "facility": "Facility",
    "resiliency": "Resiliency",
    "economics": "Economics",
    "construction": "Construction",
    "licensing": "Regulation",
    "reactor design": "SMR Technology",
    "grid integration": "Grid Integration",
    "fuel cycle": "Fuel Cycle",
    "deployment timeline": "Deployment Timeline",
    "safety": "Safety",
    "waste management": "Waste Management",
    "bwrx": "SMR Technology",
    "nuscale": "SMR Technology",
    "other": "Other",
}


def _map_evidence_to_areas(
    evidence_items: list[Any],
    investigation_areas: list[str],
) -> dict[str, list[str]]:
    """Map each evidence item to matching investigation areas.

    An item can appear in multiple areas (one-to-many).
    Matching uses the item's category and topics fields.
    """
    mapping: dict[str, list[str]] = {area: [] for area in investigation_areas}

    for item in evidence_items:
        eid = getattr(item, "evidence_id", "") or ""
        category = getattr(item, "category", "") or ""
        topics = getattr(item, "topics", []) or []

        # Signals: category name → canonical area + topics text
        signal_texts = [category] + list(topics)
        combined = " ".join(signal_texts).lower()

        for area in investigation_areas:
            area_tokens = _tokens(area)
            if area_tokens & _tokens(combined):
                mapping[area].append(eid)
            # Also check via _CATEGORY_TO_AREA lookup
            elif _CATEGORY_TO_AREA.get(category) == area:
                mapping[area].append(eid)

    return mapping


def _compute_coverage(
    evidence_by_subquestion: dict[str, list[str]],
    subquestions: list[str],
) -> dict[str, dict[str, Any]]:
    """Compute coverage level per subquestion."""
    return {
        sq: {
            "evidence_count": len(evidence_by_subquestion.get(sq, [])),
            "coverage": _coverage_level(len(evidence_by_subquestion.get(sq, []))),
        }
        for sq in subquestions
    }


def _build_evidence_summary(
    evidence_items: list[Any],
    evidence_by_subquestion: dict[str, list[str]],
    evidence_by_area: dict[str, list[str]],
    subquestions: list[str],
) -> dict[str, Any]:
    coverage = _compute_coverage(evidence_by_subquestion, subquestions)
    covered = sum(1 for sq in subquestions if coverage[sq]["coverage"] != "NONE")
    return {
        "total_evidence_items": len(evidence_items),
        "subquestions_with_evidence": covered,
        "subquestions_without_evidence": len(subquestions) - covered,
        "investigation_areas_with_evidence": sum(
            1 for items in evidence_by_area.values() if items
        ),
        "coverage_distribution": {
            level: sum(1 for sq in subquestions if coverage[sq]["coverage"] == level)
            for level in ("STRONG", "MODERATE", "WEAK", "NONE")
        },
    }


class EvidenceAgent(FunctionalAgent):
    """Runs the research engine and organizes evidence around the research plan (J5.2).

    Responsibilities:
    - Run the extraction pipeline (unchanged from J5.0b)
    - Map evidence to subquestions from PlannerAgent
    - Map evidence to investigation areas
    - Compute per-subquestion coverage levels
    - Build evidence summary
    - Write all structured evidence data into context and Research Object
    """

    def __init__(
        self,
        *,
        sources_dir: str | Path = "sources",
        client: Any = None,
        top_evidence: int = 50,
        top_chunks: int = 20,
        domain_profile: Any = None,
    ) -> None:
        self._sources_dir = Path(sources_dir)
        self._client = client
        self._top_evidence = top_evidence
        self._top_chunks = top_chunks
        self._domain_profile = domain_profile

    def _execute(self, context: AgentContext) -> AgentContext:
        from research_agent.log import PROGRESS
        from research_agent.agent import DcPowerAgent
        from research_agent.loaders import load_sources

        # --- 1. Run extraction pipeline (unchanged) ---
        collection = load_sources(self._sources_dir)
        if collection.errors:
            for err in collection.errors:
                LOGGER.warning("Source load error: %s — %s", err.path.name, err.message)

        agent = DcPowerAgent(
            client=self._client,
            top_evidence=self._top_evidence,
            top_chunks=self._top_chunks,
            profile=self._domain_profile,
        )
        memo = agent.analyze(context.question, collection.documents)
        evidence_items = memo.source_notes or memo.evidence

        LOGGER.log(PROGRESS, "[EvidenceAgent] extracted %d items", len(evidence_items))

        # --- 2. Read planner outputs ---
        subquestions: list[str] = context.plan.get("subquestions", [])
        investigation_areas: list[str] = context.plan.get("investigation_areas", [])

        # --- 3. Map evidence to subquestions and investigation areas ---
        evidence_by_subquestion: dict[str, list[str]] = {}
        evidence_by_area: dict[str, list[str]] = {}

        if subquestions:
            evidence_by_subquestion = _map_evidence_to_subquestions(
                evidence_items, subquestions
            )
        if investigation_areas:
            evidence_by_area = _map_evidence_to_areas(
                evidence_items, investigation_areas
            )

        # --- 4. Coverage and summary ---
        coverage_by_subquestion = _compute_coverage(evidence_by_subquestion, subquestions)
        evidence_summary = _build_evidence_summary(
            evidence_items, evidence_by_subquestion, evidence_by_area, subquestions
        )

        covered = evidence_summary["subquestions_with_evidence"]
        uncovered = evidence_summary["subquestions_without_evidence"]
        mapped_areas = evidence_summary["investigation_areas_with_evidence"]

        LOGGER.log(
            PROGRESS,
            "[EvidenceAgent] mapped: subquestions=%d/%d covered, areas=%d/%d covered",
            covered,
            len(subquestions),
            mapped_areas,
            len(investigation_areas),
        )

        # --- 5. Write structured evidence context ---
        context.evidence_notes = [
            {
                "evidence_items": [
                    {
                        "evidence_id": getattr(e, "evidence_id", ""),
                        "claim": getattr(e, "claim", ""),
                        "category": getattr(e, "category", ""),
                        "topics": getattr(e, "topics", []),
                        "relevance_score": getattr(e, "relevance_score", 0),
                        "source_document": getattr(e, "source_document", ""),
                    }
                    for e in evidence_items
                ],
                "evidence_by_subquestion": evidence_by_subquestion,
                "evidence_by_area": evidence_by_area,
                "coverage_by_subquestion": coverage_by_subquestion,
                "evidence_summary": evidence_summary,
            }
        ]

        # --- 6. Update Research Object (J5.2.6) ---
        if context.research_object:
            ro = context.research_object
            ro["evidence_by_subquestion"] = {
                sq: ids for sq, ids in evidence_by_subquestion.items() if sq != "_unmapped"
            }
            ro["evidence_by_area"] = evidence_by_area
            ro["coverage_by_subquestion"] = coverage_by_subquestion
            ro["evidence_summary"] = evidence_summary

        # --- 7. Agent history (J5.2.8) ---
        self._record(
            context,
            status="success",
            summary=(
                f"Retrieved {len(evidence_items)} evidence items, confirmed "
                f"{len(memo.confirmed_facts or [])} facts. "
                f"Mapped {covered}/{len(subquestions)} subquestions, "
                f"{mapped_areas}/{len(investigation_areas)} investigation areas."
            ),
            evidence_count=len(evidence_items),
            confirmed_facts=len(memo.confirmed_facts or []),
            mapped_subquestions=covered,
            mapped_areas=mapped_areas,
            uncovered_subquestions=uncovered,
        )

        # Stash memo and documents for downstream agents
        context.trace["_memo"] = memo
        context.trace["_documents"] = collection.documents
        return context
