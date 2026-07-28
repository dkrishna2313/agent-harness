"""PH12.2b — DiscriminationCalculator: post-hoc discrimination enrichment.

Computes discrimination scores for all content items across a set of TheoryContent
objects, then enriches each with:
  - relationship_classification on each lineage entry
  - distinctive_*/shared_* ID splits on TheoryContent
  - updated ContentCoverage.distinctive/shared_context/evidence_distinctive
  - updated ContentConfidence with discrimination-aware shares

Algorithm:
  1. Build item_id → frozenset[theory_ids] per content category
  2. discrimination_score = 1.0 - (shared_count / total_theories)
     - score == 0.0 → global_shared (present in ALL theories)
     - 0 < score < 1.0 → theory_subset
     - score == 1.0 → theory_unique (present in exactly ONE theory)
  3. Classify lineage entries:
     - explicit assignment + score > 0  → explicit_discriminating
     - explicit assignment + score == 0 → explicit_shared
     - posture assignment + score > 0   → posture_discriminating
     - posture assignment + score == 0  → semantic_discriminating
     - sensitivity                      → sensitivity
     - fallback / other                 → fallback
  4. Populate distinctive_*/shared_* lists and update coverage/confidence.

No LLM, no external services, no new option-mapping logic.
"""

from __future__ import annotations

from typing import Any

from .theory_content import (
    ContentLineageEntry,
    EvidenceLineageEntry,
    TheoryContent,
)


# Assignment types considered "explicit" for discrimination classification
_EXPLICIT_TYPES: frozenset[str] = frozenset({
    "option_link",
    "recommendation_link",
    "risk_link",
    "opportunity_link",
    "assumption_link",
})

# Assignment types considered "posture match"
_POSTURE_TYPES: frozenset[str] = frozenset({
    "posture_match",
    "semantic_inference",
})

# Content categories tracked for discrimination
_CONTENT_CATEGORIES: tuple[str, ...] = (
    "assumptions",
    "risks",
    "opportunities",
    "recommendations",
)


def _compute_discrimination_score(item_count: int, total_theories: int) -> float:
    """Return discrimination_score = 1.0 - (item_count / total_theories)."""
    if total_theories <= 0:
        return 0.0
    return round(1.0 - (item_count / total_theories), 6)


def _relationship_scope(item_count: int, total_theories: int) -> str:
    if item_count == total_theories:
        return "global_shared"
    if item_count == 1:
        return "theory_unique"
    return "theory_subset"


def _classify_lineage(
    assignment_type: str,
    discrimination_score: float,
) -> str:
    """Map assignment_type + discrimination_score to relationship_classification."""
    if assignment_type == "sensitivity":
        return "sensitivity"
    if assignment_type in _EXPLICIT_TYPES:
        return "explicit_discriminating" if discrimination_score > 0.0 else "explicit_shared"
    if assignment_type in _POSTURE_TYPES:
        return "posture_discriminating" if discrimination_score > 0.0 else "semantic_discriminating"
    return "fallback"


def _build_item_map(
    all_contents: list[TheoryContent],
    category: str,
    id_field: str,
) -> dict[str, set[str]]:
    """Build {item_id: set_of_theory_ids} for one content category."""
    item_map: dict[str, set[str]] = {}
    for tc in all_contents:
        ids: list[str] = getattr(tc, id_field, [])
        for item_id in ids:
            if item_id not in item_map:
                item_map[item_id] = set()
            item_map[item_id].add(tc.theory_id)
    return item_map


def _enrich_lineage_entries(
    entries: list[ContentLineageEntry],
    item_map: dict[str, set[str]],
    total_theories: int,
    all_theory_ids: list[str],
) -> tuple[list[ContentLineageEntry], int, int, int]:
    """
    Enrich lineage entries in place with discrimination fields.
    Returns (enriched_entries, discriminating_count, shared_count, posture_disc_count).
    """
    enriched = []
    discriminating_count = 0
    shared_count = 0
    posture_disc_count = 0

    for entry in entries:
        theory_ids_with_item = item_map.get(entry.source_id, set())
        item_count = len(theory_ids_with_item)
        disc_score = _compute_discrimination_score(item_count, total_theories)
        scope = _relationship_scope(item_count, total_theories)
        classification = _classify_lineage(entry.assignment_type, disc_score)
        shared_across = sorted(theory_ids_with_item)

        updated = entry.model_copy(update={
            "relationship_classification": classification,
            "discrimination_score": disc_score,
            "relationship_scope": scope,
            "shared_across_theory_ids": shared_across,
        })
        enriched.append(updated)

        if classification == "explicit_discriminating":
            discriminating_count += 1
        elif classification == "explicit_shared":
            shared_count += 1
        elif classification == "posture_discriminating":
            posture_disc_count += 1

    return enriched, discriminating_count, shared_count, posture_disc_count


def _enrich_evidence_entries(
    entries: list[EvidenceLineageEntry],
    evidence_map: dict[str, set[str]],
    total_theories: int,
) -> tuple[list[EvidenceLineageEntry], int]:
    """
    Enrich evidence lineage entries.
    Returns (enriched_entries, distinctive_evidence_count).
    """
    enriched = []
    distinctive_count = 0

    for entry in entries:
        theory_ids_with_item = evidence_map.get(entry.evidence_id, set())
        item_count = len(theory_ids_with_item)
        disc_score = _compute_discrimination_score(item_count, total_theories)
        scope = _relationship_scope(item_count, total_theories)
        classification = _classify_lineage(entry.assignment_type, disc_score)
        shared_across = sorted(theory_ids_with_item)

        updated = entry.model_copy(update={
            "relationship_classification": classification,
            "discrimination_score": disc_score,
            "relationship_scope": scope,
            "shared_across_theory_ids": shared_across,
        })
        enriched.append(updated)

        if disc_score > 0.0:
            distinctive_count += 1

    return enriched, distinctive_count


def enrich_with_discrimination(
    all_contents: list[TheoryContent],
    min_discrimination_score: float = 0.20,
) -> list[TheoryContent]:
    """Enrich a list of TheoryContent with discrimination scores and splits.

    Args:
        all_contents: All TheoryContent objects for the current run.
        min_discrimination_score: Items with discrimination_score < this threshold
            are considered shared for the distinctive/shared split. This parameter
            is reserved for future use — currently, the split uses score > 0.0.

    Returns:
        New list of TheoryContent with discrimination fields populated.
        Original objects are not mutated.
    """
    if not all_contents:
        return all_contents

    total_theories = len(all_contents)
    all_theory_ids = [tc.theory_id for tc in all_contents]

    # Build item → theory_ids maps per category
    category_maps: dict[str, dict[str, set[str]]] = {
        "assumptions": _build_item_map(all_contents, "assumptions", "assumption_ids"),
        "risks": _build_item_map(all_contents, "risks", "risk_ids"),
        "opportunities": _build_item_map(all_contents, "opportunities", "opportunity_ids"),
        "recommendations": _build_item_map(all_contents, "recommendations", "recommendation_ids"),
    }
    evidence_map = _build_item_map(all_contents, "evidence", "evidence_ids")

    enriched_contents: list[TheoryContent] = []

    for tc in all_contents:
        # --- Enrich content_lineage entries per category ---
        new_content_lineage: dict[str, list[ContentLineageEntry]] = {}
        total_discriminating = 0
        total_shared_explicit = 0
        total_posture_disc = 0

        for cat in _CONTENT_CATEGORIES:
            cat_entries = tc.content_lineage.get(cat, [])
            item_map = category_maps.get(cat, {})
            enriched_cat, d_count, s_count, pd_count = _enrich_lineage_entries(
                cat_entries, item_map, total_theories, all_theory_ids
            )
            new_content_lineage[cat] = enriched_cat
            total_discriminating += d_count
            total_shared_explicit += s_count
            total_posture_disc += pd_count

        # Carry forward non-enriched categories (success_conditions, evidence)
        for cat in tc.content_lineage:
            if cat not in new_content_lineage:
                new_content_lineage[cat] = tc.content_lineage[cat]

        # --- Enrich evidence_lineage ---
        new_evidence_lineage, distinctive_ev_count = _enrich_evidence_entries(
            tc.evidence_lineage, evidence_map, total_theories
        )

        # --- Compute distinctive/shared ID splits ---
        def _split_ids(ids: list[str], item_map: dict[str, set[str]]) -> tuple[list[str], list[str]]:
            distinctive = [i for i in ids if len(item_map.get(i, set())) < total_theories]
            shared = [i for i in ids if len(item_map.get(i, set())) == total_theories]
            return sorted(distinctive), sorted(shared)

        dist_assumptions, shared_assumptions = _split_ids(
            tc.assumption_ids, category_maps["assumptions"])
        dist_risks, shared_risks = _split_ids(
            tc.risk_ids, category_maps["risks"])
        dist_opps, shared_opps = _split_ids(
            tc.opportunity_ids, category_maps["opportunities"])
        dist_recs, shared_recs = _split_ids(
            tc.recommendation_ids, category_maps["recommendations"])
        dist_ev, shared_ev = _split_ids(tc.evidence_ids, evidence_map)

        # --- Update ContentCoverage discrimination fractions ---
        total_assigned = (
            len(tc.assumption_ids) + len(tc.risk_ids) +
            len(tc.opportunity_ids) + len(tc.recommendation_ids)
        )
        total_distinctive = len(dist_assumptions) + len(dist_risks) + len(dist_opps) + len(dist_recs)
        total_shared_ids = len(shared_assumptions) + len(shared_risks) + len(shared_opps) + len(shared_recs)
        total_evidence = len(tc.evidence_ids)

        new_coverage = tc.coverage.model_copy(update={
            "canonical": tc.coverage.overall,  # fraction of global pool assigned — same as overall
            "distinctive": round(total_distinctive / max(total_assigned, 1), 4),
            "shared_context": round(total_shared_ids / max(total_assigned, 1), 4),
            "evidence_distinctive": round(
                len(dist_ev) / max(total_evidence, 1), 4
            ),
        })

        # --- Update ContentConfidence with discrimination-aware shares ---
        new_confidence = ContentConfidence_update(
            tc.confidence,
            explicit_discriminating_count=total_discriminating,
            explicit_shared_count=total_shared_explicit,
            posture_discriminating_count=total_posture_disc,
            distinctive_evidence_count=distinctive_ev_count,
            total_evidence_count=total_evidence,
            total_content_count=(tc.coverage.explicit_count + tc.coverage.fallback_count),
        )

        # --- Build updated TheoryContent ---
        updated = tc.model_copy(update={
            "content_lineage": new_content_lineage,
            "evidence_lineage": new_evidence_lineage,
            "distinctive_assumption_ids": dist_assumptions,
            "shared_assumption_ids": shared_assumptions,
            "distinctive_risk_ids": dist_risks,
            "shared_risk_ids": shared_risks,
            "distinctive_opportunity_ids": dist_opps,
            "shared_opportunity_ids": shared_opps,
            "distinctive_recommendation_ids": dist_recs,
            "shared_recommendation_ids": shared_recs,
            "distinctive_evidence_ids": dist_ev,
            "shared_evidence_ids": shared_ev,
            "coverage": new_coverage,
            "confidence": new_confidence,
        })
        enriched_contents.append(updated)

    return enriched_contents


def ContentConfidence_update(
    confidence: Any,
    explicit_discriminating_count: int,
    explicit_shared_count: int,
    posture_discriminating_count: int,
    distinctive_evidence_count: int,
    total_evidence_count: int,
    total_content_count: int,
) -> Any:
    """Return a new ContentConfidence with discrimination shares set.

    Recalculates the confidence level using the PH12.2b matrix:
    High requires discriminating_share >= 0.40 AND mapping not None.
    """
    total = total_content_count
    if total <= 0:
        return confidence

    explicit_disc_share = round(explicit_discriminating_count / total, 4)
    explicit_sh_share   = round(explicit_shared_count / total, 4)
    posture_disc_share  = round(posture_discriminating_count / total, 4)
    dist_ev_share       = round(
        distinctive_evidence_count / total_evidence_count if total_evidence_count > 0 else 0.0, 4
    )

    # Recompute level only if discrimination data is non-trivial
    has_discrimination_data = (explicit_discriminating_count + explicit_shared_count) > 0
    discriminating_share = explicit_disc_share + posture_disc_share

    if has_discrimination_data:
        if (confidence.level == "High"
                and discriminating_share < 0.40):
            # Downgrade: High requires discriminating content
            new_level = "Medium"
        else:
            new_level = confidence.level
    else:
        new_level = confidence.level

    return confidence.model_copy(update={
        "level": new_level,
        "explicit_discriminating_share": explicit_disc_share,
        "explicit_shared_share": explicit_sh_share,
        "posture_discriminating_share": posture_disc_share,
        "distinctive_evidence_share": dist_ev_share,
    })
