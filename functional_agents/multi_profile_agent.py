"""MultiProfileAgent – prove multi-profile execution is genuine (J5.6a).

Runs after RecommendationAgent, before ScenarioAgent.  Reads the evidence
attribution that EvidenceAgent already computed (source_profile on every
evidence item) and propagates it upward:

  evidence items     → source_profile (already set by EvidenceAgent)
  findings (hyps)    → contributing_profiles  (derived from supporting_evidence)
  recommendations    → contributing_profiles  (derived from supporting_evidence
                                               + supported_by_hypotheses)
  profile_influence  → {profile: {evidence, findings, recommendations}}
  missing diagnostics→ [{profile, status, reason}]

Public API
----------
build_evidence_profile_map(context)          – {evidence_id: profile}
attribute_findings(hypotheses, ev_map)       – findings with contributing_profiles
attribute_recommendations(recs, hyp_map, ev_map) – recs with contributing_profiles
compute_profile_influence(profiles, ev_items, findings, recs) – influence summary
diagnose_missing_profiles(profiles, ev_coverage) – diagnostic list
build_multi_profile_analysis(context)        – full analysis dict
MultiProfileAgent                            – FunctionalAgent subclass
"""

from __future__ import annotations

import logging
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evidence profile map
# ---------------------------------------------------------------------------

def build_evidence_profile_map(context: AgentContext) -> dict[str, str]:
    """Return {evidence_id: source_profile} from context.evidence_notes.

    Falls back to ``context.execution_profile`` when an item has no
    ``source_profile`` / ``profile`` field.
    """
    fallback = context.execution_profile or (context.profiles[0] if context.profiles else "unknown")
    ev_map: dict[str, str] = {}

    for note in context.evidence_notes:
        for item in note.get("evidence_items", []):
            eid = item.get("evidence_id", "")
            if not eid:
                continue
            profile = (
                item.get("source_profile")
                or item.get("profile")
                or item.get("originating_profile")
                or fallback
            )
            ev_map[eid] = profile

    return ev_map


# ---------------------------------------------------------------------------
# Finding attribution
# ---------------------------------------------------------------------------

def attribute_findings(
    hypotheses: list[dict[str, Any]],
    evidence_profile_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Return hypotheses enriched with ``contributing_profiles``.

    For each hypothesis the contributing profiles are the union of
    source_profiles for all evidence IDs in ``supporting_evidence``.
    If ``supporting_evidence`` is absent or empty, ``contributing_profiles``
    is an empty list (no manufactured attribution).
    """
    enriched: list[dict[str, Any]] = []
    for hyp in hypotheses:
        ev_ids: list[str] = hyp.get("supporting_evidence", [])
        profiles: list[str] = sorted({
            evidence_profile_map[eid]
            for eid in ev_ids
            if eid in evidence_profile_map
        })
        enriched.append({**hyp, "contributing_profiles": profiles})
    return enriched


# ---------------------------------------------------------------------------
# Recommendation attribution
# ---------------------------------------------------------------------------

def attribute_recommendations(
    recommendations: list[dict[str, Any]],
    hypothesis_profile_map: dict[str, list[str]],
    evidence_profile_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Return recommendations enriched with ``contributing_profiles``.

    Sources of profile attribution (unioned):
    - ``supporting_evidence`` IDs → evidence_profile_map
    - ``supported_by_hypotheses`` IDs → hypothesis_profile_map
    """
    enriched: list[dict[str, Any]] = []
    for rec in recommendations:
        ev_ids: list[str] = rec.get("supporting_evidence", [])
        hyp_ids: list[str] = rec.get("supported_by_hypotheses", [])

        from_evidence: set[str] = {
            evidence_profile_map[eid]
            for eid in ev_ids
            if eid in evidence_profile_map
        }
        from_hypotheses: set[str] = {
            p
            for hid in hyp_ids
            for p in hypothesis_profile_map.get(hid, [])
        }
        profiles: list[str] = sorted(from_evidence | from_hypotheses)
        enriched.append({**rec, "contributing_profiles": profiles})
    return enriched


# ---------------------------------------------------------------------------
# Profile influence summary
# ---------------------------------------------------------------------------

def compute_profile_influence(
    profiles: list[str],
    evidence_items: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Return ``{profile: {evidence, findings, recommendations}}`` counts."""
    influence: dict[str, dict[str, int]] = {
        p: {"evidence": 0, "findings": 0, "recommendations": 0}
        for p in profiles
    }

    for item in evidence_items:
        p = (
            item.get("source_profile")
            or item.get("profile")
            or item.get("originating_profile")
        )
        if p and p in influence:
            influence[p]["evidence"] += 1

    for f in findings:
        for p in f.get("contributing_profiles", []):
            if p in influence:
                influence[p]["findings"] += 1

    for r in recommendations:
        for p in r.get("contributing_profiles", []):
            if p in influence:
                influence[p]["recommendations"] += 1

    return influence


# ---------------------------------------------------------------------------
# Missing profile diagnostics
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Profile balance, synthesis validation, recommendation audit (J6.8b)
# ---------------------------------------------------------------------------

def _compute_profile_balance(
    recommendations: list[dict[str, Any]],
    profiles: list[str],
) -> dict[str, float]:
    """Return fraction of recommendation profile-touches per profile.

    Each recommendation may contribute to multiple profiles (once per profile
    in ``contributing_profiles``).  The result sums to 1.0 over all profiles.
    If no profile touches are recorded, equal weight is returned.
    """
    counts: dict[str, int] = {p: 0 for p in profiles}
    total = 0
    for r in recommendations:
        for p in r.get("contributing_profiles", []):
            if p in counts:
                counts[p] += 1
                total += 1
    if total == 0:
        n = len(profiles)
        return {p: round(1.0 / n, 3) if n else 0.0 for p in profiles}
    return {p: round(counts[p] / total, 3) for p in profiles}


def _compute_synthesis_validation(
    profiles_requested: list[str],
    profiles_contributing: list[str],
    attributed_findings: list[dict[str, Any]],
    attributed_recs: list[dict[str, Any]],
) -> dict[str, int]:
    """Return synthesis coverage counts for all four required fields."""
    profiles_in_findings = {
        p
        for f in attributed_findings
        for p in f.get("contributing_profiles", [])
    }
    profiles_in_recs = {
        p
        for r in attributed_recs
        for p in r.get("contributing_profiles", [])
    }
    return {
        "profiles_requested":                  len(profiles_requested),
        "profiles_contributing":               len(profiles_contributing),
        "profiles_represented_in_findings":    len(profiles_in_findings),
        "profiles_represented_in_recommendations": len(profiles_in_recs),
    }


def _build_recommendation_profile_audit(
    attributed_recs: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Return {rec_id: [profiles]} for every recommendation."""
    audit: dict[str, list[str]] = {}
    for i, r in enumerate(attributed_recs):
        rid = r.get("id") or r.get("recommendation_id") or f"R{i + 1}"
        audit[str(rid)] = list(r.get("contributing_profiles", []))
    return audit


def diagnose_missing_profiles(
    profiles_requested: list[str],
    profiles_contributing: list[str],
    profile_coverage: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    """Return a diagnostic entry for each non-contributing profile."""
    contributing_set = set(profiles_contributing)
    diagnostics: list[dict[str, str]] = []

    for p in profiles_requested:
        if p in contributing_set:
            continue
        cov = profile_coverage.get(p, {})
        count = cov.get("evidence_count", 0)
        if count == 0:
            reason = "no evidence retrieved"
        else:
            reason = "evidence retrieved but keyword overlap insufficient for attribution"
        diagnostics.append({
            "profile": p,
            "status": "missing",
            "reason": reason,
        })

    return diagnostics


# ---------------------------------------------------------------------------
# Profile coverage summary from evidence notes
# ---------------------------------------------------------------------------

def _extract_coverage_from_notes(
    context: AgentContext,
) -> tuple[list[str], list[str], list[str], dict[str, Any], list[dict[str, Any]]]:
    """Pull profile tracking data out of evidence_notes.

    Returns (profiles_requested, profiles_contributing, profiles_missing,
             profile_coverage_by_profile, all_evidence_items).
    """
    profiles_requested = list(context.profiles)
    profiles_contributing: list[str] = []
    profiles_missing: list[str] = []
    profile_coverage: dict[str, Any] = {}
    all_items: list[dict[str, Any]] = []

    for note in context.evidence_notes:
        if note.get("profiles_contributing"):
            profiles_contributing = note["profiles_contributing"]
        if note.get("profiles_missing"):
            profiles_missing = note["profiles_missing"]
        if note.get("profile_coverage_by_profile"):
            profile_coverage = note["profile_coverage_by_profile"]
        all_items.extend(note.get("evidence_items", []))

    # Derive from profile_coverage if not set directly
    if not profiles_contributing and profile_coverage:
        profiles_contributing = [
            p for p, cov in profile_coverage.items()
            if cov.get("evidence_count", 0) > 0
        ]
    if not profiles_missing:
        contributing_set = set(profiles_contributing)
        profiles_missing = [p for p in profiles_requested if p not in contributing_set]

    return (
        profiles_requested,
        profiles_contributing,
        profiles_missing,
        profile_coverage,
        all_items,
    )


# ---------------------------------------------------------------------------
# Full analysis builder
# ---------------------------------------------------------------------------

def build_multi_profile_analysis(context: AgentContext) -> dict[str, Any]:
    """Build the complete multi-profile analysis dict for this context."""
    (
        profiles_requested,
        profiles_contributing,
        profiles_missing,
        profile_coverage,
        all_evidence_items,
    ) = _extract_coverage_from_notes(context)

    # Evidence → profile map
    ev_map = build_evidence_profile_map(context)

    # Per-profile evidence counts
    ev_counts: dict[str, int] = {p: 0 for p in profiles_requested}
    for eid, p in ev_map.items():
        if p in ev_counts:
            ev_counts[p] += 1

    profile_coverage_metrics: dict[str, dict[str, int]] = {
        p: {"evidence_count": ev_counts.get(p, 0)}
        for p in profiles_requested
    }

    # Finding attribution
    hypotheses = context.hypotheses or []
    attributed_findings = attribute_findings(hypotheses, ev_map)
    hyp_profile_map: dict[str, list[str]] = {
        f.get("hypothesis_id", f.get("id", "")): f.get("contributing_profiles", [])
        for f in attributed_findings
    }

    # Recommendation attribution
    recommendations = context.recommendations or []
    attributed_recs = attribute_recommendations(recommendations, hyp_profile_map, ev_map)

    # Profile influence summary
    influence = compute_profile_influence(
        profiles_requested, all_evidence_items, attributed_findings, attributed_recs
    )

    # Missing profile diagnostics
    diagnostics = diagnose_missing_profiles(
        profiles_requested, profiles_contributing, profile_coverage
    )

    # Coverage status
    n_req = len(profiles_requested)
    n_con = len(profiles_contributing)
    coverage_status = (
        "sufficient" if n_req == 0 or n_con == n_req
        else ("partial" if n_con > 0 else "insufficient")
    )

    # J6.8b additions
    profile_balance = _compute_profile_balance(attributed_recs, profiles_requested)
    synthesis_validation = _compute_synthesis_validation(
        profiles_requested, profiles_contributing, attributed_findings, attributed_recs
    )
    recommendation_profile_audit = _build_recommendation_profile_audit(attributed_recs)

    return {
        "profiles_requested":   profiles_requested,
        "profiles_contributing": profiles_contributing,
        "profiles_missing":     profiles_missing,
        "coverage_status":      coverage_status,
        "profile_coverage":     profile_coverage_metrics,
        "profile_influence":    influence,
        "missing_profile_diagnostics": diagnostics,
        "attributed_findings":  attributed_findings,
        "attributed_recommendations": attributed_recs,
        "evidence_profile_map_size": len(ev_map),
        # J6.8b
        "profile_balance":               profile_balance,
        "synthesis_validation":          synthesis_validation,
        "recommendation_profile_audit":  recommendation_profile_audit,
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MultiProfileAgent(FunctionalAgent):
    """Prove multi-profile execution is genuine by propagating profile
    attribution from evidence through findings to recommendations (J5.6a).

    Reads:
        context.profiles
        context.evidence_notes   (source_profile on every item)
        context.hypotheses
        context.recommendations

    Writes:
        context.multi_profile_analysis
        context.recommendations  (adds contributing_profiles to each rec)
        context.hypotheses       (adds contributing_profiles to each hyp)
        context.qa["multi_profile_validation"]
        context.research_object["multi_profile_analysis"]
        context.trace["_multi_profile"]
    """

    def _execute(self, context: AgentContext) -> AgentContext:
        from research_agent.log import PROGRESS

        analysis = build_multi_profile_analysis(context)

        # Write enriched findings and recommendations back to context
        context.hypotheses = analysis["attributed_findings"]
        context.recommendations = analysis["attributed_recommendations"]
        context.multi_profile_analysis = analysis

        # QA validation block
        context.qa["multi_profile_validation"] = {
            "requested_profiles":    len(analysis["profiles_requested"]),
            "contributing_profiles": len(analysis["profiles_contributing"]),
            "coverage_status":       analysis["coverage_status"],
            "synthesis_validation":  analysis["synthesis_validation"],
        }

        # Research Object
        ro = context.research_object
        if ro is not None:
            ro["multi_profile_analysis"] = {
                "profiles_requested":   analysis["profiles_requested"],
                "profiles_contributing": analysis["profiles_contributing"],
                "profiles_missing":     analysis["profiles_missing"],
                "coverage_status":      analysis["coverage_status"],
                "profile_coverage":     analysis["profile_coverage"],
                "profile_influence":    analysis["profile_influence"],
                "missing_profile_diagnostics": analysis["missing_profile_diagnostics"],
                # J6.8b
                "profile_balance":              analysis["profile_balance"],
                "synthesis_validation":         analysis["synthesis_validation"],
                "recommendation_profile_audit": analysis["recommendation_profile_audit"],
            }

        # Trace
        context.trace["_multi_profile"] = {
            "multi_profile_validation": {
                "profiles_requested":   analysis["profiles_requested"],
                "profiles_contributing": analysis["profiles_contributing"],
                "profiles_missing":     analysis["profiles_missing"],
                "coverage_status":      analysis["coverage_status"],
                "profile_coverage":     analysis["profile_coverage"],
                "profile_influence":    analysis["profile_influence"],
                "missing_profile_diagnostics": analysis["missing_profile_diagnostics"],
                # J6.8b
                "profile_balance":              analysis["profile_balance"],
                "synthesis_validation":         analysis["synthesis_validation"],
                "recommendation_profile_audit": analysis["recommendation_profile_audit"],
            }
        }

        summary = (
            f"requested={len(analysis['profiles_requested'])} "
            f"contributing={len(analysis['profiles_contributing'])} "
            f"missing={len(analysis['profiles_missing'])} "
            f"status={analysis['coverage_status']}"
        )
        LOGGER.log(PROGRESS, "[MultiProfileAgent] %s", summary)
        self._record(context, status="success", summary=summary)
        return context
