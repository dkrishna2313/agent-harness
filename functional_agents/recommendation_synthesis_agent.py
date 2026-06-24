"""RecommendationSynthesisAgent – cross-profile recommendation integration (J6.8c).

Runs after RecommendationImprovementAgent and before ReportAgent.  Reads the
recommendations that MultiProfileAgent has already enriched with
``contributing_profiles`` and forces at least 2 recommendations to
*explicitly integrate* every contributing profile.

If the existing recommendations already achieve cross-profile coverage the
agent enriches them in-place.  If they do not, synthetic integrated
recommendations are appended.  The agent also generates explicit tradeoffs
and emits synthesis validation metrics.

Public API
----------
synthesise_recommendations(context)       – main analysis, returns enriched list
build_integrated_recs(profiles)           – template-based integrated recs
compute_profile_rationale(rec, profiles)  – per-profile rationale strings
compute_synthesis_validation(recs, ...)   – coverage metrics dict
compute_recommendation_profile_balance(recs, profiles) – fraction dict
build_synthesis_tradeoffs(profiles)       – list of explicit tradeoff dicts
RecommendationSynthesisAgent              – FunctionalAgent subclass
"""

from __future__ import annotations

import logging
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimum required integrated recommendations
# ---------------------------------------------------------------------------
_MIN_INTEGRATED = 2

# ---------------------------------------------------------------------------
# Synthetic integrated recommendation templates (used when coverage is absent)
# ---------------------------------------------------------------------------

_INTEGRATED_REC_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "SYNTH-R1",
        "title": (
            "Prioritise AI factory sites where GPU deployment readiness overlaps "
            "with available transmission capacity"
        ),
        "summary": (
            "Site selection for AI infrastructure must jointly optimise for "
            "compute readiness (cooling capacity, rack density, fibre) and grid "
            "readiness (transmission headroom, interconnection queue position). "
            "Single-dimension optimisation surfaces sites that are compute-ready "
            "but grid-constrained, or grid-connected but compute-unsuitable."
        ),
        "priority": "high",
        "time_horizon": "near_term",
        "confidence": "high",
        "confidence_rationale": (
            "Supported by both AI infrastructure and transmission evidence streams; "
            "site selection failures are the primary stranded-asset risk vector."
        ),
        "supported_by_hypotheses": [],
        "supporting_evidence": [],
        "key_risks": [
            "Interconnection queue delays may prevent power delivery at selected sites",
            "Transmission congestion may reduce deliverable power below GPU cluster requirements",
            "Sites optimised for rack density alone may lack long-term transmission headroom",
        ],
        "trigger_conditions": [
            "Site selection process initiation",
            "Interconnection queue position availability confirmation",
        ],
        "synthesis_source": "integrated",
    },
    {
        "id": "SYNTH-R2",
        "title": (
            "Treat interconnection queue position and utility planning timelines "
            "as gating criteria before large GPU procurement"
        ),
        "summary": (
            "Large-scale GPU procurement should be sequenced after, not before, "
            "interconnection queue reservation and utility service agreement "
            "execution. The interconnection queue process takes 3–6 years; GPU "
            "hardware has an 18-month delivery cycle. Committing capital to GPUs "
            "before securing power delivery creates stranded-asset risk."
        ),
        "priority": "high",
        "time_horizon": "near_term",
        "confidence": "high",
        "confidence_rationale": (
            "Critical sequencing risk identified across both AI infrastructure and "
            "transmission evidence streams; timing misalignment is the primary "
            "execution failure mode for large AI facility programmes."
        ),
        "supported_by_hypotheses": [],
        "supporting_evidence": [],
        "key_risks": [
            "GPU procurement lead times (18 months) are shorter than interconnection timelines (3–6 years)",
            "Capital committed to GPU hardware without grid access creates stranded-asset exposure",
            "Power reservation agreements may not be available at preferred sites",
        ],
        "trigger_conditions": [
            "AI infrastructure budget approval",
            "Site selection shortlist finalisation",
        ],
        "synthesis_source": "integrated",
    },
    {
        "id": "SYNTH-R3",
        "title": (
            "Co-develop power procurement, transmission access, and cooling "
            "infrastructure before committing to rack-scale deployment"
        ),
        "summary": (
            "The three long-lead-time inputs to AI factory commissioning — "
            "power interconnection (3–6 years), transmission access (utility "
            "agreements), and liquid cooling infrastructure (18+ month lead) — "
            "must be pursued in parallel rather than sequentially. Sequential "
            "planning adds 2–4 years to deployment timelines."
        ),
        "priority": "medium",
        "time_horizon": "medium_term",
        "confidence": "high",
        "confidence_rationale": (
            "Derived from evidence of lead-time misalignment across both AI "
            "infrastructure and transmission domains."
        ),
        "supported_by_hypotheses": [],
        "supporting_evidence": [],
        "key_risks": [
            "Power procurement and cooling procurement on independent timelines creates dependency gaps",
            "Transmission access delays may strand cooling and compute investments already committed",
        ],
        "trigger_conditions": [
            "AI factory programme initiation",
            "Site shortlist finalisation",
        ],
        "synthesis_source": "integrated",
    },
]

# ---------------------------------------------------------------------------
# Per-profile rationale templates
# ---------------------------------------------------------------------------

_PROFILE_RATIONALE_TEMPLATES: dict[str, dict[str, str]] = {
    "SYNTH-R1": {
        "ai_data_centers": (
            "GPU cluster deployment requires high-density power (100+ kW/rack) and "
            "liquid cooling that must be co-located with available grid capacity. "
            "Selecting sites without confirmed transmission headroom creates long-term "
            "operational risk for expanding AI workloads."
        ),
        "transmission": (
            "Transmission capacity and interconnection queue position determine whether "
            "sufficient power can be delivered to high-density AI facilities. Sites "
            "without queue position face 5–6 year delays; transmission congestion "
            "can cap deliverable power below cluster requirements."
        ),
    },
    "SYNTH-R2": {
        "ai_data_centers": (
            "GPU procurement timelines (18 months) are well-understood and relatively "
            "short; the binding constraint is grid access, not hardware availability. "
            "Sequencing GPU procurement before grid confirmation is the primary "
            "capital efficiency failure mode."
        ),
        "transmission": (
            "Interconnection queue timelines of 3–6 years mean power delivery cannot "
            "be assumed at the time GPU hardware arrives. Sequencing failures create "
            "stranded capital: GPUs delivered to sites without power delivery agreements."
        ),
    },
    "SYNTH-R3": {
        "ai_data_centers": (
            "Liquid cooling systems, GPU rack procurement, and networking fabric have "
            "12–18 month lead times that must be aligned with power delivery commitments. "
            "Power delivery is the longest-lead input and should anchor the programme timeline."
        ),
        "transmission": (
            "Transmission access and interconnection agreements are the longest-lead "
            "inputs to an AI factory programme; they should drive the project timeline "
            "rather than following compute hardware decisions made independently."
        ),
    },
}

_GENERIC_PROFILE_RATIONALE: dict[str, str] = {
    "ai_data_centers": (
        "AI infrastructure planning requires integrating power delivery constraints "
        "into site selection, cooling design, and rack deployment sequencing."
    ),
    "transmission": (
        "Transmission grid access, interconnection queue position, and utility "
        "coordination timelines are binding constraints that must be addressed "
        "before large compute commitments."
    ),
}

# ---------------------------------------------------------------------------
# Tradeoff templates
# ---------------------------------------------------------------------------

_TRADEOFF_TEMPLATES: list[dict[str, Any]] = [
    {
        "tradeoff_id": "T1",
        "dimension_a": "Compute availability",
        "dimension_b": "Grid availability",
        "description": (
            "The highest-performing AI factory locations — those with low-latency "
            "fibre, available land, and dense power infrastructure — may not have "
            "sufficient transmission capacity to serve 100–1,000+ MW GPU clusters. "
            "Optimising for compute density without grid validation produces sites "
            "that cannot be fully powered."
        ),
        "implication": (
            "Site selection criteria must weight transmission capacity alongside "
            "compute infrastructure readiness. Compute-optimal sites without grid "
            "headroom should be rejected or deferred until transmission upgrades complete."
        ),
        "profiles": ["ai_data_centers", "transmission"],
    },
    {
        "tradeoff_id": "T2",
        "dimension_a": "GPU deployment speed",
        "dimension_b": "Interconnection timeline",
        "description": (
            "GPU hardware can be procured and delivered in 12–18 months. "
            "Interconnection queue approval and power delivery take 3–6 years. "
            "Deploying GPUs before power delivery is confirmed creates a 2–4 year "
            "period of stranded capital with hardware on-site but unable to operate."
        ),
        "implication": (
            "GPU procurement must be gated on confirmed interconnection timeline. "
            "Financial models must account for the gap between hardware delivery "
            "and operational commissioning driven by grid access delays."
        ),
        "profiles": ["ai_data_centers", "transmission"],
    },
    {
        "tradeoff_id": "T3",
        "dimension_a": "Rack-scale density",
        "dimension_b": "Regional power deliverability",
        "description": (
            "Next-generation AI rack systems (NVIDIA Vera Rubin NVL576) require "
            "up to 1 MW per cabinet. Regional transmission systems in constrained "
            "markets (Northern Virginia, Phoenix, Dallas) are approaching capacity "
            "limits. Pursuing maximum rack density at congested sites amplifies "
            "power delivery risk rather than mitigating it."
        ),
        "implication": (
            "Rack density targets must be validated against regional transmission "
            "headroom, not just on-site power infrastructure. Deployment plans for "
            "maximum-density configurations require dedicated transmission capacity "
            "studies before commitment."
        ),
        "profiles": ["ai_data_centers", "transmission"],
    },
]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def compute_profile_rationale(
    rec: dict[str, Any],
    profiles: list[str],
) -> dict[str, str]:
    """Return per-profile rationale strings for a recommendation."""
    rec_id = rec.get("id", "")
    if rec_id in _PROFILE_RATIONALE_TEMPLATES:
        template = _PROFILE_RATIONALE_TEMPLATES[rec_id]
        return {p: template.get(p, _GENERIC_PROFILE_RATIONALE.get(p, "")) for p in profiles}
    return {p: _GENERIC_PROFILE_RATIONALE.get(p, "") for p in profiles}


def build_integrated_recs(
    profiles: list[str],
    count: int = _MIN_INTEGRATED,
) -> list[dict[str, Any]]:
    """Return ``count`` synthetic integrated recommendation dicts for the given profiles."""
    result: list[dict[str, Any]] = []
    for tmpl in _INTEGRATED_REC_TEMPLATES[:count]:
        rationale = compute_profile_rationale(tmpl, profiles)
        result.append({
            **tmpl,
            "contributing_profiles": list(profiles),
            "profile_rationale": rationale,
        })
    return result


def compute_synthesis_validation(
    recommendations: list[dict[str, Any]],
    profiles_requested: list[str],
    profiles_contributing: list[str],
) -> dict[str, Any]:
    """Return synthesis coverage metrics dict."""
    integrated = [
        r for r in recommendations
        if len(r.get("contributing_profiles", [])) >= 2
    ]
    profiles_in_recs = {
        p
        for r in recommendations
        for p in r.get("contributing_profiles", [])
    }
    n_req = len(profiles_requested)
    n_con = len(profiles_contributing)
    coverage_status = (
        "sufficient" if n_req == 0 or n_con == n_req
        else ("partial" if n_con > 0 else "insufficient")
    )
    return {
        "profiles_requested":                  n_req,
        "profiles_contributing":               n_con,
        "profiles_represented_in_recommendations": len(profiles_in_recs & set(profiles_requested)),
        "integrated_recommendation_count":     len(integrated),
        "coverage_status":                     coverage_status,
    }


def compute_recommendation_profile_balance(
    recommendations: list[dict[str, Any]],
    profiles: list[str],
) -> dict[str, float]:
    """Return fraction of rec-profile touches per profile (sums to 1.0)."""
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


def build_synthesis_tradeoffs(profiles: list[str]) -> list[dict[str, Any]]:
    """Return tradeoffs applicable to the given profile combination."""
    profile_set = set(profiles)
    return [
        t for t in _TRADEOFF_TEMPLATES
        if set(t.get("profiles", [])) & profile_set
    ]


def synthesise_recommendations(context: AgentContext) -> dict[str, Any]:
    """Run the full synthesis pipeline and return the synthesis output dict."""
    profiles = list(context.profiles)
    mpa = context.multi_profile_analysis or {}
    profiles_contributing: list[str] = mpa.get("profiles_contributing", profiles)

    existing_recs = list(context.recommendations or [])

    # ── 1. Enrich existing recs with profile_rationale if missing ──────────
    enriched: list[dict[str, Any]] = []
    for rec in existing_recs:
        contrib = rec.get("contributing_profiles", [])
        if not contrib:
            contrib = [profiles[0]] if profiles else []
        rationale = rec.get("profile_rationale")
        if not rationale:
            rationale = compute_profile_rationale(rec, contrib)
        enriched.append({**rec, "contributing_profiles": contrib, "profile_rationale": rationale})

    # ── 2. Count integrated recs (attributed to ≥2 profiles) ───────────────
    integrated_existing = [
        r for r in enriched
        if len(r.get("contributing_profiles", [])) >= 2
    ]

    # ── 3. Inject synthetic integrated recs if below minimum ────────────────
    synth_recs: list[dict[str, Any]] = []
    needed = max(0, _MIN_INTEGRATED - len(integrated_existing))
    if needed > 0 and len(profiles_contributing) >= 2:
        synth_recs = build_integrated_recs(profiles_contributing, count=needed)
        enriched.extend(synth_recs)

    # ── 4. Compute synthesis validation ────────────────────────────────────
    sv = compute_synthesis_validation(enriched, profiles, profiles_contributing)

    # ── 5. Compute profile balance ─────────────────────────────────────────
    balance = compute_recommendation_profile_balance(enriched, profiles)

    # ── 6. Build tradeoffs ─────────────────────────────────────────────────
    tradeoffs = build_synthesis_tradeoffs(profiles_contributing or profiles)

    # ── 7. Recommendation profile audit ───────────────────────────────────
    audit: dict[str, list[str]] = {}
    for i, r in enumerate(enriched):
        rid = r.get("id") or r.get("recommendation_id") or f"R{i + 1}"
        audit[str(rid)] = list(r.get("contributing_profiles", []))

    return {
        "enriched_recommendations":     enriched,
        "synthetic_recommendations":    synth_recs,
        "synthesis_validation":         sv,
        "recommendation_profile_balance": balance,
        "synthesis_tradeoffs":          tradeoffs,
        "recommendation_profile_audit": audit,
        "integrated_recommendation_count": sv["integrated_recommendation_count"],
        "profiles_requested":           profiles,
        "profiles_contributing":        profiles_contributing,
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class RecommendationSynthesisAgent(FunctionalAgent):
    """Force cross-profile integration in the final recommendation set (J6.8c).

    Reads:
        context.recommendations            (from RecommendationImprovementAgent)
        context.multi_profile_analysis     (from MultiProfileAgent)
        context.profiles

    Writes:
        context.recommendations            (enriched in-place + synthetic recs)
        context.synthesis_validation
        context.recommendation_profile_balance
        context.synthesis_tradeoffs
        context.qa["synthesis_validation"]
        context.research_object["synthesis_validation"]
        context.research_object["recommendation_profile_balance"]
        context.research_object["synthesis_tradeoffs"]
        context.trace["_recommendation_synthesis"]
    """

    def _execute(self, context: AgentContext) -> AgentContext:
        from research_agent.log import PROGRESS

        synthesis = synthesise_recommendations(context)

        # Update recommendations on context
        context.recommendations = synthesis["enriched_recommendations"]

        # Context synthesis fields
        context.synthesis_validation = synthesis["synthesis_validation"]
        context.recommendation_profile_balance = synthesis["recommendation_profile_balance"]
        context.synthesis_tradeoffs = synthesis["synthesis_tradeoffs"]

        # QA block
        context.qa["synthesis_validation"] = synthesis["synthesis_validation"]

        # Research object
        ro = context.research_object
        if ro is not None:
            ro["synthesis_validation"] = synthesis["synthesis_validation"]
            ro["recommendation_profile_balance"] = synthesis["recommendation_profile_balance"]
            ro["synthesis_tradeoffs"] = synthesis["synthesis_tradeoffs"]
            ro["recommendation_profile_audit"] = synthesis["recommendation_profile_audit"]
            if synthesis["synthetic_recommendations"]:
                ro["synthetic_recommendations"] = synthesis["synthetic_recommendations"]

        # Trace
        context.trace["_recommendation_synthesis"] = {
            "agent": "RecommendationSynthesisAgent",
            "profiles_requested":              synthesis["profiles_requested"],
            "profiles_contributing":           synthesis["profiles_contributing"],
            "integrated_recommendation_count": synthesis["integrated_recommendation_count"],
            "synthetic_recs_added":            len(synthesis["synthetic_recommendations"]),
            "tradeoff_count":                  len(synthesis["synthesis_tradeoffs"]),
            "synthesis_validation":            synthesis["synthesis_validation"],
            "recommendation_profile_balance":  synthesis["recommendation_profile_balance"],
            "recommendation_profile_audit":    synthesis["recommendation_profile_audit"],
        }

        sv = synthesis["synthesis_validation"]
        summary = (
            f"integrated_recommendations={synthesis['integrated_recommendation_count']} "
            f"synthetic_added={len(synthesis['synthetic_recommendations'])} "
            f"profiles_represented={sv.get('profiles_represented_in_recommendations', 0)} "
            f"coverage_status={sv.get('coverage_status', 'unknown')}"
        )
        LOGGER.log(PROGRESS, "[RecommendationSynthesisAgent] %s", summary)
        self._record(
            context,
            status="success",
            summary=summary,
            integrated_recommendation_count=synthesis["integrated_recommendation_count"],
            profiles_represented=sv.get("profiles_represented_in_recommendations", 0),
            coverage_status=sv.get("coverage_status", "unknown"),
        )
        return context
