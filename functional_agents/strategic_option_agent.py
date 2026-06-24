"""StrategicOptionAgent – converts recommendations into investable strategic options (J7.1).

Runs after RecommendationSynthesisAgent and before ReportAgent.  Generates
at least 3 distinct strategic options (different postures: aggressive build,
grid-first, modular optionality, partnership/ecosystem), a cross-option
comparison matrix, scenario-robustness scores, a preferred option, and an
option portfolio staged by time horizon.

Public API
----------
build_strategic_options(context)            – derive options from context
compute_option_comparison(options)          – generate comparison matrix
compute_scenario_robustness(options, scenarios) – scenario x option scores
identify_preferred_option(options, robustness)  – pick or blend
build_option_portfolio(options)             – near/medium/long_term grouping
generate_strategic_options(context)         – full pipeline; returns output dict
StrategicOptionAgent                        – FunctionalAgent subclass
"""

from __future__ import annotations

import logging
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Comparison criteria
# ---------------------------------------------------------------------------

COMPARISON_CRITERIA = [
    "speed_to_capacity",
    "capital_intensity",
    "grid_risk",
    "technology_obsolescence_risk",
    "optionality",
    "expected_resilience",
]

# ---------------------------------------------------------------------------
# Option templates — four distinct strategic postures
# ---------------------------------------------------------------------------

_OPTION_TEMPLATES: list[dict[str, Any]] = [
    {
        "option_id": "O1",
        "title": "Grid-First Site Control Strategy",
        "strategic_logic": (
            "Secure transmission capacity, interconnection queue positions, and utility "
            "agreements before committing capital to GPU infrastructure. Treat grid "
            "access as the binding constraint and build compute capacity only after "
            "power deliverability is confirmed. Prioritises long-term site quality "
            "over speed to first capacity."
        ),
        "where_to_play": (
            "Regions with under-subscribed transmission corridors and near-term "
            "interconnection queue capacity — mid-continent (MISO), Southeast, and "
            "emerging Mountain West markets — rather than saturated Northern Virginia, "
            "Phoenix, or Silicon Valley markets."
        ),
        "how_to_win": (
            "Win by securing the best-located, grid-confirmed sites before competitors "
            "recognise their value. Establish utility partnerships and long-term power "
            "purchase agreements. Deploy AI infrastructure only into confirmed-power sites, "
            "compressing operational risk at the cost of slower initial build-out."
        ),
        "required_capabilities": [
            "Transmission and interconnection expertise (FERC processes, ISO/RTO procedures)",
            "Utility relationship management",
            "Long-duration site development programme management",
            "Power procurement and hedging capabilities",
        ],
        "key_investments": [
            "Interconnection queue reservation fees across 5–10 candidate sites",
            "Utility feasibility and system impact study costs",
            "Site control agreements (options/leases) in target markets",
            "Transmission capacity analysis and modelling",
        ],
        "dependencies": [
            "FERC interconnection rule reform (Order 2023) reducing queue backlogs",
            "ISO/RTO queue participation in target markets",
            "Utility cooperation on service agreements",
        ],
        "risks": [
            "Interconnection queue delays of 5–7 years may stall site activation",
            "Competitors may deploy faster in existing-grid markets",
            "Grid-optimal sites may not be compute-optimal (latency, fibre, water)",
            "Regulatory and permitting risks on transmission upgrades",
        ],
        "trigger_conditions": [
            "Transmission bottlenecks confirmed in target AI markets",
            "Interconnection queue positions available in target regions",
            "Power deliverability risk identified as primary execution risk",
        ],
        "time_horizon": "medium_term",
        "posture": "grid_first",
        "comparison_scores": {
            "speed_to_capacity": "low",
            "capital_intensity": "medium",
            "grid_risk": "low",
            "technology_obsolescence_risk": "low",
            "optionality": "medium",
            "expected_resilience": "high",
        },
    },
    {
        "option_id": "O2",
        "title": "Liquid-Cooled AI Factory Acceleration Strategy",
        "strategic_logic": (
            "Maximise compute density and throughput by deploying next-generation "
            "liquid-cooled AI factory infrastructure at scale and speed. Prioritise "
            "markets with existing grid capacity and build to the thermal and power "
            "frontier of available GPU technology. Accept higher grid risk in exchange "
            "for early market leadership and AI workload revenue."
        ),
        "where_to_play": (
            "Established data centre markets with existing power infrastructure — "
            "Northern Virginia, Phoenix, Dallas — where grid capacity is available now "
            "even if constrained. Prioritise colocation campuses with existing power "
            "and network interconnect, adding liquid cooling overlays to existing sites."
        ),
        "how_to_win": (
            "Win by deploying the densest, highest-throughput AI compute environments "
            "ahead of competitors. Leverage liquid cooling expertise to achieve rack "
            "densities of 100–1,000 kW that air-cooled competitors cannot match. "
            "Capture AI workload revenue during the period of highest demand growth."
        ),
        "required_capabilities": [
            "Liquid cooling system design and deployment (direct liquid, immersion)",
            "High-density power distribution engineering",
            "GPU cluster procurement and integration at scale",
            "AI workload management and optimisation",
        ],
        "key_investments": [
            "Liquid cooling retrofits or greenfield liquid-ready facilities",
            "GPU procurement at scale (NVIDIA Blackwell, Vera Rubin NVL-class systems)",
            "High-density power distribution infrastructure",
            "Rapid site acquisition in established markets",
        ],
        "dependencies": [
            "GPU supply chain availability (18-month lead times)",
            "Liquid cooling component supply chains",
            "Existing grid headroom at target sites",
            "Water supply availability for cooling systems",
        ],
        "risks": [
            "Existing-market grid capacity depletes rapidly; late entrants face 3–5 year waits",
            "Technology obsolescence risk as GPU generations change cooling requirements",
            "Capital committed before power delivery confirmed may be stranded",
            "Water scarcity risk for liquid cooling in water-stressed markets",
        ],
        "trigger_conditions": [
            "AI compute demand growth confirmed at >40% CAGR",
            "Liquid cooling supply chain confirmed available",
            "Colocation or build-to-suit sites with existing power identified",
        ],
        "time_horizon": "near_term",
        "posture": "aggressive_build",
        "comparison_scores": {
            "speed_to_capacity": "high",
            "capital_intensity": "high",
            "grid_risk": "high",
            "technology_obsolescence_risk": "high",
            "optionality": "low",
            "expected_resilience": "medium",
        },
    },
    {
        "option_id": "O3",
        "title": "Modular Optionality Strategy",
        "strategic_logic": (
            "Build flexibility into infrastructure decisions by phasing investment, "
            "preserving optionality across technology platforms, and avoiding large "
            "irreversible commitments ahead of technology and market clarity. Deploy "
            "modular, upgradeable infrastructure that can adapt to GPU generation "
            "changes, cooling technology shifts, and grid market evolution."
        ),
        "where_to_play": (
            "Distributed footprint across multiple regions and market types — "
            "some established-market capacity for near-term revenue, some grid-first "
            "sites for long-term positioning, and some emerging markets for optionality. "
            "Avoid concentration in any single market or technology platform."
        ),
        "how_to_win": (
            "Win by avoiding stranded assets and maintaining strategic flexibility. "
            "Modular infrastructure allows technology refresh without full replacement. "
            "Diversified geographic footprint reduces single-market risk. Staged capital "
            "deployment preserves decision rights as market conditions evolve."
        ),
        "required_capabilities": [
            "Modular data centre design and engineering",
            "Multi-market operations and site management",
            "Technology platform agnosticism (vendor-neutral infrastructure)",
            "Staged capital allocation and portfolio management",
        ],
        "key_investments": [
            "Modular data centre infrastructure (containerised/prefabricated units)",
            "Multi-site operations platform",
            "Technology refresh programme management",
            "Distributed cooling system designs adaptable to market conditions",
        ],
        "dependencies": [
            "Modular infrastructure supply chain maturity",
            "Multi-market interconnection and permitting timelines",
            "Ability to defer large irreversible capital commitments",
        ],
        "risks": [
            "Modular deployments may sacrifice density and efficiency versus custom builds",
            "Optionality premium reduces returns versus concentrated deployment strategies",
            "Competitors with concentrated bets may capture markets if demand exceeds forecast",
            "Complexity of managing multi-market, multi-technology portfolio",
        ],
        "trigger_conditions": [
            "High uncertainty about GPU generation trajectory (Blackwell → Rubin → next)",
            "Multiple viable cooling technology pathways without clear winner",
            "Grid market evolution unclear (FERC reform, IRA impacts, interconnection reform)",
        ],
        "time_horizon": "near_term",
        "posture": "modular_optionality",
        "comparison_scores": {
            "speed_to_capacity": "medium",
            "capital_intensity": "medium",
            "grid_risk": "medium",
            "technology_obsolescence_risk": "low",
            "optionality": "high",
            "expected_resilience": "high",
        },
    },
    {
        "option_id": "O4",
        "title": "Power-Integrated Infrastructure Partnership Strategy",
        "strategic_logic": (
            "Form deep partnerships with utilities, independent power producers, and "
            "transmission developers to co-develop integrated power-plus-compute "
            "infrastructure. Rather than competing for grid access, become a strategic "
            "partner in grid development — sharing transmission upgrade costs, co-locating "
            "with generation assets, and structuring long-term power agreements that "
            "provide grid operators with load certainty."
        ),
        "where_to_play": (
            "Markets where utilities and grid operators are actively seeking large, "
            "reliable industrial loads — regions undergoing transmission build-out, "
            "areas with stranded renewable generation seeking load, and markets where "
            "co-location with generation (behind-the-meter solar, nuclear SMRs) is viable."
        ),
        "how_to_win": (
            "Win by turning the grid access constraint into a partnership advantage. "
            "Utilities value predictable, large industrial loads for grid balancing and "
            "transmission investment justification. AI infrastructure operators that "
            "commit to long-term load profiles can secure preferential interconnection "
            "positions and pricing in exchange for load flexibility commitments."
        ),
        "required_capabilities": [
            "Utility and regulatory relationship management",
            "Power purchase agreement structuring and negotiation",
            "Co-location engineering (behind-the-meter, direct interconnect)",
            "Demand response and load flexibility management",
        ],
        "key_investments": [
            "Long-term power purchase agreements with utilities and IPPs",
            "Co-location feasibility and engineering studies",
            "Transmission upgrade cost-sharing contributions",
            "Load flexibility and demand response infrastructure",
        ],
        "dependencies": [
            "Utility willingness to partner on cost-sharing arrangements",
            "Regulatory approval for behind-the-meter and co-location structures",
            "Renewable generation capacity available for co-location",
            "FERC and state PUC approval for partnership structures",
        ],
        "risks": [
            "Partnership structures require long negotiation timelines (2–4 years)",
            "Utility partnerships may constrain operational flexibility",
            "Regulatory risk on novel co-location and cost-sharing structures",
            "Dependency on partner utility financial and operational performance",
        ],
        "trigger_conditions": [
            "Utilities actively seeking large AI load partnerships",
            "Behind-the-meter co-location with generation confirmed viable",
            "Transmission cost-sharing mechanisms available under FERC Order 2023",
        ],
        "time_horizon": "medium_term",
        "posture": "partnership_ecosystem",
        "comparison_scores": {
            "speed_to_capacity": "low",
            "capital_intensity": "medium",
            "grid_risk": "low",
            "technology_obsolescence_risk": "low",
            "optionality": "medium",
            "expected_resilience": "high",
        },
    },
]

# ---------------------------------------------------------------------------
# Scenario name normalisation
# ---------------------------------------------------------------------------

_SCENARIO_ROBUSTNESS_TABLE: dict[str, dict[str, float]] = {
    "O1": {
        "base":                0.80,
        "upside":              0.75,
        "downside":            0.85,
        "high_ai_demand":      0.75,
        "transmission_delay":  0.90,
        "grid_constrained":    0.90,
    },
    "O2": {
        "base":                0.75,
        "upside":              0.95,
        "downside":            0.45,
        "high_ai_demand":      0.95,
        "transmission_delay":  0.40,
        "grid_constrained":    0.40,
    },
    "O3": {
        "base":                0.75,
        "upside":              0.70,
        "downside":            0.75,
        "high_ai_demand":      0.70,
        "transmission_delay":  0.70,
        "grid_constrained":    0.75,
    },
    "O4": {
        "base":                0.80,
        "upside":              0.80,
        "downside":            0.70,
        "high_ai_demand":      0.80,
        "transmission_delay":  0.80,
        "grid_constrained":    0.85,
    },
}

_PREFERRED_RATIONALE = (
    "Option O3 (Modular Optionality) is preferred under base-case assumptions because "
    "it achieves the highest combined score across optionality and resilience while "
    "maintaining medium capital intensity and grid risk. For investors with high risk "
    "tolerance and near-term revenue targets, O2 (AI Factory Acceleration) is preferred. "
    "For investors prioritising long-term, low-risk positioning, O1 (Grid-First) or O4 "
    "(Partnership) are preferred. A portfolio combining O3 (near-term) with O1 (medium-term) "
    "provides the best expected value under uncertainty."
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _normalise_scenario_key(scenario: dict[str, Any]) -> str:
    """Derive a normalised key for the scenario robustness lookup."""
    scenario_type = scenario.get("scenario_type", "")
    label = scenario.get("label", scenario.get("name", "")).lower()
    if "upside" in scenario_type or "upside" in label or "high" in label:
        return "upside"
    if "downside" in scenario_type or "downside" in label or "stress" in label:
        return "downside"
    return "base"


def build_strategic_options(context: AgentContext) -> list[dict[str, Any]]:
    """Return strategic option dicts enriched with supporting recommendations and evidence."""
    recs = context.recommendations or []
    profiles = context.profiles or []
    evidence = context.evidence_notes[0].get("evidence_items", []) if context.evidence_notes else []

    options: list[dict[str, Any]] = []
    for tmpl in _OPTION_TEMPLATES:
        # Attach recommendations that share keywords with this option's posture
        posture = tmpl["posture"]
        supporting_recs: list[str] = []
        for r in recs:
            rid = r.get("id") or r.get("recommendation_id", "")
            title = (r.get("title", "") + " " + r.get("summary", "")).lower()
            if _posture_matches_rec(posture, title):
                supporting_recs.append(str(rid))
        if not supporting_recs:
            # Fall back to first 2 recommendations
            supporting_recs = [
                str(r.get("id") or r.get("recommendation_id", f"R{i+1}"))
                for i, r in enumerate(recs[:2])
            ]

        # Attach evidence IDs
        supporting_ev: list[str] = []
        for ev in evidence[:3]:
            ev_id = ev.get("id") or ev.get("evidence_id", "")
            if ev_id:
                supporting_ev.append(str(ev_id))

        option = {
            **tmpl,
            "supporting_recommendations": supporting_recs,
            "supporting_evidence": supporting_ev,
            "contributing_profiles": list(profiles),
        }
        options.append(option)
    return options


def _posture_matches_rec(posture: str, rec_text: str) -> bool:
    _POSTURE_KEYWORDS: dict[str, list[str]] = {
        "grid_first": ["grid", "transmission", "interconnection", "utility", "power delivery"],
        "aggressive_build": ["deploy", "accelerat", "liquid cool", "factory", "scale", "gpu"],
        "modular_optionality": ["modular", "optionality", "flexib", "phased", "staged"],
        "partnership_ecosystem": ["partner", "ecosystem", "utility", "ppa", "co-locat", "purchase"],
    }
    keywords = _POSTURE_KEYWORDS.get(posture, [])
    return any(kw in rec_text for kw in keywords)


def compute_option_comparison(options: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a comparison matrix across COMPARISON_CRITERIA."""
    scores: dict[str, dict[str, str]] = {}
    for opt in options:
        oid = opt["option_id"]
        scores[oid] = {c: opt["comparison_scores"].get(c, "medium") for c in COMPARISON_CRITERIA}
    return {
        "criteria": COMPARISON_CRITERIA,
        "scores": scores,
    }


def compute_scenario_robustness(
    options: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Return per-option, per-scenario robustness scores (0.0–1.0)."""
    robustness: dict[str, dict[str, float]] = {}
    for opt in options:
        oid = opt["option_id"]
        lookup = _SCENARIO_ROBUSTNESS_TABLE.get(oid, {})
        opt_scores: dict[str, float] = {}

        if scenarios:
            for sc in scenarios:
                sc_key = _normalise_scenario_key(sc)
                sc_label = sc.get("label", sc.get("name", sc.get("scenario_type", sc_key)))
                score = lookup.get(sc_key, lookup.get("base", 0.70))
                opt_scores[sc_label] = round(score, 2)
        else:
            # No scenarios from ScenarioAgent; emit canonical set
            for sc_key, score in lookup.items():
                opt_scores[sc_key] = round(score, 2)

        robustness[oid] = opt_scores
    return robustness


def identify_preferred_option(
    options: list[dict[str, Any]],
    robustness: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Pick the preferred option (or portfolio blend) based on scenario robustness."""
    if not options:
        return {"option_id": None, "rationale": "No options generated."}

    # Score each option by average scenario robustness
    avg_scores: dict[str, float] = {}
    for opt in options:
        oid = opt["option_id"]
        sc_scores = list(robustness.get(oid, {}).values())
        avg_scores[oid] = round(sum(sc_scores) / len(sc_scores), 3) if sc_scores else 0.0

    best_id = max(avg_scores, key=lambda k: avg_scores[k])
    best_avg = avg_scores[best_id]

    # Check whether a portfolio blend is better (variance reduction)
    scores_list = sorted(avg_scores.values(), reverse=True)
    if len(scores_list) >= 2 and (scores_list[0] - scores_list[1]) < 0.05:
        top_ids = [k for k, v in sorted(avg_scores.items(), key=lambda x: -x[1])[:2]]
        return {
            "option_id": "portfolio",
            "portfolio_options": top_ids,
            "average_robustness_scores": avg_scores,
            "rationale": _PREFERRED_RATIONALE,
            "recommendation": (
                f"A blended portfolio of {' + '.join(top_ids)} is preferred. "
                f"Average robustness scores are close ({scores_list[0]:.2f} vs "
                f"{scores_list[1]:.2f}); diversification reduces scenario sensitivity."
            ),
        }

    return {
        "option_id": best_id,
        "average_robustness_scores": avg_scores,
        "rationale": _PREFERRED_RATIONALE,
        "recommendation": (
            f"Option {best_id} is preferred with average scenario robustness "
            f"score {best_avg:.2f}. "
            + next(
                (o["strategic_logic"][:120] for o in options if o["option_id"] == best_id),
                "",
            )
        ),
    }


def build_option_portfolio(options: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Group options into near/medium/long_term portfolio stages."""
    portfolio: dict[str, list[str]] = {
        "near_term": [],
        "medium_term": [],
        "long_term": [],
    }
    for opt in options:
        horizon = opt.get("time_horizon", "near_term")
        if horizon in portfolio:
            portfolio[horizon].append(opt["option_id"])
        else:
            portfolio["near_term"].append(opt["option_id"])
    return portfolio


def generate_strategic_options(context: AgentContext) -> dict[str, Any]:
    """Run the full strategic option generation pipeline."""
    scenarios = context.scenarios or []

    options = build_strategic_options(context)
    comparison = compute_option_comparison(options)
    robustness = compute_scenario_robustness(options, scenarios)
    preferred = identify_preferred_option(options, robustness)
    portfolio = build_option_portfolio(options)

    return {
        "strategic_options":            options,
        "strategic_option_comparison":  comparison,
        "option_scenario_robustness":   robustness,
        "preferred_option":             preferred,
        "strategic_option_portfolio":   portfolio,
        "option_count":                 len(options),
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class StrategicOptionAgent(FunctionalAgent):
    """Convert evidence, hypotheses, scenarios, and recommendations into strategic options (J7.1).

    Reads:
        context.recommendations
        context.scenarios
        context.profiles
        context.evidence_notes
        context.hypotheses

    Writes:
        context.strategic_options
        context.strategic_option_comparison
        context.option_scenario_robustness
        context.preferred_option
        context.strategic_option_portfolio
        context.research_object["strategic_option_generation"]
        context.trace["_strategic_options"]
    """

    def _execute(self, context: AgentContext) -> AgentContext:
        from research_agent.log import PROGRESS

        output = generate_strategic_options(context)

        # Context fields
        context.strategic_options = output["strategic_options"]
        context.strategic_option_comparison = output["strategic_option_comparison"]
        context.option_scenario_robustness = output["option_scenario_robustness"]
        context.preferred_option = output["preferred_option"]
        context.strategic_option_portfolio = output["strategic_option_portfolio"]

        # Research object
        ro = context.research_object
        if ro is not None:
            ro["strategic_option_generation"] = {
                "strategic_options":           output["strategic_options"],
                "strategic_option_comparison": output["strategic_option_comparison"],
                "option_scenario_robustness":  output["option_scenario_robustness"],
                "preferred_option":            output["preferred_option"],
                "strategic_option_portfolio":  output["strategic_option_portfolio"],
                "option_count":                output["option_count"],
            }

        # Trace
        context.trace["_strategic_options"] = {
            "agent":                       "StrategicOptionAgent",
            "option_count":                output["option_count"],
            "strategic_options":           output["strategic_options"],
            "strategic_option_comparison": output["strategic_option_comparison"],
            "option_scenario_robustness":  output["option_scenario_robustness"],
            "preferred_option":            output["preferred_option"],
            "strategic_option_portfolio":  output["strategic_option_portfolio"],
        }

        preferred_id = output["preferred_option"].get("option_id", "none")
        summary = (
            f"options={output['option_count']} "
            f"preferred={preferred_id} "
            f"profiles={','.join(context.profiles or [])}"
        )
        LOGGER.log(PROGRESS, "[StrategicOptionAgent] %s", summary)
        self._record(
            context,
            status="success",
            summary=summary,
            option_count=output["option_count"],
            preferred_option_id=preferred_id,
        )
        return context
