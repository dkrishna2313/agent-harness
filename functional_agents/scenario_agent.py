"""ScenarioAgent – generate plausible futures and stress-test recommendations (J6.8).

Three canonical scenarios are generated (Base / Upside / Downside) from
template definitions enriched with evidence IDs found in context.  Each
recommendation is then scored for fit under each scenario, producing a
robustness score and scenario-specific adjustments.

Public API
----------
generate_scenarios(context)           – returns 3 scenario dicts with evidence IDs
stress_test_recommendations(recs, scenarios) – returns per-rec stress-test records
ScenarioAgent                         – FunctionalAgent subclass
"""

from __future__ import annotations

import logging
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scenario templates
# ---------------------------------------------------------------------------

_SCENARIO_TEMPLATES: list[dict[str, Any]] = [
    {
        "scenario_id": "S1",
        "name": "Base Case",
        "description": (
            "AI demand grows strongly; grid and permitting constraints remain material "
            "but manageable with early planning and proactive interconnection agreements."
        ),
        "assumptions": {
            "ai_demand_growth": "strong",
            "power_availability": "moderate",
            "grid_interconnection_timelines": "standard_2_to_4_years",
            "transmission_constraints": "material",
            "cooling_technology_readiness": "improving",
            "capital_availability": "adequate",
            "regulatory_permitting": "standard",
        },
        "critical_uncertainties": [
            "pace of grid interconnection queue resolution",
            "cooling supply chain scale-up speed",
        ],
        "probability": 0.50,
    },
    {
        "scenario_id": "S2",
        "name": "Upside Case",
        "description": (
            "Grid interconnection, power procurement, and cooling supply chains improve "
            "faster than expected; capital remains abundant and permitting streamlines."
        ),
        "assumptions": {
            "ai_demand_growth": "very_strong",
            "power_availability": "improving",
            "grid_interconnection_timelines": "accelerated_under_2_years",
            "transmission_constraints": "easing",
            "cooling_technology_readiness": "mature",
            "capital_availability": "abundant",
            "regulatory_permitting": "streamlined",
        },
        "critical_uncertainties": [
            "pace of permitting reform implementation",
            "rate of cooling technology commoditisation",
        ],
        "probability": 0.25,
    },
    {
        "scenario_id": "S3",
        "name": "Downside Case",
        "description": (
            "AI load growth outpaces grid expansion; permitting slows and power constraints "
            "become binding, compressing development timelines and stranding early investments."
        ),
        "assumptions": {
            "ai_demand_growth": "strong",
            "power_availability": "constrained",
            "grid_interconnection_timelines": "delayed_over_5_years",
            "transmission_constraints": "binding",
            "cooling_technology_readiness": "limited",
            "capital_availability": "tightening",
            "regulatory_permitting": "restrictive",
        },
        "critical_uncertainties": [
            "severity of grid interconnection delays",
            "regulatory response to power demand growth",
            "capital market appetite for constrained-location assets",
        ],
        "probability": 0.25,
    },
]

# ---------------------------------------------------------------------------
# Keyword sets for heuristic fit scoring
# ---------------------------------------------------------------------------

_POWER_GRID_KW: frozenset[str] = frozenset({
    "power", "grid", "energy", "transmission", "interconnect", "utility",
    "electricity", "load", "capacity", "megawatt", "kilowatt", "mw", "kw",
})
_COOLING_KW: frozenset[str] = frozenset({
    "cooling", "thermal", "liquid", "pue", "hvac", "heat", "refrigerant",
})
_CAPITAL_KW: frozenset[str] = frozenset({
    "capital", "investment", "invest", "cost", "budget", "opex", "capex", "fund",
})

_FIT_TO_SCORE: dict[str, float] = {"strong": 1.0, "medium": 0.6, "weak": 0.3}


def _word_bag(rec: dict) -> set[str]:
    """Lowercase word bag from title + summary + key_risks."""
    text = (
        rec.get("title", "") + " " +
        rec.get("summary", "") + " " +
        " ".join(rec.get("key_risks", []))
    ).lower()
    return set(text.split())


def _fit_label(score: float) -> str:
    if score >= 0.75:
        return "strong"
    if score >= 0.45:
        return "medium"
    return "weak"


# ---------------------------------------------------------------------------
# Scenario fit scoring
# ---------------------------------------------------------------------------

def _compute_scenario_fit(rec: dict) -> dict[str, str]:
    """Return scenario_fit dict {base_case, upside_case, downside_case} → label."""
    words = _word_bag(rec)

    has_power_grid = bool(words & _POWER_GRID_KW)
    has_cooling = bool(words & _COOLING_KW)
    has_capital = bool(words & _CAPITAL_KW)
    n_risks = len(rec.get("key_risks", []))
    has_risks = n_risks >= 2

    # Base case: generally strong; weak if rec ignores key risks
    base = 0.75
    if has_power_grid and has_risks:
        base = 1.0
    elif not has_risks:
        base = 0.60

    # Upside case: capital-intensive and cooling recs benefit most
    upside = 0.75
    if has_capital or has_cooling:
        upside = 1.0

    # Downside case: recs addressing power/grid constraints are more robust;
    # risk-unaware recs weaken
    downside = 0.60
    if has_power_grid and has_risks:
        downside = 0.75
    if not has_risks:
        downside -= 0.30

    return {
        "base_case":     _fit_label(max(0.3, min(1.0, base))),
        "upside_case":   _fit_label(max(0.3, min(1.0, upside))),
        "downside_case": _fit_label(max(0.3, min(1.0, downside))),
    }


def _compute_robustness_score(scenario_fit: dict[str, str]) -> float:
    """Return 0–1 robustness score from three scenario fit labels."""
    total = sum(
        _FIT_TO_SCORE.get(scenario_fit.get(k, "medium"), 0.6)
        for k in ("base_case", "upside_case", "downside_case")
    )
    return round(total / 3, 3)


# ---------------------------------------------------------------------------
# Scenario-specific risk and adjustment generation
# ---------------------------------------------------------------------------

def _scenario_risks_downside(rec: dict) -> list[str]:
    words = _word_bag(rec)
    risks: list[str] = []
    if words & _POWER_GRID_KW:
        risks.append("Grid interconnection delays may extend deployment timelines by 2+ years beyond plan")
        risks.append("Power capacity constraints could force site redesign or relocation after commitment")
    if words & _COOLING_KW:
        risks.append("Cooling supply chain shortages increase equipment lead times and inflate capital cost")
    if words & _CAPITAL_KW:
        risks.append("Tightening capital markets increase cost of project finance and may reduce IRR below hurdle")
    if not risks:
        risks.append("Regulatory and permitting delays may extend project timelines significantly")
        risks.append("Macro headwinds may compress returns and reduce risk appetite for long-duration commitments")
    return risks


def _scenario_adjustment_downside(rec: dict) -> str:
    words = _word_bag(rec)
    if words & _POWER_GRID_KW:
        return (
            "Prioritise power-secured sites with existing grid interconnection rights over greenfield locations; "
            "negotiate utility capacity reservations before committing capital."
        )
    if words & _COOLING_KW:
        return (
            "Accelerate liquid cooling procurement ahead of supply chain constraints; "
            "qualify multiple vendors and negotiate long-lead-time purchase commitments now."
        )
    if words & _CAPITAL_KW:
        return (
            "Structure investments as phased options rather than committed CapEx; "
            "preserve balance-sheet flexibility for opportunistic deployment when constraints ease."
        )
    return (
        "Sequence investments to preserve flexibility; establish explicit decision gates tied to "
        "power and permitting milestones before committing to major capital expenditures."
    )


def _scenario_adjustment_upside(rec: dict) -> str:
    return (
        "Accelerate timeline and scale of investment to capture the upside opportunity "
        "ahead of competitors; front-load capital commitments while supply chains are improving."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_scenarios(context: AgentContext) -> list[dict[str, Any]]:
    """Return 3 scenario dicts enriched with evidence IDs from the RO."""
    ro = context.research_object or {}
    ev_list = ro.get("evidence", [])
    all_ev_ids: list[str] = [
        e.get("evidence_id", "") if isinstance(e, dict) else getattr(e, "evidence_id", "")
        for e in ev_list
    ]
    all_ev_ids = [x for x in all_ev_ids if x][:9]  # cap at 9

    n = len(all_ev_ids)
    chunk = max(1, n // 3)

    scenarios: list[dict[str, Any]] = []
    for i, tmpl in enumerate(_SCENARIO_TEMPLATES):
        start = i * chunk
        ev_chunk = all_ev_ids[start: start + chunk] if n else []
        scenarios.append({
            **tmpl,
            "evidence_ids": ev_chunk,
            "supporting_evidence": ev_chunk,  # spec alias (J6.8a)
        })

    return scenarios


def stress_test_recommendations(
    recommendations: list[dict],
    scenarios: list[dict],
) -> list[dict[str, Any]]:
    """Return a stress-test record for each recommendation across all scenarios."""
    results: list[dict[str, Any]] = []
    for rec in recommendations:
        fit = _compute_scenario_fit(rec)
        robustness = _compute_robustness_score(fit)

        scenario_risks: dict[str, list[str]] = {
            "base_case":     [],
            "upside_case":   [],
            "downside_case": _scenario_risks_downside(rec),
        }

        adjustments: list[dict[str, str]] = []
        if fit.get("downside_case") in ("medium", "weak"):
            adjustments.append({
                "scenario":     "downside_case",
                "scenario_id":  "S3",
                "adjustment":   _scenario_adjustment_downside(rec),
            })
        if fit.get("upside_case") == "strong" and fit.get("base_case") == "strong":
            adjustments.append({
                "scenario":     "upside_case",
                "scenario_id":  "S2",
                "adjustment":   _scenario_adjustment_upside(rec),
            })

        # Build adjustments dict (spec J6.8a) alongside the existing list
        adjustments_dict: dict[str, str] = {
            a["scenario"]: a["adjustment"] for a in adjustments
        }

        results.append({
            "recommendation_id":    rec.get("id", ""),
            "title":                rec.get("title", ""),
            "scenario_fit":         fit,
            "robustness_score":     robustness,
            "scenario_risks":       scenario_risks,
            "scenario_adjustments": adjustments,   # list form (existing)
            "adjustments":          adjustments_dict,  # dict form (J6.8a)
        })

    return results


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ScenarioAgent(FunctionalAgent):
    """Generate plausible scenarios and stress-test recommendations (J6.8).

    Reads:
        context.recommendations
        context.research_object (for evidence IDs)

    Writes:
        context.scenarios
        context.scenario_analysis
        context.qa["scenario_validation"]
        context.research_object["scenarios"]
        context.research_object["scenario_analysis"]
        context.trace["_scenario_analysis"]
    """

    def _execute(self, context: AgentContext) -> AgentContext:
        from research_agent.log import PROGRESS

        recommendations = context.recommendations
        scenarios = generate_scenarios(context)
        rec_stress_test = stress_test_recommendations(recommendations, scenarios)

        n_recs = len(recommendations)
        avg_robustness = (
            round(sum(r["robustness_score"] for r in rec_stress_test) / n_recs, 3)
            if n_recs else 0.0
        )

        scenario_analysis: dict[str, Any] = {
            "scenarios": scenarios,
            "recommendation_stress_test": rec_stress_test,
            "summary": {
                "scenario_count": len(scenarios),
                "recommendations_stress_tested": n_recs,
                "average_robustness_score": avg_robustness,
            },
        }

        context.scenarios = scenarios
        context.scenario_analysis = scenario_analysis

        # QA validation (written to context.qa now; QAAgent preserves it via _validate_scenario)
        context.qa["scenario_validation"] = {
            "scenarios_present":                  len(scenarios) > 0,
            "scenario_count":                     len(scenarios),
            "recommendation_stress_test_present": n_recs > 0,
            "robustness_scores_present":          all(
                "robustness_score" in r for r in rec_stress_test
            ),
        }

        # Research object
        ro = context.research_object or {}
        if ro:
            ro["scenarios"] = scenarios
            ro["scenario_analysis"] = scenario_analysis

        # Trace
        context.trace["_scenario_analysis"] = {
            "scenario_agent": {
                "scenarios_generated":            len(scenarios),
                "recommendations_stress_tested":  n_recs,
                "average_robustness_score":       avg_robustness,
            },
        }

        summary = (
            f"Generated {len(scenarios)} scenarios; "
            f"stress-tested {n_recs} recommendations; "
            f"avg_robustness={avg_robustness:.3f}"
        )
        LOGGER.log(PROGRESS, "[ScenarioAgent] %s", summary)
        self._record(context, status="success", summary=summary)
        return context
