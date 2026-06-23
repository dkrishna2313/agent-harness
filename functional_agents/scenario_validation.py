"""Scenario Analysis Validation Harness (J6.8a).

Builds a realistic synthetic AgentContext representing an AI infrastructure
investment research run, exercises ScenarioAgent end-to-end, and writes
j68a_scenario_validation.trace.json as proof of implementation.

Public API
----------
SYNTHETIC_RECS         – 5 AI-infrastructure recommendations
build_validation_context() – build a synthetic AgentContext
run_scenario_validation()  – main runner; returns results dict
build_validation_report()  – markdown summary
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic recommendations — AI infrastructure investment domain
# ---------------------------------------------------------------------------

SYNTHETIC_RECS: list[dict[str, Any]] = [
    {
        "id": "R1",
        "title": "Establish Power-Secured Site Portfolio with Grid Interconnection Priority",
        "summary": (
            "AI data center operators must prioritise sites with existing or near-term power "
            "and grid interconnection rights. Power availability is the binding constraint for "
            "AI infrastructure deployment. However, securing grid interconnection agreements "
            "2–4 years in advance is critical while transmission capacity remains constrained. "
            "While capital commitments are large, early movers benefit from lower interconnection "
            "costs before congestion premiums rise."
        ),
        "priority": "high",
        "time_horizon": "near_term",
        "supported_by_hypotheses": ["H1", "H2"],
        "supporting_evidence": ["E001", "E004", "E007", "E012"],
        "key_risks": [
            "Grid interconnection queue delays may exceed 5 years in constrained markets",
            "Power purchase agreement pricing may escalate with renewable energy demand competition",
            "Transmission upgrade requirements may impose significant additional capital costs",
        ],
        "trigger_conditions": ["power availability < 500 MW within 24-month delivery window"],
        "confidence": "high",
        "confidence_rationale": (
            "Convergent evidence across utility interconnection studies, operator filings, and "
            "transmission constraint analyses confirms power as the primary bottleneck."
        ),
    },
    {
        "id": "R2",
        "title": "Deploy Direct Liquid Cooling Infrastructure as Standard for High-Density AI Racks",
        "summary": (
            "AI rack densities above 30 kW require direct liquid cooling as standard infrastructure. "
            "However, the capital cost of DLC retrofit is significant and requires facility redesign. "
            "While warm-water DLC reduces PUE by 0.2–0.4 points and supports GPU power envelopes, "
            "transition costs must be planned across 2–3 year deployment cycles. Tradeoffs between "
            "proprietary and open DLC standards affect long-term vendor flexibility."
        ),
        "priority": "high",
        "time_horizon": "near_term",
        "supported_by_hypotheses": ["H2", "H3"],
        "supporting_evidence": ["E002", "E005", "E008"],
        "key_risks": [
            "Cooling supply chain for liquid manifolds faces 12–18 month lead times",
            "Retrofit complexity for existing air-cooled facilities increases capital cost by 25–40%",
            "Vendor lock-in risk with proprietary DLC solutions limits future technology transitions",
        ],
        "trigger_conditions": ["rack density target exceeds 30 kW per rack"],
        "confidence": "high",
        "confidence_rationale": (
            "Hyperscale operator case studies and GPU vendor thermal requirements consistently "
            "confirm the 30 kW threshold above which air cooling becomes technically infeasible."
        ),
    },
    {
        "id": "R3",
        "title": "Structure AI Infrastructure CapEx as Phased Options to Preserve Flexibility",
        "summary": (
            "Capital investments in AI infrastructure should be structured as staged commitments "
            "with defined decision gates rather than large upfront CapEx. However, the capital "
            "intensity of power and cooling infrastructure requires multi-year commitments that "
            "constrain optionality. While phased deployment reduces stranded asset risk, it may "
            "sacrifice scale economics. Tradeoffs between cost efficiency and strategic flexibility "
            "must be explicitly evaluated at each investment gate."
        ),
        "priority": "high",
        "time_horizon": "near_term",
        "supported_by_hypotheses": ["H1", "H4"],
        "supporting_evidence": ["E003", "E009", "E011"],
        "key_risks": [
            "Phased deployment may sacrifice scale economies available to larger committed builds",
            "Capital market conditions may tighten between phases, increasing financing costs",
            "Competitor front-loading of investment may capture scarce power and land inventory",
        ],
        "trigger_conditions": [
            "capital budget approval for Phase 1 committed",
            "Phase 2 gate: rack utilization exceeds 75% and power secured for expansion",
        ],
        "confidence": "high",
        "confidence_rationale": (
            "Portfolio theory and real-options analysis applied to infrastructure investments "
            "consistently support staged commitment under uncertainty."
        ),
    },
    {
        "id": "R4",
        "title": "Commission Regional Inference Capacity with Grid-Resilient Power Design",
        "summary": (
            "AI operators must commission regional inference clusters to meet latency requirements "
            "and geographic demand growth. However, regional sites face greater grid constraint "
            "variability than established power markets. While distributed inference reduces "
            "concentration risk, each site requires independent power and transmission assessment. "
            "Grid resilience design — including on-site generation and battery buffer — is essential "
            "to manage regional power intermittency."
        ),
        "priority": "high",
        "time_horizon": "medium_term",
        "supported_by_hypotheses": ["H3", "H5"],
        "supporting_evidence": ["E006", "E010", "E013"],
        "key_risks": [
            "Regional grid interconnection timelines may exceed 4 years in underserved markets",
            "Power contract terms in emerging markets may lack credit quality for long-term planning",
            "Latency and reliability SLA requirements may conflict with lower-cost regional power availability",
        ],
        "trigger_conditions": ["latency SLA thresholds exceeded in target region"],
        "confidence": "medium",
        "confidence_rationale": (
            "Regional demand growth supported by cloud provider capacity announcements; "
            "grid constraint data is market-specific and requires site-by-site validation."
        ),
    },
    {
        "id": "R5",
        "title": "Develop Proactive Regulatory and Permitting Strategy for Power-Intensive Facilities",
        "summary": (
            "AI data center operators must develop a proactive engagement strategy with utility "
            "regulators and permitting authorities. However, regulatory processes vary significantly "
            "across jurisdictions and timelines are difficult to predict. While early engagement "
            "reduces delay risk, it requires dedicated government-relations resources. Tradeoffs "
            "between jurisdictions with permitting-friendly environments and those with lower power "
            "costs must be explicitly evaluated in site selection."
        ),
        "priority": "medium",
        "time_horizon": "near_term",
        "supported_by_hypotheses": ["H1", "H5"],
        "supporting_evidence": ["E014", "E015"],
        "key_risks": [
            "Regulatory opposition to large power loads may extend permitting timelines by 2–3 years",
            "Changes in utility interconnection rules may impose retroactive costs on approved projects",
        ],
        "trigger_conditions": ["site acquisition decision for >50 MW load facility"],
        "confidence": "medium",
        "confidence_rationale": (
            "Regulatory risk is well-documented in utility filings and operator commentary; "
            "outcomes vary substantially by jurisdiction."
        ),
    },
]

# Synthetic evidence IDs representing the research corpus
_EVIDENCE_IDS = [f"E{i:03d}" for i in range(1, 16)]


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_validation_context():
    """Return a synthetic AgentContext representing an AI infrastructure run."""
    from .context import AgentContext

    ctx = AgentContext(goal="Develop a strategy for AI infrastructure investment over the next decade.")
    ctx.recommendations = SYNTHETIC_RECS
    ctx.hypotheses = [
        {"hypothesis_id": "H1", "statement": "Power availability is the primary constraint on AI infrastructure deployment."},
        {"hypothesis_id": "H2", "statement": "Rack densities above 30 kW require liquid cooling as the default thermal solution."},
        {"hypothesis_id": "H3", "statement": "Regional inference capacity is required to meet latency and availability requirements."},
        {"hypothesis_id": "H4", "statement": "Capital structure flexibility reduces stranded asset risk under demand uncertainty."},
        {"hypothesis_id": "H5", "statement": "Regulatory and permitting risk is a material constraint on deployment timelines."},
    ]
    ctx.hypothesis_challenges = []
    ctx.research_object = {
        "research_id": "J68A-VALIDATION",
        "question": "Develop a strategy for AI infrastructure investment over the next decade.",
        "profiles": ["ai_data_centers", "power", "transmission"],
        "evidence": [{"evidence_id": eid} for eid in _EVIDENCE_IDS],
    }
    return ctx


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_scenario_validation(out_path: Path | str | None = None) -> dict[str, Any]:
    """Run ScenarioAgent on synthetic AI infrastructure context.

    Parameters
    ----------
    out_path:
        Write ``j68a_scenario_validation.trace.json`` to this directory.
        Defaults to ``outputs/``.

    Returns
    -------
    Full validation results dict.
    """
    from .scenario_agent import ScenarioAgent, generate_scenarios, stress_test_recommendations
    from .report_agent import _build_scenario_section

    ctx = build_validation_context()

    # Run ScenarioAgent
    agent = ScenarioAgent()
    ctx = agent._execute(ctx)

    scenario_analysis = ctx.scenario_analysis
    scenarios = ctx.scenarios
    rec_stress_test = scenario_analysis.get("recommendation_stress_test", [])
    summary_block = scenario_analysis.get("summary", {})

    # Build QA validation summary
    qa_validation = ctx.qa.get("scenario_validation", {})

    # Build markdown report section
    report_section = _build_scenario_section(scenario_analysis)

    results: dict[str, Any] = {
        "scenarios": scenarios,
        "recommendation_stress_test": rec_stress_test,
        "scenario_analysis_summary": summary_block,
        "qa_validation": qa_validation,
        "report_section_preview": report_section[:2000],
        "research_object_keys": list(ctx.research_object.keys()),
    }

    if out_path:
        _write_trace(ctx, results, Path(out_path))

    return results


# ---------------------------------------------------------------------------
# Trace writer
# ---------------------------------------------------------------------------

def _write_trace(ctx, results: dict[str, Any], out_dir: Path) -> None:
    """Write j68a_scenario_validation.trace.json and update latest_research_object.json."""
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_payload = {
        "j68a_scenario_validation": {
            "schema_version": "1.0",
            "goal": "Develop a strategy for AI infrastructure investment over the next decade.",
            "profiles": ["ai_data_centers", "power", "transmission"],
        },
        "scenario_agent": ctx.trace.get("_scenario_analysis", {}).get("scenario_agent", {}),
        "scenarios": [
            {
                "scenario_id": s["scenario_id"],
                "name": s["name"],
                "description": s["description"],
                "assumptions": s["assumptions"],
                "critical_uncertainties": s["critical_uncertainties"],
                "probability": s["probability"],
                "supporting_evidence": s.get("supporting_evidence", []),
            }
            for s in ctx.scenarios
        ],
        "recommendation_stress_test": [
            {
                "recommendation_id": r["recommendation_id"],
                "title": r["title"],
                "scenario_fit": r["scenario_fit"],
                "robustness_score": r["robustness_score"],
                "adjustments": r.get("adjustments", {}),
                "downside_risks": r.get("scenario_risks", {}).get("downside_case", []),
            }
            for r in results["recommendation_stress_test"]
        ],
        "qa_validation": results["qa_validation"],
        "summary": results["scenario_analysis_summary"],
    }

    trace_path = out_dir / "j68a_scenario_validation.trace.json"
    trace_path.write_text(json.dumps(trace_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("[ScenarioValidation] Trace written to %s", trace_path)

    # Also update latest_research_object.json to show scenario fields
    ro = ctx.research_object
    latest_path = out_dir / "latest_research_object.json"

    # Load existing if present, else start from scratch
    existing: dict[str, Any] = {}
    if latest_path.exists():
        try:
            existing = json.loads(latest_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    existing.update({
        "scenarios": ro.get("scenarios", []),
        "scenario_analysis": ro.get("scenario_analysis", {}),
        "_j68a_validation_note": (
            "Scenario fields populated by J6.8a validation harness using synthetic "
            "AI infrastructure recommendations; full pipeline RO written on functional run."
        ),
    })
    latest_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("[ScenarioValidation] latest_research_object.json updated with scenario fields")


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def build_validation_report(results: dict[str, Any]) -> str:
    """Return a human-readable validation summary."""
    scenarios = results.get("scenarios", [])
    stress = results.get("recommendation_stress_test", [])
    summary = results.get("scenario_analysis_summary", {})
    qa = results.get("qa_validation", {})

    lines: list[str] = [
        "# J6.8a Scenario Validation Report",
        "",
        f"**Scenarios Generated:** {summary.get('scenario_count', 0)}",
        f"**Recommendations Stress-Tested:** {summary.get('recommendations_stress_tested', 0)}",
        f"**Average Robustness Score:** {summary.get('average_robustness_score', 0):.3f}",
        f"**QA Loop Validated:** {'✓' if qa.get('scenarios_present') else '✗'}",
        "",
        "## Scenarios",
        "",
        "| ID | Name | AI Demand | Power | Grid | Probability |",
        "|----|------|-----------|-------|------|-------------|",
    ]
    for s in scenarios:
        sid = s["scenario_id"]
        name = s["name"]
        assum = s.get("assumptions", {})
        demand = assum.get("ai_demand_growth", "—").replace("_", " ")
        power = assum.get("power_availability", "—").replace("_", " ")
        grid = assum.get("grid_interconnection_timelines", "—").replace("_", " ").split(" ")[0]
        prob = f"{int(s.get('probability', 0) * 100)}%"
        lines.append(f"| {sid} | {name} | {demand} | {power} | {grid} | {prob} |")

    lines += [
        "",
        "## Recommendation Robustness",
        "",
        "| Rec | Title (abbrev) | Base | Upside | Downside | Robustness |",
        "|-----|----------------|------|--------|----------|------------|",
    ]
    for r in stress:
        rid = r["recommendation_id"]
        title = r.get("title", "")[:45] + ("…" if len(r.get("title", "")) > 45 else "")
        fit = r.get("scenario_fit", {})
        rob = r.get("robustness_score", 0.0)
        lines.append(
            f"| {rid} | {title} | {fit.get('base_case', '—')} | "
            f"{fit.get('upside_case', '—')} | {fit.get('downside_case', '—')} | **{rob:.3f}** |"
        )

    # Adjustments
    adj_rows = [
        (r["recommendation_id"], scenario, text)
        for r in stress
        for scenario, text in r.get("adjustments", {}).items()
    ]
    if adj_rows:
        lines += [
            "",
            "## Scenario Adjustments",
            "",
            "| Rec | Scenario | Adjustment |",
            "|-----|----------|------------|",
        ]
        for rid, sc, adj in adj_rows:
            lines.append(f"| {rid} | {sc} | {adj[:100]}{'…' if len(adj) > 100 else ''} |")

    lines += [
        "",
        "## QA Validation",
        "",
        f"- scenarios_present: {qa.get('scenarios_present')}",
        f"- scenario_count: {qa.get('scenario_count')}",
        f"- recommendation_stress_test_present: {qa.get('recommendation_stress_test_present')}",
        f"- robustness_scores_present: {qa.get('robustness_scores_present')}",
        "",
    ]
    return "\n".join(lines)
