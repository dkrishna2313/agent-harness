"""ReportAgent – synthesis, narrative construction, and output writing (J5.0b / J5.4)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Narrative synthesis helpers (J5.4)
# ---------------------------------------------------------------------------

def _build_hypotheses_section(hypotheses: list[dict[str, Any]]) -> str:
    """Render the Strategic Hypotheses section for the markdown report (J6.3)."""
    if not hypotheses:
        return ""

    lines = [
        "## Strategic Hypotheses",
        "",
        "> These are candidate interpretations of the evidence, not established facts.",
        "> They should be evaluated and tested before informing decisions.",
        "",
        "| Hypothesis | Confidence | Supporting Evidence | Key Uncertainty |",
        "|---|---|---|---|",
    ]

    for h in hypotheses:
        hid = h.get("id", "?")
        title = h.get("title", "")
        conf = h.get("confidence", "—").capitalize()
        sup_ev = ", ".join(h.get("supporting_evidence", [])[:3]) or "—"
        gaps = h.get("evidence_gaps", [])
        uncertainty = gaps[0] if gaps else (h.get("confidence_rationale", "")[:60] or "—")
        lines.append(f"| **{hid}** — {title} | {conf} | {sup_ev} | {uncertainty} |")

    lines.append("")

    for h in hypotheses:
        hid = h.get("id", "?")
        title = h.get("title", "")
        summary = h.get("summary", "")
        conf = h.get("confidence", "—").capitalize()
        rationale = h.get("confidence_rationale", "")
        implications = h.get("decision_implications", [])
        disconfirming = h.get("disconfirming_evidence_needed", [])
        con_ev = h.get("contradicting_evidence", [])

        lines += [
            f"### {hid}: {title}",
            "",
            summary,
            "",
            f"**Confidence:** {conf}  ",
            f"*{rationale}*",
            "",
        ]
        if implications:
            lines.append("**Decision Implications:**")
            lines.extend(f"- {imp}" for imp in implications)
            lines.append("")
        if con_ev:
            lines.append(f"**Contradicting Evidence:** {', '.join(con_ev)}")
            lines.append("")
        if disconfirming:
            lines.append("**This hypothesis would be weakened by:**")
            lines.extend(f"- {d}" for d in disconfirming)
            lines.append("")

    return "\n".join(lines)


def _build_challenges_section(
    challenges: list[dict[str, Any]],
    surviving: list[dict[str, Any]],
) -> str:
    """Render the Hypothesis Challenges section for the markdown report (J6.4)."""
    if not challenges:
        return ""

    # Build a survival lookup
    survival_by_id: dict[str, dict] = {s["hypothesis_id"]: s for s in surviving}

    lines = [
        "## Hypothesis Challenges",
        "",
        "> The Challenge Agent adversarially stress-tested each hypothesis.",
        "> Robustness scores and survival statuses reflect weaknesses in evidence and assumptions.",
        "",
        "| Hypothesis | Robustness | Key Challenge | Survival Status |",
        "|---|---|---|---|",
    ]

    for c in challenges:
        hid = c.get("hypothesis_id", "?")
        robustness = c.get("robustness", "—").capitalize()
        summary = c.get("challenge_summary", "")[:80]
        sv = survival_by_id.get(hid, {})
        status = sv.get("survival_status", "—").capitalize()
        lines.append(f"| **{hid}** | {robustness} | {summary} | {status} |")

    lines.append("")

    for c in challenges:
        hid = c.get("hypothesis_id", "?")
        sv = survival_by_id.get(hid, {})
        robustness = c.get("robustness", "—").capitalize()
        summary = c.get("challenge_summary", "")
        assumptions = c.get("hidden_assumptions", [])
        weak_ev = c.get("weak_evidence", [])
        contra_ev = c.get("contradicting_evidence", [])
        missing = c.get("missing_evidence", [])
        falsification = c.get("falsification_tests", [])
        survival_status = sv.get("survival_status", "—").capitalize()
        survival_reason = sv.get("reason", "")

        lines += [
            f"### Challenge: {hid}",
            "",
            summary,
            "",
            f"**Robustness:** {robustness}  |  **Survival:** {survival_status}",
            "",
            f"*{survival_reason}*",
            "",
        ]
        if assumptions:
            lines.append("**Hidden Assumptions:**")
            lines.extend(f"- {a}" for a in assumptions)
            lines.append("")
        if weak_ev:
            lines.append("**Weak Evidence:**")
            lines.extend(f"- {w}" for w in weak_ev)
            lines.append("")
        if contra_ev:
            ev_str = ", ".join(contra_ev) if isinstance(contra_ev, list) else str(contra_ev)
            lines.append(f"**Contradicting Evidence:** {ev_str}")
            lines.append("")
        if missing:
            lines.append("**Missing Evidence:**")
            lines.extend(f"- {m}" for m in missing)
            lines.append("")
        if falsification:
            lines.append("**Falsification Tests:**")
            lines.extend(f"- {f}" for f in falsification)
            lines.append("")

    return "\n".join(lines)


def _build_recommendation_evaluation_section(rec_eval: dict) -> str:
    """Render a Recommendation Evaluation table section for the markdown report (J6.6a)."""
    if not rec_eval:
        return ""
    scored = rec_eval.get("recommendation_scores", [])
    if not scored:
        return ""

    lines = [
        "## Recommendation Evaluation",
        "",
        "| Recommendation | Evidence | Reasoning | Tradeoff | Risk | Actionability | Score |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in scored:
        rid = s.get("recommendation_id", "")
        title = s.get("title", "")
        label = f"{rid}: {title[:55]}…" if len(title) > 55 else f"{rid}: {title}"
        penalty = f" ⚠ {s['primary_penalty']}" if s.get("primary_penalty") else ""
        lines.append(
            f"| {label}{penalty} "
            f"| {s.get('evidence_support_score', 0):.2f} "
            f"| {s.get('reasoning_score', 0):.2f} "
            f"| {s.get('tradeoff_score', 0):.2f} "
            f"| {s.get('risk_score', 0):.2f} "
            f"| {s.get('actionability_score', 0):.2f} "
            f"| **{s.get('aggregate_score', s.get('recommendation_score', 0)):.3f}** |"
        )

    agg = rec_eval.get("aggregate", {})
    rec_summary = rec_eval.get("recommendation_summary", {})
    warnings = rec_eval.get("recommendation_warnings", [])

    lines += [
        "",
        f"**Portfolio score:** {agg.get('recommendation_score', 0):.3f}  "
        f"| Lowest: {rec_summary.get('lowest_score', 0):.3f}  "
        f"| Highest: {rec_summary.get('highest_score', 0):.3f}",
    ]

    if warnings:
        lines += ["", "**Warnings:**", ""]
        for w in warnings:
            lines.append(
                f"- `{w['recommendation_id']}` — {w['issue']} "
                f"(score: {w.get('aggregate_score', 0):.3f})"
            )

    return "\n".join(lines)


def _build_recommendation_improvement_section(improvement: dict) -> str:
    """Render a Recommendation Improvement table section for the markdown report (J6.7)."""
    records = improvement.get("improvement_records", [])
    metrics = improvement.get("improvement_metrics", {})
    if not records:
        return ""

    lines = [
        "## Recommendation Improvement",
        "",
        "| Recommendation | Before | After | Delta | Weakness Addressed |",
        "|---|---|---|---|---|",
    ]
    for r in records:
        rid = r.get("recommendation_id", "")
        before = r.get("before_score", 0.0)
        after = r.get("after_score", 0.0)
        delta = r.get("delta", 0.0)
        sign = "+" if delta >= 0 else ""
        weaknesses = ", ".join(r.get("weaknesses_addressed", []))
        lines.append(
            f"| {rid} | {before:.3f} | {after:.3f} | {sign}{delta:.3f} | {weaknesses} |"
        )

    lines += [
        "",
        f"**{metrics.get('recommendations_improved', 0)} of "
        f"{metrics.get('recommendations_improved', 0) + metrics.get('recommendations_unchanged', 0)} "
        f"recommendations improved.** "
        f"Average score: {metrics.get('average_score_before', 0):.3f} → "
        f"{metrics.get('average_score_after', 0):.3f} "
        f"(Δ {metrics.get('average_delta', 0):+.3f})",
    ]
    return "\n".join(lines)


def _build_scenario_section(scenario_analysis: dict) -> str:
    """Render a Scenario Analysis section for the markdown report (J6.8 / J6.8a)."""
    scenarios = scenario_analysis.get("scenarios", [])
    rec_stress_test = scenario_analysis.get("recommendation_stress_test", [])
    summary = scenario_analysis.get("summary", {})
    if not scenarios:
        return ""

    lines: list[str] = [
        "## Scenario Analysis",
        "",
        f"Three plausible futures were evaluated across "
        f"{summary.get('recommendations_stress_tested', 0)} recommendations. "
        f"Average robustness score: **{summary.get('average_robustness_score', 0):.3f}**.",
        "",
    ]

    # Per-scenario subsections
    for s in scenarios:
        name = s.get("name", "")
        sid = s.get("scenario_id", "")
        desc = s.get("description", "")
        prob = f"{int(s.get('probability', 0) * 100)}%"
        assum = s.get("assumptions", {})
        uncertainties = s.get("critical_uncertainties", [])

        lines += [
            f"### {name}",
            "",
            desc,
            "",
            f"**Probability:** {prob}",
            "",
            "**Key Assumptions:**",
            "",
        ]
        for k, v in assum.items():
            label = k.replace("_", " ").title()
            val = str(v).replace("_", " ")
            lines.append(f"- {label}: {val}")
        if uncertainties:
            lines += ["", "**Critical Uncertainties:**", ""]
            for u in uncertainties:
                lines.append(f"- {u}")
        lines.append("")

    # Summary table
    lines += [
        "### Scenarios Summary",
        "",
        "| Scenario | AI Demand | Power | Grid Timelines | Probability |",
        "|----------|-----------|-------|----------------|-------------|",
    ]
    for s in scenarios:
        name = s.get("name", "")
        assum = s.get("assumptions", {})
        demand = assum.get("ai_demand_growth", "—").replace("_", " ")
        power = assum.get("power_availability", "—").replace("_", " ")
        grid = assum.get("grid_interconnection_timelines", "—").replace("_", " ")
        prob = f"{int(s.get('probability', 0) * 100)}%"
        lines.append(f"| {name} | {demand} | {power} | {grid} | {prob} |")

    if rec_stress_test:
        lines += [
            "",
            "### Recommendation Robustness",
            "",
            "| Recommendation | Base Case | Upside Case | Downside Case | Robustness |",
            "|----------------|-----------|-------------|---------------|------------|",
        ]
        for r in rec_stress_test:
            rid = r.get("recommendation_id", "")
            fit = r.get("scenario_fit", {})
            rob = r.get("robustness_score", 0.0)
            lines.append(
                f"| {rid} | {fit.get('base_case', '—')} | "
                f"{fit.get('upside_case', '—')} | "
                f"{fit.get('downside_case', '—')} | "
                f"**{rob:.3f}** |"
            )

    # Downside adjustments
    downside_adj = [
        (r["recommendation_id"], r["adjustments"].get("downside_case", ""))
        for r in rec_stress_test
        if r.get("adjustments", {}).get("downside_case")
    ]
    upside_adj = [
        (r["recommendation_id"], r["adjustments"].get("upside_case", ""))
        for r in rec_stress_test
        if r.get("adjustments", {}).get("upside_case")
    ]
    if downside_adj:
        lines += [
            "",
            "### Downside Case — Scenario Adjustments",
            "",
            "| Recommendation | Adjustment |",
            "|----------------|------------|",
        ]
        for rid, adj in downside_adj:
            lines.append(f"| {rid} | {adj} |")
    if upside_adj:
        lines += [
            "",
            "### Upside Case — Scenario Adjustments",
            "",
            "| Recommendation | Adjustment |",
            "|----------------|------------|",
        ]
        for rid, adj in upside_adj:
            lines.append(f"| {rid} | {adj} |")

    lines.append("")
    return "\n".join(lines)


def _build_profile_synthesis_section(
    multi_profile_analysis: dict[str, Any],
    recommendations: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    synthesis_tradeoffs: list[dict[str, Any]] | None = None,
) -> str:
    """Render Multi-Profile Synthesis section (J6.8b).

    Emits per-profile finding groups, integrated recommendations, tradeoff
    narrative, profile balance table, and synthesis coverage validation.
    Only rendered when ≥2 profiles contributed.
    """
    profiles_contributing: list[str] = multi_profile_analysis.get("profiles_contributing", [])
    if len(profiles_contributing) < 2:
        return ""

    attributed_findings: list[dict] = multi_profile_analysis.get("attributed_findings", hypotheses)
    attributed_recs: list[dict] = multi_profile_analysis.get("attributed_recommendations", recommendations)
    profile_balance: dict[str, float] = multi_profile_analysis.get("profile_balance", {})
    synthesis_val: dict[str, int] = multi_profile_analysis.get("synthesis_validation", {})
    rec_audit: dict[str, list] = multi_profile_analysis.get("recommendation_profile_audit", {})
    profile_influence: dict = multi_profile_analysis.get("profile_influence", {})

    lines: list[str] = [
        "## Multi-Profile Synthesis",
        "",
        "> This section synthesises findings and recommendations across all contributing profiles, "
        "identifies cross-profile tradeoffs, and reports coverage balance.",
        "",
    ]

    # Per-profile perspective subsections
    for profile in profiles_contributing:
        label = profile.replace("_", " ").title()
        lines += [f"### {label} Perspective", ""]

        # Findings attributed to this profile
        profile_findings = [
            f for f in attributed_findings
            if profile in f.get("contributing_profiles", [])
        ]
        if profile_findings:
            lines.append("**Key Findings:**")
            for f in profile_findings[:3]:
                title = f.get("title", f.get("summary", ""))[:120]
                lines.append(f"- {title}")
            lines.append("")
        else:
            lines.append("*No findings exclusively attributed to this profile.*")
            lines.append("")

        # Recommendations attributed to this profile
        profile_recs = [
            r for r in attributed_recs
            if profile in r.get("contributing_profiles", [])
        ]
        if profile_recs:
            lines.append("**Recommendations:**")
            for r in profile_recs[:3]:
                rid = r.get("id", r.get("recommendation_id", ""))
                title = r.get("title", "")[:100]
                lines.append(f"- **{rid}**: {title}")
            lines.append("")

        # Evidence count from influence
        ev_count = profile_influence.get(profile, {}).get("evidence", 0)
        lines.append(f"*Evidence items: {ev_count}*")
        lines.append("")

    # Integrated recommendations (attributed to 2+ profiles)
    integrated = [
        r for r in attributed_recs
        if len(r.get("contributing_profiles", [])) >= 2
    ]
    lines += ["### Integrated Strategy", ""]
    if integrated:
        lines.append(
            "The following recommendations draw on evidence from multiple profiles, "
            "integrating compute and grid infrastructure considerations:"
        )
        lines.append("")
        for r in integrated:
            rid = r.get("id", r.get("recommendation_id", ""))
            title = r.get("title", "")
            profiles_str = ", ".join(r.get("contributing_profiles", []))
            lines.append(f"- **{rid}** ({profiles_str}): {title}")
        lines.append("")
    else:
        lines.append(
            "No single recommendation was attributed to all contributing profiles. "
            "See individual profile sections above for profile-specific recommendations."
        )
        lines.append("")

    # Tradeoffs — use structured tradeoffs from RecommendationSynthesisAgent if available
    lines += ["### Tradeoffs", ""]
    has_ai_dc = "ai_data_centers" in profiles_contributing
    has_tx    = "transmission"    in profiles_contributing
    if synthesis_tradeoffs:
        for t in synthesis_tradeoffs:
            dim_a = t.get("dimension_a", "")
            dim_b = t.get("dimension_b", "")
            desc  = t.get("description", "")
            impl  = t.get("implication", "")
            lines += [
                f"**{dim_a} vs. {dim_b}**",
                "",
                desc,
                "",
            ]
            if impl:
                lines += [f"*Implication:* {impl}", ""]
    elif has_ai_dc and has_tx:
        lines += [
            "**Compute availability vs. Grid availability**",
            "",
            "The highest-performing AI factory locations may not have sufficient "
            "transmission capacity to serve high-density GPU clusters (100–1,000+ MW). "
            "Interconnection queue timelines of 3–6 years mean grid access must be "
            "secured years before facility commissioning.",
            "",
            "**Site selection implication:** optimise jointly for compute readiness "
            "and grid readiness. Single-dimension optimisation surfaces sites that are "
            "compute-ready but grid-constrained or grid-connected but compute-unsuitable.",
            "",
            "**Capital sequencing implication:** grid access investments have long "
            "lead times and low sunk costs; they should precede large compute commitments.",
            "",
        ]
    else:
        lines.append("No cross-profile tradeoffs identified for the current profile combination.")
        lines.append("")

    # Profile balance table
    if profile_balance:
        lines += [
            "### Profile Balance",
            "",
            "Recommendation weight by contributing profile:",
            "",
            "| Profile | Weight |",
            "|---------|--------|",
        ]
        for p, frac in sorted(profile_balance.items()):
            label = p.replace("_", " ").title()
            lines.append(f"| {label} | {frac:.0%} |")
        lines.append("")

    # Synthesis validation block
    if synthesis_val:
        lines += [
            "### Synthesis Coverage Validation",
            "",
            "| Metric | Count |",
            "|--------|-------|",
        ]
        for key, val in synthesis_val.items():
            label = key.replace("_", " ").title()
            lines.append(f"| {label} | {val} |")
        lines.append("")

    # Recommendation profile audit
    if rec_audit:
        lines += [
            "### Recommendation Profile Audit",
            "",
            "| Recommendation | Contributing Profiles |",
            "|----------------|----------------------|",
        ]
        for rid, profiles in sorted(rec_audit.items()):
            ps = ", ".join(profiles) if profiles else "—"
            lines.append(f"| {rid} | {ps} |")
        lines.append("")

    return "\n".join(lines)


def _write_recommendation_observability_trace(rec_eval: dict, report_path: "Path") -> None:
    """Write j66a_recommendation_observability.trace.json alongside the report (J6.6a)."""
    import json
    from pathlib import Path

    _stem = Path(report_path).stem
    trace_path = Path(report_path).parent / f"{_stem}.recommendation_observability.trace.json"
    latest_trace_path = Path(report_path).parent / f"{_stem}.recommendation_observability.latest.json"

    scored = rec_eval.get("recommendation_scores", [])
    agg = rec_eval.get("aggregate", {})
    rec_summary = rec_eval.get("recommendation_summary", {})
    warnings = rec_eval.get("recommendation_warnings", [])

    payload = {
        "trace_type": "recommendation_observability",
        "recommendation_evaluation": {
            "recommendation_summary": rec_summary,
            "recommendation_score": agg.get("recommendation_score", 0.0),
            "recommendation_dimension_summary": {
                "evidence_support": agg.get("mean_evidence_support", 0.0),
                "reasoning": agg.get("mean_reasoning", 0.0),
                "tradeoff": agg.get("mean_tradeoff", 0.0),
                "risk": agg.get("mean_risk", 0.0),
                "actionability": agg.get("mean_actionability", 0.0),
            },
            "recommendation_warnings": warnings,
            "per_recommendation": [
                {
                    "recommendation_id": s.get("recommendation_id", ""),
                    "title": s.get("title", ""),
                    "evidence_support_score": s.get("evidence_support_score", 0.0),
                    "reasoning_score": s.get("reasoning_score", 0.0),
                    "tradeoff_score": s.get("tradeoff_score", 0.0),
                    "risk_score": s.get("risk_score", 0.0),
                    "actionability_score": s.get("actionability_score", 0.0),
                    "aggregate_score": s.get("aggregate_score", s.get("recommendation_score", 0.0)),
                    "missing_evidence_links": s.get("missing_evidence_links", False),
                    "primary_penalty": s.get("primary_penalty"),
                    "supporting_evidence": s.get("traceability", {}).get("evidence_ids", []),
                    "supporting_hypotheses": s.get("traceability", {}).get("hypothesis_ids", []),
                    "supporting_challenges": s.get("traceability", {}).get("challenge_ids", []),
                }
                for s in scored
            ],
            "traceability": rec_eval.get("traceability", []),
        },
    }

    trace_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    latest_trace_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_strategic_options_section(
    strategic_options: list[dict[str, Any]],
    comparison: dict[str, Any],
    robustness: dict[str, Any],
    preferred: dict[str, Any],
    portfolio: dict[str, Any],
) -> str:
    """Render Strategic Options section for the markdown report (J7.1)."""
    if not strategic_options:
        return ""

    criteria = comparison.get("criteria", [])
    scores = comparison.get("scores", {})

    lines: list[str] = [
        "## Strategic Options",
        "",
        "> The following strategic options represent distinct investable paths. "
        "Each is a different strategic posture — not minor variations — and should "
        "be evaluated against the organisation's risk appetite and resource position.",
        "",
    ]

    # --- Option summaries ---
    for opt in strategic_options:
        oid = opt.get("option_id", "?")
        title = opt.get("title", "")
        posture = opt.get("posture", "").replace("_", " ").title()
        logic = opt.get("strategic_logic", "")
        where = opt.get("where_to_play", "")
        how = opt.get("how_to_win", "")
        risks = opt.get("risks", [])
        caps = opt.get("required_capabilities", [])
        deps = opt.get("dependencies", [])
        horizon = opt.get("time_horizon", "near_term").replace("_", " ")
        supporting_recs = opt.get("supporting_recommendations", [])
        profiles = opt.get("contributing_profiles", [])

        lines += [
            f"### {oid}: {title}",
            "",
            f"**Posture:** {posture}  |  **Time Horizon:** {horizon}",
            "",
            f"**Strategic Logic:** {logic}",
            "",
            f"**Where to Play:** {where}",
            "",
            f"**How to Win:** {how}",
            "",
        ]
        if caps:
            lines.append("**Required Capabilities:**")
            lines.extend(f"- {c}" for c in caps)
            lines.append("")
        if deps:
            lines.append("**Dependencies:**")
            lines.extend(f"- {d}" for d in deps)
            lines.append("")
        if risks:
            lines.append("**Key Risks:**")
            lines.extend(f"- {r}" for r in risks)
            lines.append("")
        if supporting_recs:
            lines.append(f"**Supporting Recommendations:** {', '.join(supporting_recs)}")
            lines.append("")
        if profiles:
            lines.append(f"**Contributing Profiles:** {', '.join(profiles)}")
            lines.append("")

    # --- Option comparison matrix ---
    lines += [
        "## Option Comparison",
        "",
    ]
    if criteria and scores:
        header = "| Criterion | " + " | ".join(scores.keys()) + " |"
        divider = "|---|" + "---|" * len(scores)
        lines += [header, divider]
        for criterion in criteria:
            row = f"| {criterion.replace('_', ' ').title()} | "
            row += " | ".join(
                scores.get(oid, {}).get(criterion, "—") for oid in scores
            ) + " |"
            lines.append(row)
        lines.append("")

    # --- Scenario robustness ---
    if robustness:
        lines += [
            "## Scenario Sensitivities",
            "",
        ]
        # Collect all scenario names across options
        all_scenarios: list[str] = []
        for sc_map in robustness.values():
            for sc in sc_map:
                if sc not in all_scenarios:
                    all_scenarios.append(sc)

        if all_scenarios:
            opt_ids = list(robustness.keys())
            header = "| Scenario | " + " | ".join(opt_ids) + " |"
            divider = "|---|" + "---|" * len(opt_ids)
            lines += [header, divider]
            for sc in all_scenarios:
                row = f"| {sc.replace('_', ' ').title()} | "
                row += " | ".join(
                    str(robustness.get(oid, {}).get(sc, "—")) for oid in opt_ids
                ) + " |"
                lines.append(row)
            lines.append("")

    # --- Preferred option ---
    lines += ["## Preferred Strategic Posture", ""]
    pref_id = preferred.get("option_id", "none")
    pref_rationale = preferred.get("rationale", "")
    pref_recommendation = preferred.get("recommendation", "")
    if pref_id == "portfolio":
        portfolio_options = preferred.get("portfolio_options", [])
        lines += [
            f"**Preferred:** Portfolio blend of {', '.join(portfolio_options)}",
            "",
            pref_recommendation,
            "",
        ]
    else:
        lines += [
            f"**Preferred Option:** {pref_id}",
            "",
            pref_recommendation,
            "",
        ]
    if pref_rationale:
        lines += [f"*{pref_rationale}*", ""]

    # --- Option portfolio ---
    if portfolio:
        lines += ["## Recommended Next Moves", "", "**Staged Option Portfolio:**", ""]
        for horizon, option_ids in portfolio.items():
            if option_ids:
                horizon_label = horizon.replace("_", " ").title()
                lines.append(f"- **{horizon_label}:** {', '.join(option_ids)}")
        lines.append("")

    return "\n".join(lines)


def _build_recommendations_section(
    recommendations: list[dict[str, Any]],
    portfolio: dict[str, Any],
) -> str:
    """Render the Strategic Recommendations section for the markdown report (J6.5)."""
    if not recommendations:
        return ""

    _HORIZON_LABEL = {
        "near_term": "Near-term (2026–2030)",
        "medium_term": "Medium-term (2030–2035)",
        "long_term": "Long-term (2035+)",
    }

    lines = [
        "## Strategic Recommendations",
        "",
        "> Recommendations are derived from surviving hypotheses and are grounded in evidence.",
        "> They should be reviewed against the Hypothesis Challenges section before acting.",
        "",
        "| Recommendation | Priority | Confidence | Time Horizon |",
        "|---|---|---|---|",
    ]

    for r in recommendations:
        rid = r.get("id", "?")
        title = r.get("title", "")
        priority = r.get("priority", "—").capitalize()
        conf = r.get("confidence", "—").capitalize()
        horizon = _HORIZON_LABEL.get(r.get("time_horizon", ""), r.get("time_horizon", "—"))
        lines.append(f"| **{rid}** — {title} | {priority} | {conf} | {horizon} |")

    lines.append("")

    for r in recommendations:
        rid = r.get("id", "?")
        title = r.get("title", "")
        summary = r.get("summary", "")
        priority = r.get("priority", "—").capitalize()
        conf = r.get("confidence", "—").capitalize()
        rationale = r.get("confidence_rationale", "")
        horizon = _HORIZON_LABEL.get(r.get("time_horizon", ""), r.get("time_horizon", "—"))
        hyp_links = r.get("supported_by_hypotheses", [])
        ev_links = r.get("supporting_evidence", [])
        risks = r.get("key_risks", [])
        triggers = r.get("trigger_conditions", [])

        lines += [
            f"### {rid}: {title}",
            "",
            summary,
            "",
            f"**Priority:** {priority}  |  **Confidence:** {conf}  |  **Time Horizon:** {horizon}",
            "",
            f"*{rationale}*",
            "",
        ]
        contrib_profiles = r.get("contributing_profiles", [])
        if contrib_profiles:
            lines.append(f"**Profiles:** {', '.join(contrib_profiles)}")
            lines.append("")
        if hyp_links:
            lines.append(f"**Supported by:** {', '.join(hyp_links)}")
            lines.append("")
        if ev_links:
            lines.append(f"**Evidence:** {', '.join(ev_links)}")
            lines.append("")
        if risks:
            lines.append("**Key Risks:**")
            lines.extend(f"- {risk}" for risk in risks)
            lines.append("")
        if triggers:
            lines.append("**Trigger Conditions:**")
            lines.extend(f"- {t}" for t in triggers)
            lines.append("")

    # Portfolio summary
    if portfolio:
        lines += ["### Recommendation Portfolio", ""]
        for key in ("near_term", "medium_term", "long_term"):
            ids = portfolio.get(key, [])
            if ids:
                label = _HORIZON_LABEL.get(key, key)
                lines.append(f"**{label}:** {', '.join(ids)}")
        lines.append("")

    return "\n".join(lines)


def _build_executive_summary(
    question: str,
    plan: dict[str, Any],
    evidence_note: dict[str, Any],
    qa: dict[str, Any],
    findings: list[dict[str, Any]],
    risks: list[dict[str, Any]],
) -> str:
    """Build a 2–5 paragraph executive summary from structured agent outputs."""
    research_type = plan.get("research_type", "RESEARCH")
    subquestions = plan.get("subquestions", [])
    ev_summary = evidence_note.get("evidence_summary", {})
    total_ev = ev_summary.get("total_evidence_items", 0)
    covered = ev_summary.get("subquestions_with_evidence", 0)
    uncovered = ev_summary.get("subquestions_without_evidence", 0)
    confidence = qa.get("confidence_assessment", {}).get("overall_confidence", "MEDIUM")

    # Para 1: scope
    n_sq = len(subquestions)
    type_label = {
        "FACT_LOOKUP": "factual",
        "COMPARISON": "comparative",
        "EXPLANATION": "explanatory",
        "RESEARCH": "in-depth research",
    }.get(research_type, "research")
    p1 = (
        f"This {type_label} inquiry examined: \"{question}\". "
        f"The analysis was structured around {n_sq} sub-question{'' if n_sq == 1 else 's'}, "
        f"drawing on {total_ev} evidence item{'' if total_ev == 1 else 's'} "
        f"across multiple sources."
    )

    # Para 2: what the evidence shows
    finding_strs = [f["finding"] for f in findings[:3]]
    if finding_strs:
        bullets = "; ".join(finding_strs)
        p2 = f"Key findings indicate: {bullets}."
    else:
        p2 = "The available evidence did not yield strongly supported conclusions."

    # Para 3: coverage / gaps
    if uncovered == 0:
        p3 = (
            f"Evidence coverage was comprehensive: all {covered} sub-question{'' if covered == 1 else 's'} "
            f"received supporting evidence."
        )
    else:
        p3 = (
            f"Coverage was partial: {covered} of {n_sq} sub-question{'' if n_sq == 1 else 's'} "
            f"received evidence support, while {uncovered} remain{'' if uncovered == 1 else 's'} "
            f"without direct evidence."
        )

    # Para 4: risks and confidence
    high_risks = [r for r in risks if r.get("severity") == "HIGH"]
    risk_str = ""
    if high_risks:
        risk_labels = "; ".join(r["risk"] for r in high_risks[:2])
        risk_str = f" Key risks identified: {risk_labels}."
    p4 = (
        f"Overall analytical confidence is assessed as {confidence}.{risk_str} "
        f"Readers should weigh findings against the identified gaps before drawing conclusions."
    )

    return "\n\n".join([p1, p2, p3, p4])


def _format_citations(
    ids: list[str],
    id_to_source: dict[str, str],
    *,
    max_citations: int = 3,
) -> str:
    """Build benchmark-compatible citation markers for a list of evidence IDs.

    Format matches the scorer regex: [Source: <name>, Evidence: E001]
    Only IDs that have a known source document are emitted.
    """
    markers: list[str] = []
    for eid in ids:
        if len(markers) >= max_citations:
            break
        source = id_to_source.get(eid, "").strip()
        if source and eid:
            markers.append(f"[Source: {source}, Evidence: {eid}]")
    return " ".join(markers)


def _build_key_findings(
    evidence_note: dict[str, Any],
    plan: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Derive evidence-backed findings from subquestion coverage.

    Finding text is taken verbatim from the highest-relevance evidence claim
    for each covered subquestion, with benchmark-compatible citation markers
    appended (J5.4 citation-preservation fix).

    Returns (findings, grounding_counts) where grounding_counts has keys
    'supported_findings', 'unsupported_findings', and 'citation_count'.
    """
    subquestions: list[str] = plan.get("subquestions", [])
    coverage_by_sq: dict = evidence_note.get("coverage_by_subquestion", {})
    evidence_by_sq: dict = evidence_note.get("evidence_by_subquestion", {})
    evidence_items: list[dict] = evidence_note.get("evidence_items", [])

    # Build lookups keyed by evidence_id
    id_to_claim: dict[str, str] = {}
    id_to_source: dict[str, str] = {}
    for e in evidence_items:
        eid = e.get("evidence_id", "")
        if not eid:
            continue
        claim = e.get("claim", "").strip()
        if claim:
            id_to_claim[eid] = claim
        source = e.get("source_document", "").strip()
        if source:
            id_to_source[eid] = source

    findings: list[dict[str, Any]] = []
    supported = 0
    unsupported = 0
    total_citations = 0

    for sq in subquestions:
        cov = coverage_by_sq.get(sq, {})
        level = cov.get("coverage", "NONE")
        if level not in ("MODERATE", "STRONG"):
            continue

        ids = evidence_by_sq.get(sq, [])
        # Take the first claim that has non-empty text
        claim_text = next(
            (id_to_claim[eid] for eid in ids if eid in id_to_claim),
            "",
        )

        if not claim_text:
            # MODERATE/STRONG coverage but no usable claim text — skip
            unsupported += 1
            continue

        # Append citation markers to the finding text (benchmark format)
        cite_str = _format_citations(ids, id_to_source)
        finding_text = f"{claim_text} {cite_str}".strip() if cite_str else claim_text
        total_citations += cite_str.count("[Source:")

        confidence = "HIGH" if level == "STRONG" else "MEDIUM"
        findings.append({
            "finding": finding_text,
            "evidence_count": len(ids),
            "confidence": confidence,
            "supporting_evidence_ids": ids[:10],
            "source_subquestion": sq,
        })
        supported += 1

    grounding = {
        "supported_findings": supported,
        "unsupported_findings": unsupported,
        "citation_count": total_citations,
    }
    return findings, grounding


def _build_key_risks(
    qa: dict[str, Any],
    evidence_note: dict[str, Any],
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build risk list from QA issues, coverage gaps, and contradictions."""
    risks: list[dict[str, Any]] = []

    # Risks from high-severity QA coverage issues
    for issue in qa.get("coverage_issues", []):
        if issue.get("severity") == "HIGH":
            sq = issue.get("subquestion", "")[:120]
            risks.append({
                "risk": f"No evidence found for: {sq}",
                "severity": "HIGH",
                "source": "coverage_gap",
            })

    # Risks from contradictions
    for issue in qa.get("contradiction_issues", []):
        topic = issue.get("topic", "unknown topic")
        sev = issue.get("severity", "MEDIUM").upper()
        risks.append({
            "risk": f"Contradictory evidence on: {topic}",
            "severity": sev if sev in ("HIGH", "MEDIUM", "LOW") else "MEDIUM",
            "source": "contradiction",
        })

    # Risks from weak evidence on any subquestion
    coverage_by_sq: dict = evidence_note.get("coverage_by_subquestion", {})
    subquestions: list[str] = plan.get("subquestions", [])
    for sq in subquestions:
        level = coverage_by_sq.get(sq, {}).get("coverage", "NONE")
        if level == "WEAK":
            risks.append({
                "risk": f"Weak evidence for: {sq[:120]}",
                "severity": "MEDIUM",
                "source": "weak_coverage",
            })

    return risks


def _build_open_questions(
    qa: dict[str, Any],
    evidence_note: dict[str, Any],
    plan: dict[str, Any],
) -> list[str]:
    """Build open questions from uncovered subquestions and evidence gaps."""
    open_qs: list[str] = []
    seen: set[str] = set()

    coverage_by_sq: dict = evidence_note.get("coverage_by_subquestion", {})
    subquestions: list[str] = plan.get("subquestions", [])

    # Subquestions with NONE coverage become open questions
    for sq in subquestions:
        level = coverage_by_sq.get(sq, {}).get("coverage", "NONE")
        if level == "NONE" and sq not in seen:
            open_qs.append(sq)
            seen.add(sq)

    # WEAK subquestions that also have HIGH-severity evidence issues
    high_ev_sqs = {
        i.get("subquestion", "")
        for i in qa.get("evidence_issues", [])
        if i.get("severity") == "HIGH"
    }
    for sq in high_ev_sqs:
        if sq not in seen and sq:
            open_qs.append(sq)
            seen.add(sq)

    return open_qs


def _report_confidence(qa: dict[str, Any]) -> str:
    """Derive report_confidence from QA overall_confidence."""
    return qa.get("confidence_assessment", {}).get("overall_confidence", "MEDIUM")


# ---------------------------------------------------------------------------
# J7.6b – Executive Report Builder
# ---------------------------------------------------------------------------

_TIMEFRAME_ORDER = ["0-3 months", "3-12 months", "1-3 years", "3+ years"]
_TIMEFRAME_ALIASES: dict[str, str] = {
    "immediate": "0-3 months",
    "short_term": "0-3 months",
    "short-term": "0-3 months",
    "near_term": "3-12 months",
    "near-term": "3-12 months",
    "medium_term": "1-3 years",
    "medium-term": "1-3 years",
    "long_term": "3+ years",
    "long-term": "3+ years",
}
_TIMEFRAME_DISPLAY: dict[str, str] = {
    "0-3 months": "Immediate (0–3 months)",
    "3-12 months": "Near-term (3–12 months)",
    "1-3 years": "Medium-term (1–3 years)",
    "3+ years": "Long-term (>3 years)",
}

# Assumption importance tiers (J7.1 schema: "Critical" | "Important" | "Supporting")
_IMPORTANCE_ORDER = {"Critical": 0, "Important": 1, "Supporting": 2}


def _normalise_timeframe(tf: str) -> str:
    return _TIMEFRAME_ALIASES.get(tf.lower().replace(" ", "_"), tf)


def _build_j7_executive_report(context: "AgentContext") -> str:
    """Build the 14-section J7 executive report from the decision graph.

    Called only when context.strategic_options is non-empty. Never
    re-generates analysis already present in the decision graph.
    """
    da: dict = context.decision_analysis or {}
    preferred: dict = context.preferred_option or {}
    options: list[dict] = context.strategic_options or []
    assumptions: list[dict] = context.assumptions or []
    risks: list[dict] = context.risks or []
    opps: list[dict] = context.opportunities or []
    recs: list[dict] = context.recommendations or []
    ro: dict = context.research_object or {}
    question: str = context.question or ""

    recommended_id = da.get("recommended_option_id") or preferred.get("option_id") or ""
    preferred_title = preferred.get("title") or next(
        (o.get("title", "") for o in options if o.get("option_id") == recommended_id), ""
    )

    lines: list[str] = []

    # ------------------------------------------------------------------ #
    # Section 1 — Executive Summary                                        #
    # ------------------------------------------------------------------ #
    exec_summary = da.get("executive_summary") or preferred.get("rationale") or ""
    # J9.3 — lead with the Executive Decision Statement (from the Decision
    # Architecture) rather than the technical research question when available.
    _arch = ro.get("decision_architecture") or {}
    decision_statement = _arch.get("decision_statement", "")
    executive_context = _arch.get("executive_context", "")
    lines += [
        "# Executive Strategic Report",
        "",
        "## 1. Executive Summary",
        "",
    ]
    if decision_statement:
        lines += [f"**Decision:** {decision_statement}", ""]
    if preferred_title:
        lines += [f"**Recommended Option:** {preferred_title}", ""]
    if executive_context:
        lines += [executive_context, ""]
    if exec_summary:
        lines += [exec_summary, ""]

    # ------------------------------------------------------------------ #
    # Section 2 — Strategic Question                                       #
    # ------------------------------------------------------------------ #
    lines += ["## 2. Strategic Question", "", question, ""]

    # ------------------------------------------------------------------ #
    # Section 3 — Recommended Strategic Option                             #
    # ------------------------------------------------------------------ #
    lines += ["## 3. Recommended Strategic Option", ""]
    rec_opt = next((o for o in options if o.get("option_id") == recommended_id), preferred)
    if rec_opt:
        _oid = rec_opt.get("option_id", "")
        _title = rec_opt.get("title", "")
        _rationale = rec_opt.get("rationale") or rec_opt.get("strategic_logic") or ""
        _horizon = rec_opt.get("time_horizon", "").replace("_", " ")
        lines += [
            f"**{_oid}: {_title}**",
            "",
        ]
        if _horizon:
            lines += [f"**Time Horizon:** {_horizon}", ""]
        if _rationale:
            lines += [_rationale, ""]
        caps = rec_opt.get("required_capabilities") or []
        deps = rec_opt.get("dependencies") or []
        if caps:
            lines.append("**Required Capabilities:**")
            lines.extend(f"- {c}" for c in caps)
            lines.append("")
        if deps:
            lines.append("**Key Dependencies:**")
            lines.extend(f"- {d}" for d in deps)
            lines.append("")

    # ------------------------------------------------------------------ #
    # Section 4 — Why This Option Wins                                     #
    # ------------------------------------------------------------------ #
    rationale = da.get("rationale") or ""
    dimensions = da.get("comparison_dimensions") or []
    rankings = da.get("option_rankings") or []
    lines += ["## 4. Why This Option Wins", ""]
    if rationale:
        lines += [rationale, ""]
    if dimensions:
        lines.append("**Comparison Dimensions:**")
        lines.extend(f"- {d}" for d in dimensions)
        lines.append("")
    if rankings:
        lines.append("**Option Rankings (best → least preferred):**")
        lines.extend(f"{i + 1}. {r}" for i, r in enumerate(rankings))
        lines.append("")

    # ------------------------------------------------------------------ #
    # Section 5 — Executive Confidence (J7.7)                              #
    # ------------------------------------------------------------------ #
    ec: dict = context.executive_confidence or {}
    lines += ["## 5. Executive Confidence", ""]
    if ec:
        ec_conf = ec.get("overall_confidence", "")
        ec_ready = ec.get("decision_readiness", "")
        ec_board = ec.get("board_recommendation", "")
        lines += [
            f"**Overall Confidence:** {ec_conf}  |  "
            f"**Decision Readiness:** {ec_ready}  |  "
            f"**Board Recommendation:** {ec_board}",
            "",
        ]
        ec_rat = ec.get("confidence_rationale", "")
        if ec_rat:
            lines += [ec_rat, ""]

        ec_drivers = ec.get("confidence_drivers", [])
        if ec_drivers:
            lines.append("**Confidence Drivers:**")
            lines.extend(f"- {d}" for d in ec_drivers)
            lines.append("")

        ec_limiters = ec.get("confidence_limiters", [])
        if ec_limiters:
            lines.append("**Confidence Limiters:**")
            lines.extend(f"- {lim}" for lim in ec_limiters)
            lines.append("")

        vp = ec.get("validation_priorities", [])
        if vp:
            lines.append("**Validation Priorities (Due Diligence Checklist):**")
            lines.extend(f"{i + 1}. {p}" for i, p in enumerate(vp))
            lines.append("")

        cu = ec.get("critical_unknowns", [])
        if cu:
            lines.append("**Critical Unknowns:**")
            lines.extend(f"- {u}" for u in cu)
            lines.append("")

        if_hold = ec.get("confidence_if_assumptions_hold", "")
        if_fail = ec.get("confidence_if_assumptions_fail", "")
        if if_hold or if_fail:
            lines.append("**Conditional Assessment:**")
            if if_hold:
                lines += [f"- *If assumptions hold:* {if_hold}", ""]
            if if_fail:
                lines += [f"- *If assumptions fail:* {if_fail}", ""]

        horizon = ec.get("decision_horizon", "")
        if horizon:
            lines += [f"**Decision Horizon:** {horizon}", ""]
    else:
        lines += ["*Executive confidence assessment not available for this run.*", ""]

    # ------------------------------------------------------------------ #
    # Section 6 — Strategic Assumptions (was §5)                           #
    # ------------------------------------------------------------------ #
    lines += ["## 6. Strategic Assumptions", ""]
    if assumptions:
        sorted_assumptions = sorted(
            assumptions,
            key=lambda a: _IMPORTANCE_ORDER.get(a.get("importance", "Supporting"), 99),
        )
        lines += [
            "| ID | Assumption | Importance | Confidence | Evidence Support |",
            "|---|---|---|---|---|",
        ]
        for a in sorted_assumptions:
            aid = a.get("assumption_id", "")
            stmt = a.get("statement", "").replace("|", "\\|")
            imp = a.get("importance", "")
            conf = a.get("confidence", "")
            ev_sup = a.get("evidence_support", "")
            lines.append(f"| {aid} | {stmt} | {imp} | {conf} | {ev_sup} |")
        lines.append("")
    else:
        lines += ["*No assumptions recorded.*", ""]

    # ------------------------------------------------------------------ #
    # Section 6 — Strategic Risks                                          #
    # ------------------------------------------------------------------ #
    lines += ["## 7. Strategic Risks", ""]
    if risks:
        lines += [
            "| ID | Risk | Severity | Likelihood | Related Assumptions | Affected Recommendations |",
            "|---|---|---|---|---|---|",
        ]
        for r in risks:
            rid = r.get("risk_id", "")
            stmt = r.get("statement", r.get("title", r.get("description", ""))).replace("|", "\\|")[:120]
            sev = r.get("severity", "")
            lhood = r.get("likelihood", "")
            rel_assump = ", ".join(r.get("related_assumption_ids", [])) or "—"
            aff_rec = ", ".join(r.get("affected_recommendation_ids", [])) or "—"
            lines.append(f"| {rid} | {stmt} | {sev} | {lhood} | {rel_assump} | {aff_rec} |")
        lines.append("")
        # Mitigation notes as sub-bullets
        for r in risks:
            mit = r.get("mitigation_notes") or r.get("mitigation") or ""
            if mit:
                rid = r.get("risk_id", "")
                lines += [f"**{rid} Mitigation:** {mit}", ""]
    else:
        lines += ["*No risks recorded.*", ""]

    # ------------------------------------------------------------------ #
    # Section 7 — Strategic Opportunities                                  #
    # ------------------------------------------------------------------ #
    lines += ["## 8. Strategic Opportunities", ""]
    if opps:
        lines += [
            "| ID | Statement | Category | Likelihood | Impact |",
            "|---|---|---|---|---|",
        ]
        for o in opps:
            oid2 = o.get("opportunity_id", "")
            stmt = o.get("statement", o.get("title", o.get("description", ""))).replace("|", "\\|")[:120]
            cat = o.get("category", "")
            lhood = o.get("likelihood", o.get("probability", ""))
            impact = o.get("impact", "")
            lines.append(f"| {oid2} | {stmt} | {cat} | {lhood} | {impact} |")
        lines.append("")
    else:
        lines += ["*No opportunities recorded.*", ""]

    # ------------------------------------------------------------------ #
    # Section 8 — Strategic Options                                        #
    # ------------------------------------------------------------------ #
    lines += ["## 9. Strategic Options", ""]
    for opt in options:
        oid3 = opt.get("option_id", "")
        t = opt.get("title", "")
        is_rec = oid3 == recommended_id
        heading = f"### {oid3}: {t}"
        if is_rec:
            heading += " *(Recommended)*"
        lines += [heading, ""]
        opt_rationale = opt.get("rationale") or opt.get("strategic_logic") or ""
        if opt_rationale:
            lines += [opt_rationale, ""]
        horizon = opt.get("time_horizon", "").replace("_", " ")
        posture = opt.get("posture", "").replace("_", " ").title()
        meta_parts = []
        if posture:
            meta_parts.append(f"**Posture:** {posture}")
        if horizon:
            meta_parts.append(f"**Time Horizon:** {horizon}")
        if meta_parts:
            lines += ["  ".join(meta_parts), ""]
        for field_label, field_key in [
            ("Required Capabilities", "required_capabilities"),
            ("Key Dependencies", "dependencies"),
            ("Key Risks", "risks"),
        ]:
            items = opt.get(field_key) or []
            if items:
                lines.append(f"**{field_label}:**")
                lines.extend(f"- {x}" for x in items)
                lines.append("")
        sup_recs = opt.get("supporting_recommendations") or []
        if sup_recs:
            lines += [f"**Supporting Recommendations:** {', '.join(sup_recs)}", ""]

    # ------------------------------------------------------------------ #
    # Section 9 — Decision Matrix                                          #
    # ------------------------------------------------------------------ #
    matrix: list[dict] = da.get("decision_matrix") or []
    lines += ["## 10. Decision Matrix", ""]
    if matrix:
        _SCORE_COLS = [
            ("strategic_fit", "Strategic Fit"),
            ("implementation_risk", "Impl. Risk"),
            ("execution_complexity", "Exec. Complexity"),
            ("capital_requirement", "Capital Req."),
            ("expected_return", "Expected Return"),
            ("time_to_value", "Time to Value"),
            ("dependency_strength", "Dependency"),
            ("assumption_strength", "Assumption"),
            ("risk_exposure", "Risk Exposure"),
            ("opportunity_capture", "Opportunity"),
            ("overall_score", "Overall"),
        ]
        col_headers = "| Option | " + " | ".join(h for _, h in _SCORE_COLS) + " |"
        col_divider = "|---|" + "---|" * len(_SCORE_COLS)
        lines += [col_headers, col_divider]
        # Build option_id → title lookup
        _opt_titles = {o.get("option_id", ""): o.get("title", "") for o in options}
        for entry in matrix:
            eid = entry.get("option_id", "")
            opt_label = f"{eid}: {_opt_titles.get(eid, '')}" if _opt_titles.get(eid) else eid
            if eid == recommended_id:
                opt_label += " ✓"
            scores = " | ".join(entry.get(k, "—") for k, _ in _SCORE_COLS)
            lines.append(f"| {opt_label} | {scores} |")
        lines.append("")
        # Strengths/weaknesses per option
        for entry in matrix:
            eid = entry.get("option_id", "")
            strengths = entry.get("strengths") or []
            weaknesses = entry.get("weaknesses") or []
            if strengths or weaknesses:
                lines += [f"**{eid} — Strengths & Weaknesses**", ""]
                if strengths:
                    lines.append("Strengths:")
                    lines.extend(f"+ {s}" for s in strengths)
                    lines.append("")
                if weaknesses:
                    lines.append("Weaknesses:")
                    lines.extend(f"- {w}" for w in weaknesses)
                    lines.append("")
    else:
        lines += ["*No decision matrix available.*", ""]

    # ------------------------------------------------------------------ #
    # Section 10 — Key Tradeoffs                                           #
    # ------------------------------------------------------------------ #
    tradeoffs = da.get("key_tradeoffs") or []
    lines += ["## 11. Key Tradeoffs", ""]
    if tradeoffs:
        lines.extend(f"- {t}" for t in tradeoffs)
        lines.append("")
    else:
        lines += ["*No tradeoffs recorded.*", ""]

    # ------------------------------------------------------------------ #
    # Section 11 — Sensitivity Analysis                                    #
    # ------------------------------------------------------------------ #
    sensitivity = da.get("sensitivity_analysis") or ""
    lines += ["## 12. Sensitivity Analysis", ""]
    if sensitivity:
        lines += [sensitivity, ""]
    else:
        lines += ["*No sensitivity analysis available.*", ""]

    # ------------------------------------------------------------------ #
    # Section 12 — Confidence Assessment                                   #
    # ------------------------------------------------------------------ #
    conf_summary = da.get("confidence_summary") or ""
    conf_level = da.get("confidence") or ""
    uncertainties = da.get("key_uncertainties") or []
    lines += ["## 13. Confidence Assessment", ""]
    if conf_level:
        lines += [f"**Overall Confidence:** {conf_level}", ""]
    if conf_summary:
        lines += [conf_summary, ""]
    if uncertainties:
        lines.append("**Key Uncertainties:**")
        lines.extend(f"- {u}" for u in uncertainties)
        lines.append("")

    # ------------------------------------------------------------------ #
    # Section 13 — Immediate Actions                                       #
    # ------------------------------------------------------------------ #
    lines += ["## 14. Immediate Actions", ""]
    if recs:
        grouped: dict[str, list[dict]] = {}
        for rec in recs:
            # Recommendations use time_horizon (DM schema) or timeframe (legacy)
            raw_tf = rec.get("time_horizon") or rec.get("timeframe") or ""
            tf = _normalise_timeframe(raw_tf)
            grouped.setdefault(tf, []).append(rec)
        ordered_tfs = [tf for tf in _TIMEFRAME_ORDER if tf in grouped]
        for tf in grouped:
            if tf not in ordered_tfs:
                ordered_tfs.append(tf)
        for tf in ordered_tfs:
            tf_recs = grouped.get(tf, [])
            if not tf_recs:
                continue
            display = _TIMEFRAME_DISPLAY.get(tf, tf)
            lines += [f"### {display}", ""]
            for rec in tf_recs:
                rid = rec.get("recommendation_id", "")
                rtitle = rec.get("title", "")
                # summary is the DM field; fall back to rationale/description for legacy
                rbody = rec.get("summary") or rec.get("rationale") or rec.get("description") or ""
                rpri = rec.get("priority", "")
                lines.append(f"**{rid}: {rtitle}**" + (f" *(Priority: {rpri})*" if rpri else ""))
                if rbody:
                    lines += [rbody, ""]
                else:
                    lines.append("")
    else:
        lines += ["*No recommendations recorded.*", ""]

    # ------------------------------------------------------------------ #
    # Section 14 — Supporting Evidence                                     #
    # ------------------------------------------------------------------ #
    # evidence_summary is populated by EvidenceAgent before ReportAgent runs;
    # summary.evidence_count is only set by update_research_object() which runs
    # after _build_j7_executive_report(). Use evidence_summary as primary source.
    ev_summary = ro.get("evidence_summary") or ro.get("summary") or {}
    ev_count = ev_summary.get("total_evidence_items", ev_summary.get("evidence_count", 0))
    # J8.10 — citation_count is computed by _build_key_findings() and stored in
    # context.report["report_grounding_score"]; the TRACE reads it from there.
    # The report previously read it from evidence_summary (which never carries
    # the key), so it always rendered "Citations: 0" while the trace recorded a
    # non-zero count. Read from the same source the trace uses to keep them
    # consistent, falling back to evidence_summary / the RO for legacy objects.
    report_block = context.report or ro.get("report") or {}
    grounding = report_block.get("report_grounding_score") or {}
    cite_count = grounding.get("citation_count", ev_summary.get("citation_count", 0))
    # Fall back to evidence_ids list length when dedicated counts are missing
    if not ev_count:
        ev_count = len(ro.get("evidence_ids", []))
    profiles = ro.get("profiles") or context.profiles or []
    lines += [
        "## 15. Supporting Evidence",
        "",
        f"**Evidence Items:** {ev_count}  |  **Citations:** {cite_count}  |  "
        f"**Source Profiles:** {', '.join(profiles) if profiles else 'N/A'}",
        "",
    ]

    # J8.10 — surface the evidence-backed findings so conclusions are traceable.
    # Each finding carries inline [Source: <doc>, Evidence: <id>] markers plus
    # its supporting evidence IDs, making the chain from claim → evidence visible
    # in the report itself (not only in the trace).
    key_findings = report_block.get("key_findings") or []
    if key_findings:
        lines.append("**Evidence-Backed Findings:**")
        lines.append("")
        for f in key_findings:
            finding_text = (f.get("finding") or "").strip()
            if not finding_text:
                continue
            conf = f.get("confidence", "")
            ids = f.get("supporting_evidence_ids") or []
            header = f"- {finding_text}"
            if conf:
                header += f" *(Confidence: {conf})*"
            lines.append(header)
            if ids:
                lines.append(f"  - Supporting evidence: {', '.join(ids)}")
        lines.append("")

    topics = ro.get("evidence_topics") or {}
    if topics:
        top_topics = sorted(topics.items(), key=lambda x: -x[1])[:8]
        lines.append("**Top Evidence Topics:**")
        lines.extend(f"- {topic} ({count} items)" for topic, count in top_topics)
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ReportAgent(FunctionalAgent):
    """Synthesises research into narrative and writes all outputs (J5.0b / J5.4).

    J5.4 additions:
      - Generates executive_summary, key_findings, key_risks, open_questions
      - Derives report_confidence from QAAgent output
      - Maintains evidence traceability (supporting_evidence_ids per finding)
      - Writes context.report, research_object["report"], trace["report_agent"]
    """

    def __init__(self, *, out_path: Path, domain_profile: Any = None) -> None:
        self._out_path = out_path
        self._domain_profile = domain_profile

    def _execute(self, context: AgentContext) -> AgentContext:
        from research_agent.log import PROGRESS
        from research_agent.markdown import memo_to_markdown, write_markdown
        from research_agent.trace import build_trace, write_trace

        memo = context.trace.get("_memo")
        documents = context.trace.get("_documents", [])

        if memo is None:
            LOGGER.error("ReportAgent: no memo on context — cannot write report")
            self._record(context, status="error", summary="No memo available; report not written.")
            return context

        # ------------------------------------------------------------------
        # J5.4 – Synthesis
        # ------------------------------------------------------------------
        plan = context.plan
        evidence_note = context.evidence_notes[0] if context.evidence_notes else {}
        qa = context.qa

        findings, grounding = _build_key_findings(evidence_note, plan)
        risks = _build_key_risks(qa, evidence_note, plan)
        open_questions = _build_open_questions(qa, evidence_note, plan)
        report_conf = _report_confidence(qa)
        executive_summary = _build_executive_summary(
            context.question, plan, evidence_note, qa, findings, risks
        )

        report_summary = {
            "finding_count": len(findings),
            "risk_count": len(risks),
            "open_question_count": len(open_questions),
            "report_confidence": report_conf,
        }

        context.report = {
            "executive_summary": executive_summary,
            "key_findings": findings,
            "key_risks": risks,
            "open_questions": open_questions,
            "report_confidence": report_conf,
            "report_summary": report_summary,
            "report_grounding_score": grounding,
            "profiles_requested": list(context.profiles),
            "profiles_contributing": qa.get("profiles_contributing", []),
            "profiles_missing": qa.get("profiles_missing", []),
            "coverage_status": qa.get("coverage_status", "sufficient"),
            "hypotheses": context.hypotheses,
            "hypothesis_challenges": context.hypothesis_challenges,
            "surviving_hypotheses": context.surviving_hypotheses,
            "recommendations": context.recommendations,
            "recommendation_portfolio": context.recommendation_portfolio,
        }

        LOGGER.log(
            PROGRESS,
            "[ReportAgent] findings=%d  risks=%d  open_questions=%d  confidence=%s",
            len(findings), len(risks), len(open_questions), report_conf,
        )

        # ------------------------------------------------------------------
        # Write markdown report
        # J7.6b: when strategic_options are present, produce the J7 executive
        # report (14 sections driven by the decision graph). Otherwise fall
        # back to the legacy memo-based path for backward compatibility.
        # ------------------------------------------------------------------
        if context.strategic_options:
            report_content = _build_j7_executive_report(context)
        else:
            report_content = memo_to_markdown(memo)
            if context.hypotheses:
                report_content = report_content.rstrip("\n") + "\n\n" + _build_hypotheses_section(context.hypotheses)
            if context.hypothesis_challenges:
                report_content = report_content.rstrip("\n") + "\n\n" + _build_challenges_section(
                    context.hypothesis_challenges, context.surviving_hypotheses
                )
            if context.recommendations:
                report_content = report_content.rstrip("\n") + "\n\n" + _build_recommendations_section(
                    context.recommendations, context.recommendation_portfolio
                )
            _rec_eval_for_md = (
                context.research_object.get("recommendation_evaluation")
                if context.research_object else None
            )
            if _rec_eval_for_md:
                _rec_eval_section = _build_recommendation_evaluation_section(_rec_eval_for_md)
                if _rec_eval_section:
                    report_content = report_content.rstrip("\n") + "\n\n" + _rec_eval_section
            _improvement_data = context.recommendation_improvement
            if _improvement_data and _improvement_data.get("improvement_records"):
                _improvement_section = _build_recommendation_improvement_section(_improvement_data)
                if _improvement_section:
                    report_content = report_content.rstrip("\n") + "\n\n" + _improvement_section
            if context.scenario_analysis:
                _scenario_section = _build_scenario_section(context.scenario_analysis)
                if _scenario_section:
                    report_content = report_content.rstrip("\n") + "\n\n" + _scenario_section
            if context.multi_profile_analysis and len(context.profiles) > 1:
                _ps_section = _build_profile_synthesis_section(
                    context.multi_profile_analysis,
                    context.recommendations,
                    context.hypotheses,
                    synthesis_tradeoffs=context.synthesis_tradeoffs or None,
                )
                if _ps_section:
                    report_content = report_content.rstrip("\n") + "\n\n" + _ps_section
        output_path = write_markdown(report_content, self._out_path)
        context.artifacts["report_path"] = str(output_path)
        context.artifacts["trace_path"] = str(output_path.with_suffix(".trace.json"))

        # Record in history before trace so it appears in agents_run (J5.4.9)
        self._record(
            context,
            status="success",
            summary=(
                f"Report written to {output_path}. "
                f"Findings={len(findings)}, risks={len(risks)}, "
                f"open_questions={len(open_questions)}, confidence={report_conf}."
            ),
            report_path=str(output_path),
            finding_count=len(findings),
            risk_count=len(risks),
            report_confidence=report_conf,
        )

        # ------------------------------------------------------------------
        # Build trace
        # ------------------------------------------------------------------
        trace_payload = build_trace(
            question=context.question,
            source_directory=Path("sources"),
            output_path=output_path,
            documents=documents,
            memo=memo,
            mock_mode=False,
            profile=self._domain_profile,
        )
        trace_payload["functional_agents"] = context.to_functional_trace()

        # J9.1 – record run mode (research vs strategic_engagement) and, when an
        # engagement drove the run, its structured metadata. Defaults preserve
        # existing behaviour for goal/question runs.
        trace_payload["run_mode"] = context.trace.get("_run_mode", "research")
        _engagement_meta = context.trace.get("_engagement")
        if _engagement_meta:
            trace_payload["engagement"] = _engagement_meta
        # J9.1a – condensed Strategic Framing Summary: shows the bounded framing
        # that propagates downstream vs the raw brief size, for prompt-growth audits.
        _framing_summary = context.trace.get("_strategic_framing_summary")
        if _framing_summary:
            trace_payload["strategic_framing_summary"] = _framing_summary
        # J9.1b – ResearchStrategy generation diagnostics (prompt estimate,
        # max_tokens, stop_reason, truncation flag, output shape).
        _rs_diag = context.trace.get("_research_strategy_diagnostics")
        if _rs_diag:
            trace_payload["research_strategy_diagnostics"] = _rs_diag
        # J9.2 – Decision Architecture: full structure + compact metadata counts.
        _arch = context.trace.get("_decision_architecture")
        if _arch:
            trace_payload["decision_architecture"] = _arch
        _arch_meta = context.trace.get("_decision_architecture_meta")
        if _arch_meta:
            trace_payload["decision_architecture_meta"] = _arch_meta
        # J10.1 – reasoning-target seam diagnostics (legacy: single question).
        from .reasoning_target import reasoning_targets_diagnostics, KIND_DECISION_DOMAIN
        _rt = context.get_reasoning_targets()
        _rt_source = (
            "decision_architecture"
            if (_rt and _rt[0].kind == KIND_DECISION_DOMAIN)
            else "context.question"
        )
        trace_payload["reasoning_targets"] = reasoning_targets_diagnostics(_rt, source=_rt_source)
        # J10.2 – planner reasoning-target diagnostics (additive).
        _planner_reasoning = context.trace.get("_planner_reasoning")
        if _planner_reasoning:
            trace_payload["planner_reasoning"] = _planner_reasoning
        # J10.5 – evidence reasoning (per-domain) diagnostics (additive).
        _evidence_reasoning = context.trace.get("_evidence_reasoning")
        if _evidence_reasoning:
            trace_payload["evidence_reasoning"] = _evidence_reasoning
        # J10.6 – hypothesis reasoning (per-domain) diagnostics (additive).
        _hypothesis_reasoning = context.trace.get("_hypothesis_reasoning")
        if _hypothesis_reasoning:
            trace_payload["hypothesis_reasoning"] = _hypothesis_reasoning
        # PH1 – LLM output normalization diagnostics (additive).
        _llm_norm = context.trace.get("_llm_normalization")
        if _llm_norm:
            trace_payload["llm_normalization"] = _llm_norm

        # Multi-profile block (J5.6 / J5.6a)
        trace_payload["profiles_requested"] = context.profiles
        trace_payload["profile_count"] = len(context.profiles)
        trace_payload["profiles_contributing"] = qa.get("profiles_contributing", [])
        trace_payload["profiles_missing"] = qa.get("profiles_missing", [])
        trace_payload["coverage_status"] = qa.get("coverage_status", "sufficient")
        # Rich profile coverage: {name: {coverage, evidence_count}}
        raw_coverage = (
            evidence_note.get("profile_coverage_by_profile", {}) if evidence_note else {}
        )
        if raw_coverage:
            trace_payload["profile_coverage"] = {
                pname: {
                    "coverage": entry.get("coverage_level", "NONE").lower(),
                    "evidence_count": entry.get("evidence_count", 0),
                }
                for pname, entry in raw_coverage.items()
            }
        elif qa.get("profile_coverage"):
            # Fall back to flat dict when raw data not available
            trace_payload["profile_coverage"] = {
                k: {"coverage": v, "evidence_count": 0}
                for k, v in qa["profile_coverage"].items()
            }

        # Planner block (J5.1.7)
        if plan:
            trace_payload["planner"] = {
                "research_type": plan.get("research_type", ""),
                "subquestion_count": len(plan.get("subquestions", [])),
                "investigation_area_count": len(plan.get("investigation_areas", [])),
                "profiles_used": plan.get("profiles_used", []),
                "reasoning": plan.get("reasoning", ""),
            }

        # Evidence agent block (J5.2.7)
        ev_summary = evidence_note.get("evidence_summary", {})
        if evidence_note:
            trace_payload["evidence_agent"] = {
                "evidence_count": ev_summary.get("total_evidence_items", 0),
                "mapped_subquestions": ev_summary.get("subquestions_with_evidence", 0),
                "mapped_areas": ev_summary.get("investigation_areas_with_evidence", 0),
                "uncovered_subquestions": ev_summary.get("subquestions_without_evidence", 0),
                "coverage_distribution": ev_summary.get("coverage_distribution", {}),
            }

        # QA block (J5.3.8)
        if qa:
            qa_summary = qa.get("qa_summary", {})
            confidence_assessment = qa.get("confidence_assessment", {})
            trace_payload["qa_agent"] = {
                "issues_found": qa_summary.get("issues_found", 0),
                "overall_confidence": confidence_assessment.get("overall_confidence", ""),
                "coverage_issues": qa_summary.get("coverage_issues", 0),
                "evidence_issues": qa_summary.get("evidence_issues", 0),
                "contradiction_issues": qa_summary.get("contradiction_issues", 0),
            }

        # Orchestrator block (J5.5.7)
        orchestrator_meta = context.trace.get("_orchestrator", {})
        if orchestrator_meta:
            trace_payload["orchestrator"] = {
                "iterations": orchestrator_meta.get("iterations", 0),
                "workflow_path": orchestrator_meta.get("workflow_path", []),
                "termination_reason": orchestrator_meta.get("termination_reason", "COMPLETE"),
            }

        # Problem framing block (J6.1a) — present only in goal-driven runs
        pf_data = context.trace.get("_problem_framing")
        if pf_data:
            trace_payload["problem_framing"] = {
                "objective": pf_data.get("objective", ""),
                "decision_areas": pf_data.get("decision_areas", []),
                "critical_uncertainties": pf_data.get("critical_uncertainties", []),
                "research_questions": pf_data.get("research_questions", []),
                "evidence_requirements": pf_data.get("evidence_requirements", []),
            }

        # Research strategy block (J6.2) — present only in goal-driven runs
        rs_data = context.trace.get("_research_strategy")
        if rs_data:
            trace_payload["research_strategy"] = {
                "profile_priorities": rs_data.get("profile_priorities", {}),
                "research_question_priorities": rs_data.get("research_question_priorities", []),
                "required_evidence": rs_data.get("required_evidence", []),
                "source_priorities": rs_data.get("source_priorities", []),
                "coverage_targets": rs_data.get("coverage_targets", {}),
                "strategy_rationale": rs_data.get("strategy_rationale", ""),
            }

        # Hypothesis generation block (J6.3)
        hyp_data = context.trace.get("_hypotheses")
        if hyp_data:
            hypotheses = hyp_data.get("hypotheses", [])
            trace_payload["hypothesis_generation"] = {
                "hypothesis_count": len(hypotheses),
                "synthesis_note": hyp_data.get("synthesis_note", ""),
                "hypotheses": hypotheses,
            }

        # Recommendation generation block (J6.5)
        rec_data = context.trace.get("_recommendations")
        if rec_data:
            recs = rec_data.get("recommendations", [])
            portfolio = rec_data.get("recommendation_portfolio", {})
            trace_payload["recommendation_generation"] = {
                "recommendation_count": len(recs),
                "recommendations": recs,
                "recommendation_portfolio": portfolio,
                "synthesis_note": rec_data.get("synthesis_note", ""),
            }

        # Challenge generation block (J6.4)
        chal_data = context.trace.get("_challenges")
        if chal_data:
            challenges = chal_data.get("hypothesis_challenges", [])
            surviving = chal_data.get("surviving_hypotheses", [])
            trace_payload["challenge_generation"] = {
                "challenge_count": len(challenges),
                "surviving_hypotheses": surviving,
                "hypothesis_challenges": challenges,
                "challenge_synthesis": chal_data.get("challenge_synthesis", ""),
            }

        # Recommendation quality evaluation block (J6.6 / J6.6a)
        rec_eval = context.research_object.get("recommendation_evaluation") if context.research_object else None
        if rec_eval:
            agg = rec_eval.get("aggregate", {})
            rec_summary = rec_eval.get("recommendation_summary", {})
            trace_payload["recommendation_quality"] = {
                "recommendation_count": agg.get("recommendation_count", 0),
                "recommendation_score": agg.get("recommendation_score", 0.0),
                "recommendation_summary": rec_summary,
                "recommendation_warnings": rec_eval.get("recommendation_warnings", []),
                "dimension_summary": {
                    "evidence_support": agg.get("mean_evidence_support", 0.0),
                    "reasoning": agg.get("mean_reasoning", 0.0),
                    "tradeoff": agg.get("mean_tradeoff", 0.0),
                    "risk": agg.get("mean_risk", 0.0),
                    "actionability": agg.get("mean_actionability", 0.0),
                },
                "per_recommendation": [
                    {
                        "recommendation_id": s.get("recommendation_id", ""),
                        "title": s.get("title", ""),
                        "evidence_support_score": s.get("evidence_support_score", 0.0),
                        "reasoning_score": s.get("reasoning_score", 0.0),
                        "tradeoff_score": s.get("tradeoff_score", 0.0),
                        "risk_score": s.get("risk_score", 0.0),
                        "actionability_score": s.get("actionability_score", 0.0),
                        "aggregate_score": s.get("aggregate_score", s.get("recommendation_score", 0.0)),
                        "missing_evidence_links": s.get("missing_evidence_links", False),
                        "primary_penalty": s.get("primary_penalty"),
                        "supporting_evidence": s.get("traceability", {}).get("evidence_ids", []),
                        "supporting_hypotheses": s.get("traceability", {}).get("hypothesis_ids", []),
                        "supporting_challenges": s.get("traceability", {}).get("challenge_ids", []),
                    }
                    for s in rec_eval.get("recommendation_scores", [])
                ],
                "traceability": rec_eval.get("traceability", []),
            }
            # Write dedicated observability trace file (J6.6a)
            _write_recommendation_observability_trace(rec_eval, output_path)

        # Contradiction hardening block (J6.5a/b/c)
        if context.contradiction_metrics:
            metrics = context.contradiction_metrics
            eligibility = metrics.get("eligibility_engine", {})
            trace_payload["contradiction_hardening"] = {
                "candidate_count": metrics.get("candidate_count", 0),
                "suppressed_count": metrics.get("suppressed_count", 0),
                "final_count": metrics.get("final_count", 0),
                "by_reason": metrics.get("by_reason", {}),
                "scope_filtering_present": metrics.get("scope_filtering_present", False),
                "entity_filtering_present": metrics.get("entity_filtering_present", False),
                "temporal_filtering_present": metrics.get("temporal_filtering_present", False),
                "product_filtering_present": metrics.get("product_filtering_present", False),
                "context_filtering_present": metrics.get("context_filtering_present", False),
                "eligibility_engine": {
                    "candidate_pairs": eligibility.get("candidate_pairs", 0),
                    "eligible_pairs": eligibility.get("eligible_pairs", 0),
                    "suppressed_pairs": eligibility.get("suppressed_pairs", 0),
                },
                "numeric_semantics": metrics.get("numeric_semantics", {}),
            }

        # Recommendation improvement block (J6.7)
        improvement_data = context.trace.get("_recommendation_improvement")
        if improvement_data:
            trace_payload["recommendation_improvement"] = improvement_data

        # Recommendation synthesis block (J6.8c)
        synthesis_data = context.trace.get("_recommendation_synthesis")
        if synthesis_data:
            trace_payload["recommendation_synthesis"] = synthesis_data

        # Strategic options block (J7.1)
        strategic_options_data = context.trace.get("_strategic_options")
        if strategic_options_data:
            trace_payload["strategic_options"] = strategic_options_data

        # Multi-profile validation block (J5.6a)
        multi_profile_data = context.trace.get("_multi_profile")
        if multi_profile_data:
            trace_payload["multi_profile_validation"] = multi_profile_data.get(
                "multi_profile_validation", multi_profile_data
            )
        elif context.multi_profile_analysis:
            mpa = context.multi_profile_analysis
            trace_payload["multi_profile_validation"] = {
                "profiles_requested":   mpa.get("profiles_requested", []),
                "profiles_contributing": mpa.get("profiles_contributing", []),
                "profiles_missing":     mpa.get("profiles_missing", []),
                "coverage_status":      mpa.get("coverage_status", "unknown"),
                "profile_coverage":     mpa.get("profile_coverage", {}),
                "profile_influence":    mpa.get("profile_influence", {}),
                "missing_profile_diagnostics": mpa.get("missing_profile_diagnostics", []),
            }

        # Scenario analysis block (J6.8)
        scenario_data = context.trace.get("_scenario_analysis")
        if scenario_data:
            trace_payload["scenario_analysis"] = scenario_data

        # Contract validation block (J5.5a follow-up)
        # ReportAgent reads _contract_runtime before _step() can record its own
        # result (the trace is written here, mid-execution). base.run() always
        # wraps _execute() in AgentResult, so we pre-populate a valid entry.
        from .contract import build_contract_validation, validate_all_classes
        class_checks = validate_all_classes()
        runtime_checks = dict(context.trace.get("_contract_runtime", {}))
        runtime_checks.setdefault("ReportAgent", {
            "returns_agent_result": True,
            "missing_fields": [],
            "error": None,
        })
        trace_payload["contract_validation"] = build_contract_validation(
            class_checks, runtime_checks
        )

        # Performance instrumentation block (J8.8a)
        # Snapshot tracker here — all agents except ReportAgent itself have completed
        perf_tracker = context.trace.get("_perf_tracker")
        if perf_tracker is not None:
            trace_payload["performance"] = perf_tracker.summary()

        # Report agent block (J5.4.8)
        trace_payload["report_agent"] = {
            "finding_count": len(findings),
            "risk_count": len(risks),
            "open_question_count": len(open_questions),
            "report_confidence": report_conf,
            "report_grounding_score": grounding,
            "citation_count": grounding.get("citation_count", 0),
        }

        # ------------------------------------------------------------------
        # Update Research Object (J5.0b.4 + J5.4.7)
        # ------------------------------------------------------------------
        if context.research_object:
            from research_agent.research_object import (
                research_object_trace_stub,
                update_research_object,
                write_research_object,
            )

            ro = update_research_object(
                context.research_object,
                memo=memo,
                output_path=output_path,
                trace_path=context.artifacts["trace_path"],
            )
            ro.setdefault("outputs", {})["agent_history"] = context.agent_history

            # J5.5.9 – inject workflow block
            if orchestrator_meta:
                ro["workflow"] = {
                    "iterations": orchestrator_meta.get("iterations", 0),
                    "workflow_path": orchestrator_meta.get("workflow_path", []),
                    "termination_reason": orchestrator_meta.get("termination_reason", "COMPLETE"),
                }

            # J5.4.7 – inject report block
            ro["report"] = {
                "executive_summary": executive_summary,
                "key_findings": findings,
                "key_risks": risks,
                "open_questions": open_questions,
                "report_confidence": report_conf,
                "report_grounding_score": grounding,
            }

            # J5.6a – inject profile coverage metadata
            ro["profiles_requested"] = context.profiles
            ro["profile_count"] = len(context.profiles)
            ro["profiles_contributing"] = qa.get("profiles_contributing", [])
            ro["profiles_missing"] = qa.get("profiles_missing", [])
            ro["coverage_status"] = qa.get("coverage_status", "sufficient")
            if raw_coverage:
                ro["profile_coverage"] = {
                    pname: {
                        "coverage": entry.get("coverage_level", "NONE").lower(),
                        "evidence_count": entry.get("evidence_count", 0),
                    }
                    for pname, entry in raw_coverage.items()
                }

            ro_path = write_research_object(ro, out_dir=output_path.parent)
            ro_stub = research_object_trace_stub(ro, ro_path)
            trace_payload["research_object"] = ro_stub
            # J6.1a — surface RO validation as a top-level trace block
            trace_payload["research_object_validation"] = ro_stub.get("research_object_validation", {})
            context.research_object = ro
            context.artifacts["research_object_path"] = str(ro_path)

        write_trace(trace_payload, output_path)
        return context
