"""Recommendation Improvement Stress Test (J6.7a).

Generates four synthetic weak recommendations — one isolated weakness each —
then runs them through the full Evaluator → ImprovementAgent → Re-Evaluator
cycle and produces a before/after proof report.

Public API
----------
SYNTHETIC_WEAK_RECS   – the four canonical synthetic recommendations
run_stress_test()     – execute the full loop; returns a results dict
build_report_section()– markdown table from results
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic weak recommendations (one isolated weakness each)
# ---------------------------------------------------------------------------

SYNTHETIC_WEAK_RECS: list[dict[str, Any]] = [
    # ---- 1. No tradeoff language ----------------------------------------
    {
        "id": "STRESS_TRADEOFF",
        "_weakness_type": "no_tradeoff_awareness",
        "title": "Adopt Direct Liquid Cooling Infrastructure for AI Racks",
        # Summary intentionally < 150 chars so has_length bonus does not inflate tradeoff_score.
        "summary": (
            "Facilities must adopt direct liquid cooling systems for AI data center "
            "deployments to enable rack densities above 30 kilowatts per rack unit."
        ),
        "priority": "high",
        "time_horizon": "near_term",
        "supported_by_hypotheses": ["H1"],
        "supporting_evidence": ["E001"],
        "key_risks": [
            "Capital cost may exceed budget for greenfield deployments",
            "Vendor lock-in risk with proprietary cooling manifolds",
        ],
        "trigger_conditions": [],
        "confidence": "high",
        "confidence_rationale": (
            "Direct liquid cooling performance advantages are consistently reported "
            "across hyperscale operator case studies and equipment vendor benchmarks."
        ),
    },
    # ---- 2. No risk identification ---------------------------------------
    {
        "id": "STRESS_RISK",
        "_weakness_type": "no_risk_identification",
        "title": "Deploy Distributed Power Grid Management Infrastructure",
        "summary": (
            "However, AI data centers must deploy distributed power grid management "
            "software to optimize energy consumption across compute facilities. "
            "Vendor-neutral power APIs enable integration with existing PDUs and "
            "support AI workload scheduling based on real-time power availability."
        ),
        "priority": "high",
        "time_horizon": "medium_term",
        "supported_by_hypotheses": ["H2"],
        "supporting_evidence": ["E002", "E003"],
        "key_risks": [],
        "trigger_conditions": ["grid capacity utilization exceeds 85%"],
        "confidence": "medium",
        "confidence_rationale": (
            "Grid constraints are well documented in utility interconnection studies "
            "and are confirmed by multiple data center operator case analyses."
        ),
    },
    # ---- 3. No evidence references ---------------------------------------
    {
        "id": "STRESS_EVIDENCE",
        "_weakness_type": "missing_evidence_links",
        "title": "Commission Regional Inference Capacity Expansion",
        "summary": (
            "Operators must commission regional inference capacity expansions to "
            "reduce latency and meet demand growth. However, distributed inference "
            "infrastructure requires careful site selection, power contracting, and "
            "network interconnection planning to avoid stranded capital investments."
        ),
        "priority": "high",
        "time_horizon": "near_term",
        "supported_by_hypotheses": ["H3"],
        "supporting_evidence": [],
        "key_risks": [
            "Permitting and zoning delays may extend timelines by 12–18 months",
            "Power contract terms may limit flexibility for demand-responsive operation",
            "Network interconnection costs may vary significantly by region",
        ],
        "trigger_conditions": ["latency SLA thresholds exceeded in target region"],
        "confidence": "medium",
        "confidence_rationale": (
            "Regional demand growth patterns are supported by cloud provider capacity "
            "announcements and operator colocation demand projections."
        ),
    },
    # ---- 4. Weak actionability (hedge-dominated) -------------------------
    {
        "id": "STRESS_ACTIONABILITY",
        "_weakness_type": "low_actionability",
        # No action-verb substrings in title or summary — actionability_score = 0.0
        "title": "Continue Monitoring Technology Trends",
        # No action verbs present; hedge_dominated=True so actionability_score = 0.0
        "summary": (
            "Monitor and evaluate cooling technology trends as they emerge. "
            "Consider exploring potential new solutions when appropriate conditions allow. "
            "Continue assessing emerging GPU architectures and review market conditions "
            "regularly to determine if further study is warranted."
        ),
        "priority": None,
        "time_horizon": None,
        "supported_by_hypotheses": [],
        "supporting_evidence": ["E004"],
        "key_risks": ["Risk of premature technology commitment"],
        "trigger_conditions": [],
        "confidence": "low",
        "confidence_rationale": "Insufficient evidence to recommend a definitive action.",
    },
]

_DIMENSION_KEYS = (
    "evidence_support_score",
    "reasoning_score",
    "tradeoff_score",
    "risk_score",
    "actionability_score",
)

_AVAILABLE_EVIDENCE_IDS = ["E001", "E002", "E003", "E004", "E005", "E006"]


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_stress_test(
    out_path: Path | str | None = None,
    available_evidence_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Run the full Evaluator → ImprovementAgent → Re-Evaluator stress test.

    Parameters
    ----------
    out_path:
        When provided, writes two artefacts alongside the markdown report:
        ``j67a_stress_test_results.json`` and ``j67a_stress_test_report.md``.
    available_evidence_ids:
        Evidence IDs that the improvement agent can link to unlinked
        recommendations.  Defaults to a small synthetic set.

    Returns
    -------
    Full results dict (see below).
    """
    from research_agent.evaluation.recommendation_evaluator import evaluate_recommendations
    from .recommendation_improvement_agent import improve_recommendations

    ev_ids = available_evidence_ids or _AVAILABLE_EVIDENCE_IDS

    # ------------------------------------------------------------------
    # Step 1 — Evaluate synthetic weak recommendations
    # ------------------------------------------------------------------
    before_eval = evaluate_recommendations(SYNTHETIC_WEAK_RECS)
    before_map: dict[str, dict] = {
        s["recommendation_id"]: s
        for s in before_eval["recommendation_scores"]
    }

    # ------------------------------------------------------------------
    # Step 2 — Run improvement agent
    # ------------------------------------------------------------------
    improvement_result = improve_recommendations(
        SYNTHETIC_WEAK_RECS,
        before_eval,
        available_evidence_ids=ev_ids,
    )
    improved_recs: list[dict] = improvement_result["improved_recommendations"]

    # ------------------------------------------------------------------
    # Step 3 — Re-evaluate improved recommendations
    # ------------------------------------------------------------------
    after_eval = evaluate_recommendations(improved_recs)
    after_map: dict[str, dict] = {
        s["recommendation_id"]: s
        for s in after_eval["recommendation_scores"]
    }

    # ------------------------------------------------------------------
    # Step 4 — Build per-recommendation comparison table
    # ------------------------------------------------------------------
    comparison: list[dict] = []
    for rec in SYNTHETIC_WEAK_RECS:
        rid = rec["id"]
        b = before_map.get(rid, {})
        a = after_map.get(rid, {})
        before_score = b.get("recommendation_score", 0.0)
        after_score = a.get("recommendation_score", 0.0)
        delta = round(after_score - before_score, 3)
        comparison.append({
            "recommendation_id": rid,
            "title": rec["title"],
            "weakness_type": rec.get("_weakness_type", "unknown"),
            "before_score": before_score,
            "after_score": after_score,
            "delta": delta,
            "improved": delta > 0,
            "primary_penalty_before": b.get("primary_penalty"),
            "primary_penalty_after": a.get("primary_penalty"),
            "before_dimensions": {k: b.get(k) for k in _DIMENSION_KEYS},
            "after_dimensions": {k: a.get(k) for k in _DIMENSION_KEYS},
        })

    # ------------------------------------------------------------------
    # Step 5 — QA validation
    # ------------------------------------------------------------------
    metrics = improvement_result["improvement_metrics"]
    improvement_loop_validated = any(c["improved"] for c in comparison)

    qa_validation = {
        "improvement_loop_validated": improvement_loop_validated,
        "recommendations_improved": metrics["recommendations_improved"],
        "recommendations_unchanged": metrics["recommendations_unchanged"],
        "average_score_before": metrics["average_score_before"],
        "average_score_after": metrics["average_score_after"],
        "average_delta": metrics["average_delta"],
    }

    # Traceability list
    traceability = [
        {
            "recommendation_id": c["recommendation_id"],
            "before_score": c["before_score"],
            "after_score": c["after_score"],
            "delta": c["delta"],
        }
        for c in comparison
    ]

    results: dict[str, Any] = {
        "synthetic_recommendations": SYNTHETIC_WEAK_RECS,
        "before_evaluation": before_eval,
        "improvement_result": improvement_result,
        "after_evaluation": after_eval,
        "comparison": comparison,
        "improvement_metrics": metrics,
        "recommendation_history": improvement_result["recommendation_history"],
        "qa_validation": qa_validation,
        "recommendation_improvements": traceability,
    }

    if out_path:
        _write_artefacts(results, Path(out_path))

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_report_section(results: dict[str, Any]) -> str:
    """Return a markdown ``## Recommendation Improvement Validation`` section."""
    comparison = results.get("comparison", [])
    metrics = results.get("improvement_metrics", {})
    qa = results.get("qa_validation", {})

    lines: list[str] = [
        "## Recommendation Improvement Validation",
        "",
        "Stress-test results: four synthetic weak recommendations processed through "
        "Evaluator → ImprovementAgent → Re-Evaluator.",
        "",
        "| Recommendation | Weakness Type | Before | After | Delta |",
        "|----------------|---------------|--------|-------|-------|",
    ]
    for c in comparison:
        rid = c["recommendation_id"]
        wtype = c["weakness_type"].replace("_", " ")
        before = f"{c['before_score']:.3f}"
        after = f"{c['after_score']:.3f}"
        delta_str = f"+{c['delta']:.3f}" if c["delta"] > 0 else f"{c['delta']:.3f}"
        lines.append(f"| {rid} | {wtype} | {before} | {after} | {delta_str} |")

    lines += [
        "",
        f"**Improvement Loop Validated:** {'✓' if qa.get('improvement_loop_validated') else '✗'}  ",
        f"**Recommendations Improved:** {metrics.get('recommendations_improved', 0)}"
        f" / {metrics.get('recommendations_improved', 0) + metrics.get('recommendations_unchanged', 0)}  ",
        f"**Average Score Before:** {metrics.get('average_score_before', 0.0):.3f}  ",
        f"**Average Score After:** {metrics.get('average_score_after', 0.0):.3f}  ",
        f"**Average Delta:** +{metrics.get('average_delta', 0.0):.3f}",
        "",
    ]

    # Per-rec dimension breakdown
    lines.append("### Dimension Score Breakdown")
    lines.append("")
    dim_labels = {
        "evidence_support_score": "Evidence",
        "reasoning_score": "Reasoning",
        "tradeoff_score": "Tradeoff",
        "risk_score": "Risk",
        "actionability_score": "Action",
    }
    header = "| Rec | " + " | ".join(dim_labels.values()) + " (Before → After) |"
    separator = "|-----|" + "|".join(["------"] * len(dim_labels)) + "|"
    lines.append(header)
    lines.append(separator)
    for c in comparison:
        cells: list[str] = [c["recommendation_id"]]
        bd = c.get("before_dimensions", {})
        ad = c.get("after_dimensions", {})
        for k in _DIMENSION_KEYS:
            bv = bd.get(k, 0.0) or 0.0
            av = ad.get(k, 0.0) or 0.0
            cells.append(f"{bv:.2f}→{av:.2f}")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Artefact writers
# ---------------------------------------------------------------------------

def _write_artefacts(results: dict[str, Any], out_path: Path) -> None:
    """Write JSON results and markdown report alongside out_path."""
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = out_path.stem.replace("j67a_stress_test", "j67a_stress_test")

    json_path = out_dir / f"{stem}_results.json"
    md_path = out_dir / f"{stem}_report.md"

    # Serialisable copy (strip internal _weakness_type keys from artefact)
    serialisable = {k: v for k, v in results.items() if k != "synthetic_recommendations"}
    serialisable["synthetic_recommendations"] = [
        {k2: v2 for k2, v2 in rec.items() if not k2.startswith("_")}
        for rec in results.get("synthetic_recommendations", [])
    ]

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2, default=str)
    LOGGER.info("[StressTest] Results written to %s", json_path)

    report_md = build_report_section(results)
    md_path.write_text(report_md, encoding="utf-8")
    LOGGER.info("[StressTest] Report written to %s", md_path)
