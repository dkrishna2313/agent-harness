"""QAAgent – research review: coverage, evidence sufficiency, contradictions, confidence (J5.3)."""

from __future__ import annotations

import logging
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coverage helpers
# ---------------------------------------------------------------------------

def _validate_coverage(
    coverage_by_subquestion: dict[str, dict],
    subquestions: list[str],
) -> list[dict[str, Any]]:
    """Flag subquestions with NONE or WEAK coverage."""
    issues = []
    for sq in subquestions:
        cov = coverage_by_subquestion.get(sq, {})
        level = cov.get("coverage", "NONE")
        count = cov.get("evidence_count", 0)
        if level in ("NONE", "WEAK"):
            issues.append({
                "type": "coverage",
                "subquestion": sq,
                "coverage_level": level,
                "evidence_count": count,
                "severity": "HIGH" if level == "NONE" else "MEDIUM",
                "message": f"Subquestion has {level} coverage ({count} evidence item(s))",
            })
    return issues


def _validate_evidence_sufficiency(
    evidence_by_subquestion: dict[str, list[str]],
    subquestions: list[str],
) -> list[dict[str, Any]]:
    """Flag subquestions with zero or one evidence item."""
    issues = []
    for sq in subquestions:
        items = evidence_by_subquestion.get(sq, [])
        count = len(items)
        if count == 0:
            issues.append({
                "type": "evidence_sufficiency",
                "subquestion": sq,
                "evidence_count": count,
                "severity": "HIGH",
                "message": "Subquestion has no supporting evidence (unsupported)",
            })
        elif count == 1:
            issues.append({
                "type": "evidence_sufficiency",
                "subquestion": sq,
                "evidence_count": count,
                "severity": "MEDIUM",
                "message": "Subquestion is weakly supported (single evidence item)",
            })
    return issues


def _review_contradictions(research_object: dict) -> list[dict[str, Any]]:
    """Surface any contradictions recorded on the Research Object."""
    issues = []
    for c in research_object.get("contradictions", []):
        issues.append({
            "type": "contradiction",
            "contradiction_id": c.get("contradiction_id", ""),
            "severity": c.get("severity", "unknown"),
            "topic": c.get("topic", ""),
            "message": (
                f"Contradiction detected: {c.get('contradiction_id', '?')} "
                f"[{c.get('severity', '?')}] topic={c.get('topic', '?')}"
            ),
        })
    return issues


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _confidence_level(score: float) -> str:
    if score >= 0.7:
        return "HIGH"
    if score >= 0.4:
        return "MEDIUM"
    return "LOW"


def _assess_confidence(
    subquestions: list[str],
    coverage_issues: list[dict],
    evidence_issues: list[dict],
    contradiction_issues: list[dict],
    total_evidence: int,
) -> dict[str, Any]:
    n = len(subquestions) or 1

    # Coverage confidence: fraction of subquestions with better than NONE coverage
    none_count = sum(1 for i in coverage_issues if i["coverage_level"] == "NONE")
    coverage_score = max(0.0, 1.0 - (none_count / n))
    coverage_conf = _confidence_level(coverage_score)

    # Evidence confidence: penalise for unsupported/weakly-supported subquestions
    high_ev = sum(1 for i in evidence_issues if i["severity"] == "HIGH")
    med_ev = sum(1 for i in evidence_issues if i["severity"] == "MEDIUM")
    evidence_score = max(0.0, 1.0 - (high_ev * 0.3 + med_ev * 0.1) / n)
    # Also reward having enough total evidence
    if total_evidence >= 20:
        evidence_score = min(1.0, evidence_score + 0.1)
    evidence_conf = _confidence_level(evidence_score)

    # Contradiction confidence: penalise for high-severity contradictions
    high_contra = sum(1 for i in contradiction_issues if i["severity"] in ("high", "HIGH"))
    contradiction_score = max(0.0, 1.0 - high_contra * 0.2)
    contradiction_conf = _confidence_level(contradiction_score)

    overall_score = (coverage_score + evidence_score + contradiction_score) / 3.0
    overall_conf = _confidence_level(overall_score)

    return {
        "overall_confidence": overall_conf,
        "coverage_confidence": coverage_conf,
        "evidence_confidence": evidence_conf,
        "contradiction_confidence": contradiction_conf,
        "scores": {
            "coverage": round(coverage_score, 3),
            "evidence": round(evidence_score, 3),
            "contradiction": round(contradiction_score, 3),
            "overall": round(overall_score, 3),
        },
    }


# ---------------------------------------------------------------------------
# Per-profile coverage (J5.6)
# ---------------------------------------------------------------------------

def _check_profile_coverage(
    evidence_note: dict,
    profiles: list[str],
) -> dict:
    """Compute per-profile coverage and surface profile-level gaps (J5.6 / J5.6a).

    Returns a dict with keys:
      profile_coverage     – {name: "strong"|"moderate"|"weak"|"none"}  (flat, for routing)
      profile_issues       – detailed issue dicts (existing format)
      profile_gap_issues   – simplified gap dicts: [{profile, issue}]  (spec format)
      profiles_contributing – names with ≥1 attributed evidence item
      profiles_missing      – names with 0 attributed evidence items
      coverage_status       – "sufficient" | "insufficient"
    """
    raw: dict = evidence_note.get("profile_coverage_by_profile", {})
    # Prefer pre-computed lists from EvidenceAgent; fall back to deriving them
    contributing_pre: list = evidence_note.get("profiles_contributing", [])
    missing_pre: list = evidence_note.get("profiles_missing", [])

    coverage: dict[str, str] = {}
    issues: list[dict] = []
    gap_issues: list[dict] = []

    for pname in profiles:
        entry = raw.get(pname, {})
        level_raw = entry.get("coverage_level", "NONE")
        level = level_raw.lower()
        coverage[pname] = level
        if level == "none":
            count = entry.get("evidence_count", 0)
            issues.append({
                "type": "profile_coverage",
                "profile": pname,
                "coverage_level": "NONE",
                "severity": "HIGH",
                "message": f"Profile '{pname}' has no attributed evidence",
            })
            gap_issues.append({"profile": pname, "issue": "no evidence contributed"})
        elif level == "weak":
            count = entry.get("evidence_count", 0)
            issues.append({
                "type": "profile_coverage",
                "profile": pname,
                "coverage_level": "WEAK",
                "severity": "MEDIUM",
                "message": f"Profile '{pname}' has weak evidence coverage ({count} item(s))",
            })
            gap_issues.append({"profile": pname, "issue": f"weak coverage ({count} item(s))"})

    # Use pre-computed lists when available; otherwise derive from coverage dict
    if contributing_pre or missing_pre:
        profiles_contributing = contributing_pre
        profiles_missing = missing_pre
    else:
        profiles_contributing = [p for p in profiles if coverage.get(p, "none") != "none"]
        profiles_missing = [p for p in profiles if coverage.get(p, "none") == "none"]

    coverage_status = "insufficient" if profiles_missing else "sufficient"

    return {
        "profile_coverage": coverage,
        "profile_issues": issues,
        "profile_gap_issues": gap_issues,
        "profiles_contributing": profiles_contributing,
        "profiles_missing": profiles_missing,
        "coverage_status": coverage_status,
    }


# ---------------------------------------------------------------------------
# Next-action decision (J5.5.4)
# ---------------------------------------------------------------------------

def _decide_next_action(
    overall_confidence: str,
    coverage_issues: list[dict],
    evidence_issues: list[dict],
    iteration_count: int,
) -> str:
    """Choose the next workflow action based on QA findings.

    REQUEST_REPLAN  — confidence is LOW AND coverage is very poor (< 30 % covered):
                      the research plan itself needs rethinking.
    REQUEST_EVIDENCE — confidence is LOW but partial coverage exists:
                       more evidence retrieval may fill the gaps.
    CONTINUE        — confidence is MEDIUM or HIGH, or iteration limit exceeded.
    """
    from .context import NextAction

    if overall_confidence != "LOW":
        return NextAction.CONTINUE

    high_none_issues = sum(
        1 for i in coverage_issues if i.get("coverage_level") == "NONE"
    )
    high_ev_issues = sum(
        1 for i in evidence_issues if i.get("severity") == "HIGH"
    )

    # Very poor coverage → suggest re-plan
    if high_none_issues >= 3 and high_ev_issues >= 2:
        return NextAction.REQUEST_REPLAN

    # Moderate gaps → request more evidence
    if high_none_issues >= 1 or high_ev_issues >= 1:
        return NextAction.REQUEST_EVIDENCE

    return NextAction.CONTINUE


# ---------------------------------------------------------------------------
# Hypothesis validation (J6.3)
# ---------------------------------------------------------------------------

def _validate_hypotheses(hypotheses: list[dict]) -> dict[str, Any]:
    """Check that the hypothesis set meets structural quality requirements."""
    if not hypotheses:
        return {
            "hypotheses_present": False,
            "hypothesis_count": 0,
            "all_have_evidence_mapping": False,
            "all_have_confidence": False,
            "all_have_decision_implications": False,
            "all_have_disconfirming_evidence_needs": False,
            "issues": ["No hypotheses generated"],
        }

    issues: list[str] = []

    has_mapping = all(
        isinstance(h.get("supporting_evidence"), list)
        or isinstance(h.get("contradicting_evidence"), list)
        or isinstance(h.get("evidence_gaps"), list)
        for h in hypotheses
    )
    has_confidence = all(
        h.get("confidence") in ("high", "medium", "low") for h in hypotheses
    )
    has_implications = all(
        isinstance(h.get("decision_implications"), list) and len(h.get("decision_implications", [])) > 0
        for h in hypotheses
    )
    has_disconfirming = all(
        isinstance(h.get("disconfirming_evidence_needed"), list)
        and len(h.get("disconfirming_evidence_needed", [])) > 0
        for h in hypotheses
    )

    if not has_mapping:
        issues.append("One or more hypotheses missing evidence mapping")
    if not has_confidence:
        issues.append("One or more hypotheses missing valid confidence level")
    if not has_implications:
        issues.append("One or more hypotheses missing decision implications")
    if not has_disconfirming:
        issues.append("One or more hypotheses missing disconfirming evidence needs")
    if len(hypotheses) < 3:
        issues.append(f"Only {len(hypotheses)} hypothesis/hypotheses generated (minimum 3 required)")

    return {
        "hypotheses_present": True,
        "hypothesis_count": len(hypotheses),
        "all_have_evidence_mapping": has_mapping,
        "all_have_confidence": has_confidence,
        "all_have_decision_implications": has_implications,
        "all_have_disconfirming_evidence_needs": has_disconfirming,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Challenge validation (J6.4)
# ---------------------------------------------------------------------------

def _validate_challenges(
    challenges: list[dict],
    surviving: list[dict],
    hypotheses: list[dict],
) -> dict[str, Any]:
    """Check that challenge output meets structural quality requirements."""
    hypothesis_ids = {h.get("id", "") for h in hypotheses}
    challenged_ids = {c.get("hypothesis_id", "") for c in challenges}
    surviving_ids = {s.get("hypothesis_id", "") for s in surviving}

    issues: list[str] = []

    if not challenges:
        return {
            "challenges_present": False,
            "challenge_count": 0,
            "all_hypotheses_challenged": False,
            "all_have_falsification_tests": False,
            "surviving_hypotheses_present": False,
            "issues": ["No challenges generated"],
        }

    all_challenged = hypothesis_ids <= challenged_ids
    all_have_falsification = all(
        isinstance(c.get("falsification_tests"), list) and len(c.get("falsification_tests", [])) > 0
        for c in challenges
    )
    all_have_hidden_assumptions = all(
        isinstance(c.get("hidden_assumptions"), list) and len(c.get("hidden_assumptions", [])) > 0
        for c in challenges
    )
    valid_robustness = all(
        c.get("robustness") in ("low", "medium", "high") for c in challenges
    )
    valid_survival = all(
        s.get("survival_status") in ("strong", "moderate", "weak") for s in surviving
    )

    if not all_challenged:
        missing = hypothesis_ids - challenged_ids
        issues.append(f"Hypotheses not challenged: {', '.join(sorted(missing))}")
    if not all_have_falsification:
        issues.append("One or more challenges missing falsification tests")
    if not all_have_hidden_assumptions:
        issues.append("One or more challenges missing hidden assumptions")
    if not valid_robustness:
        issues.append("One or more challenges have invalid robustness value")
    if surviving and not valid_survival:
        issues.append("One or more surviving hypotheses have invalid survival_status")

    return {
        "challenges_present": True,
        "challenge_count": len(challenges),
        "all_hypotheses_challenged": all_challenged,
        "all_have_falsification_tests": all_have_falsification,
        "all_have_hidden_assumptions": all_have_hidden_assumptions,
        "robustness_valid": valid_robustness,
        "surviving_hypotheses_present": len(surviving) > 0,
        "surviving_hypothesis_count": len(surviving),
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Recommendation validation (J6.5)
# ---------------------------------------------------------------------------

def _validate_contradiction_hardening(context: "AgentContext") -> dict[str, Any]:
    """Check that the contradiction hardening pipeline ran and suppressed false positives (J6.5a/b/c)."""
    metrics: dict = context.contradiction_metrics or {}
    suppressed_count: int = metrics.get("suppressed_count", 0)
    final_count: int = metrics.get("final_count", 0)
    by_reason: dict = metrics.get("by_reason", {})

    scope_filtering_present = bool(
        by_reason.get("scope_mismatch", 0) or by_reason.get("metric_scope_mismatch", 0)
        or metrics.get("scope_filtering_present")
    )
    entity_filtering_present = bool(
        by_reason.get("entity_mismatch", 0) or metrics.get("entity_filtering_present")
    )
    temporal_filtering_present = bool(
        by_reason.get("temporal_progression", 0) or metrics.get("temporal_filtering_present")
    )
    context_filtering_present = bool(
        by_reason.get("context_mismatch", 0) or metrics.get("context_filtering_present")
    )

    eligibility_engine = metrics.get("eligibility_engine", {})

    issues: list[str] = []
    if not metrics:
        issues.append("contradiction_metrics not populated — EvidenceAgent may not have run")

    return {
        "scope_filtering_present": scope_filtering_present,
        "entity_filtering_present": entity_filtering_present,
        "temporal_filtering_present": temporal_filtering_present,
        "context_filtering_present": context_filtering_present,
        "suppressed_count": suppressed_count,
        "final_count": final_count,
        "by_reason": by_reason,
        "eligibility_engine": eligibility_engine,
        "issues": issues,
    }


def _validate_numeric_semantics(context: "AgentContext") -> dict[str, Any]:
    """J6.5d: Validate that numeric semantic gates ran and classified correctly."""
    metrics: dict = context.contradiction_metrics or {}
    by_reason: dict = metrics.get("by_reason", {})
    ns: dict = metrics.get("numeric_semantics", {})

    threshold_filtering_present = bool(
        by_reason.get("threshold_vs_measurement", 0)
        or metrics.get("threshold_filtering_present")
    )
    historical_filtering_present = bool(
        by_reason.get("historical_progression", 0)
        or metrics.get("historical_filtering_present")
    )
    range_logic_present = bool(
        by_reason.get("range_average_compatible", 0)
        or metrics.get("range_filtering_present")
    )

    return {
        "range_logic_present": range_logic_present,
        "threshold_logic_present": threshold_filtering_present,
        "historical_logic_present": historical_filtering_present,
        "projection_logic_present": bool(
            by_reason.get("temporal_progression", 0)
            or by_reason.get("projection_progression", 0)
            or metrics.get("temporal_filtering_present")
        ),
        "numeric_semantics": ns,
        "issues": [],
    }


def _validate_contradiction_decision_logic(context: "AgentContext") -> dict[str, Any]:
    """J6.5c: Validate that the eligibility engine ran and key filtering gates are active."""
    metrics: dict = context.contradiction_metrics or {}
    eligibility_engine: dict = metrics.get("eligibility_engine", {})
    by_reason: dict = metrics.get("by_reason", {})

    eligibility_engine_present = bool(eligibility_engine)
    scope_filtering_present = bool(
        by_reason.get("scope_mismatch", 0) or by_reason.get("metric_scope_mismatch", 0)
        or metrics.get("scope_filtering_present")
    )
    entity_filtering_present = bool(
        by_reason.get("entity_mismatch", 0) or metrics.get("entity_filtering_present")
    )
    context_filtering_present = bool(
        by_reason.get("context_mismatch", 0) or metrics.get("context_filtering_present")
    )

    suppressed = eligibility_engine.get("suppressed_pairs", metrics.get("suppressed_count", 0))
    eligible = eligibility_engine.get("eligible_pairs", metrics.get("final_count", 0))
    candidate = eligibility_engine.get("candidate_pairs", metrics.get("candidate_count", 0))

    issues: list[str] = []
    if not eligibility_engine_present:
        issues.append("eligibility_engine not present in contradiction_metrics")

    return {
        "eligibility_engine_present": eligibility_engine_present,
        "scope_filtering_present": scope_filtering_present,
        "entity_filtering_present": entity_filtering_present,
        "context_filtering_present": context_filtering_present,
        "candidate_pairs": candidate,
        "suppressed_pairs": suppressed,
        "eligible_pairs": eligible,
        "issues": issues,
    }


def _validate_recommendations(
    recommendations: list[dict],
    portfolio: dict,
) -> dict[str, Any]:
    """Check that recommendation output meets structural quality requirements."""
    if not recommendations:
        return {
            "recommendations_present": False,
            "recommendation_count": 0,
            "all_have_evidence": False,
            "all_have_hypothesis_links": False,
            "all_have_confidence": False,
            "all_have_time_horizon": False,
            "portfolio_present": False,
            "issues": ["No recommendations generated"],
        }

    issues: list[str] = []
    valid_horizons = {"near_term", "medium_term", "long_term"}
    valid_confidence = {"high", "medium", "low"}
    valid_priority = {"high", "medium", "low"}

    all_have_evidence = all(
        isinstance(r.get("supporting_evidence"), list) and len(r.get("supporting_evidence", [])) > 0
        for r in recommendations
    )
    all_have_hyp_links = all(
        isinstance(r.get("supported_by_hypotheses"), list) and len(r.get("supported_by_hypotheses", [])) > 0
        for r in recommendations
    )
    all_have_confidence = all(
        r.get("confidence") in valid_confidence for r in recommendations
    )
    all_have_horizon = all(
        r.get("time_horizon") in valid_horizons for r in recommendations
    )
    all_have_priority = all(
        r.get("priority") in valid_priority for r in recommendations
    )

    if not all_have_evidence:
        issues.append("One or more recommendations missing supporting evidence IDs")
    if not all_have_hyp_links:
        issues.append("One or more recommendations missing hypothesis links")
    if not all_have_confidence:
        issues.append("One or more recommendations have invalid confidence value")
    if not all_have_horizon:
        issues.append("One or more recommendations have invalid time_horizon value")
    if not all_have_priority:
        issues.append("One or more recommendations have invalid priority value")

    portfolio_present = bool(portfolio and (
        portfolio.get("near_term") or portfolio.get("medium_term") or portfolio.get("long_term")
    ))
    if not portfolio_present:
        issues.append("Recommendation portfolio is empty or missing")

    return {
        "recommendations_present": True,
        "recommendation_count": len(recommendations),
        "all_have_evidence": all_have_evidence,
        "all_have_hypothesis_links": all_have_hyp_links,
        "all_have_confidence": all_have_confidence,
        "all_have_time_horizon": all_have_horizon,
        "all_have_priority": all_have_priority,
        "portfolio_present": portfolio_present,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class QAAgent(FunctionalAgent):
    """Research reviewer: validates coverage, evidence sufficiency, contradictions,
    and produces a confidence assessment (J5.3).

    Reads:  context.plan, context.evidence_notes, context.research_object
    Writes: context.qa, context.qa_notes, context.research_object["qa"],
            agent_history entry
    """

    def _execute(self, context: AgentContext) -> AgentContext:
        from research_agent.log import PROGRESS

        # --- inputs from upstream agents ---
        plan = context.plan
        subquestions: list[str] = plan.get("subquestions", [])

        evidence_note = context.evidence_notes[0] if context.evidence_notes else {}
        coverage_by_subquestion: dict = evidence_note.get("coverage_by_subquestion", {})
        evidence_by_subquestion: dict = evidence_note.get("evidence_by_subquestion", {})
        evidence_summary: dict = evidence_note.get("evidence_summary", {})
        total_evidence = evidence_summary.get("total_evidence_items", 0)

        ro = context.research_object

        # --- run QA checks ---
        coverage_issues = _validate_coverage(coverage_by_subquestion, subquestions)
        evidence_issues = _validate_evidence_sufficiency(evidence_by_subquestion, subquestions)
        contradiction_issues = _review_contradictions(ro)

        confidence = _assess_confidence(
            subquestions,
            coverage_issues,
            evidence_issues,
            contradiction_issues,
            total_evidence,
        )

        covered = sum(
            1 for sq in subquestions
            if coverage_by_subquestion.get(sq, {}).get("coverage", "NONE") != "NONE"
        )

        # --- per-profile coverage (J5.6 / J5.6a) ---
        pc_result = _check_profile_coverage(evidence_note, context.profiles)
        profile_coverage     = pc_result["profile_coverage"]
        profile_issues       = pc_result["profile_issues"]
        profile_gap_issues   = pc_result["profile_gap_issues"]
        profiles_contributing = pc_result["profiles_contributing"]
        profiles_missing      = pc_result["profiles_missing"]
        coverage_status       = pc_result["coverage_status"]

        total_issues = (
            len(coverage_issues) + len(evidence_issues)
            + len(contradiction_issues) + len(profile_issues)
        )

        qa_summary = {
            "subquestions_total": len(subquestions),
            "subquestions_covered": covered,
            "subquestions_uncovered": len(subquestions) - covered,
            "coverage_issues": len(coverage_issues),
            "evidence_issues": len(evidence_issues),
            "contradiction_issues": len(contradiction_issues),
            "profile_issues": len(profile_issues),
            "issues_found": total_issues,
            "profiles_evaluated": len(context.profiles),
        }

        # --- hypothesis validation (J6.3) ---
        hypothesis_validation = _validate_hypotheses(context.hypotheses)

        # --- challenge validation (J6.4) ---
        challenge_validation = _validate_challenges(
            context.hypothesis_challenges,
            context.surviving_hypotheses,
            context.hypotheses,
        )

        # --- recommendation validation (J6.5) ---
        recommendation_validation = _validate_recommendations(
            context.recommendations,
            context.recommendation_portfolio,
        )

        # --- contradiction hardening validation (J6.5a) ---
        contradiction_hardening_validation = _validate_contradiction_hardening(context)

        # --- contradiction decision logic validation (J6.5c) ---
        contradiction_decision_validation = _validate_contradiction_decision_logic(context)

        # --- numeric semantic validation (J6.5d) ---
        numeric_semantic_validation = _validate_numeric_semantics(context)

        # --- write context.qa (J5.3.1) ---
        context.qa = {
            "coverage_issues": coverage_issues,
            "evidence_issues": evidence_issues,
            "contradiction_issues": contradiction_issues,
            "profile_coverage": profile_coverage,
            "profile_gap_issues": profile_gap_issues,
            "profiles_requested": list(context.profiles),
            "profiles_contributing": profiles_contributing,
            "profiles_missing": profiles_missing,
            "coverage_status": coverage_status,
            "confidence_assessment": confidence,
            "qa_summary": qa_summary,
            "hypothesis_validation": hypothesis_validation,
            "challenge_validation": challenge_validation,
            "recommendation_validation": recommendation_validation,
            "contradiction_validation": contradiction_hardening_validation,
            "contradiction_decision_validation": contradiction_decision_validation,
            "numeric_semantic_validation": numeric_semantic_validation,
        }

        # --- update Research Object (J5.3.7) ---
        if ro:
            ro["qa"] = context.qa

        # --- decide next workflow action (J5.5.4 / J5.5.8) ---
        next_action = _decide_next_action(
            overall_confidence=confidence["overall_confidence"],
            coverage_issues=coverage_issues,
            evidence_issues=evidence_issues,
            iteration_count=context.iteration_count,
        )

        LOGGER.log(
            PROGRESS,
            "[QAAgent] confidence=%s  issues=%d (coverage=%d evidence=%d "
            "contradictions=%d profile=%d)  next_action=%s",
            confidence["overall_confidence"],
            total_issues,
            len(coverage_issues),
            len(evidence_issues),
            len(contradiction_issues),
            len(profile_issues),
            next_action,
        )

        # --- qa_notes (backward compat with existing callers) ---
        status = "success" if total_issues == 0 else "warning"
        summary = (
            f"QA complete: confidence={confidence['overall_confidence']}, "
            f"{total_issues} issue(s) found across "
            f"{len(subquestions)} subquestions. "
            f"next_action={next_action}"
        )
        context.qa_notes.append(
            self._make_note(status=status, summary=summary, qa_summary=qa_summary)
        )

        # --- agent history (J5.3.9 / J5.5.8) ---
        self._record(
            context,
            status=status,
            summary=summary,
            issues_found=total_issues,
            overall_confidence=confidence["overall_confidence"],
            next_action=next_action,
        )
        return context
