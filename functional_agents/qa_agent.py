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
