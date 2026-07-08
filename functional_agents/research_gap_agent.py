"""ResearchGapAgent — deterministic research completeness assessment (J12.0).

Runs immediately after HypothesisAgent and before StrategicSynthesisAgent.
Assesses the quality and completeness of the current research using heuristics
derived from evidence coverage, hypothesis support, and contradiction records.

This agent is ADDITIVE and OBSERVATIONAL only:
  - Does NOT modify retrieval
  - Does NOT generate new research questions
  - Does NOT trigger additional searches
  - Does NOT change downstream recommendations
  - Does NOT alter any existing reasoning outputs
  - Does NOT add to the report

Its sole responsibility is to produce a structured research quality assessment.

Writes to:
  - context.research_gap_analysis
  - context.research_object["research_gap_analysis"]
  - context.trace["_research_gap"]
"""

from __future__ import annotations

import logging
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)

# ── Health scoring weights ────────────────────────────────────────────────────
_PENALTY_NONE_COVERAGE    = 0.15   # per subquestion with NONE coverage
_PENALTY_WEAK_COVERAGE    = 0.07   # per subquestion with WEAK coverage
_PENALTY_MISSING_AREA     = 0.05   # per investigation area with 0 evidence
_PENALTY_HIGH_CONTRADICTION = 0.10 # per HIGH-severity contradiction

_HEALTH_GOOD = 0.75
_HEALTH_FAIR = 0.45

# A hypothesis is assumption-heavy when its confidence is high but its
# supporting evidence count falls below this threshold.
_ASSUMPTION_EVIDENCE_THRESHOLD = 2


# ── Pure heuristic functions (testable without a context) ─────────────────────

def _question_coverage_list(
    subquestions: list[str],
    coverage_by_subquestion: dict[str, dict],
) -> list[dict[str, Any]]:
    return [
        {
            "subquestion": sq,
            "coverage": coverage_by_subquestion.get(sq, {}).get("coverage", "NONE"),
            "evidence_count": coverage_by_subquestion.get(sq, {}).get("evidence_count", 0),
        }
        for sq in subquestions
    ]


def _weak_questions(question_coverage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [q for q in question_coverage if q["coverage"] in ("NONE", "WEAK")]


def _missing_areas(
    evidence_by_area: dict[str, list],
    investigation_areas: list[str],
) -> list[str]:
    return [
        area for area in investigation_areas
        if not evidence_by_area.get(area)
    ]


def _assumption_heavy_topics(hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hypotheses that claim high confidence but have sparse supporting evidence."""
    result = []
    for h in hypotheses:
        confidence = (h.get("confidence") or "").lower()
        if confidence != "high":
            continue
        supporting = h.get("supporting_evidence") or []
        count = len(supporting)
        if count < _ASSUMPTION_EVIDENCE_THRESHOLD:
            result.append({
                "hypothesis_id": h.get("id", ""),
                "title": h.get("title", ""),
                "confidence": confidence,
                "supporting_evidence_count": count,
            })
    return result


def _extract_contradictions(
    validated_contradictions: list[dict],
    research_object: dict,
) -> list[dict[str, Any]]:
    """Surface contradictions already detected elsewhere in the pipeline."""
    source = validated_contradictions or research_object.get("contradictions", [])
    out = []
    for c in source:
        entry: dict[str, Any] = {}
        for key in ("contradiction_id", "severity", "topic"):
            v = c.get(key)
            if v is not None:
                entry[key] = v
        if entry:
            out.append(entry)
    return out


def _recommended_followups(
    weak_qs: list[dict[str, Any]],
    missing: list[str],
) -> list[str]:
    followups: list[str] = []
    for q in weak_qs:
        sq = q["subquestion"]
        short = sq[:80] + "…" if len(sq) > 80 else sq
        followups.append(f"Gather additional evidence for: {short}")
    for area in missing:
        followups.append(f"Investigate area with no current evidence: {area}")
    return followups


def _overall_health(
    question_coverage: list[dict[str, Any]],
    missing_area_count: int,
    contradictions: list[dict[str, Any]],
) -> str:
    score = 1.0
    for q in question_coverage:
        lvl = q["coverage"]
        if lvl == "NONE":
            score -= _PENALTY_NONE_COVERAGE
        elif lvl == "WEAK":
            score -= _PENALTY_WEAK_COVERAGE
    score -= missing_area_count * _PENALTY_MISSING_AREA
    high_severity = sum(
        1 for c in contradictions
        if (c.get("severity") or "").lower() == "high"
    )
    score -= high_severity * _PENALTY_HIGH_CONTRADICTION
    score = max(0.0, min(1.0, score))
    if score >= _HEALTH_GOOD:
        return "GOOD"
    if score >= _HEALTH_FAIR:
        return "FAIR"
    return "POOR"


def _confidence_score(question_coverage: list[dict[str, Any]]) -> float:
    if not question_coverage:
        return 0.0
    covered = sum(
        1 for q in question_coverage
        if q["coverage"] in ("STRONG", "MODERATE")
    )
    return round(covered / len(question_coverage), 3)


def compute_research_gap_analysis(
    *,
    subquestions: list[str],
    investigation_areas: list[str],
    coverage_by_subquestion: dict[str, dict],
    evidence_by_area: dict[str, list],
    hypotheses: list[dict[str, Any]],
    validated_contradictions: list[dict],
    research_object: dict,
) -> dict[str, Any]:
    """Build the full ResearchGapAnalysis dict from deterministic heuristics."""
    qcov = _question_coverage_list(subquestions, coverage_by_subquestion)
    weak = _weak_questions(qcov)
    missing = _missing_areas(evidence_by_area, investigation_areas)
    assumption_heavy = _assumption_heavy_topics(hypotheses)
    contradictions = _extract_contradictions(validated_contradictions, research_object)
    followups = _recommended_followups(weak, missing)
    health = _overall_health(qcov, len(missing), contradictions)
    confidence = _confidence_score(qcov)

    return {
        "question_coverage": qcov,
        "weak_questions": weak,
        "missing_investigation_areas": missing,
        "assumption_heavy_topics": assumption_heavy,
        "contradictions": contradictions,
        "recommended_followups": followups,
        "overall_research_health": health,
        "confidence": confidence,
    }


# ── Agent ─────────────────────────────────────────────────────────────────────

class ResearchGapAgent(FunctionalAgent):
    """Deterministic research completeness assessment (J12.0)."""

    def _execute(self, context: AgentContext) -> AgentContext:
        from research_agent.log import PROGRESS

        # No-op guard: nothing to assess when evidence or plan is absent.
        if not context.evidence_notes or not context.plan:
            LOGGER.log(
                PROGRESS,
                "[ResearchGapAgent] skipped — evidence_notes=%d plan_keys=%d",
                len(context.evidence_notes), len(context.plan),
            )
            context.trace["_research_gap"] = {
                "skipped": True,
                "reason": "no_evidence_or_plan",
            }
            self._record(
                context,
                status="skipped",
                summary="No evidence or plan available — research gap analysis skipped.",
            )
            return context

        note = context.evidence_notes[0]
        coverage_by_subquestion: dict = note.get("coverage_by_subquestion", {})
        evidence_by_area: dict = note.get("evidence_by_area", {})

        subquestions: list = context.plan.get("subquestions", [])
        investigation_areas: list = context.plan.get("investigation_areas", [])
        hypotheses: list = context.hypotheses or []
        validated_contradictions: list = context.validated_contradictions or []
        research_object: dict = context.research_object or {}

        analysis = compute_research_gap_analysis(
            subquestions=subquestions,
            investigation_areas=investigation_areas,
            coverage_by_subquestion=coverage_by_subquestion,
            evidence_by_area=evidence_by_area,
            hypotheses=hypotheses,
            validated_contradictions=validated_contradictions,
            research_object=research_object,
        )

        context.research_gap_analysis = analysis
        if context.research_object is not None:
            context.research_object["research_gap_analysis"] = analysis
        context.trace["_research_gap"] = analysis

        LOGGER.log(
            PROGRESS,
            "[ResearchGapAgent] health=%s  confidence=%.2f  weak_questions=%d  "
            "missing_areas=%d  assumption_heavy=%d  contradictions=%d  followups=%d",
            analysis["overall_research_health"],
            analysis["confidence"],
            len(analysis["weak_questions"]),
            len(analysis["missing_investigation_areas"]),
            len(analysis["assumption_heavy_topics"]),
            len(analysis["contradictions"]),
            len(analysis["recommended_followups"]),
        )

        self._record(
            context,
            status="success",
            summary=(
                f"Research health: {analysis['overall_research_health']}  "
                f"confidence={analysis['confidence']:.2f}  "
                f"weak_questions={len(analysis['weak_questions'])}  "
                f"missing_areas={len(analysis['missing_investigation_areas'])}"
            ),
            overall_research_health=analysis["overall_research_health"],
            confidence=analysis["confidence"],
            weak_question_count=len(analysis["weak_questions"]),
            missing_area_count=len(analysis["missing_investigation_areas"]),
            assumption_heavy_count=len(analysis["assumption_heavy_topics"]),
            contradiction_count=len(analysis["contradictions"]),
            followup_count=len(analysis["recommended_followups"]),
        )
        return context
