"""ResearchGapAgent — deterministic research completeness & decision-support assessment.

J12.0 — Research coverage checker (evidence coverage, hypothesis quality, contradictions).
J12.1 — Decision-support evaluator (assumption validation, recommendation dependencies,
         strategic option support, executive confidence alignment).

Runs immediately after HypothesisAgent and before StrategicSynthesisAgent.
Assesses research quality and decision-support completeness using only deterministic
heuristics derived from pipeline context fields.

NOTE (J12.1): At this pipeline position the decision-layer agents
(AssumptionAgent, RecommendationAgent, etc.) have not yet run on the first
pass. Decision-support gaps will be populated on QA-loop second passes when
those fields are available in context and research_object.

This agent is ADDITIVE and OBSERVATIONAL only:
  - Does NOT modify retrieval
  - Does NOT generate new research questions
  - Does NOT trigger additional searches
  - Does NOT change downstream recommendations
  - Does NOT alter any existing reasoning outputs
  - Does NOT add to the report

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

# ── J12.0 Health scoring weights ─────────────────────────────────────────────
_PENALTY_NONE_COVERAGE      = 0.15   # per subquestion with NONE coverage
_PENALTY_WEAK_COVERAGE      = 0.07   # per subquestion with WEAK coverage
_PENALTY_MISSING_AREA       = 0.05   # per investigation area with 0 evidence
_PENALTY_HIGH_CONTRADICTION = 0.10   # per HIGH-severity contradiction

_HEALTH_GOOD = 0.75
_HEALTH_FAIR = 0.45

# Hypothesis flagged when it claims high confidence but has fewer than this
# many supporting evidence items.
_ASSUMPTION_EVIDENCE_THRESHOLD = 2

# ── J12.1 Decision-support health penalties ───────────────────────────────────
_PENALTY_DSG_NO_EVIDENCE   = 0.20   # Critical assumption with zero evidence items
_PENALTY_DSG_WEAK_EVIDENCE = 0.15   # Critical/weak evidence or low-confidence assumption
_PENALTY_DSG_REC_DEP       = 0.10   # Recommendation depending on unsupported assumption
_PENALTY_DSG_OPTION_DEP    = 0.10   # Recommended option depending on unsupported assumption
_PENALTY_DSG_EXEC_CONF     = 0.10   # Executive confidence below High


# ── J12.0 pure heuristic functions ───────────────────────────────────────────

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


def _overall_health(
    question_coverage: list[dict[str, Any]],
    missing_area_count: int,
    contradictions: list[dict[str, Any]],
    decision_support_gaps: list[dict[str, Any]] | None = None,
) -> str:
    """Compute overall research health incorporating J12.0 and J12.1 penalties."""
    score = 1.0

    # J12.0 — coverage-based penalties
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

    # J12.1 — decision-support penalties
    if decision_support_gaps:
        for gap in decision_support_gaps:
            gap_type = gap.get("gap_type", "")
            artifact_type = gap.get("artifact_type", "")
            if gap_type == "no_evidence" and artifact_type == "assumption":
                score -= _PENALTY_DSG_NO_EVIDENCE
            elif gap_type in ("weak_evidence", "low_confidence") and artifact_type == "assumption":
                score -= _PENALTY_DSG_WEAK_EVIDENCE
            elif gap_type == "unsupported_dependency" and artifact_type == "recommendation":
                score -= _PENALTY_DSG_REC_DEP
            elif gap_type == "unsupported_dependency" and artifact_type == "strategic_option":
                score -= _PENALTY_DSG_OPTION_DEP
            elif gap_type == "confidence_misalignment" and artifact_type == "executive_confidence":
                score -= _PENALTY_DSG_EXEC_CONF

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


def _recommended_followups(
    weak_qs: list[dict[str, Any]],
    missing: list[str],
    decision_support_gaps: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Build ordered followup list: decision-support gaps first, then coverage gaps."""
    followups: list[str] = []

    # J12.1 — decision-support followups take priority
    if decision_support_gaps:
        for gap in decision_support_gaps:
            rec = (gap.get("recommended_followup") or "").strip()
            if rec and rec not in followups:
                followups.append(rec)

    # J12.0 — coverage-gap followups
    for q in weak_qs:
        sq = q["subquestion"]
        short = sq[:80] + "…" if len(sq) > 80 else sq
        followup = f"Gather additional evidence for: {short}"
        if followup not in followups:
            followups.append(followup)
    for area in missing:
        followup = f"Investigate area with no current evidence: {area}"
        if followup not in followups:
            followups.append(followup)

    return followups


# ── J12.1 pure heuristic functions ───────────────────────────────────────────

def _assumption_support_analysis(assumptions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag Critical assumptions with no evidence, weak evidence, or low confidence."""
    gaps: list[dict[str, Any]] = []
    for a in assumptions:
        importance = (a.get("importance") or "").strip()
        confidence = (a.get("confidence") or "").strip()
        evidence_support = (a.get("evidence_support") or "").strip()
        evidence_ids = a.get("evidence_ids") or []
        assumption_id = a.get("assumption_id") or ""
        statement = a.get("statement") or a.get("title") or ""
        supported_recs = a.get("supported_recommendation_ids") or []

        if importance == "Critical" and len(evidence_ids) == 0:
            gaps.append({
                "artifact_type": "assumption",
                "artifact_id": assumption_id,
                "artifact_title": statement[:100],
                "gap_type": "no_evidence",
                "severity": "HIGH",
                "why_it_matters": (
                    f"Assumption {assumption_id} is Critical but has no supporting evidence."
                ),
                "supporting_evidence_count": 0,
                "related_recommendation_ids": list(supported_recs),
                "related_assumption_ids": [],
                "recommended_followup": (
                    f"Conduct targeted research to validate {assumption_id}: {statement[:80]}"
                ),
            })
        elif importance == "Critical" and evidence_support == "Weak":
            gaps.append({
                "artifact_type": "assumption",
                "artifact_id": assumption_id,
                "artifact_title": statement[:100],
                "gap_type": "weak_evidence",
                "severity": "MEDIUM",
                "why_it_matters": (
                    f"Assumption {assumption_id} is Critical with only weak evidence support."
                ),
                "supporting_evidence_count": len(evidence_ids),
                "related_recommendation_ids": list(supported_recs),
                "related_assumption_ids": [],
                "recommended_followup": (
                    f"Strengthen evidence for {assumption_id}: {statement[:80]}"
                ),
            })

        if confidence == "Low" and importance in ("Critical", "Important"):
            severity = "HIGH" if importance == "Critical" else "MEDIUM"
            gaps.append({
                "artifact_type": "assumption",
                "artifact_id": assumption_id,
                "artifact_title": statement[:100],
                "gap_type": "low_confidence",
                "severity": severity,
                "why_it_matters": (
                    f"Assumption {assumption_id} has Low confidence but supports "
                    f"{importance.lower()} decisions."
                ),
                "supporting_evidence_count": len(evidence_ids),
                "related_recommendation_ids": list(supported_recs),
                "related_assumption_ids": [],
                "recommended_followup": (
                    f"Validate and strengthen confidence in {assumption_id}"
                ),
            })
    return gaps


def _recommendation_dependency_analysis(
    recommendations: list[dict[str, Any]],
    assumptions_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flag recommendations whose supporting assumptions have evidence gaps."""
    weak_ids: set[str] = set()
    for aid, a in assumptions_by_id.items():
        importance = (a.get("importance") or "").strip()
        confidence = (a.get("confidence") or "").strip()
        evidence_ids = a.get("evidence_ids") or []
        if importance == "Critical" and len(evidence_ids) == 0:
            weak_ids.add(aid)
        elif confidence == "Low":
            weak_ids.add(aid)

    gaps: list[dict[str, Any]] = []
    for rec in recommendations:
        rec_id = rec.get("recommendation_id") or rec.get("id") or ""
        supported_assumptions = (
            rec.get("supported_assumption_ids")
            or rec.get("supporting_assumption_ids")
            or []
        )
        title = rec.get("title") or rec.get("summary") or rec.get("text") or ""

        weak_deps = [aid for aid in supported_assumptions if aid in weak_ids]
        if weak_deps:
            gaps.append({
                "artifact_type": "recommendation",
                "artifact_id": rec_id,
                "artifact_title": str(title)[:100],
                "gap_type": "unsupported_dependency",
                "severity": "MEDIUM",
                "why_it_matters": (
                    f"{rec_id} depends on weakly supported assumption(s): "
                    f"{', '.join(weak_deps)}"
                ),
                "supporting_evidence_count": 0,
                "related_recommendation_ids": [rec_id],
                "related_assumption_ids": weak_deps,
                "recommended_followup": (
                    f"Validate {', '.join(weak_deps)} before committing to {rec_id}"
                ),
            })
    return gaps


def _strategic_option_support_analysis(
    recommended_option_id: str,
    strategic_options: list[dict[str, Any]],
    assumptions_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flag the recommended strategic option if its assumptions have critical evidence gaps."""
    if not recommended_option_id or not strategic_options:
        return []

    recommended_option = next(
        (o for o in strategic_options if o.get("option_id") == recommended_option_id),
        None,
    )
    if not recommended_option:
        return []

    supporting_assumptions = (
        recommended_option.get("supporting_assumption_ids")
        or recommended_option.get("assumption_ids")
        or []
    )

    critical_gap_ids: list[str] = []
    for aid in supporting_assumptions:
        a = assumptions_by_id.get(aid)
        if a is None:
            continue
        evidence_ids = a.get("evidence_ids") or []
        evidence_support = (a.get("evidence_support") or "").strip()
        if len(evidence_ids) == 0 or evidence_support == "Weak":
            critical_gap_ids.append(aid)

    if not critical_gap_ids:
        return []

    title = recommended_option.get("title") or recommended_option_id
    return [{
        "artifact_type": "strategic_option",
        "artifact_id": recommended_option_id,
        "artifact_title": str(title)[:100],
        "gap_type": "unsupported_dependency",
        "severity": "HIGH",
        "why_it_matters": (
            f"Recommended option {recommended_option_id} depends on "
            f"weakly supported assumption(s): {', '.join(critical_gap_ids)}"
        ),
        "supporting_evidence_count": 0,
        "related_recommendation_ids": [],
        "related_assumption_ids": critical_gap_ids,
        "recommended_followup": (
            f"Before committing to {recommended_option_id}, validate: "
            f"{', '.join(critical_gap_ids)}"
        ),
    }]


def _executive_confidence_gap_analysis(
    executive_confidence: dict[str, Any],
) -> list[dict[str, Any]]:
    """Flag when overall executive confidence is below High."""
    if not executive_confidence:
        return []
    overall = (executive_confidence.get("overall_confidence") or "").strip()
    if not overall or overall == "High":
        return []

    severity = "MEDIUM" if overall == "Medium" else "HIGH"
    return [{
        "artifact_type": "executive_confidence",
        "artifact_id": executive_confidence.get("confidence_id", ""),
        "artifact_title": f"Executive Confidence: {overall}",
        "gap_type": "confidence_misalignment",
        "severity": severity,
        "why_it_matters": (
            f"Overall executive confidence is {overall}, not High. "
            "The decision cannot be committed without further validation."
        ),
        "supporting_evidence_count": 0,
        "related_recommendation_ids": [],
        "related_assumption_ids": [],
        "recommended_followup": (
            "Address the validation priorities identified in the Executive Confidence assessment."
        ),
    }]


def _decision_support_gaps(
    assumptions: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    strategic_options: list[dict[str, Any]],
    decision_analysis: dict[str, Any],
    executive_confidence: dict[str, Any],
) -> list[dict[str, Any]]:
    """Aggregate all decision-support gap findings and assign sequential DSG IDs."""
    assumptions_by_id: dict[str, dict[str, Any]] = {
        a.get("assumption_id"): a
        for a in assumptions
        if a.get("assumption_id")
    }
    recommended_option_id = (decision_analysis.get("recommended_option_id") or "").strip()

    all_gaps: list[dict[str, Any]] = []
    all_gaps.extend(_assumption_support_analysis(assumptions))
    all_gaps.extend(_recommendation_dependency_analysis(recommendations, assumptions_by_id))
    all_gaps.extend(
        _strategic_option_support_analysis(
            recommended_option_id, strategic_options, assumptions_by_id
        )
    )
    all_gaps.extend(_executive_confidence_gap_analysis(executive_confidence))

    for i, gap in enumerate(all_gaps, start=1):
        gap["gap_id"] = f"DSG-{i:03d}"

    return all_gaps


def _confidence_alignment(
    research_gap_confidence: float,
    decision_analysis: dict[str, Any],
    executive_confidence: dict[str, Any],
) -> dict[str, Any]:
    """Compare research coverage confidence against decision and executive confidence levels."""
    da_confidence = (
        (decision_analysis.get("confidence") or "").strip() if decision_analysis else ""
    )
    ec_overall = (
        (executive_confidence.get("overall_confidence") or "").strip()
        if executive_confidence
        else ""
    )

    if not da_confidence and not ec_overall:
        return {
            "research_gap_confidence": research_gap_confidence,
            "decision_analysis_confidence": "",
            "executive_confidence": "",
            "alignment_status": "UNKNOWN",
            "explanation": (
                "Decision analysis and executive confidence are not yet available at this "
                "pipeline stage. Alignment can be assessed in a subsequent iteration."
            ),
        }

    is_aligned = True
    explanation = ""

    if research_gap_confidence >= 0.8 and ec_overall in ("Low", "Medium"):
        is_aligned = False
        explanation = (
            f"Research coverage is comprehensive (confidence={research_gap_confidence:.2f}), "
            f"but executive confidence remains {ec_overall}. "
            "This indicates that critical assumptions lack supporting evidence, "
            "validation priorities remain unresolved, or strategic options carry "
            "evidence risk — factors not captured by evidence coverage alone."
        )
    elif research_gap_confidence < 0.5 and ec_overall == "High":
        is_aligned = False
        explanation = (
            f"Research coverage is weak (confidence={research_gap_confidence:.2f}), "
            f"but executive confidence is {ec_overall}. "
            "The decision may be relying on expert judgment rather than evidence."
        )
    else:
        explanation = (
            f"Research confidence ({research_gap_confidence:.2f}) is broadly consistent "
            f"with decision analysis confidence ({da_confidence or 'N/A'}) "
            f"and executive confidence ({ec_overall or 'N/A'})."
        )

    return {
        "research_gap_confidence": research_gap_confidence,
        "decision_analysis_confidence": da_confidence,
        "executive_confidence": ec_overall,
        "alignment_status": "ALIGNED" if is_aligned else "MISALIGNED",
        "explanation": explanation,
    }


def compute_research_gap_analysis(
    *,
    subquestions: list[str],
    investigation_areas: list[str],
    coverage_by_subquestion: dict[str, dict],
    evidence_by_area: dict[str, list],
    hypotheses: list[dict[str, Any]],
    validated_contradictions: list[dict],
    research_object: dict,
    # J12.1 — decision-support inputs (optional; absent = graceful no-op)
    assumptions: list[dict[str, Any]] | None = None,
    recommendations: list[dict[str, Any]] | None = None,
    strategic_options: list[dict[str, Any]] | None = None,
    decision_analysis: dict[str, Any] | None = None,
    executive_confidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the full ResearchGapAnalysis dict from deterministic heuristics."""
    # J12.0 — coverage heuristics
    qcov = _question_coverage_list(subquestions, coverage_by_subquestion)
    weak = _weak_questions(qcov)
    missing = _missing_areas(evidence_by_area, investigation_areas)
    assumption_heavy = _assumption_heavy_topics(hypotheses)
    contradictions = _extract_contradictions(validated_contradictions, research_object)
    confidence = _confidence_score(qcov)

    # J12.1 — decision-support heuristics
    dsgs = _decision_support_gaps(
        assumptions=assumptions or [],
        recommendations=recommendations or [],
        strategic_options=strategic_options or [],
        decision_analysis=decision_analysis or {},
        executive_confidence=executive_confidence or {},
    )
    conf_alignment = _confidence_alignment(
        research_gap_confidence=confidence,
        decision_analysis=decision_analysis or {},
        executive_confidence=executive_confidence or {},
    )

    # Combined health: J12.0 coverage penalties + J12.1 decision-support penalties
    health = _overall_health(qcov, len(missing), contradictions, dsgs)

    # Ordered followups: decision-support gaps first, then coverage gaps
    followups = _recommended_followups(weak, missing, dsgs)

    return {
        "question_coverage": qcov,
        "weak_questions": weak,
        "missing_investigation_areas": missing,
        "assumption_heavy_topics": assumption_heavy,
        "contradictions": contradictions,
        "recommended_followups": followups,
        "overall_research_health": health,
        "confidence": confidence,
        # J12.1 additions
        "decision_support_gaps": dsgs,
        "confidence_alignment": conf_alignment,
    }


# ── Agent ─────────────────────────────────────────────────────────────────────

class ResearchGapAgent(FunctionalAgent):
    """Deterministic research completeness and decision-support assessment (J12.1)."""

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

        # J12.1 — decision-support inputs.
        # At this pipeline position (after HypothesisAgent, before StrategicSynthesisAgent)
        # these fields are empty on the first pass. They become populated on QA-loop
        # second passes when downstream agents have already enriched the context.
        assumptions = (
            context.assumptions
            or research_object.get("strategic_assumptions", [])
            or []
        )
        recommendations = (
            context.recommendations
            or research_object.get("recommendations", [])
            or []
        )
        strategic_options = (
            context.strategic_options
            or research_object.get("strategic_options", [])
            or []
        )
        decision_analysis = (
            context.decision_analysis
            or research_object.get("decision_analysis")
            or {}
        )
        executive_confidence = (
            context.executive_confidence
            or research_object.get("executive_confidence")
            or {}
        )

        analysis = compute_research_gap_analysis(
            subquestions=subquestions,
            investigation_areas=investigation_areas,
            coverage_by_subquestion=coverage_by_subquestion,
            evidence_by_area=evidence_by_area,
            hypotheses=hypotheses,
            validated_contradictions=validated_contradictions,
            research_object=research_object,
            assumptions=assumptions,
            recommendations=recommendations,
            strategic_options=strategic_options,
            decision_analysis=decision_analysis,
            executive_confidence=executive_confidence,
        )

        context.research_gap_analysis = analysis
        if context.research_object is not None:
            context.research_object["research_gap_analysis"] = analysis
        context.trace["_research_gap"] = analysis

        LOGGER.log(
            PROGRESS,
            "[ResearchGapAgent] health=%s  confidence=%.2f  weak_questions=%d  "
            "missing_areas=%d  assumption_heavy=%d  contradictions=%d  "
            "decision_support_gaps=%d  followups=%d",
            analysis["overall_research_health"],
            analysis["confidence"],
            len(analysis["weak_questions"]),
            len(analysis["missing_investigation_areas"]),
            len(analysis["assumption_heavy_topics"]),
            len(analysis["contradictions"]),
            len(analysis["decision_support_gaps"]),
            len(analysis["recommended_followups"]),
        )

        self._record(
            context,
            status="success",
            summary=(
                f"Research health: {analysis['overall_research_health']}  "
                f"confidence={analysis['confidence']:.2f}  "
                f"weak_questions={len(analysis['weak_questions'])}  "
                f"missing_areas={len(analysis['missing_investigation_areas'])}  "
                f"decision_support_gaps={len(analysis['decision_support_gaps'])}"
            ),
            overall_research_health=analysis["overall_research_health"],
            confidence=analysis["confidence"],
            weak_question_count=len(analysis["weak_questions"]),
            missing_area_count=len(analysis["missing_investigation_areas"]),
            assumption_heavy_count=len(analysis["assumption_heavy_topics"]),
            contradiction_count=len(analysis["contradictions"]),
            decision_support_gap_count=len(analysis["decision_support_gaps"]),
            followup_count=len(analysis["recommended_followups"]),
        )
        return context
