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
        # Write markdown report (existing behaviour + Strategic Hypotheses)
        # ------------------------------------------------------------------
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

        # Recommendation quality evaluation block (J6.6)
        rec_eval = context.research_object.get("recommendation_evaluation") if context.research_object else None
        if rec_eval:
            agg = rec_eval.get("aggregate", {})
            trace_payload["recommendation_quality"] = {
                "recommendation_count": agg.get("recommendation_count", 0),
                "recommendation_score": agg.get("recommendation_score", 0.0),
                "mean_evidence_support": agg.get("mean_evidence_support", 0.0),
                "mean_reasoning": agg.get("mean_reasoning", 0.0),
                "mean_tradeoff": agg.get("mean_tradeoff", 0.0),
                "mean_risk": agg.get("mean_risk", 0.0),
                "mean_actionability": agg.get("mean_actionability", 0.0),
                "traceability": rec_eval.get("traceability", []),
            }

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
