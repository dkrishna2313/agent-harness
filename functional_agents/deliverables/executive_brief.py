"""ExecutiveBriefGenerator — the second concrete DeliverableGenerator (J11.1).

Renders a concise (2-4 page) executive brief from the same completed
Strategic Reasoning Graph MarkdownReportGenerator already renders the full
report from. Pure presentation: every value below is read directly off
AgentContext fields already computed by upstream Functional Agents
(RiskAgent, AssumptionAgent, DecisionAnalysisAgent, ExecutiveConfidenceAgent,
StrategicSynthesisAgent, RecommendationAgent) — no new inference, no LLM
call, no Functional Agent invoked. Sorting/truncating already-computed
records (top-5 risks by severity, top-5 assumptions by importance) is
presentation, not reasoning — the same category of operation the existing
J7 executive report already performs (e.g. sorting assumptions by
importance in report_agent.py's `_build_j7_executive_report`).

Field note: `context.strategic_option_portfolio` is never populated by any
agent in this codebase (always `{}` in practice) — "Immediate Executive
Decisions" instead reads `context.recommendation_portfolio["near_term"]`,
which IS populated by RecommendationAgent.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .artifact import DeliverableArtifact
from .base import DeliverableGenerator

if TYPE_CHECKING:
    from ..context import AgentContext

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
_IMPORTANCE_ORDER = {"critical": 0, "important": 1, "supporting": 2}


def _rank(value: str, order: dict[str, int]) -> int:
    return order.get(str(value or "").lower(), 99)


def _build_executive_decision(context: "AgentContext") -> str:
    ro = context.research_object or {}
    arch = ro.get("decision_architecture") or {}
    decision_statement = arch.get("decision_statement", "")

    da = context.decision_analysis or {}
    preferred = context.preferred_option or {}
    recommended_id = da.get("recommended_option_id") or preferred.get("option_id", "")
    title = preferred.get("title", "") or next(
        (o.get("title", "") for o in (context.strategic_options or []) if o.get("option_id") == recommended_id),
        "",
    )

    lines = ["## 1. Executive Decision", ""]
    if decision_statement:
        lines += [f"**Decision:** {decision_statement}", ""]
    if title:
        lines += [f"**Recommended Option:** {title}", ""]
    if not decision_statement and not title:
        lines += ["*No decision statement or recommended option available for this run.*", ""]
    return "\n".join(lines)


def _build_executive_summary(context: "AgentContext") -> str:
    summary = (context.strategic_synthesis or {}).get("executive_summary", "")
    if not summary:
        summary = (context.decision_analysis or {}).get("executive_summary", "")

    lines = ["## 2. Executive Summary", ""]
    lines += [summary, ""] if summary else ["*No executive summary available for this run.*", ""]
    return "\n".join(lines)


def _build_recommended_option(context: "AgentContext") -> str:
    da = context.decision_analysis or {}
    preferred = context.preferred_option or {}
    options = context.strategic_options or []
    recommended_id = da.get("recommended_option_id") or preferred.get("option_id", "")
    option = next((o for o in options if o.get("option_id") == recommended_id), preferred)

    lines = ["## 3. Recommended Strategic Option", ""]
    if not option:
        lines += ["*No strategic option data available for this run.*", ""]
        return "\n".join(lines)

    title = option.get("title", "")
    description = option.get("description", "")
    horizon = (option.get("estimated_time_horizon") or "").replace("_", " ")
    capital = option.get("capital_intensity", "")
    confidence = option.get("confidence", "")

    lines += [f"**{option.get('option_id', '')}: {title}**", ""]
    if description:
        lines += [description, ""]
    meta = []
    if horizon:
        meta.append(f"**Time Horizon:** {horizon}")
    if capital:
        meta.append(f"**Capital Intensity:** {capital}")
    if confidence:
        meta.append(f"**Confidence:** {confidence}")
    if meta:
        lines += ["  ".join(meta), ""]
    return "\n".join(lines)


def _build_why_this_option(context: "AgentContext") -> str:
    da = context.decision_analysis or {}
    rationale = da.get("rationale", "")
    dimensions = da.get("comparison_dimensions") or []
    rankings = da.get("option_rankings") or []

    lines = ["## 4. Why This Option", ""]
    if rationale:
        lines += [rationale, ""]
    if dimensions:
        lines.append("**Comparison Dimensions:**")
        lines.extend(f"- {d}" for d in dimensions)
        lines.append("")
    if rankings:
        lines.append("**Option Rankings (best -> least preferred):**")
        lines.extend(f"{i + 1}. {r}" for i, r in enumerate(rankings))
        lines.append("")
    if not rationale and not dimensions and not rankings:
        lines += ["*No decision rationale available for this run.*", ""]
    return "\n".join(lines)


def _build_key_risks(context: "AgentContext", limit: int = 5) -> str:
    risks = sorted(
        context.risks or [], key=lambda r: _rank(r.get("severity", ""), _SEVERITY_ORDER)
    )[:limit]

    lines = ["## 5. Key Strategic Risks (Top 5)", ""]
    if not risks:
        lines += ["*No risks recorded for this run.*", ""]
        return "\n".join(lines)
    lines += ["| ID | Risk | Severity | Likelihood |", "|---|---|---|---|"]
    for r in risks:
        rid = r.get("risk_id", "")
        stmt = (r.get("statement", "") or "").replace("|", "\\|")[:140]
        sev = r.get("severity", "")
        lhood = r.get("likelihood", "")
        lines.append(f"| {rid} | {stmt} | {sev} | {lhood} |")
    lines.append("")
    return "\n".join(lines)


def _build_critical_assumptions(context: "AgentContext", limit: int = 5) -> str:
    assumptions = sorted(
        context.assumptions or [], key=lambda a: _rank(a.get("importance", ""), _IMPORTANCE_ORDER)
    )[:limit]

    lines = ["## 6. Critical Assumptions", ""]
    if not assumptions:
        lines += ["*No assumptions recorded for this run.*", ""]
        return "\n".join(lines)
    lines += ["| ID | Assumption | Importance | Confidence |", "|---|---|---|---|"]
    for a in assumptions:
        aid = a.get("assumption_id", "")
        stmt = (a.get("statement", "") or "").replace("|", "\\|")[:140]
        imp = a.get("importance", "")
        conf = a.get("confidence", "")
        lines.append(f"| {aid} | {stmt} | {imp} | {conf} |")
    lines.append("")
    return "\n".join(lines)


def _build_executive_confidence(context: "AgentContext") -> str:
    ec = context.executive_confidence or {}
    lines = ["## 7. Executive Confidence", ""]
    if not ec:
        lines += ["*Executive confidence assessment not available for this run.*", ""]
        return "\n".join(lines)

    lines += [
        f"**Overall Confidence:** {ec.get('overall_confidence', '')}  |  "
        f"**Decision Readiness:** {ec.get('decision_readiness', '')}  |  "
        f"**Board Recommendation:** {ec.get('board_recommendation', '')}",
        "",
    ]
    rationale = ec.get("confidence_rationale", "")
    if rationale:
        lines += [rationale, ""]
    drivers = (ec.get("confidence_drivers") or [])[:3]
    if drivers:
        lines.append("**Confidence Drivers:**")
        lines.extend(f"- {d}" for d in drivers)
        lines.append("")
    limiters = (ec.get("confidence_limiters") or [])[:3]
    if limiters:
        lines.append("**Confidence Limiters:**")
        lines.extend(f"- {lim}" for lim in limiters)
        lines.append("")
    return "\n".join(lines)


def _build_immediate_decisions(context: "AgentContext") -> str:
    near_term_ids = (context.recommendation_portfolio or {}).get("near_term", [])
    by_id = {r.get("id", r.get("recommendation_id", "")): r for r in (context.recommendations or [])}
    items = [by_id[rid] for rid in near_term_ids if rid in by_id]

    lines = ["## 8. Immediate Executive Decisions", ""]
    if not items:
        lines += ["*No near-term recommendations flagged for immediate executive action.*", ""]
        return "\n".join(lines)
    for r in items:
        rid = r.get("id", r.get("recommendation_id", ""))
        title = r.get("title", "")
        lines.append(f"- **{rid}**: {title}")
    lines.append("")
    return "\n".join(lines)


def _build_validation_priorities(context: "AgentContext") -> str:
    priorities = (context.executive_confidence or {}).get("validation_priorities") or []
    lines = ["## 9. Validation Priorities", ""]
    if not priorities:
        lines += ["*No validation priorities recorded for this run.*", ""]
        return "\n".join(lines)
    lines.extend(f"{i + 1}. {p}" for i, p in enumerate(priorities))
    lines.append("")
    return "\n".join(lines)


def _build_appendix(context: "AgentContext") -> str:
    critical_unknowns = (context.executive_confidence or {}).get("critical_unknowns") or []
    profiles = context.profiles or []
    if not critical_unknowns and not profiles:
        return ""
    lines = ["## 10. Appendix", ""]
    if profiles:
        lines += [f"**Profiles:** {', '.join(profiles)}", ""]
    if critical_unknowns:
        lines.append("**Critical Unknowns:**")
        lines.extend(f"- {u}" for u in critical_unknowns)
        lines.append("")
    return "\n".join(lines)


def build_executive_brief_content(context: "AgentContext") -> str:
    """Assemble the executive brief markdown body from an already-completed context.

    Pure presentation over already-computed Strategic Reasoning Graph fields.
    No new inference, no LLM call, no Functional Agent invoked.
    """
    sections = [
        _build_executive_decision(context),
        _build_executive_summary(context),
        _build_recommended_option(context),
        _build_why_this_option(context),
        _build_key_risks(context),
        _build_critical_assumptions(context),
        _build_executive_confidence(context),
        _build_immediate_decisions(context),
        _build_validation_priorities(context),
        _build_appendix(context),
    ]
    body = "\n".join(s for s in sections if s)
    return "# Executive Strategic Brief\n\n" + body.rstrip("\n") + "\n"


class ExecutiveBriefGenerator(DeliverableGenerator):
    deliverable_type = "executive_brief"

    def generate(self, context: "AgentContext", output_path: Path) -> DeliverableArtifact:
        from research_agent.markdown import write_markdown

        content = build_executive_brief_content(context)
        written_path = write_markdown(content, output_path)

        return DeliverableArtifact(
            type=self.deliverable_type,
            path=str(written_path),
            mime_type="text/markdown",
            metadata={"status": "generated"},
        )
