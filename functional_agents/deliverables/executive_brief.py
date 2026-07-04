"""ExecutiveBriefGenerator — narrative-driven executive brief (J12.1).

Migrated from direct AgentContext inspection (J11.1) to consuming
ExecutiveNarrative (J12.0). The generator is a presentation renderer: it
renders the canonical executive story assembled by ExecutiveNarrativeBuilder.
No reasoning is re-derived here — sorting, truncation, and field extraction
all happen in the builder, not here.

Context.profiles is still read by the top-level entry point for the appendix:
it is metadata, not a reasoning field, and is explicitly outside the set of
AgentContext reasoning fields that generators should stop reading directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .artifact import DeliverableArtifact
from .base import DeliverableGenerator
from ..narrative import ExecutiveNarrative, ExecutiveNarrativeBuilder

if TYPE_CHECKING:
    from ..context import AgentContext


def _build_executive_decision(narrative: ExecutiveNarrative) -> str:
    decision = narrative.decision
    rec = narrative.recommended_option
    title = rec.get("title", "") if rec else ""

    lines = ["## 1. Executive Decision", ""]
    if decision:
        lines += [f"**Decision:** {decision}", ""]
    if title:
        lines += [f"**Recommended Option:** {title}", ""]
    if not decision and not title:
        lines += ["*No decision statement or recommended option available for this run.*", ""]
    return "\n".join(lines)


def _build_executive_summary(narrative: ExecutiveNarrative) -> str:
    summary = narrative.executive_summary
    lines = ["## 2. Executive Summary", ""]
    lines += [summary, ""] if summary else ["*No executive summary available for this run.*", ""]
    return "\n".join(lines)


def _build_recommended_option(narrative: ExecutiveNarrative) -> str:
    option = narrative.recommended_option
    lines = ["## 3. Recommended Strategic Option", ""]
    if not option:
        lines += ["*No strategic option data available for this run.*", ""]
        return "\n".join(lines)

    option_id = option.get("option_id", "")
    title = option.get("title", "")
    description = option.get("description", "")
    horizon = (option.get("estimated_time_horizon") or "").replace("_", " ")
    capital = option.get("capital_intensity", "")
    confidence = option.get("confidence", "")

    lines += [f"**{option_id}: {title}**", ""]
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


def _build_why_this_option(narrative: ExecutiveNarrative) -> str:
    rationale = narrative.why_this_option
    tradeoffs = narrative.key_tradeoffs
    rankings = narrative.option_rankings

    lines = ["## 4. Why This Option", ""]
    if rationale:
        lines += [rationale, ""]
    if tradeoffs:
        lines.append("**Comparison Dimensions:**")
        lines.extend(f"- {d}" for d in tradeoffs)
        lines.append("")
    if rankings:
        lines.append("**Option Rankings (best -> least preferred):**")
        lines.extend(f"{i + 1}. {r}" for i, r in enumerate(rankings))
        lines.append("")
    if not rationale and not tradeoffs and not rankings:
        lines += ["*No decision rationale available for this run.*", ""]
    return "\n".join(lines)


def _build_key_risks(narrative: ExecutiveNarrative) -> str:
    risks = narrative.key_risks
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


def _build_critical_assumptions(narrative: ExecutiveNarrative) -> str:
    assumptions = narrative.critical_assumptions
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


def _build_executive_confidence(narrative: ExecutiveNarrative) -> str:
    ec = narrative.executive_confidence
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


def _build_immediate_decisions(narrative: ExecutiveNarrative) -> str:
    items = narrative.immediate_actions
    lines = ["## 8. Immediate Executive Decisions", ""]
    if not items:
        lines += ["*No near-term recommendations flagged for immediate executive action.*", ""]
        return "\n".join(lines)
    for action in items:
        rid = action.get("id", "")
        title = action.get("title", "")
        lines.append(f"- **{rid}**: {title}")
    lines.append("")
    return "\n".join(lines)


def _build_validation_priorities(narrative: ExecutiveNarrative) -> str:
    priorities = narrative.validation_priorities
    lines = ["## 9. Validation Priorities", ""]
    if not priorities:
        lines += ["*No validation priorities recorded for this run.*", ""]
        return "\n".join(lines)
    lines.extend(f"{i + 1}. {p}" for i, p in enumerate(priorities))
    lines.append("")
    return "\n".join(lines)


def _build_appendix(narrative: ExecutiveNarrative, profiles: list[str]) -> str:
    critical_unknowns = narrative.critical_unknowns
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

    Builds ExecutiveNarrative (idempotent; sets context.executive_narrative),
    then delegates each section to a renderer. No AgentContext reasoning fields
    are read below this point — all content flows through ExecutiveNarrative.
    """
    narrative = ExecutiveNarrativeBuilder().build(context)
    profiles = list(context.profiles or [])
    sections = [
        _build_executive_decision(narrative),
        _build_executive_summary(narrative),
        _build_recommended_option(narrative),
        _build_why_this_option(narrative),
        _build_key_risks(narrative),
        _build_critical_assumptions(narrative),
        _build_executive_confidence(narrative),
        _build_immediate_decisions(narrative),
        _build_validation_priorities(narrative),
        _build_appendix(narrative, profiles),
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
