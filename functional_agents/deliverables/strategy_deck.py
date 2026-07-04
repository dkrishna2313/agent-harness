"""StrategyDeckGenerator — narrative-driven strategy deck (J12.2).

Migrated from direct AgentContext inspection (J11.3) to consuming
ExecutiveNarrative (J12.0). The generator is a presentation renderer: it
renders the canonical executive story assembled by ExecutiveNarrativeBuilder
into 12 slides separated by Markdown horizontal rules — the stable parse
contract for future PowerPoint conversion (J11.4).

Slides that require engagement metadata (client, industry, decision horizon)
or multi-profile metadata still receive those as explicit parameters from
the top-level entry point, since engagement, decision_architecture, question,
profiles, and multi_profile_analysis are not reasoning outputs and are not in
the replace list.

Each slide is: `# Slide X — Title` header, content, then `---` separator.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .artifact import DeliverableArtifact
from .base import DeliverableGenerator
from ..narrative import ExecutiveNarrative, ExecutiveNarrativeBuilder

if TYPE_CHECKING:
    from ..context import AgentContext

_PORTFOLIO_LABELS = {
    "near_term": "30 Days (Near-Term)",
    "medium_term": "90 Days (Medium-Term)",
    "long_term": "180 Days (Long-Term)",
}


# ---------------------------------------------------------------------------
# Slide builders — primary slides consume ExecutiveNarrative only
# ---------------------------------------------------------------------------

def _build_slide_01_executive_decision(narrative: ExecutiveNarrative) -> str:
    decision = narrative.decision
    rec = narrative.recommended_option
    rec_id = rec.get("option_id", "") if rec else ""
    rec_title = rec.get("title", "") if rec else ""

    lines = ["# Slide 1 — Executive Decision", ""]
    if decision:
        lines += [f"**Decision:** {decision}", ""]
    if rec_id and rec_title:
        lines += [f"**Recommended Option:** {rec_id}: {rec_title}", ""]
    elif rec_id:
        lines += [f"**Recommended Option:** {rec_id}", ""]
    if not decision and not rec_id:
        lines += ["*Not available for this run.*", ""]
    return "\n".join(lines)


def _build_slide_02_client_situation(
    narrative: ExecutiveNarrative,
    engagement: dict,
    decision_arch: dict,
    question: str,
) -> str:
    client = engagement.get("client", "") or engagement.get("client_name", "")
    industry = engagement.get("industry", "")
    title = engagement.get("title", "")
    strategic_q = (
        engagement.get("strategic_question", "")
        or narrative.decision  # covers decision_architecture.decision_statement
        or question
    )
    horizon = (
        engagement.get("decision_horizon", "")
        or decision_arch.get("decision_horizon", "")
    )
    constraints = engagement.get("constraints") or engagement.get("key_constraints") or []
    if isinstance(constraints, str):
        constraints = [constraints]

    lines = ["# Slide 2 — Client Situation", ""]
    if title:
        lines += [f"**Engagement:** {title}", ""]
    if client:
        lines += [f"**Client:** {client}", ""]
    if industry:
        lines += [f"**Industry:** {industry}", ""]
    if strategic_q:
        lines += [f"**Strategic Question:** {strategic_q}", ""]
    if horizon:
        lines += [f"**Decision Horizon:** {horizon}", ""]
    if constraints:
        lines.append("**Key Constraints:**")
        lines.extend(f"- {c}" for c in constraints[:5])
        lines.append("")
    if not any([client, industry, strategic_q, horizon, constraints]):
        lines += ["*Not available for this run.*", ""]
    return "\n".join(lines)


def _build_slide_03_executive_summary(narrative: ExecutiveNarrative) -> str:
    summary = narrative.executive_summary
    lines = ["# Slide 3 — Executive Summary", ""]
    lines += [summary, ""] if summary else ["*Not available for this run.*", ""]
    return "\n".join(lines)


def _build_slide_04_strategic_options(narrative: ExecutiveNarrative) -> str:
    options = narrative.strategic_options
    rec_id = (narrative.recommended_option or {}).get("option_id", "")

    lines = ["# Slide 4 — Strategic Options", ""]
    if not options:
        lines += ["*No strategic options available for this run.*", ""]
        return "\n".join(lines)

    lines += ["| Option | Description | Time Horizon | Capital | Confidence |",
              "|---|---|---|---|---|"]
    for o in options:
        oid = o.get("option_id", "")
        title = o.get("title", "")
        desc = (o.get("description", "") or "")[:80]
        horizon = (o.get("estimated_time_horizon") or "").replace("_", " ")
        capital = o.get("capital_intensity", "")
        conf = o.get("confidence", "")
        marker = " ✓" if oid == rec_id else ""
        label = f"**{oid}: {title}{marker}**" if marker else f"{oid}: {title}"
        lines.append(f"| {label} | {desc} | {horizon} | {capital} | {conf} |")
    lines.append("")

    if rec_id:
        rec = next((o for o in options if o.get("option_id") == rec_id), None)
        if rec:
            advantages = (rec.get("advantages") or [])[:3]
            if advantages:
                lines.append(f"**Why {rec_id} — {rec.get('title', '')}:**")
                lines.extend(f"- {a}" for a in advantages)
                lines.append("")
    return "\n".join(lines)


def _build_slide_05_decision_matrix(narrative: ExecutiveNarrative) -> str:
    dimensions = narrative.key_tradeoffs
    rankings = narrative.option_rankings
    rationale = narrative.why_this_option

    lines = ["# Slide 5 — Decision Matrix", ""]
    if dimensions:
        lines.append("**Comparison Dimensions:**")
        lines.extend(f"{i + 1}. {d}" for i, d in enumerate(dimensions))
        lines.append("")
    if rankings:
        lines.append("**Option Rankings (best → least preferred):**")
        lines.extend(f"{i + 1}. {r}" for i, r in enumerate(rankings))
        lines.append("")
    if rationale:
        lines += [f"**Rationale:** {rationale}", ""]
    if not dimensions and not rankings and not rationale:
        lines += ["*Not available for this run.*", ""]
    return "\n".join(lines)


def _build_slide_06_strategic_risks(narrative: ExecutiveNarrative) -> str:
    risks = narrative.key_risks
    lines = ["# Slide 6 — Strategic Risks", ""]
    if not risks:
        lines += ["*No risks recorded for this run.*", ""]
        return "\n".join(lines)

    for r in risks:
        rid = r.get("risk_id", "")
        stmt = r.get("statement", "")
        sev = r.get("severity", "")
        mitigation = r.get("mitigation", "")
        lines += [f"**{rid}** _{sev}_", f"{stmt}"]
        if mitigation:
            lines.append(f"*Mitigation: {mitigation}*")
        lines.append("")
    return "\n".join(lines)


def _build_slide_07_strategic_opportunities(narrative: ExecutiveNarrative) -> str:
    opportunities = narrative.key_opportunities
    lines = ["# Slide 7 — Strategic Opportunities", ""]
    if not opportunities:
        lines += ["*No opportunities recorded for this run.*", ""]
        return "\n".join(lines)

    for o in opportunities:
        oid = o.get("opportunity_id", "")
        title = o.get("title", "")
        impact = o.get("impact", "")
        desc = o.get("description", "")
        header = f"**{oid}: {title}**" if oid and title else f"**{title or oid}**"
        lines.append(header)
        if impact:
            lines.append(f"Impact: {impact}")
        if desc:
            lines.append(desc[:120])
        lines.append("")
    return "\n".join(lines)


def _build_slide_08_critical_assumptions(narrative: ExecutiveNarrative) -> str:
    assumptions = narrative.critical_assumptions
    lines = ["# Slide 8 — Critical Assumptions", ""]
    if not assumptions:
        lines += ["*No assumptions recorded for this run.*", ""]
        return "\n".join(lines)

    lines += ["| ID | Assumption | Importance | Confidence |", "|---|---|---|---|"]
    for a in assumptions:
        aid = a.get("assumption_id", "")
        stmt = (a.get("statement", "") or "").replace("|", "\\|")[:120]
        imp = a.get("importance", "")
        conf = a.get("confidence", "")
        lines.append(f"| {aid} | {stmt} | {imp} | {conf} |")
    lines.append("")
    return "\n".join(lines)


def _build_slide_09_executive_confidence(narrative: ExecutiveNarrative) -> str:
    ec = narrative.executive_confidence
    lines = ["# Slide 9 — Executive Confidence", ""]

    if not ec:
        lines += ["*Executive confidence assessment not available for this run.*", ""]
        return "\n".join(lines)

    overall = ec.get("overall_confidence", "")
    readiness = ec.get("decision_readiness", "")
    board_rec = ec.get("board_recommendation", "")
    if overall or readiness or board_rec:
        lines.append(
            "  |  ".join(filter(None, [
                f"**Overall Confidence:** {overall}" if overall else "",
                f"**Decision Readiness:** {readiness}" if readiness else "",
                f"**Board Recommendation:** {board_rec}" if board_rec else "",
            ]))
        )
        lines.append("")

    unknowns = narrative.critical_unknowns[:4]
    if unknowns:
        lines.append("**Critical Unknowns:**")
        lines.extend(f"- {u}" for u in unknowns)
        lines.append("")

    priorities = narrative.validation_priorities[:4]
    if priorities:
        lines.append("**Validation Priorities:**")
        lines.extend(f"{i + 1}. {p}" for i, p in enumerate(priorities))
        lines.append("")

    return "\n".join(lines)


def _build_slide_10_immediate_actions(narrative: ExecutiveNarrative) -> str:
    buckets = {
        "near_term": narrative.immediate_actions,
        "medium_term": narrative.medium_term_actions,
        "long_term": narrative.long_term_actions,
    }

    lines = ["# Slide 10 — Immediate Actions", ""]
    any_content = False
    for key, label in _PORTFOLIO_LABELS.items():
        items = buckets.get(key, [])
        if items:
            lines.append(f"**{label}:**")
            for action in items:
                rid = action.get("id", "")
                title = action.get("title", "")
                lines.append(f"- **{rid}**: {title}")
            lines.append("")
            any_content = True

    if not any_content:
        lines += ["*No near-term recommendations available for this run.*", ""]
    return "\n".join(lines)


def _build_slide_11_supporting_evidence(narrative: ExecutiveNarrative) -> str:
    evidence = narrative.supporting_evidence
    lines = ["# Slide 11 — Supporting Evidence", ""]
    if not evidence:
        lines += ["*No supporting evidence recorded for this run.*", ""]
        return "\n".join(lines)

    for h in evidence:
        hid = h.get("id", "")
        title = h.get("title", "")
        confidence = h.get("confidence", "")
        header = f"**{hid}: {title}**" if hid else f"**{title}**"
        lines.append(header)
        if confidence:
            lines.append(f"Confidence: {confidence}")
        lines.append("")
    return "\n".join(lines)


def _build_slide_12_appendix(
    narrative: ExecutiveNarrative,
    profiles: list[str],
    multi_analysis: dict,
) -> str:
    critical_unknowns = narrative.critical_unknowns
    multi = (multi_analysis or {}).get("profiles") or []

    if not profiles and not multi and not critical_unknowns:
        return ""

    lines = ["# Slide 12 — Appendix", ""]
    if profiles:
        lines += [f"**Profiles:** {', '.join(profiles)}", ""]
    if multi:
        lines.append("**Multi-Profile Analysis:**")
        for p in multi[:4]:
            pname = p.get("profile", "") or p.get("profile_name", "")
            if pname:
                lines.append(f"- {pname}")
        lines.append("")
    if critical_unknowns:
        lines.append("**Critical Unknowns:**")
        lines.extend(f"- {u}" for u in critical_unknowns)
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level assembler
# ---------------------------------------------------------------------------

def build_strategy_deck_content(context: "AgentContext") -> str:
    """Assemble the strategy deck markdown from an already-completed context.

    Builds ExecutiveNarrative (idempotent; sets context.executive_narrative),
    then delegates each slide to a renderer. Engagement metadata (client,
    industry, decision_horizon, profiles, multi_profile_analysis) is passed
    explicitly — these are input/metadata fields not in the reasoning replace list.
    """
    narrative = ExecutiveNarrativeBuilder().build(context)
    engagement = dict(context.engagement or {})
    decision_arch = dict(context.decision_architecture or {})
    question = context.question or ""
    profiles = list(context.profiles or [])
    multi_analysis = dict(context.multi_profile_analysis or {})

    slides = [
        _build_slide_01_executive_decision(narrative),
        _build_slide_02_client_situation(narrative, engagement, decision_arch, question),
        _build_slide_03_executive_summary(narrative),
        _build_slide_04_strategic_options(narrative),
        _build_slide_05_decision_matrix(narrative),
        _build_slide_06_strategic_risks(narrative),
        _build_slide_07_strategic_opportunities(narrative),
        _build_slide_08_critical_assumptions(narrative),
        _build_slide_09_executive_confidence(narrative),
        _build_slide_10_immediate_actions(narrative),
        _build_slide_11_supporting_evidence(narrative),
        _build_slide_12_appendix(narrative, profiles, multi_analysis),
    ]
    body = "\n---\n\n".join(s for s in slides if s)
    return "# Strategy Deck\n\n" + body.rstrip("\n") + "\n"


class StrategyDeckGenerator(DeliverableGenerator):
    deliverable_type = "strategy_deck"

    def generate(self, context: "AgentContext", output_path: Path) -> DeliverableArtifact:
        from research_agent.markdown import write_markdown

        content = build_strategy_deck_content(context)
        written_path = write_markdown(content, output_path)

        return DeliverableArtifact(
            type=self.deliverable_type,
            path=str(written_path),
            mime_type="text/markdown",
            metadata={"status": "generated"},
        )
