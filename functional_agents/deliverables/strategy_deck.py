"""StrategyDeckGenerator — the third concrete DeliverableGenerator (J11.3).

Renders a 12-slide strategy deck in Markdown from the same completed
Strategic Reasoning Graph the other generators consume. Pure presentation:
every value is read directly from AgentContext fields already computed by
upstream Functional Agents — no new inference, no LLM call, no Functional
Agent invoked.

Each slide is separated by a Markdown horizontal rule (---) and begins with
a `# Slide X — Title` heading, suitable as a source file for future
PowerPoint conversion (J11.4).

Field notes:
- `context.strategic_option_portfolio` is always `{}` in production — slide 10
  reads `recommendation_portfolio["near_term"/"medium_term"/"long_term"]`
  instead (populated by RecommendationAgent).
- `context.surviving_hypotheses` (post-ChallengeAgent) is preferred for
  slide 11; falls back to `context.hypotheses` when empty.
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
_PORTFOLIO_LABELS = {
    "near_term": "30 Days (Near-Term)",
    "medium_term": "90 Days (Medium-Term)",
    "long_term": "180 Days (Long-Term)",
}


def _rank(value: str, order: dict[str, int]) -> int:
    return order.get(str(value or "").lower(), 99)


def _recommended_id(context: "AgentContext") -> str:
    da = context.decision_analysis or {}
    preferred = context.preferred_option or {}
    return da.get("recommended_option_id") or preferred.get("option_id", "")


# ---------------------------------------------------------------------------
# Slide builders
# ---------------------------------------------------------------------------

def _build_slide_01_executive_decision(context: "AgentContext") -> str:
    da = context.decision_analysis or {}
    preferred = context.preferred_option or {}
    ro = context.research_object or {}
    da_arch = ro.get("decision_architecture") or context.decision_architecture or {}

    decision_stmt = da_arch.get("decision_statement", "") or preferred.get("decision_statement", "")
    rec_id = _recommended_id(context)
    options = context.strategic_options or []
    rec_option = next((o for o in options if o.get("option_id") == rec_id), preferred or {})
    rec_title = rec_option.get("title", "") if rec_option else ""

    lines = ["# Slide 1 — Executive Decision", ""]
    if decision_stmt:
        lines += [f"**Decision:** {decision_stmt}", ""]
    if rec_id and rec_title:
        lines += [f"**Recommended Option:** {rec_id}: {rec_title}", ""]
    elif rec_id:
        lines += [f"**Recommended Option:** {rec_id}", ""]
    if not decision_stmt and not rec_id:
        lines += ["*Not available for this run.*", ""]
    return "\n".join(lines)


def _build_slide_02_client_situation(context: "AgentContext") -> str:
    eng = context.engagement or {}
    da_arch = context.decision_architecture or {}
    ro = context.research_object or {}
    ro_arch = ro.get("decision_architecture") or {}

    client = eng.get("client", "") or eng.get("client_name", "")
    industry = eng.get("industry", "")
    title = eng.get("title", "")
    strategic_q = (
        eng.get("strategic_question", "")
        or da_arch.get("decision_statement", "")
        or ro_arch.get("decision_statement", "")
        or context.question
    )
    horizon = (
        eng.get("decision_horizon", "")
        or da_arch.get("decision_horizon", "")
        or ro_arch.get("decision_horizon", "")
    )
    constraints = eng.get("constraints") or eng.get("key_constraints") or []
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


def _build_slide_03_executive_summary(context: "AgentContext") -> str:
    summary = (context.strategic_synthesis or {}).get("executive_summary", "")
    if not summary:
        summary = (context.decision_analysis or {}).get("executive_summary", "")

    lines = ["# Slide 3 — Executive Summary", ""]
    lines += [summary, ""] if summary else ["*Not available for this run.*", ""]
    return "\n".join(lines)


def _build_slide_04_strategic_options(context: "AgentContext") -> str:
    options = context.strategic_options or []
    rec_id = _recommended_id(context)

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


def _build_slide_05_decision_matrix(context: "AgentContext") -> str:
    da = context.decision_analysis or {}
    dimensions = da.get("comparison_dimensions") or []
    rankings = da.get("option_rankings") or []
    rationale = da.get("rationale", "")

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


def _build_slide_06_strategic_risks(context: "AgentContext") -> str:
    risks = sorted(
        context.risks or [],
        key=lambda r: _rank(r.get("severity", ""), _SEVERITY_ORDER),
    )[:5]

    lines = ["# Slide 6 — Strategic Risks", ""]
    if not risks:
        lines += ["*No risks recorded for this run.*", ""]
        return "\n".join(lines)

    for r in risks:
        rid = r.get("risk_id", "")
        stmt = r.get("statement", "")
        sev = r.get("severity", "")
        mitigation = (r.get("mitigation") or r.get("mitigation_strategy") or
                      r.get("mitigation_approach") or "")
        lines += [f"**{rid}** _{sev}_", f"{stmt}"]
        if mitigation:
            lines.append(f"*Mitigation: {mitigation}*")
        lines.append("")
    return "\n".join(lines)


def _build_slide_07_strategic_opportunities(context: "AgentContext") -> str:
    opportunities = context.opportunities or []

    lines = ["# Slide 7 — Strategic Opportunities", ""]
    if not opportunities:
        lines += ["*No opportunities recorded for this run.*", ""]
        return "\n".join(lines)

    for o in opportunities[:6]:
        oid = o.get("opportunity_id", "") or o.get("id", "")
        title = o.get("title", "") or o.get("name", "")
        impact = o.get("impact", "") or o.get("strategic_value", "")
        desc = o.get("description", "") or o.get("statement", "")
        header = f"**{oid}: {title}**" if oid and title else f"**{title or oid}**"
        lines.append(header)
        if impact:
            lines.append(f"Impact: {impact}")
        if desc:
            lines.append(desc[:120])
        lines.append("")
    return "\n".join(lines)


def _build_slide_08_critical_assumptions(context: "AgentContext") -> str:
    assumptions = sorted(
        context.assumptions or [],
        key=lambda a: _rank(a.get("importance", ""), _IMPORTANCE_ORDER),
    )[:5]

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


def _build_slide_09_executive_confidence(context: "AgentContext") -> str:
    ec = context.executive_confidence or {}
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

    unknowns = (ec.get("critical_unknowns") or [])[:4]
    if unknowns:
        lines.append("**Critical Unknowns:**")
        lines.extend(f"- {u}" for u in unknowns)
        lines.append("")

    priorities = (ec.get("validation_priorities") or [])[:4]
    if priorities:
        lines.append("**Validation Priorities:**")
        lines.extend(f"{i + 1}. {p}" for i, p in enumerate(priorities))
        lines.append("")

    return "\n".join(lines)


def _build_slide_10_immediate_actions(context: "AgentContext") -> str:
    portfolio = context.recommendation_portfolio or {}
    by_id = {
        r.get("id", r.get("recommendation_id", "")): r
        for r in (context.recommendations or [])
    }

    lines = ["# Slide 10 — Immediate Actions", ""]
    any_content = False
    for key, label in _PORTFOLIO_LABELS.items():
        ids = portfolio.get(key, [])
        items = [by_id[rid] for rid in ids if rid in by_id]
        if items:
            lines.append(f"**{label}:**")
            for r in items:
                rid = r.get("id", r.get("recommendation_id", ""))
                title = r.get("title", "")
                lines.append(f"- **{rid}**: {title}")
            lines.append("")
            any_content = True

    if not any_content:
        lines += ["*No near-term recommendations available for this run.*", ""]
    return "\n".join(lines)


def _build_slide_11_supporting_evidence(context: "AgentContext") -> str:
    hypotheses = context.surviving_hypotheses or context.hypotheses or []

    lines = ["# Slide 11 — Supporting Evidence", ""]
    if not hypotheses:
        lines += ["*No supporting evidence recorded for this run.*", ""]
        return "\n".join(lines)

    for h in hypotheses[:6]:
        hid = h.get("id", h.get("hypothesis_id", ""))
        title = h.get("title", "") or h.get("statement", "")
        confidence = h.get("confidence", "") or h.get("confidence_level", "")
        header = f"**{hid}: {title}**" if hid else f"**{title}**"
        lines.append(header)
        if confidence:
            lines.append(f"Confidence: {confidence}")
        lines.append("")
    return "\n".join(lines)


def _build_slide_12_appendix(context: "AgentContext") -> str:
    critical_unknowns = (context.executive_confidence or {}).get("critical_unknowns") or []
    profiles = context.profiles or []
    multi = (context.multi_profile_analysis or {}).get("profiles") or []

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

    Pure presentation over already-computed Strategic Reasoning Graph fields.
    No new inference, no LLM call, no Functional Agent invoked.
    """
    slides = [
        _build_slide_01_executive_decision(context),
        _build_slide_02_client_situation(context),
        _build_slide_03_executive_summary(context),
        _build_slide_04_strategic_options(context),
        _build_slide_05_decision_matrix(context),
        _build_slide_06_strategic_risks(context),
        _build_slide_07_strategic_opportunities(context),
        _build_slide_08_critical_assumptions(context),
        _build_slide_09_executive_confidence(context),
        _build_slide_10_immediate_actions(context),
        _build_slide_11_supporting_evidence(context),
        _build_slide_12_appendix(context),
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
