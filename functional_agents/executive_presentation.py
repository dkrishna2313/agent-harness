"""ExecutivePresentationGenerator — builds an executive slide deck from AgentContext.

Does NOT re-run any reasoning.  Reads exclusively from fields already
populated on the context by the pipeline.

Public API::

    from functional_agents.executive_presentation import (
        ExecutivePresentationGenerator,
        render_markdown,
        context_from_session,
        context_from_research_object,
    )

    gen  = ExecutivePresentationGenerator()
    pres = gen.generate(context)
    md   = render_markdown(pres)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from .presentation_model import Presentation, Slide, SlideTable, TableRow

if TYPE_CHECKING:
    from .context import AgentContext

# IDs that must not appear on slides (A-001, RSK-002, OPT-A, EC-001, etc.)
_ID_RE = re.compile(
    r"\b(?:A|RSK|OPP|REC|OPT|EC|DA|SC)-[A-Z0-9]+(?:[,;]\s*)?",
    re.IGNORECASE,
)

_MAX_BULLETS = 5
_MAX_BULLET_WORDS = 15


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _strip_ids(text: str) -> str:
    return _ID_RE.sub("", text).strip().strip(",").strip()


def _truncate(text: str, max_words: int = _MAX_BULLET_WORDS) -> str:
    """Return *text* cut to *max_words* words, appending '…' if trimmed."""
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text.rstrip(".")
    return " ".join(words[:max_words]) + "…"


def _bullets(items: list[str], max_words: int = _MAX_BULLET_WORDS) -> list[str]:
    """Sanitise and truncate a list of bullet strings."""
    out = []
    for item in items:
        cleaned = _truncate(_strip_ids(item), max_words)
        if cleaned:
            out.append(cleaned)
        if len(out) >= _MAX_BULLETS:
            break
    return out


def _severity_rank(item: dict) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(
        (item.get("severity") or item.get("impact") or "").lower(), 3
    )


def _importance_rank(item: dict) -> int:
    return {"critical": 0, "important": 1, "low": 2}.get(
        (item.get("importance") or "").lower(), 3
    )


def _find_option(options: list[dict], option_id: str) -> dict:
    for opt in options:
        if opt.get("option_id") == option_id:
            return opt
    return {}


def _first_sentence(text: str) -> str:
    """Return the first sentence of *text* (split on '. ')."""
    if not text:
        return ""
    parts = text.split(". ")
    return parts[0].rstrip(".") if parts else text.rstrip(".")


def _display_title(engagement: dict, research_object: dict, question: str) -> str:
    for key in ("title", "name"):
        val = engagement.get(key) or research_object.get(key)
        if val:
            return val
    return question[:60] if question else "Executive Strategic Briefing"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class ExecutivePresentationGenerator:
    """Assemble a ``Presentation`` from an ``AgentContext`` without re-running
    any reasoning agents."""

    def generate(self, context: "AgentContext") -> Presentation:
        da      = context.decision_analysis or {}
        ec      = context.executive_confidence or {}
        options = list(context.strategic_options or [])
        assums  = sorted(context.assumptions or [], key=_importance_rank)
        risks   = sorted(context.risks or [], key=_severity_rank)
        opps    = context.opportunities or []
        recs    = context.recommendations or []
        ro      = context.research_object or {}
        eng     = context.engagement or {}

        rec_id     = da.get("recommended_option_id") or (context.preferred_option or {}).get("option_id") or ""
        rec_option = _find_option(options, rec_id) or context.preferred_option or (options[0] if options else {})
        rec_title  = rec_option.get("title") or "Recommended Strategy"

        pres = Presentation(
            title=_display_title(eng, ro, context.question),
            subtitle=context.question or "",
            client=eng.get("client") or eng.get("organization") or "",
            metadata={
                "run_id": context.run_id,
                "profiles": context.profiles,
                "execution_profile": context.execution_profile,
            },
        )

        builders = [
            self._slide_title,
            self._slide_recommendation,
            self._slide_market_opportunity,
            self._slide_strategic_options,
            self._slide_why_this_wins,
            self._slide_key_risks,
            self._slide_critical_assumptions,
            self._slide_executive_confidence,
            self._slide_next_steps,
        ]

        slides = []
        for i, builder in enumerate(builders, start=1):
            slide = builder(i, context, da, ec, options, assums, risks, opps, recs, rec_option, rec_title)
            if slide is not None:
                slides.append(slide)

        # Re-number after any skipped slides
        for i, slide in enumerate(slides, start=1):
            slide.slide_number = i

        pres.slides = slides
        return pres

    # ------------------------------------------------------------------
    # Slide builders — each returns a Slide or None (skip if no data)
    # ------------------------------------------------------------------

    def _slide_title(self, n, ctx, da, ec, options, assums, risks, opps, recs, rec_opt, rec_title) -> Slide:
        eng = ctx.engagement or {}
        subtitle = ctx.question or ""
        confidence = da.get("confidence") or ec.get("overall_confidence") or ""
        return Slide(
            slide_number=n,
            slide_type="title",
            title=_display_title(eng, ctx.research_object or {}, ctx.question),
            key_message=f"Recommendation: {rec_title}" if rec_title else "",
            bullets=[
                _truncate(subtitle) if subtitle else "",
                f"Confidence: {confidence}" if confidence else "",
            ],
            notes="Opening title slide. Presenter introduces the engagement and the recommendation at a high level.",
        )

    def _slide_recommendation(self, n, ctx, da, ec, options, assums, risks, opps, recs, rec_opt, rec_title) -> Slide | None:
        if not rec_opt:
            return None

        summary = _first_sentence(da.get("executive_summary") or da.get("rationale") or "")
        readiness = ec.get("decision_readiness") or ""
        board_rec = ec.get("board_recommendation") or ""

        advantages = [_truncate(_strip_ids(a)) for a in (rec_opt.get("advantages") or [])[:4] if a]
        rationale_bullets = [_truncate(_strip_ids(da.get("rationale") or ""))] if da.get("rationale") else []
        raw_bullets = (advantages or rationale_bullets)[:_MAX_BULLETS]

        status_parts = [p for p in [readiness, board_rec] if p]
        title_suffix = " — ".join(status_parts) if status_parts else "Strongest Risk-Adjusted Return"

        return Slide(
            slide_number=n,
            slide_type="content",
            title=f"{rec_title}: {title_suffix}",
            key_message=_truncate(_strip_ids(summary), 20) if summary else "",
            bullets=raw_bullets,
            notes=_strip_ids(da.get("executive_summary") or ""),
        )

    def _slide_market_opportunity(self, n, ctx, da, ec, options, assums, risks, opps, recs, rec_opt, rec_title) -> Slide | None:
        if not opps:
            return None

        top = opps[:_MAX_BULLETS]
        opp_bullets = _bullets([o.get("statement") or "" for o in top])
        if not opp_bullets:
            return None

        return Slide(
            slide_number=n,
            slide_type="content",
            title="The Market Opportunity Justifies Decisive Action Now",
            key_message=f"{len(opps)} strategic opportunities identified across the landscape",
            bullets=opp_bullets,
            notes="Highlight the top opportunities before presenting the options — establishes the prize worth competing for.",
        )

    def _slide_strategic_options(self, n, ctx, da, ec, options, assums, risks, opps, recs, rec_opt, rec_title) -> Slide | None:
        if not options:
            return None

        rec_id     = da.get("recommended_option_id") or (rec_opt or {}).get("option_id") or ""
        rankings   = da.get("option_rankings") or [o.get("option_id") for o in options]
        rank_map   = {oid: i + 1 for i, oid in enumerate(rankings)}

        headers = ["Option", "Time Horizon", "Capital", "Recommended"]
        rows = []
        for opt in options:
            oid   = opt.get("option_id", "")
            title = opt.get("title") or oid
            rows.append(TableRow(cells=[
                f"{rank_map.get(oid, '—')}. {title}",
                _normalise_horizon(opt.get("estimated_time_horizon") or opt.get("time_horizon") or ""),
                opt.get("capital_intensity") or "—",
                "✓" if oid == rec_id else "—",
            ]))

        dimension_count = len(da.get("comparison_dimensions") or [])
        dim_note = f" across {dimension_count} dimensions" if dimension_count else ""

        return Slide(
            slide_number=n,
            slide_type="comparison",
            title=f"{len(options)} Strategic Paths Evaluated — One Clear Recommendation",
            key_message=f"Each option assessed{dim_note}; one recommended based on risk-adjusted fit",
            table=SlideTable(headers=headers, rows=rows),
            notes="Walk the audience through the options briefly before revealing the recommendation rationale.",
        )

    def _slide_why_this_wins(self, n, ctx, da, ec, options, assums, risks, opps, recs, rec_opt, rec_title) -> Slide | None:
        if not rec_opt:
            return None

        advantages  = [_truncate(_strip_ids(a)) for a in (rec_opt.get("advantages") or [])[:3] if a]
        tradeoffs   = [_truncate(_strip_ids(t)) for t in (da.get("key_tradeoffs") or [])[:2] if t]
        all_bullets = (advantages + tradeoffs)[:_MAX_BULLETS]

        if not all_bullets:
            return None

        rationale = _first_sentence(da.get("rationale") or rec_opt.get("rationale") or "")

        # Comparison table: recommended vs next-best across key dimensions
        table = None
        dm = da.get("decision_matrix") or []
        rec_row  = next((r for r in dm if r.get("option_id") == rec_opt.get("option_id")), {})
        dims = ["strategic_fit", "implementation_risk", "capital_requirement", "expected_return", "time_to_value"]
        dim_labels = ["Strategic Fit", "Impl. Risk", "Capital", "Expected Return", "Time to Value"]
        if rec_row:
            headers = ["Dimension", "Recommended"] if len(dm) == 1 else ["Dimension", "Recommended", "Nearest Alternative"]
            next_row = next((r for r in dm if r.get("option_id") != rec_opt.get("option_id")), {})
            rows = []
            for dim, label in zip(dims, dim_labels):
                cells = [label, rec_row.get(dim) or "—"]
                if next_row:
                    cells.append(next_row.get(dim) or "—")
                rows.append(TableRow(cells=cells))
            table = SlideTable(headers=headers, rows=rows, caption="Key decision dimensions")

        return Slide(
            slide_number=n,
            slide_type="comparison",
            title=f"Why {rec_title} Wins Across All Key Dimensions",
            key_message=_truncate(_strip_ids(rationale), 20) if rationale else "",
            bullets=all_bullets,
            table=table,
            notes=_strip_ids(da.get("rationale") or ""),
        )

    def _slide_key_risks(self, n, ctx, da, ec, options, assums, risks, opps, recs, rec_opt, rec_title) -> Slide | None:
        if not risks:
            return None

        high_risks = [r for r in risks if (r.get("severity") or "").lower() == "high"]
        top_risks  = (high_risks or risks)[:5]

        headers = ["Risk", "Severity", "Likelihood", "Mitigation"]
        rows = [
            TableRow(cells=[
                _truncate(_strip_ids(r.get("statement") or r.get("title") or ""), 10),
                r.get("severity") or "—",
                r.get("likelihood") or "—",
                _truncate(_strip_ids(r.get("mitigation_notes") or r.get("mitigation") or ""), 10),
            ])
            for r in top_risks
        ]

        high_count = len(high_risks)
        title = (
            f"{high_count} High-Severity Risks Require Active Management"
            if high_count
            else f"{len(risks)} Risks Identified Across the Strategy"
        )

        mit_bullets = _bullets([
            r.get("mitigation_notes") or r.get("mitigation") or ""
            for r in top_risks
            if r.get("mitigation_notes") or r.get("mitigation")
        ])

        return Slide(
            slide_number=n,
            slide_type="comparison",
            title=title,
            key_message="Mitigations identified and available for all high-severity risks",
            bullets=mit_bullets,
            table=SlideTable(headers=headers, rows=rows),
            notes="Presenter should focus on the top two risks and their specific mitigations.",
        )

    def _slide_critical_assumptions(self, n, ctx, da, ec, options, assums, risks, opps, recs, rec_opt, rec_title) -> Slide | None:
        if not assums:
            return None

        critical = [a for a in assums if (a.get("importance") or "").lower() == "critical"]
        top      = (critical or assums)[:5]

        headers = ["Assumption", "Importance", "Confidence"]
        rows = [
            TableRow(cells=[
                _truncate(_strip_ids(a.get("statement") or ""), 12),
                a.get("importance") or "—",
                a.get("confidence") or "—",
            ])
            for a in top
        ]

        weak_critical = [a for a in critical if (a.get("confidence") or "").lower() in ("low", "weak")]
        if weak_critical:
            title = (
                f"Strategy Rests on {len(critical)} Critical Assumption"
                + ("s" if len(critical) != 1 else "")
                + " With Low Confidence"
            )
        else:
            title = (
                f"{len(critical) or len(assums)} Assumption"
                + ("s" if (len(critical) or len(assums)) != 1 else "")
                + " Underpin the Strategy"
            )

        assump_bullets = _bullets([a.get("statement") or "" for a in top])

        return Slide(
            slide_number=n,
            slide_type="comparison",
            title=title,
            key_message="Validation required before full board approval",
            bullets=assump_bullets,
            table=SlideTable(headers=headers, rows=rows),
            notes="Walk through the critical assumptions and flag where evidence is weakest.",
        )

    def _slide_executive_confidence(self, n, ctx, da, ec, options, assums, risks, opps, recs, rec_opt, rec_title) -> Slide | None:
        if not ec and not da:
            return None

        confidence    = ec.get("overall_confidence") or da.get("confidence") or ""
        readiness     = ec.get("decision_readiness") or ""
        board_rec     = ec.get("board_recommendation") or ""
        horizon       = ec.get("decision_horizon") or ""
        drivers       = ec.get("confidence_drivers") or []
        limiters      = ec.get("confidence_limiters") or []

        hold_scenario = ec.get("confidence_if_assumptions_hold") or ""
        fail_scenario = ec.get("confidence_if_assumptions_fail") or ""

        title_parts = [p for p in [confidence, board_rec] if p]
        title = (
            f"Overall Confidence: {' — '.join(title_parts)}"
            if title_parts
            else "Executive Confidence Assessment"
        )

        key_msg_parts = [p for p in [readiness, horizon] if p]
        key_msg = " | ".join(key_msg_parts) if key_msg_parts else ""

        driver_bullets  = _bullets([f"Driver: {_strip_ids(d)}" for d in drivers[:2] if d])
        limiter_bullets = _bullets([f"Limiter: {_strip_ids(l)}" for l in limiters[:2] if l])
        all_bullets = (driver_bullets + limiter_bullets)[:_MAX_BULLETS]

        table = None
        if hold_scenario and fail_scenario:
            table = SlideTable(
                headers=["Scenario", "Confidence"],
                rows=[
                    TableRow(cells=["If critical assumptions hold", _truncate(hold_scenario, 8)]),
                    TableRow(cells=["If critical assumptions fail", _truncate(fail_scenario, 8)]),
                ],
                caption="Sensitivity: confidence range under key assumption outcomes",
            )

        return Slide(
            slide_number=n,
            slide_type="content",
            title=title,
            key_message=key_msg,
            bullets=all_bullets,
            table=table,
            notes=_strip_ids(ec.get("confidence_rationale") or da.get("confidence_summary") or ""),
        )

    def _slide_next_steps(self, n, ctx, da, ec, options, assums, risks, opps, recs, rec_opt, rec_title) -> Slide | None:
        validation_priorities = ec.get("validation_priorities") or []
        rec_bullets = [
            f"{r.get('title') or 'Action'}: {_truncate(_strip_ids(r.get('summary') or ''), 10)}"
            for r in sorted(recs, key=lambda r: {"high": 0, "medium": 1, "low": 2}.get((r.get("priority") or "").lower(), 3))
            if r.get("title")
        ]

        raw = (validation_priorities or rec_bullets)
        action_bullets = _bullets(raw)

        if not action_bullets:
            return None

        horizon = ec.get("decision_horizon") or "Next 60–90 days"
        count   = len(action_bullets)

        return Slide(
            slide_number=n,
            slide_type="content",
            title=f"{count} Action{'s' if count != 1 else ''} Required Before Strategy Is Approved",
            key_message=f"Decision horizon: {horizon}",
            bullets=action_bullets,
            notes="Each action should be owned by a named individual with a concrete deadline.",
        )


# ---------------------------------------------------------------------------
# Timeframe normaliser (matches report_agent convention)
# ---------------------------------------------------------------------------

_HORIZON_MAP = {
    "immediate":   "0–3 months",
    "short_term":  "0–3 months",
    "short-term":  "0–3 months",
    "near_term":   "3–12 months",
    "near-term":   "3–12 months",
    "medium_term": "1–3 years",
    "medium-term": "1–3 years",
    "long_term":   "3+ years",
    "long-term":   "3+ years",
}


def _normalise_horizon(raw: str) -> str:
    return _HORIZON_MAP.get(raw.lower().strip(), raw) if raw else "—"


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def render_markdown(pres: Presentation) -> str:
    """Render *pres* as a structured Markdown presentation outline."""
    lines: list[str] = []

    lines.append(f"# {pres.title}")
    if pres.subtitle:
        lines.append(f"\n*{pres.subtitle}*")
    if pres.client:
        lines.append(f"\n**Client:** {pres.client}")
    if pres.date:
        lines.append(f"\n**Date:** {pres.date}")
    lines.append("\n---\n")

    for slide in pres.slides:
        _render_slide(lines, slide)

    return "\n".join(lines)


def _render_slide(lines: list[str], slide: Slide) -> None:
    type_label = slide.slide_type.capitalize()
    lines.append(f"## Slide {slide.slide_number} — {type_label}")
    lines.append(f"\n**{slide.title}**\n")

    if slide.key_message:
        lines.append(f"> {slide.key_message}\n")

    if slide.bullets:
        for b in slide.bullets:
            if b:
                lines.append(f"- {b}")
        lines.append("")

    if slide.table:
        _render_table(lines, slide.table)

    if slide.notes:
        lines.append(f"*Notes: {slide.notes}*\n")

    lines.append("---\n")


def _render_table(lines: list[str], table: SlideTable) -> None:
    if not table.headers:
        return
    sep = "|".join(["---"] * len(table.headers))
    lines.append("| " + " | ".join(table.headers) + " |")
    lines.append(f"|{sep}|")
    for row in table.rows:
        padded = row.cells + [""] * (len(table.headers) - len(row.cells))
        lines.append("| " + " | ".join(padded) + " |")
    if table.caption:
        lines.append(f"\n*{table.caption}*")
    lines.append("")


# ---------------------------------------------------------------------------
# Context reconstruction from persisted artifacts
# ---------------------------------------------------------------------------

def context_from_session(session_path: Path) -> "AgentContext":
    """Reconstruct a minimal ``AgentContext`` from a session JSON file."""
    raw = json.loads(session_path.read_text(encoding="utf-8"))
    rs  = raw.get("research_state", raw)  # handle both wrapped and flat formats
    ro  = rs.get("research_object") or {}
    eng = rs.get("engagement") or ro.get("engagement") or {}
    ec  = rs.get("executive_confidence") or ro.get("executive_confidence") or {}
    dm  = rs.get("decision_model") or ro.get("decision_model") or {}
    return _build_context(ro=ro, engagement=eng, executive_confidence=ec, decision_model=dm)


def context_from_research_object(ro_path: Path) -> "AgentContext":
    """Reconstruct a minimal ``AgentContext`` from a research-object JSON file."""
    ro  = json.loads(ro_path.read_text(encoding="utf-8"))
    eng = ro.get("engagement") or {}
    ec  = ro.get("executive_confidence") or {}
    dm  = ro.get("decision_model") or {}
    return _build_context(ro=ro, engagement=eng, executive_confidence=ec, decision_model=dm)


def _build_context(ro: dict, engagement: dict, executive_confidence: dict, decision_model: dict) -> "AgentContext":
    from .context import AgentContext

    da  = ro.get("decision_analysis") or {}
    options = ro.get("strategic_options") or []
    rec_id  = da.get("recommended_option_id") or ""

    # preferred_option: explicit key first, then derived from decision_analysis
    preferred = (
        ro.get("preferred_option")
        or _find_option(options, rec_id)
        or {}
    )

    profiles = (
        ro.get("profiles")
        or engagement.get("profiles")
        or []
    )
    question = (
        ro.get("question")
        or ro.get("research_objective")
        or engagement.get("strategic_question")
        or engagement.get("question")
        or ""
    )

    return AgentContext(
        question=question,
        profiles=profiles,
        execution_profile=profiles[0] if profiles else "",
        engagement=engagement,
        research_object=ro,
        decision_model=decision_model,
        strategic_options=options,
        preferred_option=preferred,
        decision_analysis=da,
        assumptions=ro.get("strategic_assumptions") or ro.get("assumptions") or [],
        risks=ro.get("strategic_risks") or ro.get("risks") or [],
        opportunities=ro.get("strategic_opportunities") or ro.get("opportunities") or [],
        recommendations=ro.get("recommendations") or [],
        executive_confidence=executive_confidence,
    )


def _latest_session(sessions_dir: Path) -> Path | None:
    """Return the most-recently modified session file in *sessions_dir*."""
    files = sorted(sessions_dir.glob("SS-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None
