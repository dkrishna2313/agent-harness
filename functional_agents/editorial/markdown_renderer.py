"""MarkdownRenderer — PH7.

Renders an EditorialManuscript (with optional EditorialBrief for structured data)
into a complete Markdown executive report string.

Design constraints (inherited from PH7 spec):
- Reads only EditorialManuscript and EditorialBrief — never AgentContext
- Does not touch ReportAgent, reasoning fields, or section assembly logic
- Does not change EditorialBrief schema, EditorialWriter implementations, or
  EditorialManuscript schema
- Reuses _build_glossary_section() from report_agent (pure function, no side effects)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .editorial_brief import EditorialBrief
    from .editorial_manuscript import EditorialManuscript

LOGGER = logging.getLogger(__name__)

_TIMEFRAME_DISPLAY: dict[str, str] = {
    "near_term": "Near-term (3–12 months)",
    "medium_term": "Medium-term (1–3 years)",
    "long_term": "Long-term (3+ years)",
    "near-term": "Near-term (3–12 months)",
    "medium-term": "Medium-term (1–3 years)",
    "long-term": "Long-term (3+ years)",
}


class MarkdownRenderer:
    """Renders EditorialManuscript → Markdown report string.

    Usage::

        renderer = MarkdownRenderer()
        md = renderer.render(manuscript, brief=brief)
    """

    def render(
        self,
        manuscript: "EditorialManuscript",
        brief: "EditorialBrief | None" = None,
    ) -> str:
        lines: list[str] = ["# Executive Strategic Report", ""]
        lines += self._s1_executive_summary(manuscript)
        lines += self._s2_strategic_context(manuscript, brief)
        lines += self._s3_strategic_recommendation(manuscript, brief)
        lines += self._s4_recommendation_rationale(manuscript)
        lines += self._s_strategic_direction(manuscript, brief)
        lines += self._s5_decision_readiness(manuscript, brief)
        lines += self._s6_critical_assumptions(brief)
        lines += self._s7_key_risks(manuscript, brief)
        lines += self._s8_strategic_opportunities(manuscript, brief)
        lines += self._s9_immediate_actions(manuscript, brief)

        body = "\n".join(lines)

        # Glossary — auto-extracted from rendered text, inserted before appendix
        try:
            from ..report_agent import _build_glossary_section  # type: ignore[import]
            glossary_lines = _build_glossary_section(body)
        except Exception:
            glossary_lines = []
        if glossary_lines:
            body = body + "\n" + "\n".join(glossary_lines)

        app_lines = self._s10_appendix(manuscript, brief)
        return body + "\n" + "\n".join(app_lines)

    # -----------------------------------------------------------------------
    # Section renderers
    # -----------------------------------------------------------------------

    def _s1_executive_summary(self, ms: "EditorialManuscript") -> list[str]:
        sec = ms.executive_summary
        lines: list[str] = ["---", "", "## 1. Executive Summary", ""]

        for para in (sec.paragraphs or []):
            lines.append(f"> {para}")
            lines.append(">")
        # Trim trailing lone ">"
        while lines and lines[-1] == ">":
            lines.pop()

        bgs = sec.bullet_groups or []
        if len(bgs) > 0 and bgs[0]:
            lines += ["", "**Key Conditions:**"]
            for b in bgs[0]:
                lines.append(f"- {b}")
        if len(bgs) > 1 and bgs[1]:
            lines += ["", "**Critical Unknowns:**"]
            for b in bgs[1]:
                lines.append(f"- {b}")

        lines.append("")
        return lines

    def _s2_strategic_context(
        self,
        ms: "EditorialManuscript",
        brief: "EditorialBrief | None",
    ) -> list[str]:
        question = ""
        if brief:
            question = brief.metadata.question
        elif ms.metadata:
            question = getattr(ms.metadata, "question", "")

        return [
            "---",
            "",
            "## 2. Strategic Context",
            "",
            "*What decision does the Board need to make, and why now?*",
            "",
            question or "*Strategic question not specified.*",
            "",
        ]

    def _s3_strategic_recommendation(
        self,
        ms: "EditorialManuscript",
        brief: "EditorialBrief | None",
    ) -> list[str]:
        lines: list[str] = [
            "---",
            "",
            "## 3. Strategic Recommendation",
            "",
            "*Which path does the team recommend, and on what terms?*",
            "",
            "*Summary of key decision parameters.*",
            "",
        ]

        if not brief:
            lines.append("*Decision parameters not available.*")
            lines.append("")
            return lines

        rec_opt = next(
            (o for o in brief.strategic_options.options if o.recommended),
            None,
        )
        if rec_opt is None and brief.strategic_options.options:
            rec_opt = brief.strategic_options.options[0]

        conf = brief.executive_confidence

        # Find top risk
        top_risk = ""
        if brief.strategic_risks.top_risk_id:
            top_risk_entry = next(
                (r for r in brief.strategic_risks.risks
                 if r.risk_id == brief.strategic_risks.top_risk_id),
                None,
            )
            if top_risk_entry:
                top_risk = top_risk_entry.statement
        if not top_risk and brief.strategic_risks.risks:
            top_risk = brief.strategic_risks.risks[0].statement

        lines += ["| Field | Value |", "|---|---|"]
        if rec_opt:
            lines.append(f"| Recommended Option | {rec_opt.title} |")
            if rec_opt.estimated_time_horizon:
                th_key = rec_opt.estimated_time_horizon.lower().replace(" ", "_")
                th = _TIMEFRAME_DISPLAY.get(th_key, rec_opt.estimated_time_horizon)
                lines.append(f"| Time Horizon | {th} |")
        lines.append(f"| Decision Readiness | {conf.decision_readiness} |")
        lines.append(f"| Confidence | {conf.overall_confidence} |")
        lines.append(f"| Board Recommendation | {conf.board_recommendation} |")
        if rec_opt and rec_opt.advantages:
            lines.append(f"| Primary Benefit | {rec_opt.advantages[0]} |")
        if top_risk:
            lines.append(f"| Principal Risk | {top_risk.replace('|', chr(124))} |")
        lines.append("")

        if rec_opt and rec_opt.description:
            lines.append(rec_opt.description)
            lines.append("")

        return lines

    def _s4_recommendation_rationale(self, ms: "EditorialManuscript") -> list[str]:
        sec = ms.decision_analysis
        lines: list[str] = [
            "---",
            "",
            "## 4. Recommendation Rationale",
            "",
            "*Why does this option win, and how does it compare to alternatives?*",
            "",
        ]

        for para in (sec.paragraphs or []):
            lines.append(para)
            lines.append("")

        for table in (sec.tables or []):
            lines += self._render_table(table)

        bgs = sec.bullet_groups or []
        if len(bgs) > 0 and bgs[0]:
            lines += ["**Key Trade-offs:**"]
            for b in bgs[0]:
                lines.append(f"- {b}")
            lines.append("")
        if len(bgs) > 1 and bgs[1]:
            lines += ["**Key Uncertainties:**"]
            for b in bgs[1]:
                lines.append(f"- {b}")
            lines.append("")

        return lines

    def _s_strategic_direction(
        self,
        ms: "EditorialManuscript",
        brief: "EditorialBrief | None",
    ) -> list[str]:
        """Render the optional Strategic Direction section (PH11.4).

        Returns an empty list when the section is absent or unpopulated,
        preserving the existing report path unchanged.
        """
        sec = getattr(ms, "strategic_direction", None)
        if not sec:
            return []
        has_content = bool(
            getattr(sec, "paragraphs", None) or getattr(sec, "tables", None)
        )
        if not has_content:
            return []

        # Resolve strategy_narrative from brief for structured data
        sn = getattr(brief, "strategy_narrative", None) if brief else None

        lines: list[str] = ["---", "", "## Strategic Direction", ""]

        # Header table: winner identity and scores
        if sn is not None:
            lines += ["| Field | Value |", "|---|---|"]
            lines.append(f"| Selected Theory | {sn.winner_theory_id} |")
            if sn.winner_option_title:
                lines.append(f"| Recommended Strategy | {sn.winner_option_title} |")
            lines.append(f"| Winner Score | {sn.winner_score:.3f} |")
            if sn.runner_up_theory_id:
                lines.append(f"| Runner-up Theory | {sn.runner_up_theory_id} |")
            if sn.runner_up_score is not None:
                lines.append(f"| Runner-up Score | {sn.runner_up_score:.3f} |")
            if sn.score_margin is not None:
                lines.append(f"| Score Margin | {sn.score_margin:.3f} |")
            if sn.overall_confidence:
                lines.append(f"| Confidence | {sn.overall_confidence} |")
            if sn.framework:
                lines.append(f"| Framework | {sn.framework} |")
            if sn.tie_breaker_used:
                lines.append(f"| Tie-breaker Used | {sn.tie_breaker_used} |")
            lines.append("")

        # Recommended Strategy subsection
        lines += ["### Recommended Strategy", ""]
        for para in (sec.paragraphs or []):
            lines.append(para)
            lines.append("")

        # Why This Strategy Won: evaluation criteria scores + evaluation strengths
        bgs = sec.bullet_groups or []
        if len(bgs) > 0 and bgs[0]:
            # Separate criterion lines (no "+" prefix) from strengths lines ("+" prefix)
            crit_bullets = [b for b in bgs[0] if not b.startswith("+ ")]
            strength_bullets = [b[2:] for b in bgs[0] if b.startswith("+ ")]
            if crit_bullets or strength_bullets:
                lines += ["### Why This Strategy Won", ""]
            if crit_bullets:
                lines += ["**Evaluation Criteria Scores:**"]
                for b in crit_bullets:
                    lines.append(f"- {b}")
                lines.append("")
            if strength_bullets:
                lines += ["**Evaluation Strengths:**"]
                for b in strength_bullets:
                    lines.append(f"- {b}")
                lines.append("")

        # Alternatives Considered
        if sec.tables:
            lines += ["### Alternatives Considered", ""]
            for table in sec.tables:
                lines += self._render_table(table)

        # Strategic Choices (when present — bullet group 4)
        if len(bgs) > 4 and bgs[4]:
            lines += ["**Strategic Choices:**"]
            for b in bgs[4]:
                lines.append(f"- {b}")
            lines.append("")

        # Assumptions and Conditions for Success
        has_assumptions = len(bgs) > 1 and bgs[1]
        has_conditions = len(bgs) > 2 and bgs[2]
        if has_assumptions or has_conditions:
            lines += ["### Assumptions and Conditions for Success", ""]
            if has_assumptions:
                lines += ["**Key Assumptions:**"]
                for b in bgs[1]:
                    lines.append(f"- {b}")
                lines.append("")
            if has_conditions:
                lines += ["**Success Conditions:**"]
                for b in bgs[2]:
                    lines.append(f"- {b}")
                lines.append("")

        # Risks and Failure Modes
        if len(bgs) > 3 and bgs[3]:
            lines += ["### Risks and Failure Modes", ""]
            for b in bgs[3]:
                lines.append(f"- {b}")
            lines.append("")

        return lines

    def _s5_decision_readiness(
        self,
        ms: "EditorialManuscript",
        brief: "EditorialBrief | None",
    ) -> list[str]:
        sec = ms.executive_confidence
        lines: list[str] = [
            "---",
            "",
            "## 5. Decision Readiness",
            "",
            "*Can the Board approve this decision today?*",
            "",
        ]

        for para in (sec.paragraphs or []):
            lines.append(para)
            lines.append("")

        bgs = sec.bullet_groups or []
        if len(bgs) > 0 and bgs[0]:
            lines += ["**Validation Priorities:**"]
            for i, b in enumerate(bgs[0], start=1):
                lines.append(f"{i}. {b}")
            lines.append("")
        if len(bgs) > 1 and bgs[1]:
            lines += ["**Critical Unknowns:**"]
            for b in bgs[1]:
                lines.append(f"- {b}")
            lines.append("")

        if brief and brief.executive_confidence.decision_horizon:
            lines.append(f"**Decision Horizon:** {brief.executive_confidence.decision_horizon}")
            lines.append("")

        return lines

    def _s6_critical_assumptions(self, brief: "EditorialBrief | None") -> list[str]:
        lines: list[str] = [
            "---",
            "",
            "## 6. Critical Assumptions",
            "",
            "*What must be true for this strategy to succeed?*",
            "",
        ]

        if not brief or not brief.strategic_assumptions.assumptions:
            lines.append("*No assumptions recorded.*")
            lines.append("")
            return lines

        assumptions = brief.strategic_assumptions.assumptions
        n_critical = brief.strategic_assumptions.critical_count or sum(
            1 for a in assumptions if a.importance == "Critical"
        )
        lines.append(
            f"This recommendation depends on {n_critical} critical assumption(s). "
            "If any of these fails, the investment case changes materially."
        )
        lines.append("")

        critical = [a for a in assumptions if a.importance == "Critical"]
        important = [a for a in assumptions if a.importance == "Important"]
        others = [a for a in assumptions if a.importance not in ("Critical", "Important")]

        lines += ["| Assumption | Importance | Impact if Invalid |", "|---|---|---|"]
        for a in critical + important + others:
            impact = self._assumption_impact(a, brief)
            stmt = a.statement.replace("|", "\\|")
            lines.append(f"| {stmt} | {a.importance} | {impact} |")
        lines.append("")
        lines.append("*Full assumption register in Appendix B.*")
        lines.append("")
        return lines

    def _assumption_impact(
        self,
        assumption: Any,
        brief: "EditorialBrief | None",
    ) -> str:
        if not brief:
            return "Impact not specified"
        for risk in (brief.strategic_risks.risks or []):
            if assumption.assumption_id in (risk.related_assumption_ids or []):
                return risk.statement.replace("|", "\\|")
        return f"{assumption.importance} assumption; failure would materially affect the recommended path"

    def _s7_key_risks(
        self,
        ms: "EditorialManuscript",
        brief: "EditorialBrief | None",
    ) -> list[str]:
        sec = ms.strategic_risks
        lines: list[str] = [
            "---",
            "",
            "## 7. Key Risks",
            "",
            "*What could derail execution, and how exposed are we?*",
            "",
        ]

        for para in (sec.paragraphs or []):
            lines.append(para)
            lines.append("")

        if brief and brief.strategic_risks.risks:
            lines += [
                "| Risk | Likelihood | Business Impact | Mitigation |",
                "|---|---|---|---|",
            ]
            for r in brief.strategic_risks.risks:
                mit = (r.mitigation_notes or "Not specified").replace("|", "\\|")
                stmt = r.statement.replace("|", "\\|")
                lines.append(f"| {stmt} | {r.likelihood} | {r.severity} | {mit} |")
            lines.append("")
        elif sec.tables:
            lines += self._render_table(sec.tables[0])

        lines.append("*Full risk register in Appendix C.*")
        lines.append("")

        bgs = sec.bullet_groups or []
        if len(bgs) > 0 and bgs[0]:
            lines += ["", "**High-Severity Risks:**"]
            for b in bgs[0]:
                lines.append(f"- {b}")
            lines.append("")
        if len(bgs) > 1 and bgs[1]:
            lines += ["**Mitigation Strategies:**"]
            for b in bgs[1]:
                lines.append(f"- {b}")
            lines.append("")

        return lines

    def _s8_strategic_opportunities(
        self,
        ms: "EditorialManuscript",
        brief: "EditorialBrief | None",
    ) -> list[str]:
        sec = ms.strategic_opportunities
        lines: list[str] = [
            "---",
            "",
            "## 8. Strategic Opportunities",
            "",
            "*Where is the upside if conditions prove better than expected?*",
            "",
        ]

        for para in (sec.paragraphs or []):
            lines.append(para)
            lines.append("")

        if brief and brief.strategic_opportunities.opportunities:
            lines += [
                "| Opportunity | Category | Expected Benefit | Likelihood |",
                "|---|---|---|---|",
            ]
            for o in brief.strategic_opportunities.opportunities:
                stmt = o.statement.replace("|", "\\|")
                lines.append(f"| {stmt} | {o.category} | {o.impact} | {o.likelihood} |")
            lines.append("")
        elif sec.tables:
            lines += self._render_table(sec.tables[0])

        lines.append("*Full opportunity register in Appendix D.*")
        lines.append("")

        bgs = sec.bullet_groups or []
        if len(bgs) > 0 and bgs[0]:
            lines += ["**High-Impact Opportunities:**"]
            for b in bgs[0]:
                lines.append(f"- {b}")
            lines.append("")
        if len(bgs) > 1 and bgs[1]:
            lines += ["**Enabling Conditions:**"]
            for b in bgs[1]:
                lines.append(f"- {b}")
            lines.append("")

        return lines

    def _s9_immediate_actions(
        self,
        ms: "EditorialManuscript",
        brief: "EditorialBrief | None",
    ) -> list[str]:
        sec = ms.recommendations
        lines: list[str] = [
            "---",
            "",
            "## 9. Immediate Actions",
            "",
            "*What needs to happen in the next 90 days?*",
            "",
        ]

        for para in (sec.paragraphs or []):
            lines.append(para)
            lines.append("")

        if brief and brief.recommendations.recommendations:
            recs = brief.recommendations.recommendations
            by_horizon: dict[str, list] = {}
            for r in recs:
                key = r.time_horizon or "near_term"
                by_horizon.setdefault(key, []).append(r)

            for horizon_key in ["near_term", "medium_term", "long_term"]:
                horizon_recs = by_horizon.get(horizon_key, [])
                if not horizon_recs:
                    continue
                display = _TIMEFRAME_DISPLAY.get(horizon_key, horizon_key.replace("_", " ").title())
                lines += ["", f"### {display}", ""]
                lines += ["| Priority | Action | Details |", "|---|---|---|"]
                for r in horizon_recs:
                    title = r.title.replace("|", "\\|")
                    summary = (r.summary or "")[:100].replace("|", "\\|")
                    lines.append(f"| {r.priority.title()} | {title} | {summary} |")
                lines.append("")
        elif sec.tables:
            for table in sec.tables:
                lines += self._render_table(table)

        bgs = sec.bullet_groups or []
        if len(bgs) > 0 and bgs[0]:
            lines += ["**High-Priority Recommendations:**"]
            for b in bgs[0]:
                lines.append(f"- {b}")
            lines.append("")

        return lines

    def _s10_appendix(
        self,
        ms: "EditorialManuscript",
        brief: "EditorialBrief | None",
    ) -> list[str]:
        lines: list[str] = ["---", "", "## 10. Appendix", ""]

        # A — Strategic Options Detail
        if brief and brief.strategic_options.options:
            lines += ["---", "", "### Appendix A — Strategic Options Detail", ""]
            for opt in brief.strategic_options.options:
                suffix = " *(Recommended)*" if opt.recommended else ""
                lines.append(f"#### {opt.option_id}: {opt.title}{suffix}")
                lines.append("")
                lines.append(opt.description or "*No description.*")
                lines.append("")

        # B — Complete Assumption Register
        if brief and brief.strategic_assumptions.assumptions:
            lines += [
                "---",
                "",
                "### Appendix B — Complete Assumption Register",
                "",
                "| ID | Assumption | Importance | Confidence | Evidence Support |",
                "|---|---|---|---|---|",
            ]
            for a in brief.strategic_assumptions.assumptions:
                stmt = a.statement.replace("|", "\\|")
                lines.append(
                    f"| {a.assumption_id} | {stmt} | {a.importance} | {a.confidence} | {a.evidence_support} |"
                )
            lines.append("")

        # C — Complete Risk Register
        if brief and brief.strategic_risks.risks:
            lines += [
                "---",
                "",
                "### Appendix C — Complete Risk Register",
                "",
                "| ID | Risk | Severity | Likelihood | Related Assumptions | Affected Recommendations |",
                "|---|---|---|---|---|---|",
            ]
            for r in brief.strategic_risks.risks:
                stmt = r.statement.replace("|", "\\|")
                rel_ass = ", ".join(r.related_assumption_ids or []) or "—"
                aff_recs = ", ".join(r.affected_recommendation_ids or []) or "—"
                lines.append(
                    f"| {r.risk_id} | {stmt} | {r.severity} | {r.likelihood} | {rel_ass} | {aff_recs} |"
                )
            lines.append("")

        # D — Complete Opportunity Register
        if brief and brief.strategic_opportunities.opportunities:
            lines += [
                "---",
                "",
                "### Appendix D — Complete Opportunity Register",
                "",
                "| ID | Statement | Category | Likelihood | Impact |",
                "|---|---|---|---|---|",
            ]
            for o in brief.strategic_opportunities.opportunities:
                stmt = o.statement.replace("|", "\\|")
                lines.append(
                    f"| {o.opportunity_id} | {stmt} | {o.category} | {o.likelihood} | {o.impact} |"
                )
            lines.append("")

        # E — Detailed Confidence Analysis
        if brief:
            conf = brief.executive_confidence
            lines += ["---", "", "### Appendix E — Detailed Confidence Analysis", ""]
            # Additional paragraphs from manuscript (skip first — already in body)
            for para in (ms.executive_confidence.paragraphs or [])[1:]:
                lines.append(para)
                lines.append("")
            lines.append(f"**Overall Confidence:** {conf.overall_confidence}")
            lines.append("")
            if conf.confidence_drivers:
                lines += ["**Key Confidence Drivers:**"]
                for d in conf.confidence_drivers:
                    lines.append(f"- {d}")
                lines.append("")
            if conf.confidence_limiters:
                lines += ["**Key Confidence Limiters:**"]
                for lim in conf.confidence_limiters:
                    lines.append(f"- {lim}")
                lines.append("")
            if conf.critical_unknowns:
                lines += ["**Key Uncertainties:**"]
                for u in conf.critical_unknowns:
                    lines.append(f"- {u}")
                lines.append("")

        # F — Supporting Evidence (from manuscript.appendix)
        app_sec = ms.appendix
        if app_sec and (app_sec.paragraphs or app_sec.tables):
            lines += ["---", "", "### Appendix F — Supporting Evidence", ""]
            for para in (app_sec.paragraphs or []):
                lines.append(para)
                lines.append("")
            for table in (app_sec.tables or []):
                lines += self._render_table(table)

        return lines

    # -----------------------------------------------------------------------
    # Table renderer
    # -----------------------------------------------------------------------

    def _render_table(self, table: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        title = table.get("title", "")
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        notes = table.get("notes", "")

        if not headers:
            return lines
        if title:
            lines += ["", f"**{title}:**", ""]
        sep = "|---|" * len(headers)
        lines.append("| " + " | ".join(str(h) for h in headers) + " |")
        lines.append("|" + sep)
        for row in rows:
            cells = [str(c).replace("|", "\\|") for c in row]
            lines.append("| " + " | ".join(cells) + " |")
        if notes:
            lines += ["", f"*{notes}*"]
        lines.append("")
        return lines
