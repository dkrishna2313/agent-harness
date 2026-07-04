"""ExecutiveNarrativeComposer — narrative composition engine (J12.4).

Transforms an already-extracted ExecutiveNarrative into a coherent executive
storyline.  Two kinds of work happen here:

1. **Enrichment** of existing field ``why_this_option``: the extracted rationale
   from ``decision_analysis`` is extended with the recommended option's
   advantages (sourced from ``strategic_options``), making the Executive Brief
   and Strategy Deck "Why This Option" section richer without any generator
   change.

2. **Composition** of three new story fields: ``decision_story``,
   ``risk_story``, ``confidence_story``.  These answer the seven composition
   principles below and are available for J12.5 generators that want a
   single coherent paragraph rather than structured sub-fields.

Composition principles (in narrative order)
-------------------------------------------
1. What decision is being made?        → decision_story
2. Why now?                            → decision_story (executive_summary)
3. Which option wins?                  → decision_story (recommended_option)
4. Why does it win?                    → decision_story (why_this_option)
5. What could invalidate it?           → risk_story
6. What should executives do next?     → risk_story (immediate_actions)
7. What evidence still needs collecting? → confidence_story (validation_priorities)

Design invariants
-----------------
- Every sentence traces to a specific named field of ExecutiveNarrative.
- No new reasoning: composition is pure string assembly over already-extracted
  facts.
- Deterministic: the same ExecutiveNarrative always produces the same output.
- Never reads AgentContext directly.
- Never mutates key_risks, critical_assumptions, strategic_options, or any
  other structured field — only writes to why_this_option and the three
  story fields.
"""

from __future__ import annotations

from .executive_narrative import ExecutiveNarrative


class ExecutiveNarrativeComposer:
    """Composes story fields from an already-extracted ExecutiveNarrative.

    Usage::

        narrative = ExecutiveNarrativeComposer().compose(narrative)

    Mutates the narrative in-place (story fields set, why_this_option enriched)
    and returns the same object.  Designed to run immediately after
    ExecutiveNarrativeBuilder extracts fields from AgentContext.
    """

    def compose(self, narrative: ExecutiveNarrative) -> ExecutiveNarrative:
        """Enrich why_this_option and set story fields; returns the same instance."""
        # Enrich why_this_option first — _compose_decision_story reads it.
        narrative.why_this_option = self._enrich_why_this_option(narrative)
        narrative.decision_story = self._compose_decision_story(narrative)
        narrative.risk_story = self._compose_risk_story(narrative)
        narrative.confidence_story = self._compose_confidence_story(narrative)
        return narrative

    # ------------------------------------------------------------------
    # Enrichment of existing field
    # ------------------------------------------------------------------

    def _enrich_why_this_option(self, n: ExecutiveNarrative) -> str:
        """Append recommended-option advantages to the extracted rationale.

        Source: ``why_this_option`` (extracted rationale from decision_analysis)
                + ``strategic_options[recommended_id].advantages``.

        The original rationale is always preserved as the leading sentence.
        Does nothing if why_this_option is empty or no advantages are found.
        """
        rationale = n.why_this_option
        if not rationale:
            return rationale

        rec_id = (n.recommended_option or {}).get("option_id", "")
        advantages: list[str] = []
        if rec_id:
            opt = next(
                (o for o in (n.strategic_options or []) if o.get("option_id") == rec_id),
                None,
            )
            if opt:
                advantages = list((opt.get("advantages") or [])[:3])

        if not advantages:
            return rationale

        adv_text = f"Key advantages: {'; '.join(advantages)}."
        base = rationale.rstrip(".")
        return f"{base}.  {adv_text}"

    # ------------------------------------------------------------------
    # Story field composers — one per narrative question
    # ------------------------------------------------------------------

    def _compose_decision_story(self, n: ExecutiveNarrative) -> str:
        """Compose a decision narrative paragraph (composition principles 1–4).

        Sources (in order): decision, executive_summary, recommended_option,
        why_this_option (already enriched), option_rankings / strategic_options
        (count), key_tradeoffs.
        """
        parts: list[str] = []

        # Q1: What decision is being made?
        if n.decision:
            d = n.decision.rstrip(".")
            parts.append(f"{d}.")

        # Q2: Why now? / strategic context
        if n.executive_summary:
            s = n.executive_summary.strip().rstrip(".")
            parts.append(f"{s}.")

        # Q3: Which option wins?
        rec = n.recommended_option or {}
        if rec.get("option_id") and rec.get("title"):
            horizon = (rec.get("estimated_time_horizon") or "").replace("_", " ")
            rec_line = f"Recommended: {rec['option_id']} — {rec['title']}"
            if horizon:
                rec_line += f" ({horizon})"
            parts.append(f"{rec_line}.")

        # Q4: Why does it win? (uses the enriched rationale)
        if n.why_this_option:
            w = n.why_this_option.strip()
            if not w.endswith("."):
                w += "."
            parts.append(w)

        # Option landscape context
        n_options = len(n.option_rankings or n.strategic_options or [])
        if n_options > 1:
            parts.append(f"{n_options} strategic options were evaluated.")

        # Key evaluation dimensions
        tradeoffs = (n.key_tradeoffs or [])[:3]
        if tradeoffs:
            parts.append(f"The decision turns on: {', '.join(tradeoffs)}.")

        return "  ".join(parts) if parts else ""

    def _compose_risk_story(self, n: ExecutiveNarrative) -> str:
        """Compose a risk narrative connecting identified risks to near-term actions
        (composition principles 5–6).

        Sources: key_risks (risk_id, statement, severity, mitigation),
                 immediate_actions (id, title).
        """
        risks = n.key_risks or []
        if not risks:
            return ""

        parts: list[str] = []

        # Risk count and severity distribution
        severity_counts: dict[str, int] = {}
        for r in risks:
            sev = str(r.get("severity") or "").lower()
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        sev_summary_parts = []
        for sev in ("critical", "high", "medium", "low"):
            if sev in severity_counts:
                sev_summary_parts.append(f"{severity_counts[sev]} {sev}")
        sev_summary = ", ".join(sev_summary_parts) if sev_summary_parts else ""

        count_line = f"{len(risks)} material risk(s) identified"
        if sev_summary:
            count_line += f" ({sev_summary})"
        parts.append(f"{count_line}.")

        # Top 3 risk details with mitigations
        for r in risks[:3]:
            rid = r.get("risk_id", "")
            stmt = r.get("statement", "")
            sev = r.get("severity", "")
            mitigation = r.get("mitigation", "")
            line = f"{rid}: {stmt} (severity: {sev})"
            if mitigation:
                line += f"; mitigation: {mitigation}"
            parts.append(f"{line}.")

        # Connect risks to near-term actions (Q6)
        actions = n.immediate_actions or []
        if actions:
            titles = [a.get("title", "") for a in actions if a.get("title")]
            if titles:
                parts.append(f"Near-term actions address these risks: {'; '.join(titles[:3])}.")

        return "  ".join(parts)

    def _compose_confidence_story(self, n: ExecutiveNarrative) -> str:
        """Compose a confidence narrative connecting assessment, assumptions,
        and unknowns (composition principle 7).

        Sources: executive_confidence (overall_confidence, decision_readiness,
                 board_recommendation, confidence_rationale, confidence_drivers,
                 confidence_limiters), critical_assumptions, validation_priorities,
                 critical_unknowns.
        """
        ec = n.executive_confidence or {}
        if not ec:
            return ""

        parts: list[str] = []

        # Opening assessment
        assessment_parts: list[str] = []
        overall = ec.get("overall_confidence", "")
        readiness = ec.get("decision_readiness", "")
        board_rec = ec.get("board_recommendation", "")
        if overall:
            assessment_parts.append(f"overall confidence: {overall}")
        if readiness:
            assessment_parts.append(f"decision readiness: {readiness}")
        if board_rec:
            assessment_parts.append(f"board recommendation: {board_rec}")
        if assessment_parts:
            parts.append("Assessment — " + "; ".join(assessment_parts) + ".")

        # Rationale
        rationale = ec.get("confidence_rationale", "")
        if rationale:
            parts.append(rationale.rstrip(".") + ".")

        # Critical assumptions underpinning confidence
        critical_assum = [
            a for a in (n.critical_assumptions or [])
            if str(a.get("importance") or "").lower() == "critical"
        ]
        if critical_assum:
            texts = [a.get("statement", "") for a in critical_assum[:2] if a.get("statement")]
            if texts:
                parts.append(f"Confidence rests on: {'; '.join(texts)}.")

        # Drivers and limiters
        drivers = list((ec.get("confidence_drivers") or [])[:3])
        if drivers:
            parts.append(f"Drivers: {'; '.join(drivers)}.")
        limiters = list((ec.get("confidence_limiters") or [])[:2])
        if limiters:
            parts.append(f"Limiters: {'; '.join(limiters)}.")

        # Q7: What evidence still needs to be collected?
        vp = (n.validation_priorities or [])[:3]
        if vp:
            parts.append(f"Before committing: {'; '.join(vp)}.")

        unknowns = (n.critical_unknowns or [])[:2]
        if unknowns:
            parts.append(f"Critical unknowns: {'; '.join(unknowns)}.")

        return "  ".join(parts) if parts else ""
