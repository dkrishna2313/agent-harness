"""ExecutiveNarrativeBuilder — assembles ExecutiveNarrative from AgentContext (J12.0).

Pure extraction and narrative composition over an already-completed Strategic
Reasoning Graph. No LLM call, no Functional Agent invoked, no reasoning field
mutated. The only AgentContext side effect is setting
``context.executive_narrative = narrative.to_dict()`` — presentation state,
not reasoning state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .composer import ExecutiveNarrativeComposer
from .executive_narrative import ExecutiveNarrative

if TYPE_CHECKING:
    from ..context import AgentContext

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
_IMPORTANCE_ORDER = {"critical": 0, "important": 1, "supporting": 2}


def _rank(value: str, order: dict[str, int]) -> int:
    return order.get(str(value or "").lower(), 99)


class ExecutiveNarrativeBuilder:
    """Assembles an ExecutiveNarrative from a completed AgentContext.

    Usage::

        narrative = ExecutiveNarrativeBuilder().build(context)
        # context.executive_narrative is now set

    The builder reads from the Strategic Reasoning Graph (risks, assumptions,
    decision_analysis, strategic_options, executive_confidence, …) and
    produces a single canonical executive view. No field is re-inferred or
    re-ranked beyond the presentation-layer sorting the existing J7 report
    already performs.
    """

    def build(self, context: "AgentContext") -> ExecutiveNarrative:
        """Build and return an ExecutiveNarrative; sets context.executive_narrative."""
        narrative = ExecutiveNarrative(
            decision=self._extract_decision(context),
            executive_summary=self._extract_executive_summary(context),
            recommended_option=self._extract_recommended_option(context),
            why_this_option=self._extract_why_this_option(context),
            key_tradeoffs=self._extract_key_tradeoffs(context),
            key_risks=self._extract_key_risks(context),
            key_opportunities=self._extract_key_opportunities(context),
            critical_assumptions=self._extract_critical_assumptions(context),
            executive_confidence=self._extract_executive_confidence(context),
            immediate_actions=self._extract_immediate_actions(context),
            validation_priorities=self._extract_validation_priorities(context),
            option_rankings=self._extract_option_rankings(context),
            critical_unknowns=self._extract_critical_unknowns(context),
            strategic_options=self._extract_strategic_options(context),
            medium_term_actions=self._extract_portfolio_actions(context, "medium_term"),
            long_term_actions=self._extract_portfolio_actions(context, "long_term"),
            supporting_evidence=self._extract_supporting_evidence(context),
        )
        # J12.4 — compose story fields and enrich why_this_option with advantages.
        ExecutiveNarrativeComposer().compose(narrative)
        context.executive_narrative = narrative.to_dict()
        return narrative

    # ------------------------------------------------------------------
    # Private extractors — one per ExecutiveNarrative field
    # ------------------------------------------------------------------

    def _extract_decision(self, context: "AgentContext") -> str:
        ro = context.research_object or {}
        arch = ro.get("decision_architecture") or context.decision_architecture or {}
        return arch.get("decision_statement", "")

    def _extract_executive_summary(self, context: "AgentContext") -> str:
        summary = (context.strategic_synthesis or {}).get("executive_summary", "")
        if not summary:
            summary = (context.decision_analysis or {}).get("executive_summary", "")
        return summary

    def _extract_recommended_option(self, context: "AgentContext") -> dict[str, Any]:
        da = context.decision_analysis or {}
        preferred = context.preferred_option or {}
        options = context.strategic_options or []
        rec_id = da.get("recommended_option_id") or preferred.get("option_id", "")
        option = next((o for o in options if o.get("option_id") == rec_id), preferred or {})
        if not option:
            return {}
        return {
            "option_id": option.get("option_id", ""),
            "title": option.get("title", ""),
            "description": option.get("description", ""),
            "estimated_time_horizon": option.get("estimated_time_horizon", ""),
            "capital_intensity": option.get("capital_intensity", ""),
            "confidence": option.get("confidence", ""),
        }

    def _extract_why_this_option(self, context: "AgentContext") -> str:
        return (context.decision_analysis or {}).get("rationale", "")

    def _extract_key_tradeoffs(self, context: "AgentContext") -> list[str]:
        # Prefer explicit tradeoffs from strategic_synthesis if present.
        tradeoffs = (context.strategic_synthesis or {}).get("key_tradeoffs") or []
        if tradeoffs:
            return list(tradeoffs)
        # Fall back to comparison dimensions — the axes along which options
        # were evaluated (presentation-layer label for the tradeoff space).
        dimensions = (context.decision_analysis or {}).get("comparison_dimensions") or []
        return list(dimensions)

    def _extract_key_risks(self, context: "AgentContext") -> list[dict[str, Any]]:
        risks = sorted(
            context.risks or [],
            key=lambda r: _rank(r.get("severity", ""), _SEVERITY_ORDER),
        )[:5]
        return [
            {
                "risk_id": r.get("risk_id", ""),
                "statement": r.get("statement", ""),
                "severity": r.get("severity", ""),
                "likelihood": r.get("likelihood", ""),
                # J12.2 — deck slide 6 renders mitigation without re-reading AgentContext.
                "mitigation": (
                    r.get("mitigation") or r.get("mitigation_strategy")
                    or r.get("mitigation_approach") or ""
                ),
            }
            for r in risks
        ]

    def _extract_key_opportunities(self, context: "AgentContext") -> list[dict[str, Any]]:
        opportunities = (context.opportunities or [])[:6]  # J12.2 — deck slide 7 uses up to 6
        return [
            {
                "opportunity_id": o.get("opportunity_id", "") or o.get("id", ""),
                "title": o.get("title", "") or o.get("name", ""),
                "impact": o.get("impact", "") or o.get("strategic_value", ""),
                "description": o.get("description", "") or o.get("statement", ""),
            }
            for o in opportunities
        ]

    def _extract_critical_assumptions(self, context: "AgentContext") -> list[dict[str, Any]]:
        assumptions = sorted(
            context.assumptions or [],
            key=lambda a: _rank(a.get("importance", ""), _IMPORTANCE_ORDER),
        )[:5]
        return [
            {
                "assumption_id": a.get("assumption_id", ""),
                "statement": a.get("statement", ""),
                "importance": a.get("importance", ""),
                "confidence": a.get("confidence", ""),
            }
            for a in assumptions
        ]

    def _extract_executive_confidence(self, context: "AgentContext") -> dict[str, Any]:
        ec = context.executive_confidence or {}
        if not ec:
            return {}
        result: dict[str, Any] = {
            "overall_confidence": ec.get("overall_confidence", ""),
            "decision_readiness": ec.get("decision_readiness", ""),
            "board_recommendation": ec.get("board_recommendation", ""),
            "confidence_rationale": ec.get("confidence_rationale", ""),
        }
        # J12.1 — include drivers/limiters so ExecutiveBriefGenerator can render
        # the confidence section without reading executive_confidence directly.
        drivers = list((ec.get("confidence_drivers") or [])[:3])
        if drivers:
            result["confidence_drivers"] = drivers
        limiters = list((ec.get("confidence_limiters") or [])[:3])
        if limiters:
            result["confidence_limiters"] = limiters
        return result

    def _extract_immediate_actions(self, context: "AgentContext") -> list[dict[str, Any]]:
        near_term_ids = (context.recommendation_portfolio or {}).get("near_term", [])
        by_id = {
            r.get("id", r.get("recommendation_id", "")): r
            for r in (context.recommendations or [])
        }
        return [
            {
                "id": rid,
                "title": by_id[rid].get("title", ""),
            }
            for rid in near_term_ids
            if rid in by_id
        ]

    def _extract_validation_priorities(self, context: "AgentContext") -> list[str]:
        return list((context.executive_confidence or {}).get("validation_priorities") or [])

    def _extract_option_rankings(self, context: "AgentContext") -> list[str]:
        return list((context.decision_analysis or {}).get("option_rankings") or [])

    def _extract_critical_unknowns(self, context: "AgentContext") -> list[str]:
        return list((context.executive_confidence or {}).get("critical_unknowns") or [])

    def _extract_strategic_options(self, context: "AgentContext") -> list[dict[str, Any]]:
        """All strategic options with presentation-relevant fields (J12.2 — slide 4 table)."""
        return [
            {
                "option_id": o.get("option_id", ""),
                "title": o.get("title", ""),
                "description": o.get("description", ""),
                "estimated_time_horizon": o.get("estimated_time_horizon", ""),
                "capital_intensity": o.get("capital_intensity", ""),
                "confidence": o.get("confidence", ""),
                "advantages": list((o.get("advantages") or [])[:3]),
            }
            for o in (context.strategic_options or [])
        ]

    def _extract_portfolio_actions(
        self, context: "AgentContext", bucket: str
    ) -> list[dict[str, Any]]:
        """Extract {id, title} actions from a recommendation_portfolio bucket."""
        ids = (context.recommendation_portfolio or {}).get(bucket, [])
        by_id = {
            r.get("id", r.get("recommendation_id", "")): r
            for r in (context.recommendations or [])
        }
        return [{"id": rid, "title": by_id[rid].get("title", "")} for rid in ids if rid in by_id]

    def _extract_supporting_evidence(self, context: "AgentContext") -> list[dict[str, Any]]:
        """Surviving (post-challenge) hypotheses as supporting evidence (J12.2 — slide 11)."""
        hypotheses = context.surviving_hypotheses or context.hypotheses or []
        return [
            {
                "id": h.get("id", h.get("hypothesis_id", "")),
                "title": h.get("title", "") or h.get("statement", ""),
                "confidence": h.get("confidence", "") or h.get("confidence_level", ""),
            }
            for h in hypotheses[:6]
        ]
