"""ExecutiveNarrativeBuilder — assembles ExecutiveNarrative from AgentContext (J12.0).

Pure extraction and narrative composition over an already-completed Strategic
Reasoning Graph. No LLM call, no Functional Agent invoked, no reasoning field
mutated. The only AgentContext side effect is setting
``context.executive_narrative = narrative.to_dict()`` — presentation state,
not reasoning state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
        )
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
            }
            for r in risks
        ]

    def _extract_key_opportunities(self, context: "AgentContext") -> list[dict[str, Any]]:
        opportunities = (context.opportunities or [])[:5]
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
        return {
            "overall_confidence": ec.get("overall_confidence", ""),
            "decision_readiness": ec.get("decision_readiness", ""),
            "board_recommendation": ec.get("board_recommendation", ""),
            "confidence_rationale": ec.get("confidence_rationale", ""),
        }

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
