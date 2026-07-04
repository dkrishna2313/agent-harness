"""ExecutiveNarrative — canonical executive communication object (J12.0/J12.1).

Represents the executive storyline derived from a completed Strategic Reasoning
Graph. Fields are assembled by ExecutiveNarrativeBuilder from already-computed
AgentContext values — no new reasoning, no LLM call, no Functional Agent
invoked.

Design principle: reference/summarise, don't duplicate.
- Each field is a concise extraction from one authoritative AgentContext field.
- The Strategic Reasoning Graph on AgentContext remains the single source of
  truth; ExecutiveNarrative is a read-only view optimised for executive
  communication.
- Generators (J12.1 ExecutiveBriefGenerator, J12.2 StrategyDeckGenerator)
  read from ExecutiveNarrative instead of independently assembling from
  AgentContext, eliminating cross-generator inconsistencies.

J12.1 additions: option_rankings, critical_unknowns; extended executive_confidence
dict to carry confidence_drivers and confidence_limiters (consumed by
ExecutiveBriefGenerator's confidence section).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutiveNarrative:
    """Canonical executive storyline assembled from a completed AgentContext.

    Fields
    ------
    decision              : executive decision statement (the "what")
    executive_summary     : one-paragraph narrative summary
    recommended_option    : identifying fields of the recommended option
    why_this_option       : rationale for the recommendation
    key_tradeoffs         : dimensions along which options were evaluated
    key_risks             : top risks (risk_id, statement, severity, likelihood)
    key_opportunities     : top opportunities (id, title, impact)
    critical_assumptions  : critical/important assumptions (id, statement, importance)
    executive_confidence  : confidence summary dict (overall_confidence,
                            decision_readiness, board_recommendation,
                            confidence_rationale, confidence_drivers,
                            confidence_limiters)
    immediate_actions     : near-term recommended actions (id, title)
    validation_priorities : what must be validated before committing
    option_rankings       : ordered option IDs from decision_analysis (J12.1)
    critical_unknowns     : unknowns that must be resolved before deciding (J12.1)
    strategic_options     : all options with presentation fields (J12.2 — slide 4 table)
    medium_term_actions   : 90-day portfolio actions (J12.2 — slide 10)
    long_term_actions     : 180-day portfolio actions (J12.2 — slide 10)
    supporting_evidence   : surviving hypotheses for evidence slide (J12.2 — slide 11)
    """

    decision: str = ""
    executive_summary: str = ""
    recommended_option: dict[str, Any] = field(default_factory=dict)
    why_this_option: str = ""
    key_tradeoffs: list[str] = field(default_factory=list)
    key_risks: list[dict[str, Any]] = field(default_factory=list)
    key_opportunities: list[dict[str, Any]] = field(default_factory=list)
    critical_assumptions: list[dict[str, Any]] = field(default_factory=list)
    executive_confidence: dict[str, Any] = field(default_factory=dict)
    immediate_actions: list[dict[str, Any]] = field(default_factory=list)
    validation_priorities: list[str] = field(default_factory=list)
    # J12.1 — required by ExecutiveBriefGenerator; avoids direct decision_analysis /
    # executive_confidence reads inside the generator.
    option_rankings: list[str] = field(default_factory=list)
    critical_unknowns: list[str] = field(default_factory=list)
    # J12.2 — required by StrategyDeckGenerator.
    # strategic_options: full list (slide 4 table + recommended-option marker).
    # medium_term_actions / long_term_actions: portfolio buckets beyond near-term.
    # supporting_evidence: post-challenge hypotheses used on slide 11.
    strategic_options: list[dict[str, Any]] = field(default_factory=list)
    medium_term_actions: list[dict[str, Any]] = field(default_factory=list)
    long_term_actions: list[dict[str, Any]] = field(default_factory=list)
    supporting_evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain JSON-serialisable dict.

        All fields are always present (empty string / [] / {} for unset
        values) — consumers can rely on key presence without ``get()`` guards.
        """
        return {
            "decision": self.decision,
            "executive_summary": self.executive_summary,
            "recommended_option": self.recommended_option,
            "why_this_option": self.why_this_option,
            "key_tradeoffs": self.key_tradeoffs,
            "key_risks": self.key_risks,
            "key_opportunities": self.key_opportunities,
            "critical_assumptions": self.critical_assumptions,
            "executive_confidence": self.executive_confidence,
            "immediate_actions": self.immediate_actions,
            "validation_priorities": self.validation_priorities,
            "option_rankings": self.option_rankings,
            "critical_unknowns": self.critical_unknowns,
            "strategic_options": self.strategic_options,
            "medium_term_actions": self.medium_term_actions,
            "long_term_actions": self.long_term_actions,
            "supporting_evidence": self.supporting_evidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ExecutiveNarrative":
        """Restore from a dict (e.g. from ``context.executive_narrative``)."""
        if not data:
            return cls()
        return cls(
            decision=data.get("decision", ""),
            executive_summary=data.get("executive_summary", ""),
            recommended_option=data.get("recommended_option") or {},
            why_this_option=data.get("why_this_option", ""),
            key_tradeoffs=list(data.get("key_tradeoffs") or []),
            key_risks=list(data.get("key_risks") or []),
            key_opportunities=list(data.get("key_opportunities") or []),
            critical_assumptions=list(data.get("critical_assumptions") or []),
            executive_confidence=data.get("executive_confidence") or {},
            immediate_actions=list(data.get("immediate_actions") or []),
            validation_priorities=list(data.get("validation_priorities") or []),
            option_rankings=list(data.get("option_rankings") or []),
            critical_unknowns=list(data.get("critical_unknowns") or []),
            strategic_options=list(data.get("strategic_options") or []),
            medium_term_actions=list(data.get("medium_term_actions") or []),
            long_term_actions=list(data.get("long_term_actions") or []),
            supporting_evidence=list(data.get("supporting_evidence") or []),
        )
