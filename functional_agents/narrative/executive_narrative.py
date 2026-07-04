"""ExecutiveNarrative — canonical executive communication object (J12.0).

Represents the executive storyline derived from a completed Strategic Reasoning
Graph. Fields are assembled by ExecutiveNarrativeBuilder from already-computed
AgentContext values — no new reasoning, no LLM call, no Functional Agent
invoked.

Design principle: reference/summarise, don't duplicate.
- Each field is a concise extraction from one authoritative AgentContext field.
- The Strategic Reasoning Graph on AgentContext remains the single source of
  truth; ExecutiveNarrative is a read-only view optimised for executive
  communication.
- Future generators (J12.1 ExecutiveBriefGenerator, J12.2 StrategyDeckGenerator)
  will read from ExecutiveNarrative instead of independently assembling from
  AgentContext, eliminating cross-generator inconsistencies.
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
    executive_confidence  : confidence assessment summary dict
    immediate_actions     : near-term recommended actions (id, title)
    validation_priorities : what must be validated before committing
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

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain JSON-serialisable dict.

        All 11 fields are always present (empty string / [] / {} for unset
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
        )
