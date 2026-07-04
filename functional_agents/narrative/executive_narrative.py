"""ExecutiveNarrative — canonical executive communication contract (J12.0–J12.4).

CONTRACT VERSION: 1.1  (frozen J12.3; extended J12.4 with story fields)
========================================================================

ExecutiveNarrative is the canonical communication model for all executive-facing
deliverables. It is assembled once from a completed Strategic Reasoning Graph by
ExecutiveNarrativeBuilder, then consumed by every generator — eliminating
cross-generator inconsistencies and decoupling presentation from reasoning.

Architecture
------------
Strategic Reasoning Graph (AgentContext)
    │
    ▼  ExecutiveNarrativeBuilder.build()
    │   (field extraction)
    ▼  ExecutiveNarrativeComposer.compose()
    │   (story field composition + why_this_option enrichment)
ExecutiveNarrative  (this object — communication contract v1.1)
    │
    ├── ExecutiveBriefGenerator   (J12.1, improved J12.4)
    ├── StrategyDeckGenerator     (J12.2, improved J12.4)
    └── MarkdownReportGenerator   (J12.5, planned)

Design invariants
-----------------
1. Reference / summarise, don't duplicate.
   Each field is a concise extraction from one authoritative AgentContext field.
   The Strategic Reasoning Graph remains the single source of truth.
2. All fields always present in ``to_dict()``.
   Consumers can read ``narrative["key_risks"]`` without a ``KeyError`` guard —
   empty list is meaningful (no risks computed); absent key is not permitted.
3. No new reasoning.
   The builder performs no LLM calls, no Functional Agent invocations, and no
   inference beyond sorting/truncation of already-computed values.
4. Deterministic.
   ``build(context)`` on the same completed AgentContext always returns an
   identical ExecutiveNarrative.

Field contract (v1.0)
---------------------

version
  Purpose   Schema version for this narrative instance. Consumers may use this
            to negotiate schema differences across pipeline versions.
  Source    Dataclass default ``"1.1"``; not read from AgentContext.
  Consumers All generators (forward-compatibility guard).
  Required  Yes — always ``"1.1"`` in this release.
  Evolution ``"1.0"`` → ``"1.1"`` (J12.4): story fields added (additive; no
            semantic changes to existing fields). Increment to ``"2.0"`` only
            when an existing field changes semantics or is removed.

decision
  Purpose   Executive decision statement — the question being decided.
  Source    ``decision_architecture.decision_statement`` (from research_object
            or AgentContext.decision_architecture).
  Consumers ExecutiveBriefGenerator § 1, StrategyDeckGenerator slide 1.
  Required  Yes — present in all complete engagement runs.
  Evolution Add contextual fields; do not change the statement's meaning.

executive_summary
  Purpose   One-paragraph narrative summary of the strategic situation.
  Source    ``strategic_synthesis.executive_summary`` →
            ``decision_analysis.executive_summary`` (fallback).
  Consumers ExecutiveBriefGenerator § 2, StrategyDeckGenerator slide 3.
  Required  Yes — present in all complete engagement runs.
  Evolution Single prose field; do not split into sub-fields.

recommended_option
  Purpose   Identifying and descriptive fields of the recommended strategic option.
            Keys: option_id, title, description, estimated_time_horizon,
            capital_intensity, confidence.
  Source    ``decision_analysis.recommended_option_id`` matched against
            ``strategic_options``; falls back to ``preferred_option``.
  Consumers ExecutiveBriefGenerator § 1/3, StrategyDeckGenerator slide 1/4.
  Required  Yes — present when DecisionAnalysisAgent ran.
  Evolution Add keys to the dict; do not remove option_id or title.

why_this_option
  Purpose   Prose rationale for the recommendation.
  Source    ``decision_analysis.rationale``.
  Consumers ExecutiveBriefGenerator § 4, StrategyDeckGenerator slide 5.
  Required  Yes — present when DecisionAnalysisAgent ran.
  Evolution Single prose field.

key_tradeoffs
  Purpose   Named dimensions along which options were evaluated.
  Source    ``strategic_synthesis.key_tradeoffs`` →
            ``decision_analysis.comparison_dimensions`` (fallback).
  Consumers ExecutiveBriefGenerator § 4, StrategyDeckGenerator slide 5.
  Required  No — empty list when no dimensions computed.
  Evolution Append items; do not reorder or remove existing items mid-run.

key_risks
  Purpose   Top-5 risks sorted by severity (critical → high → medium → low).
            Each entry: risk_id, statement, severity, likelihood, mitigation.
  Source    ``AgentContext.risks`` (sorted/truncated by builder).
  Consumers ExecutiveBriefGenerator § 5, StrategyDeckGenerator slide 6.
  Required  No — empty list when RiskAgent did not run.
  Evolution Add keys to each risk dict; do not remove risk_id, statement, severity.

key_opportunities
  Purpose   Top-6 opportunities (impact-ordered by source list order).
            Each entry: opportunity_id, title, impact, description.
  Source    ``AgentContext.opportunities`` (first 6).
  Consumers StrategyDeckGenerator slide 7.
  Required  No — empty list when OpportunityAgent did not run.
  Evolution Add keys to each opportunity dict.

critical_assumptions
  Purpose   Top-5 assumptions sorted by importance (critical → important →
            supporting). Each entry: assumption_id, statement, importance,
            confidence.
  Source    ``AgentContext.assumptions`` (sorted/truncated by builder).
  Consumers ExecutiveBriefGenerator § 6, StrategyDeckGenerator slide 8.
  Required  No — empty list when AssumptionAgent did not run.
  Evolution Add keys to each assumption dict.

executive_confidence
  Purpose   Confidence assessment summary.
            Always-present keys: overall_confidence, decision_readiness,
            board_recommendation, confidence_rationale.
            Optional keys (absent when empty): confidence_drivers (list, max 3),
            confidence_limiters (list, max 3).
  Source    ``AgentContext.executive_confidence``.
  Consumers ExecutiveBriefGenerator § 7, StrategyDeckGenerator slide 9.
  Required  No — empty dict when ExecutiveConfidenceAgent did not run.
  Evolution Add optional keys; do not remove the four always-present keys.

immediate_actions
  Purpose   Near-term (≤30 day) portfolio actions. Each entry: id, title.
  Source    ``recommendation_portfolio["near_term"]`` resolved against
            ``AgentContext.recommendations``.
  Consumers ExecutiveBriefGenerator § 8, StrategyDeckGenerator slide 10.
  Required  No — empty list when portfolio not populated.
  Evolution Add keys to each action dict.

validation_priorities
  Purpose   Ordered list of validation tasks that must complete before the
            decision can be made with confidence.
  Source    ``executive_confidence.validation_priorities``.
  Consumers ExecutiveBriefGenerator § 9, StrategyDeckGenerator slide 9.
  Required  No — empty list when ExecutiveConfidenceAgent did not run.
  Evolution Append items.

option_rankings
  Purpose   Ordered list of option IDs from best to least preferred.
  Source    ``decision_analysis.option_rankings``.
  Consumers ExecutiveBriefGenerator § 4, StrategyDeckGenerator slide 5.
  Required  No — empty list when DecisionAnalysisAgent did not rank options.
  Evolution Ordered list; do not change the ordering semantics.

critical_unknowns
  Purpose   Facts that are unknown and must be resolved before committing.
  Source    ``executive_confidence.critical_unknowns``.
  Consumers ExecutiveBriefGenerator appendix, StrategyDeckGenerator slide 9/12.
  Required  No — empty list when none identified.
  Evolution Append items.

strategic_options
  Purpose   All strategic options with presentation-relevant fields.
            Each entry: option_id, title, description, estimated_time_horizon,
            capital_intensity, confidence, advantages (max 3).
  Source    ``AgentContext.strategic_options``.
  Consumers StrategyDeckGenerator slide 4.
  Required  No — empty list when StrategicOptionsAgent did not run.
  Evolution Add keys to each option dict; do not remove option_id or title.

medium_term_actions
  Purpose   Medium-term (≤90 day) portfolio actions. Each entry: id, title.
  Source    ``recommendation_portfolio["medium_term"]`` resolved against
            ``AgentContext.recommendations``.
  Consumers StrategyDeckGenerator slide 10.
  Required  No — empty list when portfolio not populated.
  Evolution Add keys to each action dict.

long_term_actions
  Purpose   Long-term (≤180 day) portfolio actions. Each entry: id, title.
  Source    ``recommendation_portfolio["long_term"]`` resolved against
            ``AgentContext.recommendations``.
  Consumers StrategyDeckGenerator slide 10.
  Required  No — empty list when portfolio not populated.
  Evolution Add keys to each action dict.

supporting_evidence
  Purpose   Post-challenge surviving hypotheses used as evidence backing.
            Each entry: id, title, confidence.
  Source    ``AgentContext.surviving_hypotheses`` → ``AgentContext.hypotheses``
            (fallback), first 6.
  Consumers StrategyDeckGenerator slide 11.
  Required  No — empty list when HypothesisAgent / ChallengeAgent did not run.
  Evolution Add keys to each evidence dict.

decision_story  [J12.4 — composed by ExecutiveNarrativeComposer]
  Purpose   Coherent prose paragraph answering: what decision, why now, which
            option wins, why it wins, option landscape, key tradeoffs.
  Source    Composed from: decision, executive_summary, recommended_option,
            why_this_option (enriched), option_rankings / strategic_options
            (count), key_tradeoffs.
  Consumers MarkdownReportGenerator (J12.5+), any consumer wanting a single
            executive paragraph instead of structured sub-fields.
  Required  No — empty string when context is insufficient.
  Evolution Append sentences; do not remove existing composition inputs.

risk_story  [J12.4 — composed by ExecutiveNarrativeComposer]
  Purpose   Coherent prose paragraph covering risk count, severity distribution,
            top-3 risks with mitigations, and connection to near-term actions.
  Source    Composed from: key_risks (risk_id, statement, severity, mitigation),
            immediate_actions (id, title).
  Consumers MarkdownReportGenerator (J12.5+).
  Required  No — empty string when key_risks is empty.
  Evolution Append sentences; do not change risk count logic.

confidence_story  [J12.4 — composed by ExecutiveNarrativeComposer]
  Purpose   Coherent prose paragraph covering confidence assessment, rationale,
            assumptions underpinning confidence, drivers/limiters, validation
            priorities, and critical unknowns.
  Source    Composed from: executive_confidence (overall_confidence,
            decision_readiness, board_recommendation, confidence_rationale,
            confidence_drivers, confidence_limiters), critical_assumptions,
            validation_priorities, critical_unknowns.
  Consumers MarkdownReportGenerator (J12.5+).
  Required  No — empty string when executive_confidence is empty.
  Evolution Append sentences; do not change assumption-filtering logic.

Evolution rules (contract v1.1)
--------------------------------
- NEW FIELDS: always additive; default to empty string / [] / {}. Minor version
  bump (e.g. 1.0→1.1) is optional but recommended when a batch of fields is
  added together, to aid schema negotiation.
- REMOVED FIELDS: requires version increment to "2.0"; must deprecate one release
  prior by emitting the field as empty and logging a deprecation notice.
- RENAMED FIELDS: not permitted; add a new field and emit both for one release.
- SEMANTIC CHANGES: requires version increment to "2.0".
- v1.0→v1.1 (J12.4): added decision_story, risk_story, confidence_story; enriched
  why_this_option with recommended-option advantages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

_LOGGER = logging.getLogger(__name__)

#: Schema version for this contract. See evolution rules in the module docstring.
#: v1.0→v1.1 (J12.4): story fields added; why_this_option enrichment introduced.
NARRATIVE_CONTRACT_VERSION = "1.1"


@dataclass
class ExecutiveNarrative:
    """Canonical executive communication contract v1.0 (see module docstring).

    All fields are always serialised by ``to_dict()`` — consumers rely on key
    presence without ``get()`` guards. Empty string, ``[]``, or ``{}`` signals
    "not computed", never ``None`` or a missing key.
    """

    # Schema version — first field so it leads in serialised output.
    version: str = NARRATIVE_CONTRACT_VERSION

    # ── Core decision narrative ──────────────────────────────────────────────
    decision: str = ""
    executive_summary: str = ""
    recommended_option: dict[str, Any] = field(default_factory=dict)
    why_this_option: str = ""

    # ── Option landscape ─────────────────────────────────────────────────────
    key_tradeoffs: list[str] = field(default_factory=list)
    option_rankings: list[str] = field(default_factory=list)
    strategic_options: list[dict[str, Any]] = field(default_factory=list)

    # ── Risk / opportunity / assumption register ─────────────────────────────
    key_risks: list[dict[str, Any]] = field(default_factory=list)
    key_opportunities: list[dict[str, Any]] = field(default_factory=list)
    critical_assumptions: list[dict[str, Any]] = field(default_factory=list)

    # ── Confidence and validation ─────────────────────────────────────────────
    executive_confidence: dict[str, Any] = field(default_factory=dict)
    validation_priorities: list[str] = field(default_factory=list)
    critical_unknowns: list[str] = field(default_factory=list)

    # ── Action portfolio (near / medium / long term) ─────────────────────────
    immediate_actions: list[dict[str, Any]] = field(default_factory=list)
    medium_term_actions: list[dict[str, Any]] = field(default_factory=list)
    long_term_actions: list[dict[str, Any]] = field(default_factory=list)

    # ── Supporting evidence ───────────────────────────────────────────────────
    supporting_evidence: list[dict[str, Any]] = field(default_factory=list)

    # ── Story fields (J12.4 — composed by ExecutiveNarrativeComposer) ─────────
    decision_story: str = ""
    risk_story: str = ""
    confidence_story: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain JSON-serialisable dict.

        All 21 fields are always present — consumers can rely on key presence
        without ``get()`` guards. Empty values (``""``, ``[]``, ``{}``) signal
        "not computed for this run"; ``None`` is never emitted.
        """
        return {
            "version": self.version,
            "decision": self.decision,
            "executive_summary": self.executive_summary,
            "recommended_option": self.recommended_option,
            "why_this_option": self.why_this_option,
            "key_tradeoffs": self.key_tradeoffs,
            "option_rankings": self.option_rankings,
            "strategic_options": self.strategic_options,
            "key_risks": self.key_risks,
            "key_opportunities": self.key_opportunities,
            "critical_assumptions": self.critical_assumptions,
            "executive_confidence": self.executive_confidence,
            "validation_priorities": self.validation_priorities,
            "critical_unknowns": self.critical_unknowns,
            "immediate_actions": self.immediate_actions,
            "medium_term_actions": self.medium_term_actions,
            "long_term_actions": self.long_term_actions,
            "supporting_evidence": self.supporting_evidence,
            "decision_story": self.decision_story,   # J12.4
            "risk_story": self.risk_story,           # J12.4
            "confidence_story": self.confidence_story,  # J12.4
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ExecutiveNarrative":
        """Restore from a dict (e.g. ``context.executive_narrative``).

        Forward-compatible: unknown keys are ignored; missing keys use defaults.
        Pre-v1.1 dicts (no ``version`` key, or version ``"1.0"``) deserialise
        cleanly — story fields default to empty string.
        """
        if not data:
            return cls()
        stored_version = data.get("version", NARRATIVE_CONTRACT_VERSION)
        # PH4.1-M1 — warn when a stored narrative is older than the current contract
        # so consumers know story fields may be empty or absent.
        if stored_version != NARRATIVE_CONTRACT_VERSION:
            _LOGGER.warning(
                "[ExecutiveNarrative] loaded narrative version %r differs from current "
                "contract version %r — story fields may be absent",
                stored_version,
                NARRATIVE_CONTRACT_VERSION,
            )
        return cls(
            version=stored_version,
            decision=data.get("decision", ""),
            executive_summary=data.get("executive_summary", ""),
            recommended_option=data.get("recommended_option") or {},
            why_this_option=data.get("why_this_option", ""),
            key_tradeoffs=list(data.get("key_tradeoffs") or []),
            option_rankings=list(data.get("option_rankings") or []),
            strategic_options=list(data.get("strategic_options") or []),
            key_risks=list(data.get("key_risks") or []),
            key_opportunities=list(data.get("key_opportunities") or []),
            critical_assumptions=list(data.get("critical_assumptions") or []),
            executive_confidence=data.get("executive_confidence") or {},
            validation_priorities=list(data.get("validation_priorities") or []),
            critical_unknowns=list(data.get("critical_unknowns") or []),
            immediate_actions=list(data.get("immediate_actions") or []),
            medium_term_actions=list(data.get("medium_term_actions") or []),
            long_term_actions=list(data.get("long_term_actions") or []),
            supporting_evidence=list(data.get("supporting_evidence") or []),
            decision_story=data.get("decision_story", ""),   # J12.4
            risk_story=data.get("risk_story", ""),           # J12.4
            confidence_story=data.get("confidence_story", ""),  # J12.4
        )
