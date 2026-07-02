"""Shared context object passed through functional agents (J5.0b).

AgentContext is the single source of truth for all workflow state.
It is created once by the Orchestrator, passed through every agent in
sequence, and written to the Research Object and trace at the end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .reasoning_target import ReasoningTarget


class ContextValidationError(ValueError):
    """Raised when AgentContext fails pre-flight validation."""


# ---------------------------------------------------------------------------
# Workflow constants (J5.5 / J5.5a / J6.1) – defined here to avoid circular imports
# ---------------------------------------------------------------------------

class WorkflowState:
    PROBLEM_FRAMING   = "PROBLEM_FRAMING"
    RESEARCH_STRATEGY = "RESEARCH_STRATEGY"
    PLANNING    = "PLANNING"
    EVIDENCE    = "EVIDENCE"
    HYPOTHESIS  = "HYPOTHESIS"
    STRATEGIC_SYNTHESIS = "STRATEGIC_SYNTHESIS"   # J10.7
    CHALLENGE       = "CHALLENGE"
    ASSUMPTION      = "ASSUMPTION"          # J7.1
    RISK            = "RISK"               # J7.3
    OPPORTUNITY     = "OPPORTUNITY"        # J7.4
    RECOMMENDATION              = "RECOMMENDATION"
    MULTI_PROFILE                = "MULTI_PROFILE"
    SCENARIO                     = "SCENARIO"
    RECOMMENDATION_IMPROVEMENT   = "RECOMMENDATION_IMPROVEMENT"
    RECOMMENDATION_SYNTHESIS     = "RECOMMENDATION_SYNTHESIS"
    STRATEGIC_OPTIONS            = "STRATEGIC_OPTIONS"
    DECISION_ANALYSIS            = "DECISION_ANALYSIS"    # J7.6
    EXECUTIVE_CONFIDENCE         = "EXECUTIVE_CONFIDENCE" # J7.7
    QA                           = "QA"
    REPORT      = "REPORT"
    COMPLETE    = "COMPLETE"
    ERROR       = "ERROR"


class NextAction:
    CONTINUE          = "CONTINUE"
    REQUEST_EVIDENCE  = "REQUEST_EVIDENCE"
    REQUEST_REPLAN    = "REQUEST_REPLAN"
    REQUEST_QA        = "REQUEST_QA"
    COMPLETE          = "COMPLETE"
    ERROR             = "ERROR"


@dataclass
class AgentContext:
    """Mutable shared state threaded through each functional agent (J5.0b.1).

    Lifecycle
    ---------
    1. Orchestrator creates one AgentContext and calls validate().
    2. Each agent receives it, mutates it, and returns it.
    3. ReportAgent writes agent_history into the RO and trace.

    Fields
    ------
    question          : the research question (may be empty if goal is provided;
                        ProblemFramingAgent populates it from the decision model)
    goal              : high-level business goal for goal-driven runs (J6.1)
    decision_model    : structured output of ProblemFramingAgent (J6.1)
    profiles          : ordered list of profile names (first = execution profile)
    execution_profile : explicit copy of profiles[0] for clarity
    research_object   : the durable Research Object dict (updated in-place)
    plan              : PlannerAgent output
    evidence_notes    : EvidenceAgent detailed notes
    qa_notes          : QAAgent detailed notes
    agent_history     : structured completion record from every agent (J5.0b.3)
    artifacts         : named output paths and blobs (report_path, trace_path, …)
    trace             : scratch space for inter-agent data (not persisted directly)
    """

    # Core research intent
    question: str = ""
    profiles: list[str] = field(default_factory=list)
    execution_profile: str = ""
    run_id: str = ""

    # Goal-driven input (J6.1) — alternative entry point to question-driven
    goal: str = ""
    # Strategic Engagement input (J9.1) — structured consulting brief that, when
    # present, drives goal-driven framing from a richer context than a one-liner.
    # Holds the validated EngagementSpec as a dict; empty for goal/question runs.
    engagement: dict[str, Any] = field(default_factory=dict)
    decision_model: dict[str, Any] = field(default_factory=dict)
    # Decision Architecture (J9.2) — executive decision framing derived by
    # ProblemFramingAgent, sitting between the engagement and the research program.
    decision_architecture: dict[str, Any] = field(default_factory=dict)
    # Research strategy output of ResearchStrategyAgent (J6.2)
    research_strategy: dict[str, Any] = field(default_factory=dict)

    # Shared durable state
    research_object: dict[str, Any] = field(default_factory=dict)

    # Per-agent detailed notes
    plan: dict[str, Any] = field(default_factory=dict)
    # J10.4 — one plan per Decision Domain (engagement mode). plan (above) remains
    # the PRIMARY plan that executes downstream; domain_plans is organizational
    # only and does not change execution. Single entry in goal/question mode.
    domain_plans: list[dict[str, Any]] = field(default_factory=list)
    evidence_notes: list[dict[str, Any]] = field(default_factory=list)
    # J10.5 — one evidence collection per Decision Domain (engagement mode).
    # evidence_notes (above) remains the PRIMARY collection that flows to
    # HypothesisAgent; domain_evidence is organizational only. Single entry in
    # goal/question mode. Each entry: {decision_domain_id, decision_domain_title,
    # evidence, mapping, coverage}.
    domain_evidence: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    # J10.6 — one hypothesis set per Decision Domain (engagement mode). hypotheses
    # (above) remains the PRIMARY set that flows to ChallengeAgent; domain_hypotheses
    # is organizational only. Each entry: {decision_domain_id, decision_domain_title,
    # hypotheses, synthesis_note, diagnostics}.
    domain_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    # J10.7 — cross-domain strategic synthesis (executive reasoning; NO
    # recommendations). Integrates the per-domain reasoning streams into one
    # executive perspective. Additive; does not change the primary path.
    strategic_synthesis: dict[str, Any] = field(default_factory=dict)
    hypothesis_challenges: list[dict[str, Any]] = field(default_factory=list)
    surviving_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    assumptions: list[dict[str, Any]] = field(default_factory=list)        # J7.1
    risks: list[dict[str, Any]] = field(default_factory=list)             # J7.3
    opportunities: list[dict[str, Any]] = field(default_factory=list)   # J7.4
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    recommendation_portfolio: dict[str, Any] = field(default_factory=dict)
    multi_profile_analysis: dict[str, Any] = field(default_factory=dict)
    scenarios: list[dict[str, Any]] = field(default_factory=list)
    scenario_analysis: dict[str, Any] = field(default_factory=dict)
    recommendation_improvement: dict[str, Any] = field(default_factory=dict)
    # J6.8c — cross-profile synthesis outputs
    synthesis_validation: dict[str, Any] = field(default_factory=dict)
    recommendation_profile_balance: dict[str, Any] = field(default_factory=dict)
    synthesis_tradeoffs: list[dict[str, Any]] = field(default_factory=list)
    # J7.1 — strategic option generation
    strategic_options: list[dict[str, Any]] = field(default_factory=list)
    strategic_option_comparison: dict[str, Any] = field(default_factory=dict)
    option_scenario_robustness: dict[str, Any] = field(default_factory=dict)
    preferred_option: dict[str, Any] = field(default_factory=dict)
    strategic_option_portfolio: dict[str, Any] = field(default_factory=dict)
    # J6.5a – validated (post-suppression) contradictions and suppression metrics
    validated_contradictions: list[dict[str, Any]] = field(default_factory=list)
    contradiction_metrics: dict[str, Any] = field(default_factory=dict)
    qa_notes: list[dict[str, Any]] = field(default_factory=list)
    qa: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    # J7.6 — decision analysis
    decision_analysis: dict[str, Any] = field(default_factory=dict)
    # J7.7 — executive confidence
    executive_confidence: dict[str, Any] = field(default_factory=dict)

    # Orchestrator state (J5.5)
    workflow_state: str = ""
    iteration_count: int = 0
    workflow_path: list[str] = field(default_factory=list)

    # Unified agent history (J5.0b.3)
    agent_history: list[dict[str, Any]] = field(default_factory=list)

    # Named output artifacts (J5.0b)
    artifacts: dict[str, Any] = field(default_factory=dict)

    # Inter-agent scratch space (not written to trace directly)
    trace: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise ContextValidationError if required fields are missing (J5.0b.7).

        Either question or goal must be provided — goal-driven runs leave
        question empty until ProblemFramingAgent populates it.
        """
        errors: list[str] = []
        has_question = bool(self.question and self.question.strip())
        has_goal = bool(self.goal and self.goal.strip())
        if not has_question and not has_goal:
            errors.append("either 'question' or 'goal' is required and must not be empty")
        if not self.profiles:
            errors.append("'profiles' must contain at least one profile name")
        if not self.execution_profile:
            errors.append("'execution_profile' must be set")
        if not isinstance(self.research_object, dict) or not self.research_object:
            errors.append("'research_object' must be a non-empty dict")
        if errors:
            raise ContextValidationError(
                f"AgentContext validation failed ({len(errors)} error(s)):\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def append_history(self, entry: dict[str, Any]) -> None:
        """Append a structured agent completion entry to agent_history."""
        self.agent_history.append(entry)

    def to_functional_trace(self) -> dict[str, Any]:
        """Return the functional_agents block for the execution trace (J5.0b.5)."""
        return {
            "enabled": True,
            "agents_run": [h["agent"] for h in self.agent_history],
            "agent_history": self.agent_history,
            "profiles": self.profiles,
            "execution_profile": self.execution_profile,
        }

    # ------------------------------------------------------------------
    # Reasoning targets (J10.1) — compatibility seam for Decision Domains
    # ------------------------------------------------------------------

    def get_reasoning_targets(self) -> list["ReasoningTarget"]:
        """Return the reasoning target(s) downstream agents should reason over.

        J10.3 — the producer now depends on run mode:

        * **Strategic Engagement mode** (``self.engagement`` populated and a
          Decision Architecture with decision streams present): return ONE target
          per Decision Domain (decision stream), ``kind='decision_domain'``.
        * **Research/goal/question mode**: return exactly ONE
          ``kind='research_question'`` target derived from ``context.question``
          (unchanged from J10.1).

        This changes the *representation*, not the reasoning: the Planner plans
        only the primary target (``[0]``), and the primary target's ``question``
        is pinned to ``context.question`` so planning output is byte-identical to
        prior behaviour. Returns an empty list only when no question is set yet
        (e.g. a goal-driven run before ProblemFramingAgent populates it).
        """
        from .reasoning_target import (
            ReasoningTarget,
            KIND_RESEARCH_QUESTION,
            KIND_DECISION_DOMAIN,
        )

        question = (self.question or "").strip()
        streams = (self.decision_architecture or {}).get("decision_streams") or []

        # Strategic Engagement mode → one target per Decision Domain.
        if self.engagement and streams:
            targets: list[ReasoningTarget] = []
            for i, s in enumerate(streams):
                title = (s.get("title") or "").strip() or f"Decision Domain {i + 1}"
                rqs = [str(q).strip() for q in (s.get("research_questions") or []) if str(q).strip()]
                # Primary domain is pinned to context.question so the Planner (which
                # plans the primary target) produces identical output to J10.2. Other
                # domains use their own primary analytical question from the stream.
                if i == 0 and question:
                    dq = question
                elif rqs:
                    dq = rqs[0]
                else:
                    dq = (s.get("executive_objective") or title).strip()
                domain_id = f"domain-{i + 1}"
                targets.append(
                    ReasoningTarget(
                        id=domain_id,
                        title=title,
                        kind=KIND_DECISION_DOMAIN,
                        question=dq,
                        decision_domain_id=domain_id,
                        decision_domain_title=title,
                        evidence_requirements=[],
                    )
                )
            return targets

        # Research / goal / question mode → single research-question target.
        if not question:
            return []
        return [
            ReasoningTarget(
                id="primary",
                title=question,
                kind=KIND_RESEARCH_QUESTION,
                question=question,
                decision_domain_id=None,
                decision_domain_title=None,
                evidence_requirements=[],
            )
        ]

    @property
    def reasoning_targets(self) -> list["ReasoningTarget"]:
        """Convenience property mirroring :meth:`get_reasoning_targets` (J10.1)."""
        return self.get_reasoning_targets()


# ---------------------------------------------------------------------------
# AgentResult (J5.5a) – standardised return value for every agent's run()
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Standardised outcome of a single agent's run() call (J5.5a).

    Fields
    ------
    status      : "success" | "warning" | "error"
    next_action : NextAction constant (read by orchestrator for routing)
    summary     : one-line human-readable outcome
    context     : the updated AgentContext (orchestrator passes it to next agent)
    outputs     : agent-specific structured output data
    metrics     : execution metrics — always includes duration_seconds
    trace       : per-agent trace block: {agent, run_id, duration_seconds, status}
    """

    status: str
    next_action: str
    summary: str
    context: "AgentContext"
    outputs: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)
