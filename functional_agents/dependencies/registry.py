"""DependencyRegistry — read-only, deterministic registry of agent dependency
declarations for all 22 production agents (J13.1).

Usage
-----
from functional_agents.dependencies import DependencyRegistry

dep = DependencyRegistry.get_dependency("EvidenceAgent")
consumers = DependencyRegistry.agents_consuming("research_object.evidence")
"""

from __future__ import annotations

from .agent_dependency import AgentDependency

# ---------------------------------------------------------------------------
# Canonical path shorthand constants — avoid string typos
# ---------------------------------------------------------------------------

_KS = "knowledge_store"
_ENG = "engagement"
_DM = "decision_model"
_DA = "decision_architecture"
_RS = "research_strategy"
_PL = "planner"
_RO_EV = "research_object.evidence"
_RO_HY = "research_object.hypotheses"
_RGA = "research_gap_analysis"
_SS = "strategic_synthesis"
_CR = "challenge_results"
_DM_AS = "decision_model.assumptions"
_DM_RC = "decision_model.recommendations"
_DM_RI = "decision_model.risks"
_DM_OP = "decision_model.opportunities"
_DM_SO = "decision_model.strategic_options"
_DM_DA = "decision_model.decision_analysis"
_EC = "executive_confidence"
_IP = "iteration_plan"
_MPA = "multi_profile_analysis"
_SCA = "scenario_analysis"
_QA = "qa"
_RI = "recommendation_improvement"
_RSY = "recommendation_synthesis"
_REP = "report"

# ---------------------------------------------------------------------------
# Downstream clusters — grouped for clarity in invalidates lists
# ---------------------------------------------------------------------------

# Everything after EvidenceAgent
_AFTER_EVIDENCE = [
    _RO_HY, _RGA, _SS, _CR,
    _DM_AS, _DM_RC, _DM_RI, _DM_OP, _DM_SO, _DM_DA,
    _EC, _IP, _MPA, _SCA, _QA, _RI, _RSY, _REP,
]
# Everything after HypothesisAgent
_AFTER_HYPOTHESES = [
    _RGA, _SS, _CR,
    _DM_AS, _DM_RC, _DM_RI, _DM_OP, _DM_SO, _DM_DA,
    _EC, _IP, _QA, _RI, _RSY, _REP,
]
# Everything after ResearchStrategyAgent (planner + evidence + below)
_AFTER_RESEARCH_STRATEGY = [_PL, _RO_EV] + _AFTER_EVIDENCE
# Everything after PlannerAgent (evidence + below — planner produces planner, so evidence becomes stale)
_AFTER_PLANNER = [_RO_EV] + _AFTER_EVIDENCE
# Everything after Assumptions
_AFTER_ASSUMPTIONS = [
    _DM_RC, _DM_RI, _DM_OP, _DM_SO, _DM_DA,
    _EC, _IP, _MPA, _SCA, _QA, _RI, _RSY, _REP,
]
# Everything after Recommendations
_AFTER_RECOMMENDATIONS = [
    _DM_RI, _DM_OP, _DM_SO, _DM_DA,
    _EC, _IP, _MPA, _SCA, _QA, _RI, _RSY, _REP,
]
# Everything after StrategicOptions
_AFTER_STRATEGIC_OPTIONS = [
    _DM_DA, _EC, _IP, _MPA, _SCA, _QA, _RI, _RSY, _REP,
]

# ---------------------------------------------------------------------------
# All 22 production agent declarations
# ---------------------------------------------------------------------------

_DECLARATIONS: list[AgentDependency] = [
    AgentDependency(
        agent_name="ProblemFramingAgent",
        consumes=[_ENG],
        produces=[_DM, _DA],
        invalidates=[
            _RS, _PL,
            _RO_EV, _RO_HY, _RGA, _SS, _CR,
            _DM_AS, _DM_RC, _DM_RI, _DM_OP, _DM_SO, _DM_DA,
            _EC, _IP, _MPA, _SCA, _QA, _RI, _RSY, _REP,
        ],
    ),
    AgentDependency(
        agent_name="ResearchStrategyAgent",
        consumes=[_ENG, _DM, _DA],
        produces=[_RS],
        invalidates=_AFTER_RESEARCH_STRATEGY,
    ),
    AgentDependency(
        agent_name="PlannerAgent",
        consumes=[_ENG, _DM, _RS],
        produces=[_PL],
        invalidates=_AFTER_PLANNER,
    ),
    AgentDependency(
        agent_name="EvidenceAgent",
        consumes=[_KS, _PL, _RS, _DM],
        produces=[_RO_EV],
        invalidates=_AFTER_EVIDENCE,
    ),
    AgentDependency(
        agent_name="HypothesisAgent",
        consumes=[_RO_EV, _PL, _DM],
        produces=[_RO_HY],
        invalidates=_AFTER_HYPOTHESES,
    ),
    AgentDependency(
        agent_name="ResearchGapAgent",
        consumes=[_RO_EV, _RO_HY, _PL],
        produces=[_RGA],
        invalidates=[_EC, _IP, _REP],
    ),
    AgentDependency(
        agent_name="StrategicSynthesisAgent",
        consumes=[_RO_EV, _RO_HY, _DM, _ENG],
        produces=[_SS],
        invalidates=[
            _DM_AS, _DM_RC, _DM_RI, _DM_OP, _DM_SO, _DM_DA,
            _EC, _IP, _REP,
        ],
    ),
    AgentDependency(
        agent_name="ChallengeAgent",
        consumes=[_RO_HY, _RO_EV],
        produces=[_CR],
        invalidates=[
            _DM_AS, _DM_RC, _DM_RI, _DM_OP, _DM_SO, _DM_DA,
            _EC, _IP, _REP,
        ],
    ),
    AgentDependency(
        agent_name="AssumptionAgent",
        consumes=[_CR, _RO_HY, _DM, _SS],
        produces=[_DM_AS],
        invalidates=_AFTER_ASSUMPTIONS,
    ),
    AgentDependency(
        agent_name="RecommendationAgent",
        consumes=[_DM_AS, _RO_EV, _RO_HY, _SS, _DM],
        produces=[_DM_RC],
        invalidates=_AFTER_RECOMMENDATIONS,
    ),
    AgentDependency(
        agent_name="RiskAgent",
        consumes=[_DM_AS, _DM_RC, _RO_EV, _DM],
        produces=[_DM_RI],
        invalidates=[_DM_SO, _DM_DA, _EC, _IP, _REP],
    ),
    AgentDependency(
        agent_name="OpportunityAgent",
        consumes=[_DM_AS, _DM_RC, _RO_EV, _DM],
        produces=[_DM_OP],
        invalidates=[_DM_SO, _DM_DA, _EC, _IP, _REP],
    ),
    AgentDependency(
        agent_name="StrategicOptionAgent",
        consumes=[_DM_RC, _DM_RI, _DM_OP, _DM_AS, _DM],
        produces=[_DM_SO],
        invalidates=_AFTER_STRATEGIC_OPTIONS,
    ),
    AgentDependency(
        agent_name="DecisionAnalysisAgent",
        consumes=[_DM_SO, _DM_AS, _DM_RC, _DM_RI, _DM],
        produces=[_DM_DA],
        invalidates=[_EC, _IP, _REP],
    ),
    AgentDependency(
        agent_name="ExecutiveConfidenceAgent",
        consumes=[_DM_DA, _DM_AS, _RGA, _DM],
        produces=[_EC],
        invalidates=[_IP, _REP],
    ),
    AgentDependency(
        agent_name="IterationPlanAgent",
        consumes=[_EC, _DM_AS, _DM_RI, _DM_RC, _DM_SO, _RGA],
        produces=[_IP],
        invalidates=[_REP],
    ),
    AgentDependency(
        agent_name="MultiProfileAgent",
        consumes=[_DM_RC, _DM_SO, _RO_EV],
        produces=[_MPA],
        invalidates=[_SCA, _QA, _RI, _RSY, _REP],
    ),
    AgentDependency(
        agent_name="ScenarioAgent",
        consumes=[_DM_RC, _MPA, _DM],
        produces=[_SCA],
        invalidates=[_QA, _RI, _RSY, _REP],
    ),
    AgentDependency(
        agent_name="QAAgent",
        consumes=[_DM_RC, _RO_EV, _RO_HY, _SCA, _MPA],
        produces=[_QA],
        invalidates=[_RI, _RSY, _REP],
    ),
    AgentDependency(
        agent_name="RecommendationImprovementAgent",
        consumes=[_DM_RC, _QA, _RO_EV],
        produces=[_RI],
        invalidates=[_RSY, _REP],
    ),
    AgentDependency(
        agent_name="RecommendationSynthesisAgent",
        consumes=[_RI, _DM_RC, _MPA, _SCA],
        produces=[_RSY],
        invalidates=[_REP],
    ),
    AgentDependency(
        agent_name="ReportAgent",
        consumes=[_RSY, _DM, _RO_EV, _IP, _EC, _QA],
        produces=[_REP],
        invalidates=[],
    ),
]


# ---------------------------------------------------------------------------
# DependencyRegistry
# ---------------------------------------------------------------------------

class DependencyRegistry:
    """Read-only, deterministic registry of all 22 production agent dependencies.

    All methods are classmethods — there is no instance state.
    """

    _registry: dict[str, AgentDependency] = {
        dep.agent_name: dep for dep in _DECLARATIONS
    }

    @classmethod
    def get_dependency(cls, agent_name: str) -> AgentDependency:
        """Return the AgentDependency for *agent_name*.

        Raises KeyError if the agent has no registered declaration.
        """
        try:
            return cls._registry[agent_name]
        except KeyError:
            raise KeyError(
                f"No dependency declaration for agent {agent_name!r}. "
                f"Registered agents: {sorted(cls._registry)}"
            ) from None

    @classmethod
    def list_dependencies(cls) -> list[AgentDependency]:
        """Return all declarations in pipeline order (declaration order)."""
        return list(_DECLARATIONS)

    @classmethod
    def agents_consuming(cls, path: str) -> list[str]:
        """Return names of agents that list *path* in their `consumes`."""
        return [
            dep.agent_name
            for dep in _DECLARATIONS
            if path in dep.consumes
        ]

    @classmethod
    def agents_producing(cls, path: str) -> list[str]:
        """Return names of agents that list *path* in their `produces`."""
        return [
            dep.agent_name
            for dep in _DECLARATIONS
            if path in dep.produces
        ]

    @classmethod
    def agents_invalidated_by(cls, path: str) -> list[str]:
        """Return names of agents whose `invalidates` list contains *path*."""
        return [
            dep.agent_name
            for dep in _DECLARATIONS
            if path in dep.invalidates
        ]
