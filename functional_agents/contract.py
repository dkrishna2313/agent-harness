"""Agent contract validation (J5.5a follow-up).

Provides static and runtime checks that prove every functional agent conforms
to the standardised run(AgentContext) -> AgentResult contract.

Public API
----------
CONTRACT_VERSION        – semver string for the validation schema
FUNCTIONAL_AGENT_CLASSES – ordered list of all known concrete agents
validate_agent_class()  – static checks: inheritance + run() signature
validate_agent_result() – runtime check: AgentResult fields present
build_contract_validation() – assemble the trace block
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

LOGGER = logging.getLogger(__name__)

CONTRACT_VERSION = "1.0"

# Required fields on every AgentResult (J5.5a)
_REQUIRED_RESULT_FIELDS = ("status", "outputs", "metrics", "trace")


def _agent_classes() -> list[type]:
    """Return all concrete agent classes (lazy import avoids cycles)."""
    from .planner_agent                       import PlannerAgent
    from .evidence_agent                      import EvidenceAgent
    from .qa_agent                            import QAAgent
    from .report_agent                        import ReportAgent
    from .problem_framing_agent               import ProblemFramingAgent
    from .research_strategy_agent             import ResearchStrategyAgent
    from .hypothesis_agent                    import HypothesisAgent
    from .challenge_agent                     import ChallengeAgent
    from .assumption_agent                    import AssumptionAgent
    from .risk_agent                          import RiskAgent
    from .opportunity_agent                   import OpportunityAgent
    from .recommendation_agent               import RecommendationAgent
    from .scenario_agent                      import ScenarioAgent
    from .recommendation_improvement_agent   import RecommendationImprovementAgent
    from .multi_profile_agent                import MultiProfileAgent
    from .recommendation_synthesis_agent    import RecommendationSynthesisAgent
    from .strategic_option_agent            import StrategicOptionAgent
    return [
        ProblemFramingAgent, ResearchStrategyAgent,
        PlannerAgent, EvidenceAgent,
        HypothesisAgent, ChallengeAgent, AssumptionAgent, RiskAgent, OpportunityAgent, RecommendationAgent,
        MultiProfileAgent, ScenarioAgent, RecommendationImprovementAgent,
        RecommendationSynthesisAgent, StrategicOptionAgent, QAAgent, ReportAgent,
    ]


FUNCTIONAL_AGENT_CLASSES: list[type] = []  # populated on first call to _agent_classes()


# ---------------------------------------------------------------------------
# Static validation
# ---------------------------------------------------------------------------

def validate_agent_class(cls: type) -> dict[str, Any]:
    """Check class-level contract compliance for one agent.

    Returns a dict with keys:
      inherits_base_agent  – bool
      implements_run       – bool
      error                – str | None   (set only when a check raises)
    """
    from .base import FunctionalAgent
    from .context import AgentContext, AgentResult

    result: dict[str, Any] = {
        "inherits_base_agent": False,
        "implements_run": False,
        "error": None,
    }

    try:
        result["inherits_base_agent"] = issubclass(cls, FunctionalAgent)
    except TypeError as exc:
        result["error"] = f"issubclass check failed: {exc}"
        return result

    # Check run() exists and has the expected signature
    run_method = getattr(cls, "run", None)
    if run_method is None or not callable(run_method):
        result["implements_run"] = False
        return result

    try:
        sig = inspect.signature(run_method)
        params = list(sig.parameters.keys())
        # Expect: self, context  (positional or keyword)
        non_self = [p for p in params if p != "self"]
        has_context_param = len(non_self) >= 1 and non_self[0] == "context"

        # Return annotation: AgentResult or the string 'AgentResult'
        ret = sig.return_annotation
        has_result_return = (
            ret is AgentResult
            or ret == "AgentResult"
            or (isinstance(ret, str) and "AgentResult" in ret)
        )

        result["implements_run"] = has_context_param and has_result_return
    except (ValueError, TypeError) as exc:
        result["error"] = f"signature inspection failed: {exc}"

    return result


def validate_all_classes() -> dict[str, dict[str, Any]]:
    """Run validate_agent_class() for every known functional agent."""
    return {cls.__name__: validate_agent_class(cls) for cls in _agent_classes()}


# ---------------------------------------------------------------------------
# Runtime validation
# ---------------------------------------------------------------------------

def validate_agent_result(result: Any, agent_name: str) -> dict[str, Any]:
    """Verify a returned value conforms to the AgentResult contract.

    Returns:
      returns_agent_result – bool
      missing_fields       – list[str]  (empty when valid)
      error                – str | None
    """
    from .context import AgentResult

    check: dict[str, Any] = {
        "returns_agent_result": False,
        "missing_fields": [],
        "error": None,
    }

    if not isinstance(result, AgentResult):
        check["error"] = (
            f"{agent_name}.run() returned {type(result).__name__!r}, expected AgentResult"
        )
        return check

    missing = [f for f in _REQUIRED_RESULT_FIELDS if not hasattr(result, f)]
    check["missing_fields"] = missing
    check["returns_agent_result"] = len(missing) == 0

    if missing:
        check["error"] = f"AgentResult missing fields: {missing}"

    return check


# ---------------------------------------------------------------------------
# Trace block assembly
# ---------------------------------------------------------------------------

def build_contract_validation(
    class_checks: dict[str, dict[str, Any]],
    runtime_checks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the contract_validation trace block.

    class_checks   – output of validate_all_classes()
    runtime_checks – {agent_name: validate_agent_result()} per executed agent
    """
    agents: dict[str, Any] = {}

    all_names = set(class_checks) | set(runtime_checks)
    for name in sorted(all_names):
        cc = class_checks.get(name, {})
        rc = runtime_checks.get(name, {})
        entry: dict[str, Any] = {
            "inherits_base_agent": cc.get("inherits_base_agent", False),
            "implements_run":      cc.get("implements_run", False),
            "returns_agent_result": rc.get("returns_agent_result", False),
        }
        if cc.get("error"):
            entry["class_error"] = cc["error"]
        if rc.get("error"):
            entry["runtime_error"] = rc["error"]
        if rc.get("missing_fields"):
            entry["missing_fields"] = rc["missing_fields"]
        agents[name] = entry

    # An agent passes if it satisfies static checks AND, if it actually ran
    # (i.e. is present in runtime_checks), the runtime check too.
    # Agents registered statically but not executed in this run (e.g.
    # ProblemFramingAgent in question-driven runs) are not penalised.
    agent_contract_valid = all(
        v["inherits_base_agent"] and v["implements_run"] and
        (v["returns_agent_result"] if name in runtime_checks else True)
        for name, v in agents.items()
    ) if agents else False

    return {
        "contract_version": CONTRACT_VERSION,
        "agent_contract_valid": agent_contract_valid,
        "agents": agents,
    }
