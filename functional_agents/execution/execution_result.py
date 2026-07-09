"""ExecutionResult — outcome of one IncrementalExecutor.execute() call (J13.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from functional_agents.session.research_session import ResearchSession


class ExecutionStatus:
    COMPLETE = "COMPLETE"   # all required agents ran successfully
    FAILED   = "FAILED"     # an agent raised an exception; execution stopped
    EMPTY    = "EMPTY"      # execution plan had no required agents


@dataclass
class ExecutionResult:
    """Outcome of one IncrementalExecutor.execute() call.

    Fields
    ------
    status              COMPLETE / FAILED / EMPTY
    session             updated ResearchSession (always consistent, never corrupted)
    completed_agents    agents that ran and returned without error
    failed_agent        name of the agent that raised, or None
    failure_reason      str(exception) from the failed agent, or None
    execution_plan_id   plan_id of the ExecutionPlan that was consumed
    trace               optional extra diagnostics from the execution run
    """

    status: str
    session: ResearchSession
    completed_agents: list[str]
    failed_agent: str | None
    failure_reason: str | None
    execution_plan_id: str
    trace: dict[str, Any] = field(default_factory=dict)
