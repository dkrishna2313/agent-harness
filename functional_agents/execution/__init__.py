"""Incremental execution package — runs only the required subset of the pipeline (J13.4).

Public API
----------
ExecutionStatus    — COMPLETE / FAILED / EMPTY constants
ExecutionResult    — outcome of one IncrementalExecutor.execute() call
IncrementalExecutor — executes an ExecutionPlan against a ResearchSession
"""

from .execution_result import ExecutionResult, ExecutionStatus
from .incremental_executor import IncrementalExecutor

__all__ = [
    "ExecutionStatus",
    "ExecutionResult",
    "IncrementalExecutor",
]
