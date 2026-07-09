"""Execution planning package — converts a StalenessPlan into an ExecutionPlan (J13.3).

Public API
----------
ExecutionPlan    — output of the planner; describes what to run and in what order
ExecutionPlanner — deterministic topological planner
"""

from .execution_plan import ExecutionPlan
from .execution_planner import ExecutionPlanner

__all__ = [
    "ExecutionPlan",
    "ExecutionPlanner",
]
