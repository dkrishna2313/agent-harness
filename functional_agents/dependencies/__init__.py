"""Agent dependency declarations package (J13.1).

Public API
----------
AgentDependency    — declarative dependency contract for one agent
DependencyRegistry — read-only registry with lookup methods
"""

from .agent_dependency import AgentDependency
from .registry import DependencyRegistry

__all__ = ["AgentDependency", "DependencyRegistry"]
