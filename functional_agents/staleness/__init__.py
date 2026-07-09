"""Staleness analysis package — dependency reasoning engine (J13.2).

Public API
----------
PathKind           — PERSISTED / EXECUTION_ONLY / EXTERNAL classification constants
classify_path      — classify a single path string
expand_path        — expand container paths to sub-paths
StalenessPlan      — output of a staleness analysis
DependencyReasoner — deterministic BFS staleness engine
"""

from .dependency_reasoner import DependencyReasoner
from .path_kind import (
    CONTAINER_EXPANSIONS,
    PATH_CLASSIFICATION,
    PathKind,
    classify_path,
    expand_path,
    is_container_path,
)
from .staleness_plan import StalenessPlan

__all__ = [
    "PathKind",
    "PATH_CLASSIFICATION",
    "CONTAINER_EXPANSIONS",
    "classify_path",
    "expand_path",
    "is_container_path",
    "StalenessPlan",
    "DependencyReasoner",
]
