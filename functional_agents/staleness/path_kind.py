"""PathKind — classification of dependency paths for staleness analysis (J13.2).

Every path in the dependency graph belongs to exactly one class:

PERSISTED       stored in ResearchState; survives pipeline runs; can be restored
                from a session file without re-running the producing agent.

EXECUTION_ONLY  lives only in AgentContext during an active pipeline run; not
                stored in ResearchState; must be re-produced before it can be used.

EXTERNAL        provided by the environment (knowledge store, user input, web);
                never computed by agents; can trigger staleness but cannot be
                restored by the pipeline itself.
"""

from __future__ import annotations


class PathKind:
    """Path classification constants."""

    PERSISTED = "PERSISTED"
    EXECUTION_ONLY = "EXECUTION_ONLY"
    EXTERNAL = "EXTERNAL"


# Canonical classification of every known dependency path.
PATH_CLASSIFICATION: dict[str, str] = {
    # -----------------------------------------------------------------------
    # EXTERNAL — provided by the environment; never produced by pipeline agents
    # -----------------------------------------------------------------------
    "knowledge_store":                  PathKind.EXTERNAL,

    # -----------------------------------------------------------------------
    # PERSISTED — stored in ResearchState; survive across pipeline runs
    # -----------------------------------------------------------------------
    # Root fields
    "engagement":                       PathKind.PERSISTED,
    "research_object":                  PathKind.PERSISTED,
    "decision_model":                   PathKind.PERSISTED,
    "research_gap_analysis":            PathKind.PERSISTED,
    "executive_confidence":             PathKind.PERSISTED,
    "iteration_plan":                   PathKind.PERSISTED,
    # research_object sub-fields
    "research_object.evidence":         PathKind.PERSISTED,
    "research_object.hypotheses":       PathKind.PERSISTED,
    # decision_model sub-fields
    "decision_model.assumptions":       PathKind.PERSISTED,
    "decision_model.recommendations":   PathKind.PERSISTED,
    "decision_model.risks":             PathKind.PERSISTED,
    "decision_model.opportunities":     PathKind.PERSISTED,
    "decision_model.strategic_options": PathKind.PERSISTED,
    "decision_model.decision_analysis": PathKind.PERSISTED,

    # -----------------------------------------------------------------------
    # EXECUTION_ONLY — AgentContext intermediate artifacts; not in ResearchState
    # -----------------------------------------------------------------------
    "decision_architecture":            PathKind.EXECUTION_ONLY,
    "research_strategy":                PathKind.EXECUTION_ONLY,
    "planner":                          PathKind.EXECUTION_ONLY,
    "strategic_synthesis":              PathKind.EXECUTION_ONLY,
    "challenge_results":                PathKind.EXECUTION_ONLY,
    "multi_profile_analysis":           PathKind.EXECUTION_ONLY,
    "scenario_analysis":                PathKind.EXECUTION_ONLY,
    "qa":                               PathKind.EXECUTION_ONLY,
    "recommendation_improvement":       PathKind.EXECUTION_ONLY,
    "recommendation_synthesis":         PathKind.EXECUTION_ONLY,
    "report":                           PathKind.EXECUTION_ONLY,
}

# Container paths that expand to their constituent sub-paths during analysis.
# A StateChange with affected_paths=["research_state"] expands to all PERSISTED
# sub-paths; a change to "decision_model" expands to its six sub-fields.
CONTAINER_EXPANSIONS: dict[str, list[str]] = {
    "research_state": [
        "engagement",
        "research_object",
        "research_object.evidence",
        "research_object.hypotheses",
        "decision_model",
        "decision_model.assumptions",
        "decision_model.recommendations",
        "decision_model.risks",
        "decision_model.opportunities",
        "decision_model.strategic_options",
        "decision_model.decision_analysis",
        "research_gap_analysis",
        "executive_confidence",
        "iteration_plan",
    ],
    "research_object": [
        "research_object.evidence",
        "research_object.hypotheses",
    ],
    "decision_model": [
        "decision_model.assumptions",
        "decision_model.recommendations",
        "decision_model.risks",
        "decision_model.opportunities",
        "decision_model.strategic_options",
        "decision_model.decision_analysis",
    ],
}


def classify_path(path: str) -> str:
    """Return the PathKind constant for *path*.

    Unknown paths are conservatively classified as EXECUTION_ONLY so they are
    never falsely assumed to be persisted.
    """
    return PATH_CLASSIFICATION.get(path, PathKind.EXECUTION_ONLY)


def is_container_path(path: str) -> bool:
    """Return True if *path* expands to constituent sub-paths."""
    return path in CONTAINER_EXPANSIONS


def expand_path(path: str) -> list[str]:
    """Return the sub-paths for a container, or ``[path]`` for a leaf."""
    return list(CONTAINER_EXPANSIONS.get(path, [path]))
