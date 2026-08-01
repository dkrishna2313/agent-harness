"""StrategyOutputView — shared downstream adapter for Strategy Layer results (PH12.2f).

A single normalized view model that all consumers (report, editorial brief,
editorial manuscript) use to avoid duplicated extraction logic.

Built from a StrategyNarrative + strategic_options for title resolution.
Never invokes StrategyCoordinator. Never generates prose. Fully deterministic.

Source-of-truth hierarchy:
  StrategicPosition / StrategySelection → winning strategy and mapped option
  StrategyNarrative → all presentation-ready fields
  StrategyOutputView → resolved titles + structured choice cascade + execution implications
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .strategy_narrative import StrategyNarrative


class StrategyChoiceCascadeItem(BaseModel):
    """One dimension in the strategic choice cascade."""

    dimension_id: str = ""
    dimension_title: str = ""
    choice_id: str = ""
    choice_title: str = ""
    choice_description: str = ""
    execution_complexity: str = ""

    model_config = {"frozen": True}


class StrategyOutputView(BaseModel):
    """Normalized downstream view for all strategy consumers (PH12.2f).

    Constructed by build_strategy_output_view().
    Immutable (frozen=True). JSON-serializable.
    No LLM calls. No file I/O. No StrategyCoordinator invocation.
    """

    # Identity
    framework: str = ""
    trace_id: str = ""
    strategic_position_id: str = ""
    winning_theory_id: str = ""

    # Strategic position display
    strategic_position: str = ""    # winning_position (executive statement)
    strategic_mechanism: str = ""   # winning_mechanism

    # Option mapping (titles resolved from strategic_options list)
    mapped_option_id: str = ""
    mapped_option_title: str = ""
    preferred_option_id: str = ""
    preferred_option_title: str = ""
    mapping_status: str = ""
    mapping_score: float | None = None
    mapping_margin: float | None = None
    mapping_confidence: str = ""
    mapping_rationale: str = ""

    # Alignment
    alignment_status: str = ""
    alignment_narrative: str = ""

    # Choice cascade (dimension-ordered, with human-readable labels)
    choice_cascade: list[StrategyChoiceCascadeItem] = Field(default_factory=list)

    # Theory content
    assumptions: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    execution_implications: list[str] = Field(default_factory=list)

    # Scores and confidence
    winner_score: float = 0.0
    runner_up_score: float | None = None
    score_margin: float | None = None
    overall_confidence: str = ""
    saturation_detected: bool = False

    # Provenance
    strategy_config_fingerprint: str = ""

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _resolve_option_title(option_id: str, strategic_options: list[dict]) -> str:
    """Look up option title from strategic_options by option_id."""
    if not option_id:
        return ""
    for opt in (strategic_options or []):
        if isinstance(opt, dict) and opt.get("option_id") == option_id:
            return opt.get("title") or option_id
    return option_id


def _build_cascade_from_narrative(narrative: "StrategyNarrative") -> list[StrategyChoiceCascadeItem]:
    """Build a structured choice cascade from StrategyNarrative.choice_cascade."""
    items: list[StrategyChoiceCascadeItem] = []
    for entry in (narrative.choice_cascade or []):
        if not isinstance(entry, dict):
            continue
        items.append(StrategyChoiceCascadeItem(
            dimension_id=entry.get("dimension_id", ""),
            dimension_title=entry.get("dimension_title", ""),
            choice_id=entry.get("choice_id", ""),
            choice_title=entry.get("choice_title", ""),
            choice_description=entry.get("choice_description", ""),
            execution_complexity=entry.get("execution_complexity", ""),
        ))
    return items


_EXECUTION_DIMENSIONS = frozenset({
    "must_have_capabilities",
    "management_systems",
    "how_to_win",
})


def _build_execution_implications(cascade: list[StrategyChoiceCascadeItem]) -> list[str]:
    """Extract execution-relevant implications from the choice cascade."""
    implications: list[str] = []
    for item in cascade:
        if item.dimension_id not in _EXECUTION_DIMENSIONS or not item.choice_title:
            continue
        label = item.dimension_title or item.dimension_id.replace("_", " ").title()
        implications.append(f"{label}: {item.choice_title}")
    return implications


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_strategy_output_view(
    narrative: "StrategyNarrative | None",
    strategic_options: list[dict] | None = None,
    *,
    strategy_config_fingerprint: str = "",
) -> "StrategyOutputView | None":
    """Build a StrategyOutputView from a StrategyNarrative.

    Returns None when narrative is None (strategy disabled or absent).
    All title resolution uses strategic_options (never re-invokes StrategyCoordinator).

    Parameters
    ----------
    narrative:
        StrategyNarrative from build_strategy_narrative(). May be None.
    strategic_options:
        List of option dicts from AgentContext/StrategicPosition for title lookup.
    strategy_config_fingerprint:
        Optional fingerprint from the resolved strategy configuration.
    """
    if narrative is None:
        return None

    opts = list(strategic_options or [])
    mapped_title = _resolve_option_title(narrative.mapped_option_id or "", opts)
    preferred_title = _resolve_option_title(narrative.preferred_option_id or "", opts)

    cascade = _build_cascade_from_narrative(narrative)
    exec_implications = _build_execution_implications(cascade)

    return StrategyOutputView(
        framework=narrative.framework,
        trace_id=narrative.trace_id,
        strategic_position_id=narrative.strategic_position_id,
        winning_theory_id=narrative.winner_theory_id,
        strategic_position=narrative.winning_position,
        strategic_mechanism=narrative.winning_mechanism,
        mapped_option_id=narrative.mapped_option_id or "",
        mapped_option_title=mapped_title,
        preferred_option_id=narrative.preferred_option_id or "",
        preferred_option_title=preferred_title,
        mapping_status=narrative.mapping_status,
        mapping_score=narrative.mapping_score,
        mapping_margin=narrative.mapping_margin,
        mapping_confidence=narrative.mapping_confidence,
        mapping_rationale=narrative.mapping_rationale,
        alignment_status=narrative.alignment_status,
        alignment_narrative=narrative.alignment_narrative,
        choice_cascade=cascade,
        assumptions=list(narrative.assumptions),
        failure_modes=list(narrative.failure_modes),
        execution_implications=exec_implications,
        winner_score=narrative.winner_score,
        runner_up_score=narrative.runner_up_score,
        score_margin=narrative.score_margin,
        overall_confidence=narrative.overall_confidence,
        saturation_detected=narrative.saturation_detected,
        strategy_config_fingerprint=strategy_config_fingerprint,
    )
