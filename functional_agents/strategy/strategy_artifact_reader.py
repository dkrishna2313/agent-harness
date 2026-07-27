"""StrategyArtifactReader — read-only access to persisted Strategy artifacts (PH11.3).

Provides canonical loading and inspection of strategy.trace.json and
artifact.index.json files produced by StrategyCoordinator.

All methods are read-only: no artifacts are modified, no reasoning is re-run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .strategic_position import TheoryOfWinning
from .strategy_trace import StrategyTrace
from .theory_evaluation import TheoryEvaluation


class StrategyArtifactReader:
    """Read-only access to persisted Strategy Layer artifacts.

    Usage::

        reader = StrategyArtifactReader()
        trace = reader.load_trace(Path("outputs/strategy.trace.json"))
        summary = reader.summarize(trace)
        theory = reader.find_theory(trace, "TH-SCS-0")
        evaluation = reader.find_evaluation(trace, "TH-SCS-0")
        index = reader.load_index(Path("outputs/artifact.index.json"))
    """

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_trace(self, path: Path) -> StrategyTrace:
        """Load and validate a StrategyTrace from a JSON file.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file contains invalid JSON or cannot be validated as a
            StrategyTrace.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Strategy trace not found: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in strategy trace {path}: {exc}") from exc
        try:
            return StrategyTrace.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(f"Invalid StrategyTrace in {path}: {exc}") from exc

    def load_index(self, path: Path) -> dict[str, Any]:
        """Load an artifact index from a JSON file.

        Returns the raw index dict as persisted. Does not invent missing entries.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file contains invalid JSON.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact index not found: {path}")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in artifact index {path}: {exc}") from exc

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summarize(self, trace: StrategyTrace) -> dict[str, Any]:
        """Return a compact read-only summary of a StrategyTrace.

        All fields reference data already in the trace — nothing is computed
        beyond simple lookups.
        """
        sel = trace.selection
        winner = next(
            (t for t in trace.theories if t.theory_id == sel.winner_theory_id), None
        )
        framework = trace.metadata.get("framework", "") or trace.plan.framework
        return {
            "trace_id": trace.trace_id,
            "created_at": trace.created_at,
            "framework": framework,
            "research_id": trace.metadata.get("research_id", ""),
            "plan_id": trace.plan.plan_id,
            "choice_set_count": len(trace.choice_sets),
            "theory_count": len(trace.theories),
            "evaluation_count": len(trace.evaluations),
            "winner_theory_id": sel.winner_theory_id,
            "winner_option_id": winner.recommended_option_id if winner else "",
            "runner_up_theory_id": sel.runner_up_theory_id,
            "winner_score": sel.winner_score,
            "runner_up_score": sel.runner_up_score,
            "score_margin": sel.score_margin,
            "tie_breaker_used": sel.tie_breaker_used,
            "strategic_position_id": trace.strategic_position.position_id,
        }

    # ------------------------------------------------------------------
    # Lookup by theory_id
    # ------------------------------------------------------------------

    def find_theory(self, trace: StrategyTrace, theory_id: str) -> TheoryOfWinning:
        """Return the TheoryOfWinning matching *theory_id* in *trace*.

        Raises
        ------
        ValueError
            If *theory_id* is blank, not found, or matches more than one theory
            (duplicate — should not occur in a valid StrategyTrace).
        """
        if not theory_id or not theory_id.strip():
            raise ValueError("theory_id must be a non-empty, non-whitespace string.")
        matches = [t for t in trace.theories if t.theory_id == theory_id]
        if not matches:
            available = sorted(t.theory_id for t in trace.theories)
            raise ValueError(
                f"theory_id={theory_id!r} not found in trace {trace.trace_id!r}. "
                f"Available: {available}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous: theory_id={theory_id!r} matched {len(matches)} theories "
                f"in trace {trace.trace_id!r}."
            )
        return matches[0]

    def find_evaluation(self, trace: StrategyTrace, theory_id: str) -> TheoryEvaluation:
        """Return the TheoryEvaluation matching *theory_id* in *trace*.

        Raises
        ------
        ValueError
            If *theory_id* is blank, not found, or matches more than one evaluation.
        """
        if not theory_id or not theory_id.strip():
            raise ValueError("theory_id must be a non-empty, non-whitespace string.")
        matches = [ev for ev in trace.evaluations if ev.theory_id == theory_id]
        if not matches:
            available = sorted(ev.theory_id for ev in trace.evaluations)
            raise ValueError(
                f"theory_id={theory_id!r} not found in evaluations for trace "
                f"{trace.trace_id!r}. Available: {available}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous: theory_id={theory_id!r} matched {len(matches)} evaluations "
                f"in trace {trace.trace_id!r}."
            )
        return matches[0]
