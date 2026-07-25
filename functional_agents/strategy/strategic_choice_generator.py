"""StrategicChoiceGenerator — produces one StrategicChoiceSet from a StrategyPlan (PH10.1).

PH10.1 is the pass-through implementation:
  - Reads active_dimensions from the StrategyPlan
  - Produces exactly one StrategicChoice per dimension
  - Assembles them into one StrategicChoiceSet
  - Reflects the strategic conclusions already present in the research output
  - No search, no alternatives, no multiple candidates, no constraints engine

Produced by: StrategyCoordinator.build()
Consumed by: StrategyCoordinator (stored as _choice_set; not yet wired into StrategicPosition)
Future phases: theory generation, theory evaluation
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from .strategic_choice import StrategicChoice
from .strategic_choice_set import StrategicChoiceSet
from .strategy_plan import StrategyPlan

if TYPE_CHECKING:
    pass

LOGGER = logging.getLogger(__name__)


class StrategicChoiceGenerator:
    """Generates a single StrategicChoiceSet from a StrategyPlan and research data.

    PH10.1: pass-through. One choice per active dimension; the selected value
    and confidence are drawn directly from the research output. No LLM, no
    search, no alternatives generated.
    """

    def build(self, plan: StrategyPlan, research: Any) -> StrategicChoiceSet:
        """Generate one StrategicChoiceSet.

        Parameters
        ----------
        plan:
            The resolved, executable StrategyPlan (produced by StrategyPlanner).
        research:
            The available research output. For PH10.1 this is an AgentContext.
            A future phase will replace this with a typed ResearchObject handoff.

        Returns
        -------
        StrategicChoiceSet
            One set containing one StrategicChoice per active dimension.
            Vacuously complete (completeness=1.0) when no dimensions are active.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

        choices = [
            self._build_choice(dimension, timestamp, research)
            for dimension in plan.active_dimensions
        ]

        n_dims = len(plan.active_dimensions)
        completeness = 1.0 if n_dims == 0 else float(len(choices)) / float(n_dims)
        overall_confidence = self._extract_confidence(research)

        n = len(choices)
        dim_label = "dimension" if n == 1 else "dimensions"
        rationale = (
            f"Pass-through extraction from research output "
            f"({n} {dim_label} covered)."
        )

        cs = StrategicChoiceSet(
            id=f"SCS-{timestamp}",
            choices=choices,
            overall_confidence=overall_confidence,
            internal_conflicts=[],
            completeness=completeness,
            rationale=rationale,
        )
        LOGGER.debug(
            "[StrategicChoiceGenerator] produced %d choice(s), completeness=%.2f",
            len(choices),
            completeness,
        )
        return cs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_choice(
        self,
        dimension: str,
        timestamp: str,
        research: Any,
    ) -> StrategicChoice:
        """Produce one StrategicChoice for a single dimension."""
        ec = self._as_dict(getattr(research, "executive_confidence", None))
        da = self._as_dict(getattr(research, "decision_analysis", None))
        preferred = self._as_dict(getattr(research, "preferred_option", None))

        # selected_value: the recommended option, reflecting existing conclusions
        selected_value = (
            preferred.get("option_id")
            or da.get("recommended_option_id")
            or ""
        )

        # rationale: from decision analysis or preferred option
        rationale = (
            da.get("rationale", "")
            or preferred.get("rationale", "")
        )

        # confidence: from executive confidence
        confidence = ec.get("overall_confidence", "")

        # supporting_assumptions: statement strings from ctx.assumptions
        raw_assumptions = getattr(research, "assumptions", None) or []
        supporting_assumptions = [
            a.get("statement", "") if isinstance(a, dict) else str(a)
            for a in raw_assumptions
            if a
        ]

        return StrategicChoice(
            id=f"SC-{dimension}-{timestamp}",
            dimension=dimension,
            selected_value=selected_value,
            rationale=rationale,
            confidence=confidence,
            supporting_assumptions=supporting_assumptions,
            requiredness="optional",
        )

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        """Return value if it is a dict, else an empty dict."""
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _extract_confidence(research: Any) -> str:
        """Extract overall_confidence from the research object."""
        ec = getattr(research, "executive_confidence", None)
        if isinstance(ec, dict):
            return ec.get("overall_confidence", "")
        return ""
