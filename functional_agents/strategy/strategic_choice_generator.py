"""StrategicChoiceGenerator — produces multiple coherent StrategicChoiceSets (PH10.2).

PH10.1: returned one StrategicChoiceSet (pass-through).
PH10.2: returns a list of exactly three StrategicChoiceSets, one per posture.

Each set:
  - contains one StrategicChoice per active plan dimension
  - has completeness=1.0
  - has no internal conflicts
  - differs from the other sets in posture key, ID, and rationale
  - where three or more strategic options exist, also differs in selected_value

Postures are deterministic and ordered: recommended → alternative-a → alternative-b.
No LLM calls, no search, no optimization, no ranking.

Produced by: StrategyCoordinator.build()
Consumed by: StrategyCoordinator (stored as _choice_sets; not yet wired into StrategicPosition)
Future phases: theory generation, theory evaluation
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from .strategic_choice import StrategicChoice
from .strategic_choice_set import StrategicChoiceSet
from .strategy_plan import StrategyPlan

LOGGER = logging.getLogger(__name__)

# Fixed posture definitions: (option_index, posture_key, set_rationale_template).
# option_index is used to select from available strategic options;
# when fewer options exist than postures, indices wrap via modulo.
_POSTURES: tuple[tuple[int, str, str], ...] = (
    (0, "recommended",   "Posture 0 (recommended): {} dimension(s) covered."),
    (1, "alternative-a", "Posture 1 (alternative-a): {} dimension(s) covered."),
    (2, "alternative-b", "Posture 2 (alternative-b): {} dimension(s) covered."),
)


class StrategicChoiceGenerator:
    """Generates a list of StrategicChoiceSets from a StrategyPlan and research data.

    PH10.2: produces exactly three sets with deterministic posture variation.
    Each set is internally coherent (one choice per active dimension),
    complete (completeness=1.0), and free of internal conflicts.
    Sets differ in posture key, ID, rationale, and selected option where
    three or more strategic options are available.
    """

    def build(self, plan: StrategyPlan, research: Any) -> list[StrategicChoiceSet]:
        """Generate three StrategicChoiceSets with deterministic posture variation.

        Parameters
        ----------
        plan:
            The resolved, executable StrategyPlan (produced by StrategyPlanner).
        research:
            The available research output. For PH10.x this is an AgentContext.
            A future phase will replace this with a typed ResearchObject handoff.

        Returns
        -------
        list[StrategicChoiceSet]
            Exactly three sets, ordered: recommended, alternative-a, alternative-b.
            All are complete (completeness=1.0) and conflict-free.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        option_ids = self._extract_option_ids(research)

        sets = [
            self._build_set(
                posture_idx, posture_key, rationale_tpl,
                plan, research, option_ids, timestamp,
            )
            for posture_idx, posture_key, rationale_tpl in _POSTURES
        ]
        LOGGER.debug(
            "[StrategicChoiceGenerator] produced %d set(s), %d option(s) available",
            len(sets), len(option_ids),
        )
        return sets

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_set(
        self,
        posture_idx: int,
        posture_key: str,
        rationale_tpl: str,
        plan: StrategyPlan,
        research: Any,
        option_ids: list[str],
        timestamp: str,
    ) -> StrategicChoiceSet:
        """Build one StrategicChoiceSet for a single posture."""
        choices = [
            self._build_choice(
                dimension, posture_idx, posture_key, option_ids, timestamp, research,
            )
            for dimension in plan.active_dimensions
        ]
        n_dims = len(plan.active_dimensions)
        completeness = 1.0 if n_dims == 0 else float(len(choices)) / float(n_dims)

        return StrategicChoiceSet(
            id=f"SCS-{posture_idx}-{timestamp}",
            choices=choices,
            overall_confidence=self._extract_confidence(research),
            internal_conflicts=[],
            completeness=completeness,
            rationale=rationale_tpl.format(len(choices)),
        )

    def _build_choice(
        self,
        dimension: str,
        posture_idx: int,
        posture_key: str,
        option_ids: list[str],
        timestamp: str,
        research: Any,
    ) -> StrategicChoice:
        """Produce one StrategicChoice for a single dimension under a posture."""
        ec = self._as_dict(getattr(research, "executive_confidence", None))
        da = self._as_dict(getattr(research, "decision_analysis", None))
        preferred = self._as_dict(getattr(research, "preferred_option", None))

        # selected_value: pick by posture index when options are available;
        # fall back to preferred/da when no strategic options exist.
        if option_ids:
            selected_value = option_ids[posture_idx % len(option_ids)]
        else:
            selected_value = (
                preferred.get("option_id")
                or da.get("recommended_option_id")
                or ""
            )

        rationale = (
            da.get("rationale", "")
            or preferred.get("rationale", "")
        )
        confidence = ec.get("overall_confidence", "")

        raw_assumptions = getattr(research, "assumptions", None) or []
        supporting_assumptions = [
            a.get("statement", "") if isinstance(a, dict) else str(a)
            for a in raw_assumptions
            if a
        ]

        return StrategicChoice(
            id=f"SC-{posture_key}-{dimension}-{timestamp}",
            dimension=dimension,
            selected_value=selected_value,
            rationale=rationale,
            confidence=confidence,
            supporting_assumptions=supporting_assumptions,
            requiredness="optional",
        )

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_option_ids(research: Any) -> list[str]:
        """Return a list of non-empty option_id strings from strategic_options."""
        opts = getattr(research, "strategic_options", None) or []
        ids = [o.get("option_id", "") for o in opts if isinstance(o, dict)]
        return [i for i in ids if i]

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        """Return value as-is if it is a dict, else an empty dict."""
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _extract_confidence(research: Any) -> str:
        """Extract overall_confidence from the research object."""
        ec = getattr(research, "executive_confidence", None)
        if isinstance(ec, dict):
            return ec.get("overall_confidence", "")
        return ""
