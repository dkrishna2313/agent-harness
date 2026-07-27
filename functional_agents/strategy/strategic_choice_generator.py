"""StrategicChoiceGenerator — produces multiple coherent StrategicChoiceSets (PH10.2/PH12.0).

PH10.1: returned one StrategicChoiceSet (pass-through).
PH10.2: returns a list of exactly three StrategicChoiceSets, one per posture.
PH12.0: configured mode — uses plan.dimension_configs to pick per-posture choices from
        the engagement-defined decision space. Validates unique signatures and coverage.

Each set:
  - contains one StrategicChoice per active plan dimension
  - has completeness=1.0
  - has no internal conflicts
  - differs from the other sets in posture key, ID, and rationale
  - where three or more strategic options exist, also differs in selected_value

A choice-set signature is deterministic: tuple(sorted((dim_id, choice_id) for ...)).

Produced by: StrategyCoordinator.build()
Consumed by: StrategyCoordinator (stored as _choice_sets)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from .strategic_choice import StrategicChoice
from .strategic_choice_set import StrategicChoiceSet
from .strategy_config import DimensionConfig
from .strategy_plan import StrategyPlan

LOGGER = logging.getLogger(__name__)

# Fixed posture definitions (legacy mode): (option_index, posture_key, set_rationale_template).
_POSTURES: tuple[tuple[int, str, str], ...] = (
    (0, "recommended",   "Posture 0 (recommended): {} dimension(s) covered."),
    (1, "alternative-a", "Posture 1 (alternative-a): {} dimension(s) covered."),
    (2, "alternative-b", "Posture 2 (alternative-b): {} dimension(s) covered."),
)

_POSTURE_KEYS = ("recommended", "alternative-a", "alternative-b")


class StrategicChoiceGenerator:
    """Generates a list of StrategicChoiceSets from a StrategyPlan and research data.

    PH10.2: produces exactly three sets with deterministic posture variation.
    PH12.0: configured mode uses plan.dimension_configs and their choices.

    Configured mode:
      - For each posture index, picks choices[posture_idx % len(choices)] per dimension.
      - Computes a deterministic signature per set; deduplicates before returning.
      - Embeds choice metadata (choice_title, dimension_title, dimension_description)
        on each StrategicChoice for downstream use by TheoryGenerator.
      - Raises ValueError when diversity_required=True and only one unique set is produced.
    """

    def build(self, plan: StrategyPlan, research: Any) -> list[StrategicChoiceSet]:
        """Generate StrategicChoiceSets with deterministic posture variation.

        Parameters
        ----------
        plan:
            The resolved, executable StrategyPlan (produced by StrategyPlanner).
        research:
            The available research output (AgentContext in PH10.x).

        Returns
        -------
        list[StrategicChoiceSet]
            Unique sets, ordered: recommended, alternative-a, alternative-b.
            All are complete (completeness=1.0) and conflict-free.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

        if plan.dimension_configs:
            sets = self._build_configured_sets(plan, research, timestamp)
        else:
            sets = self._build_legacy_sets(plan, research, timestamp)

        LOGGER.debug(
            "[StrategicChoiceGenerator] produced %d set(s) (%s mode)",
            len(sets),
            "configured" if plan.dimension_configs else "legacy",
        )
        return sets

    # ------------------------------------------------------------------
    # Configured mode (PH12.0)
    # ------------------------------------------------------------------

    def _build_configured_sets(
        self,
        plan: StrategyPlan,
        research: Any,
        timestamp: str,
    ) -> list[StrategicChoiceSet]:
        """Generate choice sets from plan.dimension_configs."""
        max_candidates = plan.generation_policy.max_candidates
        diversity_required = plan.generation_policy.diversity_required
        dims = plan.dimension_configs
        overall_conf = self._extract_confidence(research)

        seen_signatures: set[tuple] = set()
        sets: list[StrategicChoiceSet] = []

        for posture_idx, posture_key in enumerate(_POSTURE_KEYS):
            if len(sets) >= max_candidates:
                break

            choices = self._build_configured_choices(dims, posture_idx, posture_key, timestamp)
            sig = self._compute_signature(choices)

            if sig in seen_signatures:
                LOGGER.debug(
                    "[StrategicChoiceGenerator] skipping duplicate signature at posture %d",
                    posture_idx,
                )
                continue

            seen_signatures.add(sig)
            n_dims = len(dims)
            completeness = 1.0 if n_dims == 0 else float(len(choices)) / float(n_dims)

            choice_labels = " | ".join(
                f"{c.metadata.get('dimension_title', c.dimension)}: "
                f"{c.metadata.get('choice_title', c.selected_value)}"
                for c in choices
            )
            rationale = f"{posture_key.capitalize()} — {choice_labels}"

            cs = StrategicChoiceSet(
                id=f"SCS-{posture_idx}-{timestamp}",
                choices=choices,
                overall_confidence=overall_conf,
                internal_conflicts=[],
                completeness=completeness,
                rationale=rationale,
            )
            sets.append(cs)

        if diversity_required and len(sets) == 1:
            raise ValueError(
                "[StrategicChoiceGenerator] diversity_required=True but only one unique "
                "choice set could be generated. Add more choices to each dimension."
            )

        return sets

    def _build_configured_choices(
        self,
        dims: list[DimensionConfig],
        posture_idx: int,
        posture_key: str,
        timestamp: str,
    ) -> list[StrategicChoice]:
        """Build one StrategicChoice per dimension for the given posture."""
        choices: list[StrategicChoice] = []
        for dim in dims:
            available = dim.choices
            if not available:
                # Required validation happens in ConfigurationResolver; skip empty.
                continue
            chosen = available[posture_idx % len(available)]
            choices.append(StrategicChoice(
                id=f"SC-{posture_key}-{dim.id}-{timestamp}",
                dimension=dim.id,
                selected_value=chosen.id,
                rationale=dim.description or dim.title,
                confidence="",
                requiredness="required" if dim.required else "optional",
                metadata={
                    "choice_title": chosen.title,
                    "choice_description": chosen.description,
                    "dimension_title": dim.title,
                    "dimension_description": dim.description,
                },
            ))
        return choices

    @staticmethod
    def _compute_signature(choices: list[StrategicChoice]) -> tuple:
        """Deterministic signature: sorted (dimension_id, choice_id) pairs."""
        return tuple(sorted((c.dimension, c.selected_value) for c in choices))

    # ------------------------------------------------------------------
    # Legacy mode (PH10.2)
    # ------------------------------------------------------------------

    def _build_legacy_sets(
        self,
        plan: StrategyPlan,
        research: Any,
        timestamp: str,
    ) -> list[StrategicChoiceSet]:
        option_ids = self._extract_option_ids(research)
        return [
            self._build_legacy_set(
                posture_idx, posture_key, rationale_tpl,
                plan, research, option_ids, timestamp,
            )
            for posture_idx, posture_key, rationale_tpl in _POSTURES
        ]

    def _build_legacy_set(
        self,
        posture_idx: int,
        posture_key: str,
        rationale_tpl: str,
        plan: StrategyPlan,
        research: Any,
        option_ids: list[str],
        timestamp: str,
    ) -> StrategicChoiceSet:
        choices = [
            self._build_legacy_choice(
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

    def _build_legacy_choice(
        self,
        dimension: str,
        posture_idx: int,
        posture_key: str,
        option_ids: list[str],
        timestamp: str,
        research: Any,
    ) -> StrategicChoice:
        ec = self._as_dict(getattr(research, "executive_confidence", None))
        da = self._as_dict(getattr(research, "decision_analysis", None))
        preferred = self._as_dict(getattr(research, "preferred_option", None))

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
        opts = getattr(research, "strategic_options", None) or []
        ids = [o.get("option_id", "") for o in opts if isinstance(o, dict)]
        return [i for i in ids if i]

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _extract_confidence(research: Any) -> str:
        ec = getattr(research, "executive_confidence", None)
        if isinstance(ec, dict):
            return ec.get("overall_confidence", "")
        return ""
