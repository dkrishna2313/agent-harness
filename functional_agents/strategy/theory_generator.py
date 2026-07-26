"""TheoryGenerator — maps a StrategicChoiceSet to a TheoryOfWinning (PH10.3).

Public interface: build(choice_set, research) -> TheoryOfWinning

One TheoryOfWinning is produced per StrategicChoiceSet. All fields are
derived deterministically from the choice set and available research
outputs. No LLM calls, no ranking, no optimisation.

Produced by: StrategyCoordinator.build() (one per StrategicChoiceSet)
Consumed by: StrategyCoordinator (stored as _theories; not yet wired
             into StrategicPosition)
"""

from __future__ import annotations

import logging
from typing import Any

from .strategic_choice_set import StrategicChoiceSet
from .strategic_position import TheoryOfWinning

LOGGER = logging.getLogger(__name__)


class TheoryGenerator:
    """Produces a TheoryOfWinning for a given StrategicChoiceSet.

    All fields are extracted deterministically from the choice set and
    the research object. No LLM calls, no optimisation.
    """

    def build(self, choice_set: StrategicChoiceSet, research: Any) -> TheoryOfWinning:
        """Produce one TheoryOfWinning from a StrategicChoiceSet and research data.

        Parameters
        ----------
        choice_set:
            One of the StrategicChoiceSets produced by StrategicChoiceGenerator.
        research:
            Available research output (AgentContext in PH10.x).

        Returns
        -------
        TheoryOfWinning
            Populated from choice_set fields and research data.
        """
        ec = self._as_dict(getattr(research, "executive_confidence", None))
        da = self._as_dict(getattr(research, "decision_analysis", None))
        preferred = self._as_dict(getattr(research, "preferred_option", None))

        recommended_id = self._extract_recommended_id(choice_set, da, preferred)
        recommended_title = self._extract_recommended_title(recommended_id, research)
        winning_mechanism = self._extract_winning_mechanism(recommended_id, research)
        success_conditions = list(ec.get("confidence_drivers", []))
        failure_modes = self._extract_failure_modes(research)
        assumptions = list(getattr(research, "assumptions", None) or [])
        evidence = self._extract_evidence(research)
        strategic_choices = [c.to_dict() for c in choice_set.choices]

        theory = TheoryOfWinning(
            theory_id=f"TH-{choice_set.id}",
            recommended_option_id=recommended_id,
            recommended_option_title=recommended_title,
            winning_position=choice_set.rationale,
            winning_mechanism=winning_mechanism,
            strategic_choices=strategic_choices,
            success_conditions=success_conditions,
            failure_modes=failure_modes,
            assumptions=assumptions,
            evidence=evidence,
            confidence=choice_set.overall_confidence,
        )
        LOGGER.debug(
            "[TheoryGenerator] produced theory for set %s, option=%s",
            choice_set.id, recommended_id,
        )
        return theory

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_recommended_id(
        choice_set: StrategicChoiceSet,
        da: dict[str, Any],
        preferred: dict[str, Any],
    ) -> str:
        """Return the recommended option ID for this theory.

        Derived from the first choice's selected_value when choices exist.
        Falls back to preferred_option.option_id, then da.recommended_option_id.
        """
        if choice_set.choices:
            return choice_set.choices[0].selected_value or ""
        return (
            preferred.get("option_id")
            or da.get("recommended_option_id")
            or ""
        )

    @staticmethod
    def _extract_recommended_title(recommended_id: str, research: Any) -> str:
        """Look up the option title for recommended_id in research.strategic_options."""
        for opt in (getattr(research, "strategic_options", None) or []):
            if isinstance(opt, dict) and opt.get("option_id") == recommended_id:
                return opt.get("title", "")
        return ""

    @staticmethod
    def _extract_winning_mechanism(recommended_id: str, research: Any) -> str:
        """Return the description of the recommended option."""
        for opt in (getattr(research, "strategic_options", None) or []):
            if isinstance(opt, dict) and opt.get("option_id") == recommended_id:
                return opt.get("description", "")
        return ""

    @staticmethod
    def _extract_failure_modes(research: Any) -> list[dict[str, Any]]:
        """Return high-severity risks from research."""
        risks = getattr(research, "risks", None) or []
        return [
            r for r in risks
            if isinstance(r, dict) and r.get("severity", "").lower() == "high"
        ]

    @staticmethod
    def _extract_evidence(research: Any) -> list[str]:
        """Return up to 10 citation strings from research_object."""
        ro = getattr(research, "research_object", None) or {}
        if isinstance(ro, dict):
            citations_raw = ro.get("citations", []) or []
        else:
            citations_raw = []
        return [
            c if isinstance(c, str) else str(c.get("text", c.get("citation", "")))
            for c in citations_raw[:10]
        ]

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}
