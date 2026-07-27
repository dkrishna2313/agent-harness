"""TheoryGenerator — maps a StrategicChoiceSet to a TheoryOfWinning (PH10.3/PH12.0).

Public interface: build(choice_set, research) -> TheoryOfWinning

PH10.3: one TheoryOfWinning per StrategicChoiceSet, derived from research context.
PH12.0: configured mode — derives winning_position, winning_mechanism, recommended ID
        and title from the actual choice set metadata embedded by StrategicChoiceGenerator.
        Evidence and failure modes are filtered by relevance to the chosen options.

Produced by: StrategyCoordinator.build() (one per StrategicChoiceSet)
Consumed by: StrategyCoordinator (stored as _theories)
"""

from __future__ import annotations

import logging
from typing import Any

from .strategic_choice_set import StrategicChoiceSet
from .strategic_position import TheoryOfWinning

LOGGER = logging.getLogger(__name__)

_MAX_EVIDENCE = 10
_EVIDENCE_PER_THEORY = 5   # max evidence items per theory in configured mode


class TheoryGenerator:
    """Produces a TheoryOfWinning for a given StrategicChoiceSet.

    PH10.3 (legacy): fields extracted from research context.
    PH12.0 (configured): fields derived from choice set metadata when present.
    """

    def build(self, choice_set: StrategicChoiceSet, research: Any) -> TheoryOfWinning:
        """Produce one TheoryOfWinning from a StrategicChoiceSet and research data."""
        if self._is_configured_mode(choice_set):
            return self._build_configured(choice_set, research)
        return self._build_legacy(choice_set, research)

    # ------------------------------------------------------------------
    # Configured mode (PH12.0)
    # ------------------------------------------------------------------

    def _is_configured_mode(self, choice_set: StrategicChoiceSet) -> bool:
        """True when choices carry dimension metadata embedded by configured generator."""
        return bool(
            choice_set.choices
            and choice_set.choices[0].metadata.get("choice_title")
        )

    def _build_configured(
        self, choice_set: StrategicChoiceSet, research: Any
    ) -> TheoryOfWinning:
        """Build a theory from configured choice metadata."""
        first_choice = choice_set.choices[0]

        recommended_id = first_choice.selected_value
        recommended_title = first_choice.metadata.get("choice_title", recommended_id)

        winning_position = self._build_winning_position(choice_set)
        winning_mechanism = self._build_winning_mechanism(choice_set)

        ec = self._as_dict(getattr(research, "executive_confidence", None))
        success_conditions = list(ec.get("confidence_drivers", []))
        assumptions = list(getattr(research, "assumptions", None) or [])

        choice_keywords = self._extract_choice_keywords(choice_set)
        evidence = self._filter_evidence(research, choice_keywords)
        failure_modes = self._filter_failure_modes(research, choice_keywords)

        strategic_choices = [c.to_dict() for c in choice_set.choices]

        theory = TheoryOfWinning(
            theory_id=f"TH-{choice_set.id}",
            source_choice_set_id=choice_set.id,
            recommended_option_id=recommended_id,
            recommended_option_title=recommended_title,
            winning_position=winning_position,
            winning_mechanism=winning_mechanism,
            strategic_choices=strategic_choices,
            success_conditions=success_conditions,
            failure_modes=failure_modes,
            assumptions=assumptions,
            evidence=evidence,
            confidence=choice_set.overall_confidence,
        )
        LOGGER.debug(
            "[TheoryGenerator] configured theory %s: option=%s, evidence=%d, failure_modes=%d",
            theory.theory_id, recommended_id, len(evidence), len(failure_modes),
        )
        return theory

    @staticmethod
    def _build_winning_position(choice_set: StrategicChoiceSet) -> str:
        """Compose a descriptive winning position from all choices in the set."""
        parts = []
        for c in choice_set.choices:
            dim_title = c.metadata.get("dimension_title", c.dimension)
            choice_title = c.metadata.get("choice_title", c.selected_value)
            parts.append(f"{dim_title}: {choice_title}")
        if parts:
            return " | ".join(parts)
        return choice_set.rationale

    @staticmethod
    def _build_winning_mechanism(choice_set: StrategicChoiceSet) -> str:
        """Derive a winning mechanism from the first choice's dimension description."""
        if not choice_set.choices:
            return ""
        first = choice_set.choices[0]
        dim_desc = first.metadata.get("dimension_description", "")
        choice_title = first.metadata.get("choice_title", first.selected_value)
        if dim_desc:
            return f"{choice_title}: {dim_desc}"
        return choice_title

    @staticmethod
    def _extract_choice_keywords(choice_set: StrategicChoiceSet) -> list[str]:
        """Gather keywords from choice and dimension titles for relevance filtering."""
        keywords: list[str] = []
        for c in choice_set.choices:
            for key in ("choice_title", "dimension_title", "choice_description"):
                val = c.metadata.get(key, "")
                if val:
                    keywords.extend(w for w in val.lower().split() if len(w) > 3)
            if c.selected_value:
                keywords.append(c.selected_value.lower().replace("_", " "))
        return list(dict.fromkeys(keywords))  # deduplicated, ordered

    @staticmethod
    def _filter_evidence(
        research: Any,
        keywords: list[str],
    ) -> list[str]:
        """Return evidence items relevant to the theory's strategic choices.

        Algorithm:
        1. Collect citation strings from research_object.citations.
        2. When citations is empty, fall back to evidence_ids (strings).
        3. If keywords are present, filter to items containing any keyword.
        4. When keyword filtering yields results, return those (up to
           _EVIDENCE_PER_THEORY). When it yields nothing, return the first
           _EVIDENCE_PER_THEORY items from the full pool (symmetric fallback).

        The symmetric fallback assigns the same evidence set to all theories
        that cannot be distinguished by keyword relevance — candidate position
        never influences which evidence or how much evidence a theory receives.
        """
        ro = getattr(research, "research_object", None) or {}
        if isinstance(ro, dict):
            citations_raw: list = ro.get("citations", []) or []
            if not citations_raw:
                # Symmetric fallback: use evidence_ids as opaque string tokens.
                citations_raw = list(ro.get("evidence_ids", []) or [])
        else:
            citations_raw = []

        all_citations = [
            c if isinstance(c, str) else str(c.get("text", c.get("citation", "")))
            for c in citations_raw[:_MAX_EVIDENCE]
        ]

        if not all_citations:
            return []

        if not keywords:
            return all_citations[:_EVIDENCE_PER_THEORY]

        matching = [
            c for c in all_citations
            if any(kw in c.lower() for kw in keywords)
        ]
        # Symmetric fallback: when no keyword match, every theory gets the same
        # leading slice rather than a position-dependent window.
        return matching[:_EVIDENCE_PER_THEORY] if matching else all_citations[:_EVIDENCE_PER_THEORY]

    @staticmethod
    def _filter_failure_modes(research: Any, keywords: list[str]) -> list[dict[str, Any]]:
        """Return high-severity risks relevant to the given choice keywords.

        Filters by keyword presence in risk description. Falls back to all
        high-severity risks when nothing matches.
        """
        risks = getattr(research, "risks", None) or []
        high_severity = [
            r for r in risks
            if isinstance(r, dict) and r.get("severity", "").lower() == "high"
        ]

        if not high_severity:
            return []

        if not keywords:
            return high_severity

        matching = [
            r for r in high_severity
            if any(kw in str(r).lower() for kw in keywords)
        ]
        return matching if matching else high_severity

    # ------------------------------------------------------------------
    # Legacy mode (PH10.3)
    # ------------------------------------------------------------------

    def _build_legacy(self, choice_set: StrategicChoiceSet, research: Any) -> TheoryOfWinning:
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
            source_choice_set_id=choice_set.id,
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
    # Legacy extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_recommended_id(
        choice_set: StrategicChoiceSet,
        da: dict[str, Any],
        preferred: dict[str, Any],
    ) -> str:
        if choice_set.choices:
            return choice_set.choices[0].selected_value or ""
        return (
            preferred.get("option_id")
            or da.get("recommended_option_id")
            or ""
        )

    @staticmethod
    def _extract_recommended_title(recommended_id: str, research: Any) -> str:
        for opt in (getattr(research, "strategic_options", None) or []):
            if isinstance(opt, dict) and opt.get("option_id") == recommended_id:
                return opt.get("title", "")
        return ""

    @staticmethod
    def _extract_winning_mechanism(recommended_id: str, research: Any) -> str:
        for opt in (getattr(research, "strategic_options", None) or []):
            if isinstance(opt, dict) and opt.get("option_id") == recommended_id:
                return opt.get("description", "")
        return ""

    @staticmethod
    def _extract_failure_modes(research: Any) -> list[dict[str, Any]]:
        risks = getattr(research, "risks", None) or []
        return [
            r for r in risks
            if isinstance(r, dict) and r.get("severity", "").lower() == "high"
        ]

    @staticmethod
    def _extract_evidence(research: Any) -> list[str]:
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
