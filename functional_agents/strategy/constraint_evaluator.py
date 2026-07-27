"""PH12.1 — ConstraintEvaluator.

Checks each plan constraint against a theory's strategic choices.
No LLM calls. All logic is keyword-based and deterministic.
"""

from __future__ import annotations

import logging
from typing import Any

from .alignment import ConstraintResult
from .strategic_position import TheoryOfWinning
from .strategy_plan import StrategyPlan

LOGGER = logging.getLogger(__name__)

# Prefix applied by StrategyPlanner._build_constraints()
_REQUIRED_CONDITION_PREFIX = "required_condition:"
_EXCLUDED_OPTION_PREFIX = "excluded_option:"

# Triggers that indicate a choice violates an avoidance constraint.
_AVOID_TRIGGERS: frozenset[str] = frozenset({
    "concentrat",   # concentrated / concentration
    "single-state",
    "single_state",
    "undiversif",
    "focused",
})

# Tokens that fully satisfy a preservation constraint.
_PRESERVE_SATISFY: frozenset[str] = frozenset({
    "diversif",     # diversified / diversification
    "alternati",    # alternative(s)
    "multiple",
    "spread",
})

# Tokens that partially satisfy a preservation constraint.
_PRESERVE_PARTIAL: frozenset[str] = frozenset({
    "staged",
    "contingenc",   # contingency
    "monitor",
    "gated",
    "hybrid",
})


class ConstraintEvaluator:
    """Evaluates plan constraints against a theory's strategic choices.

    For each string in plan.constraints:
      - Entries prefixed "required_condition:" are parsed as either avoidance
        or preservation constraints based on the leading verb.
      - Entries prefixed "excluded_option:" result in a violation when the
        theory's recommended_option_id matches the excluded ID.
      - Unknown prefixes → not_assessable.
    """

    def evaluate(
        self,
        theory: TheoryOfWinning,
        plan: StrategyPlan,
    ) -> list[ConstraintResult]:
        """Return one ConstraintResult per entry in plan.constraints."""
        choice_ids, choice_tokens = self._extract_choice_tokens(theory)
        results: list[ConstraintResult] = []

        for raw in plan.constraints:
            if raw.startswith(_REQUIRED_CONDITION_PREFIX):
                text = raw[len(_REQUIRED_CONDITION_PREFIX):].strip()
                results.append(
                    self._check_required_condition(text, choice_ids, choice_tokens)
                )
            elif raw.startswith(_EXCLUDED_OPTION_PREFIX):
                excluded_id = raw[len(_EXCLUDED_OPTION_PREFIX):].strip()
                results.append(
                    self._check_excluded_option(raw, excluded_id, theory.recommended_option_id)
                )
            else:
                results.append(ConstraintResult(
                    constraint=raw,
                    status="not_assessable",
                    score=0.75,
                    rationale="Constraint prefix not recognised.",
                ))

        LOGGER.debug(
            "[ConstraintEvaluator] theory=%s constraints=%d violated=%d",
            theory.theory_id,
            len(results),
            sum(1 for r in results if r.status == "violated"),
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_choice_tokens(
        theory: TheoryOfWinning,
    ) -> tuple[set[str], set[str]]:
        """Return (choice_ids, choice_tokens) derived from theory's strategic_choices."""
        choice_ids: set[str] = set()
        choice_tokens: set[str] = set()

        for c in theory.strategic_choices:
            if not isinstance(c, dict):
                continue
            # Raw ID / selected_value
            for field in ("selected_value", "id"):
                v = str(c.get(field, "")).lower().strip()
                if v:
                    choice_ids.add(v)
                    choice_tokens.update(v.replace("_", " ").replace("-", " ").split())

            # Embedded metadata titles / descriptions
            meta: dict[str, Any] = c.get("metadata", {}) or {}
            for key in ("choice_title", "choice_description", "dimension_title"):
                v = str(meta.get(key, "")).lower()
                if v:
                    choice_tokens.update(w for w in v.replace("-", " ").split() if len(w) > 3)

        return choice_ids, choice_tokens

    @staticmethod
    def _check_required_condition(
        text: str,
        choice_ids: set[str],
        choice_tokens: set[str],
    ) -> ConstraintResult:
        text_lower = text.lower().strip()

        is_avoidance = text_lower.startswith(("avoid", "do not", "never", "prohibit"))
        is_preservation = text_lower.startswith(("preserve", "require", "ensure", "maintain"))

        if is_avoidance:
            if any(
                trigger in cid
                for trigger in _AVOID_TRIGGERS
                for cid in choice_ids
            ) or any(
                trigger in tok
                for trigger in _AVOID_TRIGGERS
                for tok in choice_tokens
            ):
                return ConstraintResult(
                    constraint=text,
                    status="violated",
                    score=0.0,
                    rationale="Theory choices match avoidance criteria (concentration/single-state).",
                )
            return ConstraintResult(
                constraint=text,
                status="satisfied",
                score=1.0,
                rationale="Theory choices do not trigger avoidance criteria.",
            )

        if is_preservation:
            has_full = any(
                term in cid
                for term in _PRESERVE_SATISFY
                for cid in choice_ids
            ) or any(
                term in tok
                for term in _PRESERVE_SATISFY
                for tok in choice_tokens
            )
            has_partial = any(
                term in cid
                for term in _PRESERVE_PARTIAL
                for cid in choice_ids
            ) or any(
                term in tok
                for term in _PRESERVE_PARTIAL
                for tok in choice_tokens
            )

            if has_full:
                return ConstraintResult(
                    constraint=text,
                    status="satisfied",
                    score=1.0,
                    rationale="Theory includes diversification or alternative-preserving choices.",
                )
            if has_partial:
                return ConstraintResult(
                    constraint=text,
                    status="partially_satisfied",
                    score=0.5,
                    rationale="Theory partially preserves alternatives (staged/hybrid approach).",
                )
            return ConstraintResult(
                constraint=text,
                status="violated",
                score=0.0,
                rationale="Theory does not preserve alternative pathways.",
            )

        return ConstraintResult(
            constraint=text,
            status="not_assessable",
            score=0.75,
            rationale="Constraint type (avoidance/preservation) not recognized from leading verb.",
        )

    @staticmethod
    def _check_excluded_option(
        raw: str,
        excluded_id: str,
        recommended_id: str,
    ) -> ConstraintResult:
        if excluded_id and recommended_id and excluded_id == recommended_id:
            return ConstraintResult(
                constraint=raw,
                status="violated",
                score=0.0,
                rationale=f"Theory recommends excluded option {excluded_id!r}.",
            )
        return ConstraintResult(
            constraint=raw,
            status="satisfied",
            score=1.0,
            rationale=f"Theory does not recommend excluded option {excluded_id!r}.",
        )
