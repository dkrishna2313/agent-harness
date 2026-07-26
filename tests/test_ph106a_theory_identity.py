"""PH10.6a — Theory Identity Hardening tests.

Covers:
- TheoryOfWinning has a theory_id field (distinct from recommended_option_id)
- TheoryGenerator sets theory_id = f"TH-{choice_set.id}"
- Different choice sets produce theories with different theory_ids
- Same option recommended by two choice sets → two distinct theory_ids
- TheoryEvaluator propagates theory.theory_id (not recommended_option_id)
- StrategySelector: strict theory_id matching (no positional fallback)
- StrategySelector: duplicate theory_id in theories raises ValueError
- StrategySelector: same option_id but distinct theory_ids matches correctly
- StrategyCoordinator end-to-end: each theory has a distinct theory_id
- StrategyCoordinator end-to-end: winner_theory_id == selected theory's theory_id
"""

from __future__ import annotations

import types

import pytest

from functional_agents.context import AgentContext
from functional_agents.strategy import (
    StrategicChoiceSet,
    StrategyCoordinator,
    StrategySelection,
    StrategySelector,
    TheoryGenerator,
    TheoryOfWinning,
)
from functional_agents.strategy.strategic_choice import StrategicChoice
from functional_agents.strategy.strategy_plan import StrategyPlan
from functional_agents.strategy.theory_evaluation import CriterionScore, TheoryEvaluation
from functional_agents.strategy.theory_evaluator import TheoryEvaluator
from functional_agents.strategy.strategy_plan import EvaluationModel


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _plan() -> StrategyPlan:
    return StrategyPlan(plan_id="P-TEST", framework="executive", active_dimensions=[])


def _choice(selected_value: str = "OPT-A", dimension: str = "market") -> StrategicChoice:
    return StrategicChoice(
        id=f"SC-{dimension}-20260101-000000",
        dimension=dimension,
        selected_value=selected_value,
        rationale="Best choice.",
        confidence="High",
        supporting_assumptions=[],
        requiredness="optional",
    )


def _choice_set(set_id: str, *choices: StrategicChoice) -> StrategicChoiceSet:
    return StrategicChoiceSet(
        id=set_id,
        choices=list(choices),
        overall_confidence="High",
        internal_conflicts=[],
        completeness=1.0,
        rationale=f"Posture for {set_id}.",
    )


def _eval(*, theory_id: str, overall_score: float = 0.8, confidence: str = "High") -> TheoryEvaluation:
    return TheoryEvaluation(
        theory_id=theory_id,
        criteria_scores={
            "option_identified": CriterionScore(score=1.0, rationale="present", weight=1.0),
        },
        strengths=[],
        weaknesses=[],
        residual_risks=[],
        overall_score=overall_score,
        confidence=confidence,
        metadata={},
    )


def _research() -> types.SimpleNamespace:
    ns = types.SimpleNamespace()
    ns.executive_confidence = {"overall_confidence": "High", "confidence_drivers": []}
    ns.decision_analysis = {"recommended_option_id": "OPT-A", "rationale": "Best."}
    ns.preferred_option = {}
    ns.strategic_options = [{"option_id": "OPT-A", "title": "Alpha", "description": "Desc."}]
    ns.assumptions = []
    ns.risks = []
    ns.research_object = {}
    return ns


def _full_ctx() -> AgentContext:
    return AgentContext(
        question="What should we do?",
        profiles=["test"],
        execution_profile="test",
        research_object={"id": "R-TEST"},
        run_id="run001",
        strategic_options=[
            {
                "option_id": "OPT-A",
                "title": "Option A",
                "description": "First option.",
                "strategic_objective": "Grow.",
                "expected_outcomes": ["Outcome 1"],
                "supporting_assumption_ids": [],
                "associated_risk_ids": [],
                "associated_opportunity_ids": [],
                "supporting_recommendation_ids": [],
                "advantages": ["Fast"],
                "disadvantages": ["Risky"],
                "implementation_complexity": "Low",
                "estimated_time_horizon": "Near-term",
                "capital_intensity": "Low",
                "confidence": "High",
                "recommended": True,
                "rationale": "Best return.",
            },
        ],
        assumptions=[],
        risks=[],
        opportunities=[],
        recommendations=[],
        decision_model={"strategic_question": "What should we do?"},
        decision_analysis={
            "recommended_option_id": "OPT-A",
            "rationale": "Best risk-adjusted return.",
            "key_tradeoffs": [],
            "decision_matrix": [],
        },
        executive_confidence={
            "overall_confidence": "High",
            "board_recommendation": "Proceed.",
            "decision_readiness": "Ready",
            "confidence_drivers": [],
            "confidence_limiters": [],
            "critical_unknowns": [],
            "validation_priorities": [],
        },
        preferred_option={"option_id": "OPT-A", "title": "Option A"},
        research_strategy={},
    )


# ---------------------------------------------------------------------------
# TheoryOfWinning — theory_id field
# ---------------------------------------------------------------------------

class TestTheoryOfWinningTheoryId:
    def test_theory_of_winning_has_theory_id_field(self):
        theory = TheoryOfWinning(theory_id="TH-TEST")
        assert hasattr(theory, "theory_id")

    def test_theory_id_is_required(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TheoryOfWinning()

    def test_theory_id_is_distinct_from_recommended_option_id(self):
        theory = TheoryOfWinning(
            theory_id="TH-SCS-001",
            recommended_option_id="OPT-A",
        )
        assert theory.theory_id != theory.recommended_option_id

    def test_theory_id_can_be_set_independently(self):
        theory = TheoryOfWinning(
            theory_id="TH-XYZ",
            recommended_option_id="OPT-A",
        )
        assert theory.theory_id == "TH-XYZ"
        assert theory.recommended_option_id == "OPT-A"


# ---------------------------------------------------------------------------
# TheoryGenerator — theory_id set from choice_set.id
# ---------------------------------------------------------------------------

class TestTheoryGeneratorTheoryId:
    def test_theory_id_prefixed_from_choice_set_id(self):
        cs = _choice_set("SCS-0-20260101-000000", _choice())
        theory = TheoryGenerator().build(cs, _research())
        assert theory.theory_id == "TH-SCS-0-20260101-000000"

    def test_theory_id_format_th_prefix(self):
        cs = _choice_set("SCS-99-test", _choice())
        theory = TheoryGenerator().build(cs, _research())
        assert theory.theory_id.startswith("TH-")

    def test_different_choice_sets_produce_different_theory_ids(self):
        cs1 = _choice_set("SCS-0-20260101-000000", _choice("OPT-A"))
        cs2 = _choice_set("SCS-1-20260101-000000", _choice("OPT-A"))
        gen = TheoryGenerator()
        t1 = gen.build(cs1, _research())
        t2 = gen.build(cs2, _research())
        assert t1.theory_id != t2.theory_id

    def test_same_option_different_choice_sets_distinct_theory_ids(self):
        # Both postures recommend OPT-A but have distinct theory_ids
        cs1 = _choice_set("SCS-0-20260101-000000", _choice("OPT-A"))
        cs2 = _choice_set("SCS-1-20260101-000000", _choice("OPT-A"))
        gen = TheoryGenerator()
        t1 = gen.build(cs1, _research())
        t2 = gen.build(cs2, _research())
        assert t1.recommended_option_id == t2.recommended_option_id
        assert t1.theory_id != t2.theory_id

    def test_recommended_option_id_still_set(self):
        cs = _choice_set("SCS-0-20260101-000000", _choice("OPT-A"))
        theory = TheoryGenerator().build(cs, _research())
        assert theory.recommended_option_id == "OPT-A"

    def test_same_set_same_theory_id(self):
        cs = _choice_set("SCS-0-stable", _choice())
        gen = TheoryGenerator()
        t1 = gen.build(cs, _research())
        t2 = gen.build(cs, _research())
        assert t1.theory_id == t2.theory_id


# ---------------------------------------------------------------------------
# TheoryEvaluator — propagates theory.theory_id
# ---------------------------------------------------------------------------

class TestTheoryEvaluatorTheoryId:
    def test_evaluator_propagates_theory_id(self):
        theory = TheoryOfWinning(theory_id="TH-SCS-MY-SET", recommended_option_id="OPT-B")
        ev = TheoryEvaluator().build(theory, _plan(), None)
        assert ev.theory_id == "TH-SCS-MY-SET"

    def test_evaluator_does_not_use_recommended_option_id(self):
        theory = TheoryOfWinning(theory_id="TH-CUSTOM", recommended_option_id="OPT-DIFFERENT")
        ev = TheoryEvaluator().build(theory, _plan(), None)
        assert ev.theory_id == "TH-CUSTOM"
        assert ev.theory_id != theory.recommended_option_id

    def test_evaluator_theory_id_distinct_from_option_id_with_matching_name(self):
        # theory_id and recommended_option_id can have different values independently
        theory = TheoryOfWinning(theory_id="TH-SCS-99", recommended_option_id="OPT-A")
        ev = TheoryEvaluator().build(theory, _plan(), None)
        assert ev.theory_id == "TH-SCS-99"
        assert ev.theory_id != theory.recommended_option_id


# ---------------------------------------------------------------------------
# StrategySelector — strict ID matching
# ---------------------------------------------------------------------------

class TestStrategySelectorStrictMatching:
    def test_matches_by_theory_id_not_option_id(self):
        # option_ids are the same; theory_ids are different — selector must use theory_id
        t1 = TheoryOfWinning(theory_id="TH-SCS-0", recommended_option_id="OPT-A")
        t2 = TheoryOfWinning(theory_id="TH-SCS-1", recommended_option_id="OPT-A")
        evals = [
            _eval(theory_id="TH-SCS-0", overall_score=0.6),
            _eval(theory_id="TH-SCS-1", overall_score=0.9),
        ]
        result = StrategySelector().select([t1, t2], evals, _plan())
        assert result.theory_id == "TH-SCS-1"

    def test_same_option_id_no_positional_fallback(self):
        # Previously would hit positional fallback; now must use theory_id
        t1 = TheoryOfWinning(theory_id="TH-SCS-0", recommended_option_id="OPT-A")
        t2 = TheoryOfWinning(theory_id="TH-SCS-1", recommended_option_id="OPT-A")
        t3 = TheoryOfWinning(theory_id="TH-SCS-2", recommended_option_id="OPT-A")
        evals = [
            _eval(theory_id="TH-SCS-0", overall_score=0.7),
            _eval(theory_id="TH-SCS-1", overall_score=0.5),
            _eval(theory_id="TH-SCS-2", overall_score=0.9),
        ]
        result = StrategySelector().select([t1, t2, t3], evals, _plan())
        assert result.theory_id == "TH-SCS-2"


class TestStrategySelectorDuplicateRejection:
    def test_duplicate_theory_ids_raises(self):
        t1 = TheoryOfWinning(theory_id="TH-SAME", recommended_option_id="OPT-A")
        t2 = TheoryOfWinning(theory_id="TH-SAME", recommended_option_id="OPT-B")
        evals = [
            _eval(theory_id="TH-SAME", overall_score=0.8),
            _eval(theory_id="TH-SAME", overall_score=0.5),
        ]
        with pytest.raises(ValueError, match="duplicate theory_id"):
            StrategySelector().select([t1, t2], evals, _plan())

    def test_unique_theory_ids_does_not_raise(self):
        t1 = TheoryOfWinning(theory_id="TH-A", recommended_option_id="OPT-A")
        t2 = TheoryOfWinning(theory_id="TH-B", recommended_option_id="OPT-A")
        evals = [
            _eval(theory_id="TH-A", overall_score=0.8),
            _eval(theory_id="TH-B", overall_score=0.5),
        ]
        # Must not raise
        result = StrategySelector().select([t1, t2], evals, _plan())
        assert result.theory_id == "TH-A"


# ---------------------------------------------------------------------------
# StrategyCoordinator — end-to-end theory identity
# ---------------------------------------------------------------------------

class TestStrategyCoordinatorTheoryIdentity:
    def test_each_theory_has_distinct_theory_id(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        theory_ids = [t.theory_id for t in coord._theories]
        assert len(set(theory_ids)) == len(theory_ids), "Theory IDs must be unique"

    def test_theory_ids_have_th_prefix(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        for theory in coord._theories:
            assert theory.theory_id.startswith("TH-"), (
                f"Expected theory_id to start with TH-, got {theory.theory_id!r}"
            )

    def test_winner_theory_id_matches_selected_theory_theory_id(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        assert coord._selection.winner_theory_id == coord._selected_theory.theory_id

    def test_evaluation_theory_ids_match_theory_ids(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        theory_ids = {t.theory_id for t in coord._theories}
        eval_ids = {ev.theory_id for ev in coord._evaluations}
        assert theory_ids == eval_ids

    def test_theory_id_distinct_from_recommended_option_id(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        for theory in coord._theories:
            # theory_id is TH-prefixed; recommended_option_id is OPT-prefixed
            assert theory.theory_id != theory.recommended_option_id
