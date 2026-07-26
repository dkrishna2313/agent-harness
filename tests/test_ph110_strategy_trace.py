"""PH11.0 — StrategyTrace Artifact tests.

Covers:
- StrategyTrace model construction and field access
- trace_id format is STRAT-{plan_id}
- All 10 validation rules (each violation raises ValueError)
- StrategyCoordinator._trace is populated after build()
- StrategyCoordinator._trace.trace_id = STRAT-{plan.plan_id}
- StrategyTrace serialization round-trip (to_dict / from_dict)
- build_canonical_trace exposes "strategy_trace" key
- "strategy_trace" key is None when strategy layer did not run
"""

from __future__ import annotations

import types

import pytest
from pydantic import ValidationError

from functional_agents.context import AgentContext
from functional_agents.strategy import (
    StrategicChoiceSet,
    StrategyCoordinator,
    StrategySelection,
    StrategyTrace,
    TheoryOfWinning,
)
from functional_agents.strategy.strategic_choice import StrategicChoice
from functional_agents.strategy.strategic_position import (
    StrategicExecution,
    StrategicJustification,
    StrategicPosition,
    StrategicRecommendation,
)
from functional_agents.strategy.strategy_plan import StrategyPlan
from functional_agents.strategy.theory_evaluation import CriterionScore, TheoryEvaluation


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _plan(plan_id: str = "P-TEST") -> StrategyPlan:
    return StrategyPlan(plan_id=plan_id, framework="executive", active_dimensions=[])


def _choice_set(set_id: str) -> StrategicChoiceSet:
    choice = StrategicChoice(
        id=f"SC-{set_id}", dimension="market",
        selected_value="OPT-A", rationale="r", confidence="High",
        supporting_assumptions=[], requiredness="optional",
    )
    return StrategicChoiceSet(
        id=set_id, choices=[choice], overall_confidence="High",
        internal_conflicts=[], completeness=1.0, rationale="r",
    )


def _theory(tid: str, oid: str = "OPT-A") -> TheoryOfWinning:
    return TheoryOfWinning(theory_id=tid, recommended_option_id=oid)


def _eval(tid: str, score: float = 0.8) -> TheoryEvaluation:
    return TheoryEvaluation(
        theory_id=tid,
        criteria_scores={"x": CriterionScore(score=score, rationale="r", weight=1.0)},
        strengths=[], weaknesses=[], residual_risks=[],
        overall_score=score, confidence="High", metadata={},
    )


def _selection(winner: str, runner_up: str | None = None) -> StrategySelection:
    return StrategySelection(
        winner_theory_id=winner,
        winner_score=0.8,
        runner_up_theory_id=runner_up,
    )


def _position(theory: TheoryOfWinning) -> StrategicPosition:
    return StrategicPosition(
        position_id="SP-TEST",
        created_at="2026-07-26T00:00:00+00:00",
        theory_of_winning=theory,
        recommendation=StrategicRecommendation(
            recommended_option_id=theory.recommended_option_id,
            recommended_option_title="",
            board_recommendation="Proceed.",
            decision_readiness="Ready",
            overall_confidence="High",
        ),
        justification=StrategicJustification(
            decision_analysis={}, strategic_options=[],
            assumptions=[], risks=[], opportunities=[],
        ),
        execution=StrategicExecution(recommendations=[], validation_priorities=[]),
    )


def _valid_trace(n: int = 3, plan_id: str = "P-TEST") -> StrategyTrace:
    """Build a minimal valid StrategyTrace with n theories."""
    plan = _plan(plan_id)
    choice_sets = [_choice_set(f"SCS-{i}") for i in range(n)]
    theories = [_theory(f"TH-SCS-{i}") for i in range(n)]
    evaluations = [_eval(f"TH-SCS-{i}", 0.8 - i * 0.1) for i in range(n)]
    winner_theory = theories[0]
    sel = _selection(winner_theory.theory_id)
    pos = _position(winner_theory)
    return StrategyTrace(
        trace_id=f"STRAT-{plan.plan_id}",
        created_at="2026-07-26T00:00:00+00:00",
        plan=plan,
        choice_sets=choice_sets,
        theories=theories,
        evaluations=evaluations,
        selection=sel,
        strategic_position=pos,
        metadata={},
    )


def _full_ctx() -> AgentContext:
    return AgentContext(
        question="What should we do?",
        profiles=["test"],
        execution_profile="test",
        research_object={"id": "R-TEST"},
        run_id="run001",
        strategic_options=[
            {
                "option_id": "OPT-A", "title": "Option A", "description": "First.",
                "strategic_objective": "Grow.", "expected_outcomes": ["O1"],
                "supporting_assumption_ids": [], "associated_risk_ids": [],
                "associated_opportunity_ids": [], "supporting_recommendation_ids": [],
                "advantages": ["Fast"], "disadvantages": ["Risky"],
                "implementation_complexity": "Low", "estimated_time_horizon": "Near-term",
                "capital_intensity": "Low", "confidence": "High",
                "recommended": True, "rationale": "Best.",
            },
        ],
        assumptions=[], risks=[], opportunities=[], recommendations=[],
        decision_model={"strategic_question": "What should we do?"},
        decision_analysis={
            "recommended_option_id": "OPT-A", "rationale": "Best.",
            "key_tradeoffs": [], "decision_matrix": [],
        },
        executive_confidence={
            "overall_confidence": "High", "board_recommendation": "Proceed.",
            "decision_readiness": "Ready", "confidence_drivers": [],
            "confidence_limiters": [], "critical_unknowns": [], "validation_priorities": [],
        },
        preferred_option={"option_id": "OPT-A", "title": "Option A"},
        research_strategy={},
    )


# ---------------------------------------------------------------------------
# StrategyTrace — model construction
# ---------------------------------------------------------------------------

class TestStrategyTraceConstruction:
    def test_valid_trace_constructs(self):
        st = _valid_trace()
        assert st.trace_id == "STRAT-P-TEST"

    def test_trace_id_format(self):
        st = _valid_trace(plan_id="SP-20260726-run001")
        assert st.trace_id == "STRAT-SP-20260726-run001"

    def test_plan_preserved(self):
        st = _valid_trace(plan_id="MY-PLAN")
        assert st.plan.plan_id == "MY-PLAN"

    def test_theories_preserved(self):
        st = _valid_trace(n=3)
        assert len(st.theories) == 3

    def test_evaluations_preserved(self):
        st = _valid_trace(n=2)
        assert len(st.evaluations) == 2

    def test_choice_sets_preserved(self):
        st = _valid_trace(n=3)
        assert len(st.choice_sets) == 3

    def test_selection_preserved(self):
        st = _valid_trace()
        assert st.selection.winner_theory_id == "TH-SCS-0"

    def test_strategic_position_preserved(self):
        st = _valid_trace()
        assert st.strategic_position.position_id == "SP-TEST"

    def test_immutable(self):
        st = _valid_trace()
        with pytest.raises(Exception):
            st.trace_id = "MUTATED"  # type: ignore[misc]

    def test_serialization_round_trip(self):
        st = _valid_trace(n=3, plan_id="P-ROUNDTRIP")
        d = st.to_dict()
        st2 = StrategyTrace.from_dict(d)
        assert st2.trace_id == st.trace_id
        assert st2.plan.plan_id == st.plan.plan_id
        assert len(st2.theories) == len(st.theories)
        assert len(st2.evaluations) == len(st.evaluations)
        assert len(st2.choice_sets) == len(st.choice_sets)
        assert st2.selection.winner_theory_id == st.selection.winner_theory_id

    def test_to_dict_returns_json_safe_types(self):
        st = _valid_trace()
        d = st.to_dict()
        import json
        # Must not raise — all values are JSON-serialisable
        json.dumps(d)


# ---------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------

class TestStrategyTraceValidationRules:

    def _base_kwargs(self) -> dict:
        """Minimal valid kwargs for StrategyTrace."""
        st = _valid_trace()
        return {
            "trace_id": st.trace_id,
            "created_at": st.created_at,
            "plan": st.plan,
            "choice_sets": list(st.choice_sets),
            "theories": list(st.theories),
            "evaluations": list(st.evaluations),
            "selection": st.selection,
            "strategic_position": st.strategic_position,
            "metadata": {},
        }

    # Rule 1: empty choice_sets
    def test_rule1_empty_choice_sets_rejected(self):
        kw = self._base_kwargs()
        kw["choice_sets"] = []
        with pytest.raises(ValidationError, match="choice_sets must not be empty"):
            StrategyTrace(**kw)

    # Rule 2: empty theories — choice_sets must stay non-empty so rule 1 passes first
    def test_rule2_empty_theories_rejected(self):
        kw = self._base_kwargs()
        kw["theories"] = []
        # choice_sets remains non-empty; rule 1 passes, rule 2 fires
        with pytest.raises(ValidationError, match="theories must not be empty"):
            StrategyTrace(**kw)

    # Rule 3: empty evaluations
    def test_rule3_empty_evaluations_rejected(self):
        kw = self._base_kwargs()
        kw["evaluations"] = []
        with pytest.raises(ValidationError, match="evaluations must not be empty"):
            StrategyTrace(**kw)

    # Rule 4: theory/evaluation count mismatch
    def test_rule4_count_mismatch_rejected(self):
        kw = self._base_kwargs()
        # add an extra evaluation
        extra_ev = _eval("TH-SCS-99")
        kw["evaluations"] = list(kw["evaluations"]) + [extra_ev]
        with pytest.raises(ValidationError, match="same count"):
            StrategyTrace(**kw)

    # Rule 5: theory/choice_set count mismatch
    def test_rule5_choice_set_mismatch_rejected(self):
        kw = self._base_kwargs()
        kw["choice_sets"] = [kw["choice_sets"][0]]  # one fewer set
        with pytest.raises(ValidationError, match="same count"):
            StrategyTrace(**kw)

    # Rule 6: duplicate theory_ids in theories
    def test_rule6_duplicate_theory_ids_rejected(self):
        kw = self._base_kwargs()
        t_dup = _theory("TH-SCS-0")  # duplicate of theories[0]
        kw["theories"] = [kw["theories"][0], t_dup, kw["theories"][2]]
        with pytest.raises(ValidationError, match="duplicate theory_id"):
            StrategyTrace(**kw)

    # Rule 7: duplicate theory_ids in evaluations
    def test_rule7_duplicate_eval_ids_rejected(self):
        kw = self._base_kwargs()
        ev_dup = _eval("TH-SCS-0")  # duplicate of evaluations[0]
        kw["evaluations"] = [kw["evaluations"][0], ev_dup, kw["evaluations"][2]]
        with pytest.raises(ValidationError, match="duplicate theory_id"):
            StrategyTrace(**kw)

    # Rule 8: evaluation theory_id has no matching theory
    def test_rule8_unresolved_eval_id_rejected(self):
        kw = self._base_kwargs()
        kw["evaluations"] = [
            kw["evaluations"][0],
            kw["evaluations"][1],
            _eval("TH-GHOST"),  # no theory with this ID
        ]
        with pytest.raises(ValidationError, match="no matching theory"):
            StrategyTrace(**kw)

    # Rule 9: winner_theory_id not in theories
    def test_rule9_winner_not_in_theories_rejected(self):
        kw = self._base_kwargs()
        kw["selection"] = _selection("TH-NONEXISTENT")
        with pytest.raises(ValidationError, match="winner_theory_id"):
            StrategyTrace(**kw)

    # Rule 10: position theory_id != winner_theory_id
    def test_rule10_position_winner_mismatch_rejected(self):
        kw = self._base_kwargs()
        # Build a position whose theory_of_winning.theory_id != winner
        wrong_theory = _theory("TH-SCS-1")  # exists but is not the winner
        kw["strategic_position"] = _position(wrong_theory)
        # selection.winner_theory_id is "TH-SCS-0" (from base_kwargs)
        with pytest.raises(ValidationError, match="does not match"):
            StrategyTrace(**kw)


# ---------------------------------------------------------------------------
# StrategyCoordinator — trace population
# ---------------------------------------------------------------------------

class TestStrategyCoordinatorTrace:
    def test_trace_is_none_before_build(self):
        coord = StrategyCoordinator()
        assert coord._trace is None

    def test_trace_populated_after_build(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        assert coord._trace is not None
        assert isinstance(coord._trace, StrategyTrace)

    def test_trace_id_format(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        assert coord._trace.trace_id.startswith("STRAT-")
        assert coord._trace.plan.plan_id in coord._trace.trace_id

    def test_trace_contains_theories(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        assert len(coord._trace.theories) == len(coord._theories)

    def test_trace_contains_evaluations(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        assert len(coord._trace.evaluations) == len(coord._evaluations)

    def test_trace_contains_choice_sets(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        assert len(coord._trace.choice_sets) == len(coord._choice_sets)

    def test_trace_selection_winner_matches(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        assert coord._trace.selection.winner_theory_id == coord._selection.winner_theory_id

    def test_trace_position_theory_is_winner(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        assert (
            coord._trace.strategic_position.theory_of_winning.theory_id
            == coord._trace.selection.winner_theory_id
        )

    def test_trace_is_immutable(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        with pytest.raises(Exception):
            coord._trace.trace_id = "MUTATED"  # type: ignore[misc]

    def test_trace_round_trip_serializable(self):
        import json
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        d = coord._trace.to_dict()
        json.dumps(d)  # must not raise


# ---------------------------------------------------------------------------
# pipeline_trace integration
# ---------------------------------------------------------------------------

class TestPipelineTraceIntegration:
    def _make_ctx_with_trace(self) -> object:
        """Build an AgentContext-like namespace with the strategy trace populated."""
        coord = StrategyCoordinator()
        ctx = _full_ctx()
        sp = coord.build(ctx)
        ctx.trace["_strategic_position"] = sp
        ctx.trace["_strategy_trace"] = coord._trace
        return ctx

    def test_strategy_trace_key_present_in_canonical_trace(self):
        from functional_agents.pipeline_trace import build_canonical_trace
        ctx = self._make_ctx_with_trace()
        result = build_canonical_trace(ctx)
        assert "strategy_trace" in result

    def test_strategy_trace_is_dict_when_populated(self):
        from functional_agents.pipeline_trace import build_canonical_trace
        ctx = self._make_ctx_with_trace()
        result = build_canonical_trace(ctx)
        assert isinstance(result["strategy_trace"], dict)

    def test_strategy_trace_contains_trace_id(self):
        from functional_agents.pipeline_trace import build_canonical_trace
        ctx = self._make_ctx_with_trace()
        result = build_canonical_trace(ctx)
        assert "trace_id" in result["strategy_trace"]
        assert result["strategy_trace"]["trace_id"].startswith("STRAT-")

    def test_strategy_trace_none_when_not_populated(self):
        from functional_agents.pipeline_trace import build_canonical_trace
        # Context without any strategy trace
        ctx = types.SimpleNamespace(
            trace={},
            agent_history=[],
            run_id=None, question=None, goal=None,
            profiles=[], execution_profile=None, engagement=None,
            deliverables=None, deliverable_bundle=None,
            executive_narrative=None,
        )
        result = build_canonical_trace(ctx)
        assert result["strategy_trace"] is None

    def test_strategy_key_still_present(self):
        from functional_agents.pipeline_trace import build_canonical_trace
        ctx = self._make_ctx_with_trace()
        result = build_canonical_trace(ctx)
        # Backward-compatible "strategy" summary key must still be present
        assert "strategy" in result
        assert result["strategy"] is not None
