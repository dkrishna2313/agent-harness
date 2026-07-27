"""PH11.2a — Lineage Identity Hardening tests.

Covers:
- TheoryOfWinning.source_choice_set_id is required and non-empty (no default)
- StrategyCoordinator research ID priority chain: ro["id"] → ro["research_id"] → run_id
- StrategyCoordinator raises when no valid identifier is available
- StrategyTrace rule 14 always fires (not gated on lineage)
- StrategyTrace rule 17 rejects "unknown" sentinel in lineage source/target IDs
- StrategyTrace rule 18 enforces metadata["research_id"] == research_object lineage source_id
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from functional_agents.context import AgentContext
from functional_agents.strategy import StrategyCoordinator, StrategyTrace
from functional_agents.strategy.strategy_lineage import (
    StrategyLineageLink,
    build_strategy_lineage,
)
from functional_agents.strategy.strategic_position import (
    StrategicExecution,
    StrategicJustification,
    StrategicPosition,
    StrategicRecommendation,
    TheoryOfWinning,
)
from functional_agents.strategy.strategic_choice import StrategicChoice
from functional_agents.strategy.strategic_choice_set import StrategicChoiceSet
from functional_agents.strategy.strategy_plan import StrategyPlan
from functional_agents.strategy.strategy_selector import StrategySelection
from functional_agents.strategy.theory_evaluation import CriterionScore, TheoryEvaluation


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _plan(pid: str = "P-TEST") -> StrategyPlan:
    return StrategyPlan(plan_id=pid, framework="executive", active_dimensions=[])


def _choice_set(sid: str) -> StrategicChoiceSet:
    ch = StrategicChoice(
        id=f"SC-{sid}", dimension="market", selected_value="OPT-A",
        rationale="r", confidence="High", supporting_assumptions=[], requiredness="optional",
    )
    return StrategicChoiceSet(
        id=sid, choices=[ch], overall_confidence="High",
        internal_conflicts=[], completeness=1.0, rationale="r",
    )


def _theory(tid: str, scid: str) -> TheoryOfWinning:
    return TheoryOfWinning(theory_id=tid, source_choice_set_id=scid)


def _eval(tid: str, score: float = 0.8) -> TheoryEvaluation:
    return TheoryEvaluation(
        theory_id=tid,
        criteria_scores={"x": CriterionScore(score=score, rationale="r", weight=1.0)},
        strengths=[], weaknesses=[], residual_risks=[],
        overall_score=score, confidence="High", metadata={},
    )


def _selection(winner: str, runner: str | None = None) -> StrategySelection:
    return StrategySelection(winner_theory_id=winner, winner_score=0.8,
                             runner_up_theory_id=runner)


def _position(theory: TheoryOfWinning) -> StrategicPosition:
    return StrategicPosition(
        position_id="SP-TEST", created_at="2026-07-26T00:00:00+00:00",
        theory_of_winning=theory,
        recommendation=StrategicRecommendation(
            recommended_option_id=theory.recommended_option_id,
            recommended_option_title="", board_recommendation="Go",
            decision_readiness="Ready", overall_confidence="High",
        ),
        justification=StrategicJustification(
            decision_analysis={}, strategic_options=[],
            assumptions=[], risks=[], opportunities=[],
        ),
        execution=StrategicExecution(recommendations=[], validation_priorities=[]),
    )


def _full_ctx(research_id: str = "R-TEST") -> AgentContext:
    return AgentContext(
        question="What should we do?", profiles=["test"], execution_profile="test",
        research_object={"id": research_id}, run_id="run001",
        strategic_options=[{
            "option_id": "OPT-A", "title": "Option A", "description": "First.",
            "strategic_objective": "Grow.", "expected_outcomes": ["O1"],
            "supporting_assumption_ids": [], "associated_risk_ids": [],
            "associated_opportunity_ids": [], "supporting_recommendation_ids": [],
            "advantages": ["Fast"], "disadvantages": ["Risky"],
            "implementation_complexity": "Low", "estimated_time_horizon": "Near-term",
            "capital_intensity": "Low", "confidence": "High",
            "recommended": True, "rationale": "Best.",
        }],
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


def _make_inputs(n: int = 2):
    """Return (plan, choice_sets, theories, evaluations, sel, pos, trace_id)."""
    plan = _plan()
    choice_sets = [_choice_set(f"SCS-{i}") for i in range(n)]
    theories = [_theory(f"TH-SCS-{i}", f"SCS-{i}") for i in range(n)]
    evaluations = [_eval(f"TH-SCS-{i}", 0.9 - i * 0.1) for i in range(n)]
    winner = theories[0]
    runner = theories[1].theory_id if n > 1 else None
    sel = _selection(winner.theory_id, runner)
    pos = _position(winner)
    trace_id = f"STRAT-{plan.plan_id}"
    return plan, choice_sets, theories, evaluations, sel, pos, trace_id


# ---------------------------------------------------------------------------
# TestSourceChoiceSetIdRequired
# ---------------------------------------------------------------------------

class TestSourceChoiceSetIdRequired:
    def test_construction_without_scid_raises(self):
        with pytest.raises(ValidationError):
            TheoryOfWinning(theory_id="TH-X")

    def test_construction_with_empty_scid_raises(self):
        with pytest.raises(ValidationError, match="source_choice_set_id"):
            TheoryOfWinning(theory_id="TH-X", source_choice_set_id="")

    def test_construction_with_whitespace_scid_raises(self):
        with pytest.raises(ValidationError, match="source_choice_set_id"):
            TheoryOfWinning(theory_id="TH-X", source_choice_set_id="   ")

    def test_construction_with_valid_scid_succeeds(self):
        t = TheoryOfWinning(theory_id="TH-X", source_choice_set_id="SCS-0")
        assert t.source_choice_set_id == "SCS-0"

    def test_serialization_round_trip_preserves_scid(self):
        t = TheoryOfWinning(theory_id="TH-X", source_choice_set_id="SCS-ABC")
        d = t.model_dump(mode="json")
        t2 = TheoryOfWinning.model_validate(d)
        assert t2.source_choice_set_id == "SCS-ABC"


# ---------------------------------------------------------------------------
# TestResearchIdentityResolution
# ---------------------------------------------------------------------------

class TestResearchIdentityResolution:
    def test_research_object_id_used_when_present(self):
        coord = StrategyCoordinator()
        ctx = _full_ctx(research_id="R-PRIMARY")
        coord.build(ctx)
        assert coord._trace.metadata["research_id"] == "R-PRIMARY"

    def test_research_object_research_id_field_fallback(self):
        coord = StrategyCoordinator()
        ctx = _full_ctx()
        ctx.research_object = {"research_id": "R-FALLBACK"}
        coord.build(ctx)
        assert coord._trace.metadata["research_id"] == "R-FALLBACK"

    def test_run_id_fallback_when_no_research_object(self):
        coord = StrategyCoordinator()
        ctx = _full_ctx()
        ctx.research_object = {}
        ctx.run_id = "run-xyz"
        coord.build(ctx)
        assert coord._trace.metadata["research_id"] == "run-xyz"

    def test_raises_when_no_valid_identifier(self):
        coord = StrategyCoordinator()
        ctx = _full_ctx()
        ctx.research_object = {}
        ctx.run_id = None  # type: ignore[assignment]
        with pytest.raises(ValueError, match="no valid research or run identifier"):
            coord.build(ctx)

    def test_error_message_includes_coordinator_name(self):
        coord = StrategyCoordinator()
        ctx = _full_ctx()
        ctx.research_object = {}
        ctx.run_id = None  # type: ignore[assignment]
        with pytest.raises(ValueError) as exc_info:
            coord.build(ctx)
        assert "StrategyCoordinator" in str(exc_info.value)


# ---------------------------------------------------------------------------
# TestStrategyTraceHardening
# ---------------------------------------------------------------------------

class TestStrategyTraceHardening:
    def test_rule14_fires_without_lineage(self):
        plan, choice_sets, theories, evaluations, sel, pos, trace_id = _make_inputs(n=2)
        # Replace first theory's scid with one that won't resolve
        bad_theory = TheoryOfWinning(
            theory_id=theories[0].theory_id,
            source_choice_set_id="SCS-NONEXISTENT",
        )
        theories = [bad_theory] + theories[1:]
        with pytest.raises(ValueError, match="not found in choice_sets"):
            StrategyTrace(
                trace_id=trace_id, created_at="2026-07-26T00:00:00+00:00",
                plan=plan, choice_sets=choice_sets, theories=theories,
                evaluations=evaluations, selection=sel, strategic_position=pos,
                lineage=[], metadata={},
            )

    def test_rule14_passes_with_valid_scid_no_lineage(self):
        plan, choice_sets, theories, evaluations, sel, pos, trace_id = _make_inputs(n=2)
        trace = StrategyTrace(
            trace_id=trace_id, created_at="2026-07-26T00:00:00+00:00",
            plan=plan, choice_sets=choice_sets, theories=theories,
            evaluations=evaluations, selection=sel, strategic_position=pos,
            lineage=[], metadata={},
        )
        assert len(trace.theories) == 2

    def test_rule17_unknown_source_id_rejected(self):
        plan, choice_sets, theories, evaluations, sel, pos, trace_id = _make_inputs(n=2)
        lineage = build_strategy_lineage(
            research_id="R-TEST", plan=plan, choice_sets=choice_sets,
            theories=theories, evaluations=evaluations,
            selection=sel, strategic_position=pos, trace_id=trace_id,
        )
        bad_link = StrategyLineageLink(
            source_type="research_object",
            source_id="unknown",
            target_type="strategy_plan",
            target_id=plan.plan_id,
            relationship="informs",
        )
        with pytest.raises(ValueError, match="unknown"):
            StrategyTrace(
                trace_id=trace_id, created_at="2026-07-26T00:00:00+00:00",
                plan=plan, choice_sets=choice_sets, theories=theories,
                evaluations=evaluations, selection=sel, strategic_position=pos,
                lineage=list(lineage) + [bad_link],
                metadata={"research_id": "R-TEST"},
            )

    def test_rule17_unknown_target_id_rejected(self):
        plan, choice_sets, theories, evaluations, sel, pos, trace_id = _make_inputs(n=2)
        lineage = build_strategy_lineage(
            research_id="R-TEST", plan=plan, choice_sets=choice_sets,
            theories=theories, evaluations=evaluations,
            selection=sel, strategic_position=pos, trace_id=trace_id,
        )
        bad_link = StrategyLineageLink(
            source_type="strategy_plan",
            source_id=plan.plan_id,
            target_type="strategy_trace",
            target_id="unknown",
            relationship="captured_in",
        )
        with pytest.raises(ValueError, match="unknown"):
            StrategyTrace(
                trace_id=trace_id, created_at="2026-07-26T00:00:00+00:00",
                plan=plan, choice_sets=choice_sets, theories=theories,
                evaluations=evaluations, selection=sel, strategic_position=pos,
                lineage=list(lineage) + [bad_link],
                metadata={"research_id": "R-TEST"},
            )

    def test_rule18_metadata_research_id_mismatch_rejected(self):
        plan, choice_sets, theories, evaluations, sel, pos, trace_id = _make_inputs(n=2)
        lineage = build_strategy_lineage(
            research_id="R-REAL",
            plan=plan, choice_sets=choice_sets, theories=theories,
            evaluations=evaluations, selection=sel, strategic_position=pos,
            trace_id=trace_id,
        )
        with pytest.raises(ValueError, match="research_id"):
            StrategyTrace(
                trace_id=trace_id, created_at="2026-07-26T00:00:00+00:00",
                plan=plan, choice_sets=choice_sets, theories=theories,
                evaluations=evaluations, selection=sel, strategic_position=pos,
                lineage=lineage,
                metadata={"research_id": "R-WRONG"},
            )

    def test_rule18_metadata_research_id_match_passes(self):
        plan, choice_sets, theories, evaluations, sel, pos, trace_id = _make_inputs(n=2)
        lineage = build_strategy_lineage(
            research_id="R-CORRECT",
            plan=plan, choice_sets=choice_sets, theories=theories,
            evaluations=evaluations, selection=sel, strategic_position=pos,
            trace_id=trace_id,
        )
        trace = StrategyTrace(
            trace_id=trace_id, created_at="2026-07-26T00:00:00+00:00",
            plan=plan, choice_sets=choice_sets, theories=theories,
            evaluations=evaluations, selection=sel, strategic_position=pos,
            lineage=lineage,
            metadata={"research_id": "R-CORRECT"},
        )
        assert trace.metadata["research_id"] == "R-CORRECT"


# ---------------------------------------------------------------------------
# TestCoordinatorLineageConsistency
# ---------------------------------------------------------------------------

class TestCoordinatorLineageConsistency:
    def test_metadata_research_id_matches_lineage_source(self):
        coord = StrategyCoordinator()
        ctx = _full_ctx(research_id="R-CHECK")
        coord.build(ctx)
        trace = coord._trace
        ro_links = [lk for lk in trace.lineage if lk.source_type == "research_object"]
        assert ro_links, "Expected at least one research_object lineage link"
        assert trace.metadata["research_id"] == ro_links[0].source_id

    def test_no_unknown_in_lineage(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        for link in coord._trace.lineage:
            assert link.source_id != "unknown", f"'unknown' in source_id: {link}"
            assert link.target_id != "unknown", f"'unknown' in target_id: {link}"
