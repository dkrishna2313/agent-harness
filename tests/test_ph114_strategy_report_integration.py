"""PH11.4 — Strategy Report Integration tests.

Covers:
- build_strategy_narrative: field mapping from StrategyTrace
- build_strategy_narrative: evaluation criteria and criterion scores
- build_strategy_narrative: alternatives sorted by score descending
- build_strategy_narrative: assumption/failure_mode extraction from list[dict]
- build_strategy_narrative: single-theory trace (no runner-up)
- EditorialCoordinator.build() with strategy_trace → brief.strategy_narrative populated
- EditorialCoordinator.build() without strategy_trace → brief.strategy_narrative is None
- EditorialBrief.to_dict() serializes strategy_narrative as dict (not Pydantic model)
- EditorialBrief.to_dict() serializes strategy_narrative=None as None
- EditorialManuscript.strategic_direction scaffold always created by build_manuscript()
- StrategyWriter: populates strategic_direction when strategy_narrative present
- StrategyWriter: no-ops when strategy_narrative is None (no failure)
- StrategyWriter: bullet_groups layout (4 groups: criteria, assumptions, conditions, failures)
- StrategyWriter: alternatives table in tables[0]
- run_writers(): StrategyWriter registered, completeness check skips optional sections
- run_writers(): existing 7 required sections still pass completeness check
- MarkdownRenderer: renders ## Strategic Direction when section populated
- MarkdownRenderer: omits ## Strategic Direction when section absent/empty
- MarkdownRenderer: renders score table from brief.strategy_narrative
- MarkdownRenderer: renders subsections (Recommended Strategy, Why, Alternatives, etc.)
- Missing-trace: full pipeline with no StrategyTrace produces identical report structure
- Missing-trace: run_writers does not raise when strategic_direction is empty
- Missing-trace: rendered report contains no empty ## Strategic Direction heading
- EditorialBrief backward compat: all 11 original fields still accessible
- StrategyManuscriptSection is a proper ManuscriptSection subclass
- StrategyWriter.optional == True
- EditorialWriter.optional default == False
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from functional_agents.editorial import (
    EditorialCoordinator,
    EditorialManuscript,
    MarkdownRenderer,
    StrategyAlternativeSummary,
    StrategyManuscriptSection,
    StrategyNarrative,
    StrategyWriter,
    build_strategy_narrative,
)
from functional_agents.editorial.editorial_writer import EditorialWriter
from functional_agents.editorial.editorial_manuscript import ManuscriptSection
from functional_agents.strategy import (
    StrategicChoiceSet,
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
from functional_agents.context import AgentContext


# ---------------------------------------------------------------------------
# Shared fixtures
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


def _theory(
    tid: str,
    oid: str = "OPT-A",
    scid: str = "SCS-0",
    title: str = "",
    winning_position: str = "",
    winning_mechanism: str = "",
    assumptions: list | None = None,
    success_conditions: list | None = None,
    failure_modes: list | None = None,
) -> TheoryOfWinning:
    return TheoryOfWinning(
        theory_id=tid,
        recommended_option_id=oid,
        recommended_option_title=title or f"Option {oid}",
        source_choice_set_id=scid,
        winning_position=winning_position,
        winning_mechanism=winning_mechanism,
        assumptions=assumptions or [],
        success_conditions=success_conditions or [],
        failure_modes=failure_modes or [],
    )


def _eval(
    tid: str,
    score: float = 0.8,
    confidence: str = "High",
    criteria: dict | None = None,
    strengths: list | None = None,
    weaknesses: list | None = None,
) -> TheoryEvaluation:
    cs = criteria or {"market_fit": CriterionScore(score=score, rationale="r", weight=1.0)}
    return TheoryEvaluation(
        theory_id=tid,
        criteria_scores=cs,
        strengths=strengths or [],
        weaknesses=weaknesses or [],
        residual_risks=[],
        overall_score=score,
        confidence=confidence,
        metadata={},
    )


def _selection(
    winner: str,
    winner_score: float = 0.8,
    runner_up: str | None = None,
    runner_up_score: float | None = None,
) -> StrategySelection:
    score_margin = None
    if runner_up_score is not None:
        score_margin = round(winner_score - runner_up_score, 6)
    return StrategySelection(
        winner_theory_id=winner,
        winner_score=winner_score,
        runner_up_theory_id=runner_up,
        runner_up_score=runner_up_score,
        score_margin=score_margin,
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


def _trace_with_n_theories(
    n: int = 3,
    *,
    winning_position: str = "We win by owning the premium segment.",
    winning_mechanism: str = "Through superior unit economics and brand loyalty.",
    assumptions: list | None = None,
    success_conditions: list | None = None,
    failure_modes: list | None = None,
) -> StrategyTrace:
    plan = _plan()
    choice_sets = [_choice_set(f"SCS-{i}") for i in range(n)]
    theories = [
        _theory(
            f"TH-SCS-{i}", scid=f"SCS-{i}",
            winning_position=winning_position if i == 0 else "",
            winning_mechanism=winning_mechanism if i == 0 else "",
            assumptions=assumptions or [],
            success_conditions=success_conditions or [],
            failure_modes=failure_modes or [],
        )
        for i in range(n)
    ]
    evaluations = [_eval(f"TH-SCS-{i}", score=round(0.9 - i * 0.1, 2)) for i in range(n)]
    winner = theories[0]
    runner_up_id = theories[1].theory_id if n > 1 else None
    runner_up_score = evaluations[1].overall_score if n > 1 else None
    sel = _selection(winner.theory_id, 0.9, runner_up_id, runner_up_score)
    pos = _position(winner)
    return StrategyTrace(
        trace_id="STRAT-P-TEST",
        created_at="2026-07-26T00:00:00+00:00",
        plan=plan,
        choice_sets=choice_sets,
        theories=theories,
        evaluations=evaluations,
        selection=sel,
        strategic_position=pos,
        metadata={"framework": "executive", "research_id": ""},
    )


def _full_ctx() -> AgentContext:
    return AgentContext(
        question="What should we do?",
        profiles=["test"],
        execution_profile="test",
        research_object={"id": "R-TEST"},
        run_id="run001",
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


# ---------------------------------------------------------------------------
# Section 1: build_strategy_narrative — field mapping
# ---------------------------------------------------------------------------

class TestBuildStrategyNarrativeFieldMapping:
    def test_trace_id_and_framework_copied(self):
        trace = _trace_with_n_theories(2)
        sn = build_strategy_narrative(trace)
        assert sn.trace_id == "STRAT-P-TEST"
        assert sn.framework == "executive"

    def test_winner_theory_id_matches_selection(self):
        trace = _trace_with_n_theories(2)
        sn = build_strategy_narrative(trace)
        assert sn.winner_theory_id == trace.selection.winner_theory_id

    def test_winner_score_from_selection(self):
        trace = _trace_with_n_theories(2)
        sn = build_strategy_narrative(trace)
        assert sn.winner_score == trace.selection.winner_score

    def test_winning_position_and_mechanism_from_theory(self):
        trace = _trace_with_n_theories(
            2,
            winning_position="Premium segment capture.",
            winning_mechanism="Superior unit economics.",
        )
        sn = build_strategy_narrative(trace)
        assert sn.winning_position == "Premium segment capture."
        assert sn.winning_mechanism == "Superior unit economics."

    def test_overall_confidence_from_winner_evaluation(self):
        plan = _plan()
        cs = [_choice_set("SCS-0")]
        theory = _theory("TH-0", scid="SCS-0")
        ev = _eval("TH-0", score=0.85, confidence="Medium")
        sel = _selection("TH-0", winner_score=0.85)
        pos = _position(theory)
        trace = StrategyTrace(
            trace_id="STRAT-P-TEST", created_at="2026-07-26T00:00:00+00:00",
            plan=plan, choice_sets=cs, theories=[theory], evaluations=[ev],
            selection=sel, strategic_position=pos, metadata={},
        )
        sn = build_strategy_narrative(trace)
        assert sn.overall_confidence == "Medium"

    def test_runner_up_fields_when_present(self):
        trace = _trace_with_n_theories(3)
        sn = build_strategy_narrative(trace)
        assert sn.runner_up_theory_id == "TH-SCS-1"
        assert sn.runner_up_score is not None
        assert sn.score_margin is not None

    def test_runner_up_fields_none_for_single_theory(self):
        plan = _plan()
        cs = [_choice_set("SCS-0")]
        theory = _theory("TH-0", scid="SCS-0")
        ev = _eval("TH-0")
        sel = _selection("TH-0")
        pos = _position(theory)
        trace = StrategyTrace(
            trace_id="STRAT-P-TEST", created_at="2026-07-26T00:00:00+00:00",
            plan=plan, choice_sets=cs, theories=[theory], evaluations=[ev],
            selection=sel, strategic_position=pos, metadata={},
        )
        sn = build_strategy_narrative(trace)
        assert sn.runner_up_theory_id is None
        assert sn.runner_up_score is None
        assert sn.score_margin is None

    def test_strategic_position_id_copied(self):
        trace = _trace_with_n_theories(2)
        sn = build_strategy_narrative(trace)
        assert sn.strategic_position_id == "SP-TEST"


# ---------------------------------------------------------------------------
# Section 2: build_strategy_narrative — evaluation criteria
# ---------------------------------------------------------------------------

class TestBuildStrategyNarrativeEvaluationCriteria:
    def _trace_with_criteria(self, criteria: dict) -> StrategyTrace:
        plan = _plan()
        cs = [_choice_set("SCS-0")]
        theory = _theory("TH-0", scid="SCS-0")
        ev = TheoryEvaluation(
            theory_id="TH-0",
            criteria_scores=criteria,
            overall_score=0.8,
            confidence="High",
        )
        sel = _selection("TH-0")
        pos = _position(theory)
        return StrategyTrace(
            trace_id="STRAT-P-TEST", created_at="2026-07-26T00:00:00+00:00",
            plan=plan, choice_sets=cs, theories=[theory], evaluations=[ev],
            selection=sel, strategic_position=pos, metadata={},
        )

    def test_evaluation_criteria_names_extracted(self):
        criteria = {
            "market_fit": CriterionScore(score=0.9, rationale="r", weight=1.0),
            "execution_risk": CriterionScore(score=0.7, rationale="r", weight=1.0),
        }
        sn = build_strategy_narrative(self._trace_with_criteria(criteria))
        assert set(sn.evaluation_criteria) == {"market_fit", "execution_risk"}

    def test_criterion_scores_correct(self):
        criteria = {
            "market_fit": CriterionScore(score=0.9, rationale="r", weight=1.0),
            "execution_risk": CriterionScore(score=0.7, rationale="r", weight=1.0),
        }
        sn = build_strategy_narrative(self._trace_with_criteria(criteria))
        assert abs(sn.criterion_scores["market_fit"] - 0.9) < 1e-6
        assert abs(sn.criterion_scores["execution_risk"] - 0.7) < 1e-6

    def test_empty_criteria_scores_ok(self):
        sn = build_strategy_narrative(self._trace_with_criteria({}))
        assert sn.evaluation_criteria == []
        assert sn.criterion_scores == {}


# ---------------------------------------------------------------------------
# Section 3: build_strategy_narrative — assumptions and failure modes
# ---------------------------------------------------------------------------

class TestBuildStrategyNarrativeAssumptionExtraction:
    def test_assumptions_extracted_from_dict_list(self):
        assumptions = [
            {"statement": "Policy environment remains stable."},
            {"statement": "Battery cost continues to decline."},
        ]
        trace = _trace_with_n_theories(1, assumptions=assumptions)
        sn = build_strategy_narrative(trace)
        assert "Policy environment remains stable." in sn.assumptions
        assert "Battery cost continues to decline." in sn.assumptions

    def test_failure_modes_extracted_from_dict_list(self):
        failure_modes = [
            {"description": "Regulatory reversal disrupts supply chain."},
            {"description": "Competitor cost breakthrough erodes margin."},
        ]
        trace = _trace_with_n_theories(1, failure_modes=failure_modes)
        sn = build_strategy_narrative(trace)
        assert "Regulatory reversal disrupts supply chain." in sn.failure_modes
        assert "Competitor cost breakthrough erodes margin." in sn.failure_modes

    def test_success_conditions_from_list_of_strings(self):
        conditions = ["Technology readiness by 2027.", "Policy support maintained."]
        trace = _trace_with_n_theories(1, success_conditions=conditions)
        sn = build_strategy_narrative(trace)
        assert sn.success_conditions == conditions

    def test_empty_assumptions_success_conditions_failure_modes(self):
        trace = _trace_with_n_theories(1)
        sn = build_strategy_narrative(trace)
        assert sn.assumptions == []
        assert sn.success_conditions == []
        assert sn.failure_modes == []


# ---------------------------------------------------------------------------
# Section 4: build_strategy_narrative — alternatives
# ---------------------------------------------------------------------------

class TestBuildStrategyNarrativeAlternatives:
    def test_alternatives_exclude_winner(self):
        trace = _trace_with_n_theories(3)
        sn = build_strategy_narrative(trace)
        winner_id = trace.selection.winner_theory_id
        assert all(a.theory_id != winner_id for a in sn.alternatives)

    def test_alternatives_count_equals_n_minus_1(self):
        trace = _trace_with_n_theories(3)
        sn = build_strategy_narrative(trace)
        assert len(sn.alternatives) == 2

    def test_alternatives_sorted_by_score_descending(self):
        trace = _trace_with_n_theories(3)
        sn = build_strategy_narrative(trace)
        scores = [a.score for a in sn.alternatives]
        assert scores == sorted(scores, reverse=True)

    def test_alternatives_carry_strengths_weaknesses(self):
        plan = _plan()
        cs = [_choice_set("SCS-0"), _choice_set("SCS-1")]
        t0 = _theory("TH-0", scid="SCS-0")
        t1 = _theory("TH-1", scid="SCS-1")
        ev0 = _eval("TH-0", score=0.8, strengths=["Fast"], weaknesses=["Expensive"])
        ev1 = _eval("TH-1", score=0.6, strengths=["Cheap"], weaknesses=["Slow"])
        sel = _selection("TH-0", winner_score=0.8, runner_up="TH-1", runner_up_score=0.6)
        pos = _position(t0)
        trace = StrategyTrace(
            trace_id="STRAT-P-TEST", created_at="2026-07-26T00:00:00+00:00",
            plan=plan, choice_sets=cs, theories=[t0, t1], evaluations=[ev0, ev1],
            selection=sel, strategic_position=pos, metadata={},
        )
        sn = build_strategy_narrative(trace)
        alt = next(a for a in sn.alternatives if a.theory_id == "TH-1")
        assert "Cheap" in alt.strengths
        assert "Slow" in alt.weaknesses

    def test_single_theory_has_no_alternatives(self):
        plan = _plan()
        cs = [_choice_set("SCS-0")]
        theory = _theory("TH-0", scid="SCS-0")
        ev = _eval("TH-0")
        sel = _selection("TH-0")
        pos = _position(theory)
        trace = StrategyTrace(
            trace_id="STRAT-P-TEST", created_at="2026-07-26T00:00:00+00:00",
            plan=plan, choice_sets=cs, theories=[theory], evaluations=[ev],
            selection=sel, strategic_position=pos, metadata={},
        )
        sn = build_strategy_narrative(trace)
        assert sn.alternatives == []


# ---------------------------------------------------------------------------
# Section 5: EditorialCoordinator.build() integration
# ---------------------------------------------------------------------------

class TestEditorialCoordinatorBuildWithTrace:
    def test_build_without_strategy_trace_has_none_narrative(self):
        ctx = _full_ctx()
        brief = EditorialCoordinator().build(ctx)
        assert brief.strategy_narrative is None

    def test_build_with_strategy_trace_has_narrative(self):
        ctx = _full_ctx()
        trace = _trace_with_n_theories(2)
        brief = EditorialCoordinator().build(ctx, strategy_trace=trace)
        assert brief.strategy_narrative is not None

    def test_build_with_strategy_trace_narrative_is_strategy_narrative(self):
        ctx = _full_ctx()
        trace = _trace_with_n_theories(2)
        brief = EditorialCoordinator().build(ctx, strategy_trace=trace)
        assert isinstance(brief.strategy_narrative, StrategyNarrative)

    def test_build_with_strategy_trace_narrative_has_correct_trace_id(self):
        ctx = _full_ctx()
        trace = _trace_with_n_theories(2)
        brief = EditorialCoordinator().build(ctx, strategy_trace=trace)
        assert brief.strategy_narrative.trace_id == trace.trace_id

    def test_build_original_fields_unaffected_by_strategy_trace(self):
        ctx = _full_ctx()
        trace = _trace_with_n_theories(2)
        brief_with = EditorialCoordinator().build(ctx, strategy_trace=trace)
        brief_without = EditorialCoordinator().build(ctx)
        # All original brief fields should have the same types
        for field_name in [
            "metadata", "executive_summary", "decision_analysis", "strategic_options",
            "recommendations", "strategic_assumptions", "strategic_risks",
            "strategic_opportunities", "executive_confidence", "validation_priorities",
            "appendix",
        ]:
            assert type(getattr(brief_with, field_name)) == type(getattr(brief_without, field_name))

    def test_build_backward_compat_no_strategy_trace_kwarg(self):
        ctx = _full_ctx()
        # build() with positional-only arg, no strategy_trace — must not raise
        brief = EditorialCoordinator().build(ctx)
        assert brief.strategy_narrative is None


# ---------------------------------------------------------------------------
# Section 6: EditorialBrief serialization
# ---------------------------------------------------------------------------

class TestEditorialBriefSerialization:
    def test_to_dict_without_narrative_has_none_key(self):
        ctx = _full_ctx()
        brief = EditorialCoordinator().build(ctx)
        d = brief.to_dict()
        assert "strategy_narrative" in d
        assert d["strategy_narrative"] is None

    def test_to_dict_with_narrative_is_dict_not_pydantic(self):
        ctx = _full_ctx()
        trace = _trace_with_n_theories(2)
        brief = EditorialCoordinator().build(ctx, strategy_trace=trace)
        d = brief.to_dict()
        assert isinstance(d["strategy_narrative"], dict)

    def test_to_dict_narrative_dict_has_trace_id(self):
        ctx = _full_ctx()
        trace = _trace_with_n_theories(2)
        brief = EditorialCoordinator().build(ctx, strategy_trace=trace)
        d = brief.to_dict()
        assert d["strategy_narrative"]["trace_id"] == trace.trace_id

    def test_to_dict_without_narrative_all_original_keys_present(self):
        ctx = _full_ctx()
        brief = EditorialCoordinator().build(ctx)
        d = brief.to_dict()
        for key in [
            "metadata", "executive_summary", "decision_analysis", "strategic_options",
            "recommendations", "strategic_assumptions", "strategic_risks",
            "strategic_opportunities", "executive_confidence", "validation_priorities",
            "appendix",
        ]:
            assert key in d


# ---------------------------------------------------------------------------
# Section 7: EditorialManuscript.strategic_direction scaffold
# ---------------------------------------------------------------------------

class TestEditorialManuscriptStrategicDirectionScaffold:
    def test_strategic_direction_present_after_build_manuscript(self):
        ctx = _full_ctx()
        brief = EditorialCoordinator().build(ctx)
        manuscript = EditorialCoordinator().build_manuscript(brief)
        assert hasattr(manuscript, "strategic_direction")
        assert manuscript.strategic_direction is not None

    def test_strategic_direction_is_strategy_manuscript_section(self):
        ctx = _full_ctx()
        brief = EditorialCoordinator().build(ctx)
        manuscript = EditorialCoordinator().build_manuscript(brief)
        assert isinstance(manuscript.strategic_direction, StrategyManuscriptSection)

    def test_strategic_direction_is_manuscript_section_subclass(self):
        assert issubclass(StrategyManuscriptSection, ManuscriptSection)

    def test_strategic_direction_title_is_strategic_direction(self):
        ctx = _full_ctx()
        brief = EditorialCoordinator().build(ctx)
        manuscript = EditorialCoordinator().build_manuscript(brief)
        assert manuscript.strategic_direction.title == "Strategic Direction"

    def test_strategic_direction_scaffold_empty_paragraphs(self):
        ctx = _full_ctx()
        brief = EditorialCoordinator().build(ctx)
        manuscript = EditorialCoordinator().build_manuscript(brief)
        assert manuscript.strategic_direction.paragraphs == []

    def test_to_dict_includes_strategic_direction_key(self):
        ctx = _full_ctx()
        brief = EditorialCoordinator().build(ctx)
        manuscript = EditorialCoordinator().build_manuscript(brief)
        d = manuscript.to_dict()
        assert "strategic_direction" in d
        assert d["strategic_direction"] is not None


# ---------------------------------------------------------------------------
# Section 8: StrategyWriter behavior
# ---------------------------------------------------------------------------

class TestStrategyWriter:
    def _make_manuscript(self, brief) -> EditorialManuscript:
        return EditorialCoordinator().build_manuscript(brief)

    def test_optional_flag_is_true(self):
        assert StrategyWriter.optional is True

    def test_section_name_is_strategic_direction(self):
        assert StrategyWriter.section_name == "strategic_direction"

    def test_write_noop_when_strategy_narrative_none(self):
        ctx = _full_ctx()
        brief = EditorialCoordinator().build(ctx)  # no trace
        manuscript = self._make_manuscript(brief)
        writer = StrategyWriter()
        result = writer.write(brief, manuscript)
        assert result is manuscript
        assert manuscript.strategic_direction.paragraphs == []

    def test_write_populates_paragraphs_when_narrative_present(self):
        ctx = _full_ctx()
        trace = _trace_with_n_theories(
            2,
            winning_position="We own the premium segment.",
            winning_mechanism="Via brand loyalty and unit economics.",
        )
        brief = EditorialCoordinator().build(ctx, strategy_trace=trace)
        manuscript = self._make_manuscript(brief)
        writer = StrategyWriter()
        writer.write(brief, manuscript)
        assert "We own the premium segment." in manuscript.strategic_direction.paragraphs
        assert "Via brand loyalty and unit economics." in manuscript.strategic_direction.paragraphs

    def test_write_five_bullet_groups(self):
        ctx = _full_ctx()
        trace = _trace_with_n_theories(2)
        brief = EditorialCoordinator().build(ctx, strategy_trace=trace)
        manuscript = self._make_manuscript(brief)
        StrategyWriter().write(brief, manuscript)
        # groups: [0] criteria+strengths, [1] assumptions, [2] conditions, [3] failures, [4] choices
        assert len(manuscript.strategic_direction.bullet_groups) == 5

    def test_write_criteria_bullets_in_group_0(self):
        plan = _plan()
        cs = [_choice_set("SCS-0")]
        theory = _theory("TH-0", scid="SCS-0")
        criteria = {"market_fit": CriterionScore(score=0.9, rationale="r", weight=1.0)}
        ev = TheoryEvaluation(
            theory_id="TH-0", criteria_scores=criteria,
            overall_score=0.9, confidence="High",
        )
        sel = _selection("TH-0", winner_score=0.9)
        pos = _position(theory)
        trace = StrategyTrace(
            trace_id="STRAT-P-TEST", created_at="2026-07-26T00:00:00+00:00",
            plan=plan, choice_sets=cs, theories=[theory], evaluations=[ev],
            selection=sel, strategic_position=pos, metadata={},
        )
        ctx = _full_ctx()
        brief = EditorialCoordinator().build(ctx, strategy_trace=trace)
        manuscript = self._make_manuscript(brief)
        StrategyWriter().write(brief, manuscript)
        bg0 = manuscript.strategic_direction.bullet_groups[0]
        assert any("market_fit" in b for b in bg0)

    def test_write_alternatives_table_when_alternatives_present(self):
        ctx = _full_ctx()
        trace = _trace_with_n_theories(3)
        brief = EditorialCoordinator().build(ctx, strategy_trace=trace)
        manuscript = self._make_manuscript(brief)
        StrategyWriter().write(brief, manuscript)
        assert len(manuscript.strategic_direction.tables) >= 1
        table = manuscript.strategic_direction.tables[0]
        assert table["headers"] == ["Theory ID", "Option", "Score", "Confidence", "Key Weaknesses"]

    def test_write_returns_manuscript(self):
        ctx = _full_ctx()
        trace = _trace_with_n_theories(2)
        brief = EditorialCoordinator().build(ctx, strategy_trace=trace)
        manuscript = self._make_manuscript(brief)
        result = StrategyWriter().write(brief, manuscript)
        assert result is manuscript


# ---------------------------------------------------------------------------
# Section 9: run_writers completeness contract
# ---------------------------------------------------------------------------

class TestRunWritersCompleteness:
    def test_run_writers_without_trace_does_not_raise(self):
        ctx = _full_ctx()
        coord = EditorialCoordinator()
        brief = coord.build(ctx)
        manuscript = coord.build_manuscript(brief)
        result = coord.run_writers(brief, manuscript)  # must not raise
        assert result is manuscript

    def test_run_writers_with_trace_does_not_raise(self):
        ctx = _full_ctx()
        trace = _trace_with_n_theories(2)
        coord = EditorialCoordinator()
        brief = coord.build(ctx, strategy_trace=trace)
        manuscript = coord.build_manuscript(brief)
        result = coord.run_writers(brief, manuscript)
        assert result is manuscript

    def test_run_writers_without_trace_strategic_direction_empty(self):
        ctx = _full_ctx()
        coord = EditorialCoordinator()
        brief = coord.build(ctx)
        manuscript = coord.build_manuscript(brief)
        coord.run_writers(brief, manuscript)
        sec = manuscript.strategic_direction
        assert not sec.paragraphs and not sec.tables

    def test_run_writers_with_trace_strategic_direction_populated(self):
        ctx = _full_ctx()
        trace = _trace_with_n_theories(
            2,
            winning_position="Premium segment capture.",
            winning_mechanism="Brand loyalty creates moat.",
        )
        coord = EditorialCoordinator()
        brief = coord.build(ctx, strategy_trace=trace)
        manuscript = coord.build_manuscript(brief)
        coord.run_writers(brief, manuscript)
        sec = manuscript.strategic_direction
        assert sec.paragraphs  # non-empty after writer runs

    def test_run_writers_existing_7_sections_still_populated(self):
        ctx = _full_ctx()
        coord = EditorialCoordinator()
        brief = coord.build(ctx)
        manuscript = coord.build_manuscript(brief)
        coord.run_writers(brief, manuscript)
        for attr in [
            "executive_summary", "decision_analysis", "recommendations",
            "strategic_risks", "strategic_opportunities", "executive_confidence", "appendix",
        ]:
            sec = getattr(manuscript, attr)
            assert sec.paragraphs or sec.tables, f"{attr} not populated"


# ---------------------------------------------------------------------------
# Section 10: MarkdownRenderer
# ---------------------------------------------------------------------------

class TestMarkdownRendererStrategicDirection:
    def _render_with_trace(self) -> tuple[str, Any]:
        ctx = _full_ctx()
        trace = _trace_with_n_theories(
            3,
            winning_position="We win through premium positioning.",
            winning_mechanism="Superior brand loyalty and unit economics.",
            assumptions=[{"statement": "Market remains premium-oriented."}],
            success_conditions=["Policy support maintained for 3+ years."],
            failure_modes=[{"description": "Competitor enters at lower price point."}],
        )
        coord = EditorialCoordinator()
        brief = coord.build(ctx, strategy_trace=trace)
        manuscript = coord.build_manuscript(brief)
        coord.run_writers(brief, manuscript)
        md = MarkdownRenderer().render(manuscript, brief=brief)
        return md, brief

    def _render_without_trace(self) -> str:
        ctx = _full_ctx()
        coord = EditorialCoordinator()
        brief = coord.build(ctx)
        manuscript = coord.build_manuscript(brief)
        coord.run_writers(brief, manuscript)
        return MarkdownRenderer().render(manuscript, brief=brief)

    def test_strategic_direction_heading_present_with_trace(self):
        md, _ = self._render_with_trace()
        assert "## Strategic Direction" in md

    def test_strategic_direction_heading_absent_without_trace(self):
        md = self._render_without_trace()
        assert "## Strategic Direction" not in md

    def test_recommended_strategy_subsection_present_with_trace(self):
        md, _ = self._render_with_trace()
        assert "### Recommended Strategy" in md

    def test_why_this_strategy_won_subsection_present(self):
        md, _ = self._render_with_trace()
        assert "### Why This Strategy Won" in md

    def test_alternatives_considered_subsection_present(self):
        md, _ = self._render_with_trace()
        assert "### Alternatives Considered" in md

    def test_assumptions_subsection_present(self):
        md, _ = self._render_with_trace()
        assert "### Assumptions and Conditions for Success" in md

    def test_risks_and_failure_modes_subsection_present(self):
        md, _ = self._render_with_trace()
        assert "### Risks and Failure Modes" in md

    def test_winning_position_in_recommended_strategy(self):
        md, _ = self._render_with_trace()
        assert "We win through premium positioning." in md

    def test_assumptions_bullet_rendered(self):
        md, _ = self._render_with_trace()
        assert "Market remains premium-oriented." in md

    def test_failure_mode_bullet_rendered(self):
        md, _ = self._render_with_trace()
        assert "Competitor enters at lower price point." in md

    def test_score_table_rendered_from_narrative(self):
        md, _ = self._render_with_trace()
        assert "Winner Score" in md or "Score" in md

    def test_no_empty_strategic_direction_heading_without_trace(self):
        md = self._render_without_trace()
        # No section heading should be followed immediately by end or another heading
        lines = md.splitlines()
        heading_lines = [i for i, l in enumerate(lines) if l.strip() == "## Strategic Direction"]
        assert len(heading_lines) == 0

    def test_render_returns_string(self):
        md, _ = self._render_with_trace()
        assert isinstance(md, str)
        assert len(md) > 0

    def test_existing_sections_still_present_with_trace(self):
        md, _ = self._render_with_trace()
        for heading in [
            "## 1. Executive Summary",
            "## 2. Strategic Context",
            "## 3. Strategic Recommendation",
        ]:
            assert heading in md, f"Missing: {heading}"


# ---------------------------------------------------------------------------
# Section 11: Missing-trace compatibility (full pipeline)
# ---------------------------------------------------------------------------

class TestMissingTraceCompatibility:
    def test_brief_has_no_strategy_narrative(self):
        ctx = _full_ctx()
        brief = EditorialCoordinator().build(ctx)
        assert brief.strategy_narrative is None

    def test_manuscript_strategic_direction_scaffold_present(self):
        ctx = _full_ctx()
        coord = EditorialCoordinator()
        brief = coord.build(ctx)
        manuscript = coord.build_manuscript(brief)
        assert manuscript.strategic_direction is not None

    def test_full_pipeline_without_trace_no_exception(self):
        ctx = _full_ctx()
        coord = EditorialCoordinator()
        brief = coord.build(ctx)
        manuscript = coord.build_manuscript(brief)
        manuscript = coord.run_writers(brief, manuscript)
        md = MarkdownRenderer().render(manuscript, brief=brief)
        assert isinstance(md, str)

    def test_full_pipeline_with_trace_no_exception(self):
        ctx = _full_ctx()
        trace = _trace_with_n_theories(2)
        coord = EditorialCoordinator()
        brief = coord.build(ctx, strategy_trace=trace)
        manuscript = coord.build_manuscript(brief)
        manuscript = coord.run_writers(brief, manuscript)
        md = MarkdownRenderer().render(manuscript, brief=brief)
        assert isinstance(md, str)

    def test_report_without_trace_lacks_strategy_section(self):
        ctx = _full_ctx()
        coord = EditorialCoordinator()
        brief = coord.build(ctx)
        manuscript = coord.build_manuscript(brief)
        coord.run_writers(brief, manuscript)
        md = MarkdownRenderer().render(manuscript, brief=brief)
        assert "## Strategic Direction" not in md

    def test_report_with_trace_has_strategy_section(self):
        ctx = _full_ctx()
        trace = _trace_with_n_theories(2)
        coord = EditorialCoordinator()
        brief = coord.build(ctx, strategy_trace=trace)
        manuscript = coord.build_manuscript(brief)
        coord.run_writers(brief, manuscript)
        md = MarkdownRenderer().render(manuscript, brief=brief)
        assert "## Strategic Direction" in md

    def test_editorial_writer_optional_default_false(self):
        assert EditorialWriter.optional is False

    def test_strategy_narrative_is_frozen(self):
        trace = _trace_with_n_theories(2)
        sn = build_strategy_narrative(trace)
        with pytest.raises(Exception):  # frozen Pydantic model
            sn.winner_theory_id = "modified"

    def test_strategy_alternative_summary_is_frozen(self):
        alt = StrategyAlternativeSummary(theory_id="TH-TEST", score=0.5)
        with pytest.raises(Exception):
            alt.score = 0.99
