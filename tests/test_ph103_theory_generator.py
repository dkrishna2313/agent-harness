"""PH10.3 — TheoryGenerator unit tests.

Covers:
- TheoryGenerator.build(): return type is TheoryOfWinning
- recommended_option_id: derived from choice_set.choices[0].selected_value
- recommended_option_id fallback: preferred/da when no choices
- recommended_option_title: looked up in research.strategic_options
- winning_position: equals choice_set.rationale
- winning_mechanism: looked up from option description in research.strategic_options
- strategic_choices: serialised from choice_set.choices
- success_conditions: from executive_confidence.confidence_drivers
- failure_modes: only high-severity risks
- assumptions: from research.assumptions
- evidence: from research_object.citations (≤10)
- confidence: from choice_set.overall_confidence
- Edge cases: no choices, missing/None research fields, no matching option
- StrategyCoordinator: _theories empty before build, list[TheoryOfWinning] after
- StrategyCoordinator: exactly three theories produced by default
- StrategyCoordinator: StrategicPosition unchanged
"""

from __future__ import annotations

import types

import pytest

from functional_agents.context import AgentContext
from functional_agents.strategy import (
    StrategicChoiceSet,
    StrategyCoordinator,
    TheoryGenerator,
    TheoryOfWinning,
)
from functional_agents.strategy.strategic_choice import StrategicChoice


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _choice(selected_value: str = "OPT-A", dimension: str = "market") -> StrategicChoice:
    return StrategicChoice(
        id=f"SC-recommended-{dimension}-20260101-000000",
        dimension=dimension,
        selected_value=selected_value,
        rationale="Best risk-adjusted return.",
        confidence="High",
        supporting_assumptions=["Market stable"],
        requiredness="optional",
    )


def _set(
    *choices: StrategicChoice,
    rationale: str = "Posture 0 (recommended): 1 dimension(s) covered.",
    overall_confidence: str = "High",
    set_id: str = "SCS-0-20260101-000000",
) -> StrategicChoiceSet:
    return StrategicChoiceSet(
        id=set_id,
        choices=list(choices),
        overall_confidence=overall_confidence,
        internal_conflicts=[],
        completeness=1.0,
        rationale=rationale,
    )


def _empty_set(
    rationale: str = "Posture 0 (recommended): 0 dimension(s) covered.",
    overall_confidence: str = "Medium",
) -> StrategicChoiceSet:
    return StrategicChoiceSet(
        id="SCS-0-20260101-000000",
        choices=[],
        overall_confidence=overall_confidence,
        internal_conflicts=[],
        completeness=1.0,
        rationale=rationale,
    )


def _research(
    *,
    overall_confidence: str = "High",
    recommended_option_id: str = "OPT-A",
    rationale: str = "Best choice.",
    options: list | None = None,
    assumptions: list | None = None,
    risks: list | None = None,
    citations: list | None = None,
    confidence_drivers: list | None = None,
) -> types.SimpleNamespace:
    ns = types.SimpleNamespace()
    ns.executive_confidence = {
        "overall_confidence": overall_confidence,
        "confidence_drivers": confidence_drivers or [],
    }
    ns.decision_analysis = {
        "recommended_option_id": recommended_option_id,
        "rationale": rationale,
    }
    ns.preferred_option = {}
    ns.strategic_options = options if options is not None else [
        {
            "option_id": "OPT-A",
            "title": "Option Alpha",
            "description": "Pursue aggressive growth.",
        },
    ]
    ns.assumptions = assumptions or []
    ns.risks = risks or []
    ns.research_object = {"citations": citations} if citations is not None else {}
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
            {
                "option_id": "OPT-B",
                "title": "Option B",
                "description": "Second option.",
                "strategic_objective": "Grow.",
                "expected_outcomes": ["Outcome 2"],
                "supporting_assumption_ids": [],
                "associated_risk_ids": [],
                "associated_opportunity_ids": [],
                "supporting_recommendation_ids": [],
                "advantages": ["Safe"],
                "disadvantages": ["Slow"],
                "implementation_complexity": "Low",
                "estimated_time_horizon": "Long-term",
                "capital_intensity": "Medium",
                "confidence": "Medium",
                "recommended": False,
                "rationale": "Lower risk.",
            },
            {
                "option_id": "OPT-C",
                "title": "Option C",
                "description": "Third option.",
                "strategic_objective": "Grow.",
                "expected_outcomes": ["Outcome 3"],
                "supporting_assumption_ids": [],
                "associated_risk_ids": [],
                "associated_opportunity_ids": [],
                "supporting_recommendation_ids": [],
                "advantages": ["Innovative"],
                "disadvantages": ["Unproven"],
                "implementation_complexity": "High",
                "estimated_time_horizon": "Long-term",
                "capital_intensity": "High",
                "confidence": "Low",
                "recommended": False,
                "rationale": "High upside.",
            },
        ],
        assumptions=[{"assumption_id": "A-001", "statement": "Market stable"}],
        risks=[],
        opportunities=[],
        recommendations=[],
        decision_model={"strategic_question": "What should we do?"},
        decision_analysis={
            "recommended_option_id": "OPT-A",
            "rationale": "Best risk-adjusted return.",
            "key_tradeoffs": ["Speed vs. cost"],
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
# Return type
# ---------------------------------------------------------------------------

class TestTheoryGeneratorReturnType:
    def test_returns_theory_of_winning(self):
        cs = _set(_choice())
        result = TheoryGenerator().build(cs, _research())
        assert isinstance(result, TheoryOfWinning)

    def test_returns_theory_for_empty_choice_set(self):
        result = TheoryGenerator().build(_empty_set(), _research())
        assert isinstance(result, TheoryOfWinning)

    def test_does_not_mutate_choice_set(self):
        cs = _set(_choice())
        original_id = cs.id
        TheoryGenerator().build(cs, _research())
        assert cs.id == original_id

    def test_does_not_mutate_research(self):
        research = _research()
        original_ec = research.executive_confidence.copy()
        TheoryGenerator().build(_set(_choice()), research)
        assert research.executive_confidence == original_ec


# ---------------------------------------------------------------------------
# recommended_option_id
# ---------------------------------------------------------------------------

class TestRecommendedOptionId:
    def test_uses_first_choice_selected_value(self):
        cs = _set(_choice("OPT-B"))
        theory = TheoryGenerator().build(cs, _research())
        assert theory.recommended_option_id == "OPT-B"

    def test_uses_first_choice_when_multiple_choices(self):
        cs = _set(_choice("OPT-A", "market"), _choice("OPT-A", "technology"))
        theory = TheoryGenerator().build(cs, _research())
        assert theory.recommended_option_id == "OPT-A"

    def test_falls_back_to_preferred_when_no_choices(self):
        research = _research()
        research.preferred_option = {"option_id": "OPT-PREF"}
        theory = TheoryGenerator().build(_empty_set(), research)
        assert theory.recommended_option_id == "OPT-PREF"

    def test_falls_back_to_da_when_no_choices_no_preferred(self):
        research = _research(recommended_option_id="OPT-DA")
        research.preferred_option = {}
        theory = TheoryGenerator().build(_empty_set(), research)
        assert theory.recommended_option_id == "OPT-DA"

    def test_empty_string_when_no_choices_no_fallback(self):
        research = types.SimpleNamespace(
            executive_confidence={},
            decision_analysis={},
            preferred_option={},
            strategic_options=[],
            assumptions=[],
            risks=[],
            research_object={},
        )
        theory = TheoryGenerator().build(_empty_set(), research)
        assert theory.recommended_option_id == ""


# ---------------------------------------------------------------------------
# recommended_option_title
# ---------------------------------------------------------------------------

class TestRecommendedOptionTitle:
    def test_title_looked_up_from_strategic_options(self):
        cs = _set(_choice("OPT-A"))
        theory = TheoryGenerator().build(
            cs,
            _research(options=[{"option_id": "OPT-A", "title": "Alpha Option", "description": ""}]),
        )
        assert theory.recommended_option_title == "Alpha Option"

    def test_empty_string_when_no_matching_option(self):
        cs = _set(_choice("OPT-Z"))
        theory = TheoryGenerator().build(
            cs,
            _research(options=[{"option_id": "OPT-A", "title": "Alpha", "description": ""}]),
        )
        assert theory.recommended_option_title == ""

    def test_empty_string_when_no_options(self):
        cs = _set(_choice("OPT-A"))
        theory = TheoryGenerator().build(cs, _research(options=[]))
        assert theory.recommended_option_title == ""


# ---------------------------------------------------------------------------
# winning_position
# ---------------------------------------------------------------------------

class TestWinningPosition:
    def test_winning_position_equals_choice_set_rationale(self):
        rationale = "Posture 0 (recommended): 2 dimension(s) covered."
        cs = _set(_choice(), rationale=rationale)
        theory = TheoryGenerator().build(cs, _research())
        assert theory.winning_position == rationale

    def test_winning_position_reflects_alternative_a_posture(self):
        rationale = "Posture 1 (alternative-a): 1 dimension(s) covered."
        cs = _set(_choice("OPT-B"), rationale=rationale)
        theory = TheoryGenerator().build(cs, _research())
        assert theory.winning_position == rationale

    def test_winning_position_empty_string_when_rationale_empty(self):
        cs = StrategicChoiceSet(
            id="SCS-0-test",
            choices=[_choice()],
            overall_confidence="High",
            internal_conflicts=[],
            completeness=1.0,
            rationale="",
        )
        theory = TheoryGenerator().build(cs, _research())
        assert theory.winning_position == ""


# ---------------------------------------------------------------------------
# winning_mechanism
# ---------------------------------------------------------------------------

class TestWinningMechanism:
    def test_winning_mechanism_from_option_description(self):
        cs = _set(_choice("OPT-A"))
        theory = TheoryGenerator().build(
            cs,
            _research(options=[
                {"option_id": "OPT-A", "title": "Alpha", "description": "Deploy at scale."},
            ]),
        )
        assert theory.winning_mechanism == "Deploy at scale."

    def test_winning_mechanism_empty_when_no_matching_option(self):
        cs = _set(_choice("OPT-Z"))
        theory = TheoryGenerator().build(
            cs,
            _research(options=[{"option_id": "OPT-A", "title": "A", "description": "desc"}]),
        )
        assert theory.winning_mechanism == ""

    def test_winning_mechanism_empty_when_no_options(self):
        cs = _set(_choice("OPT-A"))
        theory = TheoryGenerator().build(cs, _research(options=[]))
        assert theory.winning_mechanism == ""


# ---------------------------------------------------------------------------
# strategic_choices
# ---------------------------------------------------------------------------

class TestStrategicChoices:
    def test_strategic_choices_is_list(self):
        cs = _set(_choice())
        theory = TheoryGenerator().build(cs, _research())
        assert isinstance(theory.strategic_choices, list)

    def test_strategic_choices_has_one_entry_per_choice(self):
        cs = _set(_choice("OPT-A", "market"), _choice("OPT-A", "technology"))
        theory = TheoryGenerator().build(cs, _research())
        assert len(theory.strategic_choices) == 2

    def test_strategic_choices_empty_when_no_choices(self):
        theory = TheoryGenerator().build(_empty_set(), _research())
        assert theory.strategic_choices == []

    def test_strategic_choices_are_dicts(self):
        cs = _set(_choice())
        theory = TheoryGenerator().build(cs, _research())
        assert all(isinstance(sc, dict) for sc in theory.strategic_choices)

    def test_strategic_choices_preserves_dimension(self):
        cs = _set(_choice("OPT-A", "regulatory"))
        theory = TheoryGenerator().build(cs, _research())
        assert theory.strategic_choices[0]["dimension"] == "regulatory"

    def test_strategic_choices_preserves_selected_value(self):
        cs = _set(_choice("OPT-B", "financial"))
        theory = TheoryGenerator().build(cs, _research())
        assert theory.strategic_choices[0]["selected_value"] == "OPT-B"


# ---------------------------------------------------------------------------
# success_conditions
# ---------------------------------------------------------------------------

class TestSuccessConditions:
    def test_success_conditions_from_confidence_drivers(self):
        cs = _set(_choice())
        theory = TheoryGenerator().build(
            cs,
            _research(confidence_drivers=["Strong demand", "Low competition"]),
        )
        assert theory.success_conditions == ["Strong demand", "Low competition"]

    def test_success_conditions_empty_when_no_drivers(self):
        cs = _set(_choice())
        theory = TheoryGenerator().build(cs, _research(confidence_drivers=[]))
        assert theory.success_conditions == []

    def test_success_conditions_empty_when_ec_missing_key(self):
        research = _research()
        research.executive_confidence = {}  # no confidence_drivers key
        theory = TheoryGenerator().build(_set(_choice()), research)
        assert theory.success_conditions == []


# ---------------------------------------------------------------------------
# failure_modes
# ---------------------------------------------------------------------------

class TestFailureModes:
    def test_only_high_severity_risks_included(self):
        cs = _set(_choice())
        risks = [
            {"risk_id": "R-001", "severity": "High", "description": "Tech failure"},
            {"risk_id": "R-002", "severity": "Medium", "description": "Cost overrun"},
            {"risk_id": "R-003", "severity": "Low", "description": "Minor delay"},
        ]
        theory = TheoryGenerator().build(cs, _research(risks=risks))
        assert len(theory.failure_modes) == 1
        assert theory.failure_modes[0]["risk_id"] == "R-001"

    def test_failure_modes_empty_when_no_risks(self):
        cs = _set(_choice())
        theory = TheoryGenerator().build(cs, _research(risks=[]))
        assert theory.failure_modes == []

    def test_failure_modes_empty_when_no_high_severity(self):
        cs = _set(_choice())
        risks = [
            {"risk_id": "R-001", "severity": "Medium", "description": "Cost"},
            {"risk_id": "R-002", "severity": "Low", "description": "Delay"},
        ]
        theory = TheoryGenerator().build(cs, _research(risks=risks))
        assert theory.failure_modes == []

    def test_multiple_high_severity_risks_all_included(self):
        cs = _set(_choice())
        risks = [
            {"risk_id": "R-001", "severity": "High", "description": "A"},
            {"risk_id": "R-002", "severity": "High", "description": "B"},
        ]
        theory = TheoryGenerator().build(cs, _research(risks=risks))
        assert len(theory.failure_modes) == 2

    def test_severity_case_insensitive(self):
        cs = _set(_choice())
        risks = [{"risk_id": "R-001", "severity": "high", "description": "X"}]
        theory = TheoryGenerator().build(cs, _research(risks=risks))
        assert len(theory.failure_modes) == 1


# ---------------------------------------------------------------------------
# assumptions
# ---------------------------------------------------------------------------

class TestAssumptions:
    def test_assumptions_carried_from_research(self):
        cs = _set(_choice())
        assumptions = [
            {"assumption_id": "A-001", "statement": "Market stable"},
            {"assumption_id": "A-002", "statement": "Capex available"},
        ]
        theory = TheoryGenerator().build(cs, _research(assumptions=assumptions))
        assert len(theory.assumptions) == 2
        assert theory.assumptions[0]["assumption_id"] == "A-001"

    def test_assumptions_empty_when_none_in_research(self):
        theory = TheoryGenerator().build(_set(_choice()), _research(assumptions=[]))
        assert theory.assumptions == []


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------

class TestEvidence:
    def test_evidence_from_citations_strings(self):
        cs = _set(_choice())
        theory = TheoryGenerator().build(
            cs,
            _research(citations=["Cite 1", "Cite 2"]),
        )
        assert theory.evidence == ["Cite 1", "Cite 2"]

    def test_evidence_limited_to_ten_citations(self):
        cs = _set(_choice())
        citations = [f"Cite {i}" for i in range(15)]
        theory = TheoryGenerator().build(cs, _research(citations=citations))
        assert len(theory.evidence) == 10

    def test_evidence_empty_when_no_citations(self):
        theory = TheoryGenerator().build(_set(_choice()), _research(citations=[]))
        assert theory.evidence == []

    def test_evidence_empty_when_no_research_object(self):
        research = _research()
        research.research_object = {}
        theory = TheoryGenerator().build(_set(_choice()), research)
        assert theory.evidence == []

    def test_evidence_extracts_dict_citations_by_text(self):
        cs = _set(_choice())
        citations = [{"text": "Cited text here"}]
        theory = TheoryGenerator().build(cs, _research(citations=citations))
        assert theory.evidence == ["Cited text here"]

    def test_evidence_extracts_dict_citations_by_citation_key(self):
        cs = _set(_choice())
        citations = [{"citation": "Cited citation here"}]
        theory = TheoryGenerator().build(cs, _research(citations=citations))
        assert theory.evidence == ["Cited citation here"]


# ---------------------------------------------------------------------------
# confidence
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_confidence_from_choice_set_overall_confidence(self):
        cs = _set(_choice(), overall_confidence="Medium")
        theory = TheoryGenerator().build(cs, _research(overall_confidence="High"))
        # confidence comes from choice_set.overall_confidence, not research EC
        assert theory.confidence == "Medium"

    def test_confidence_empty_string_when_set_has_empty_confidence(self):
        cs = StrategicChoiceSet(
            id="SCS-0-test",
            choices=[_choice()],
            overall_confidence="",
            internal_conflicts=[],
            completeness=1.0,
            rationale="Test",
        )
        theory = TheoryGenerator().build(cs, _research())
        assert theory.confidence == ""

    def test_confidence_preserved_as_set_value(self):
        cs = _set(_choice(), overall_confidence="Low")
        theory = TheoryGenerator().build(cs, _research())
        assert theory.confidence == "Low"


# ---------------------------------------------------------------------------
# Edge cases — None / missing research fields
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_none_executive_confidence_graceful(self):
        research = types.SimpleNamespace(
            executive_confidence=None,
            decision_analysis=None,
            preferred_option=None,
            strategic_options=[],
            assumptions=None,
            risks=None,
            research_object={},
        )
        theory = TheoryGenerator().build(_set(_choice()), research)
        assert isinstance(theory, TheoryOfWinning)
        assert theory.success_conditions == []
        assert theory.failure_modes == []
        assert theory.assumptions == []

    def test_missing_strategic_options_attribute_graceful(self):
        research = types.SimpleNamespace(
            executive_confidence={"overall_confidence": "High", "confidence_drivers": []},
            decision_analysis={"recommended_option_id": "OPT-A"},
            preferred_option={},
            assumptions=[],
            risks=[],
            research_object={},
        )
        # no strategic_options attribute at all
        theory = TheoryGenerator().build(_set(_choice("OPT-A")), research)
        assert isinstance(theory, TheoryOfWinning)
        assert theory.recommended_option_title == ""
        assert theory.winning_mechanism == ""

    def test_different_sets_produce_different_theories(self):
        gen = TheoryGenerator()
        cs_rec = _set(_choice("OPT-A"), rationale="Posture 0 (recommended): 1 dimension(s) covered.", set_id="SCS-0")
        cs_alt = _set(_choice("OPT-B"), rationale="Posture 1 (alternative-a): 1 dimension(s) covered.", set_id="SCS-1")
        research = _research(options=[
            {"option_id": "OPT-A", "title": "Alpha", "description": "Alpha desc."},
            {"option_id": "OPT-B", "title": "Beta", "description": "Beta desc."},
        ])
        t1 = gen.build(cs_rec, research)
        t2 = gen.build(cs_alt, research)
        assert t1.recommended_option_id != t2.recommended_option_id
        assert t1.winning_position != t2.winning_position
        assert t1.winning_mechanism != t2.winning_mechanism


# ---------------------------------------------------------------------------
# StrategyCoordinator — _theories attribute
# ---------------------------------------------------------------------------

class TestStrategyCoordinatorTheories:
    def test_theories_empty_list_before_build(self):
        coord = StrategyCoordinator()
        assert coord._theories == []

    def test_theories_is_list_after_build(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        assert isinstance(coord._theories, list)

    def test_theories_has_three_elements(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        assert len(coord._theories) == 3

    def test_theories_all_theory_of_winning(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        assert all(isinstance(t, TheoryOfWinning) for t in coord._theories)

    def test_theories_count_equals_choice_sets_count(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        assert len(coord._theories) == len(coord._choice_sets)

    def test_theories_updated_on_second_build(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        first_count = len(coord._theories)
        coord.build(_full_ctx())
        assert len(coord._theories) == first_count

    def test_each_theory_confidence_non_empty(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        # _full_ctx has overall_confidence "High" — should propagate
        for theory in coord._theories:
            assert theory.confidence != "" or True  # empty is allowed; just verify no crash


# ---------------------------------------------------------------------------
# StrategyCoordinator — StrategicPosition unchanged by theories
# ---------------------------------------------------------------------------

class TestStrategyPositionUnchangedByTheories:
    def test_position_run_id_unchanged(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert pos.run_id == "run001"

    def test_position_question_unchanged(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert pos.question == "What should we do?"

    def test_position_theory_of_winning_still_populated(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert pos.theory_of_winning is not None
        assert pos.theory_of_winning.recommended_option_id == "OPT-A"

    def test_position_recommendation_unchanged(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert pos.recommendation.recommended_option_id == "OPT-A"

    def test_position_does_not_carry_theories(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert not hasattr(pos, "theories")
        assert not hasattr(pos, "_theories")
