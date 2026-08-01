"""PH12.2c — Strategy candidate differentiation and evaluation discrimination.

Root cause: without execution_complexity on choices, every theory scores 0.965 because
execution_feasibility defaults to 0.75 for all choices and every other criterion is
binary-present, yielding 0.14×0.75 + 0.86 = 0.965 for all theories.

Fix: add semantically appropriate execution_complexity values to engagement YAML choices.
The choice generator already propagates the field; the evaluator already scores it.

Test categories (10):
  1. TestExecutionComplexityPropagation  — generator embeds complexity in choice metadata
  2. TestExecutionFeasibilityScoring     — evaluator maps high/medium/low/missing correctly
  3. TestScoreSaturationWithoutComplexity — baseline: missing complexity → identical scores
  4. TestScoreDifferentiationWithComplexity — distinct complexities → distinct scores
  5. TestWinnerMarginPositive            — winner score > runner-up, margin > 0
  6. TestDistinctScoreCount              — ≥2 distinct scores across 4 theories
  7. TestSaturationResolvedByComplexity  — saturation_detected=False with varied complexity
  8. TestComplexitySemanticMapping       — high→0.5, medium→0.75, low→1.0, missing→0.75
  9. TestDeterminism                     — same inputs → same scores every call
  10. TestMonitorEngagementRegression    — full pipeline: 4 theories, distinct scores, no saturation
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from functional_agents.strategy.strategic_choice_generator import StrategicChoiceGenerator
from functional_agents.strategy.strategy_config import (
    ChoiceConfig,
    DimensionConfig,
)
from functional_agents.strategy.strategy_plan import (
    EvaluationModel,
    GenerationPolicy,
    StrategyPlan,
    ValidationPolicy,
)
from functional_agents.strategy.theory_evaluator import TheoryEvaluator
from functional_agents.strategy.theory_generator import TheoryGenerator
from functional_agents.strategy.strategic_position import TheoryOfWinning
from functional_agents.strategy.saturation_detector import SaturationDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plan(dim_configs, max_candidates=4):
    """Minimal StrategyPlan wired for configured mode."""
    return StrategyPlan(
        plan_id="TEST-PLAN",
        framework="monitor_choice_cascade",
        active_dimensions=[d.id for d in dim_configs],
        dimension_configs=dim_configs,
        generation_policy=GenerationPolicy(
            max_candidates=max_candidates,
            diversity_required=True,
        ),
        evaluation_model=EvaluationModel(
            weights={
                "strategic_fit":        0.16,
                "execution_feasibility": 0.14,
                "evidence_quality":      0.12,
                "opportunity_capture":   0.12,
                "risk_resilience":       0.10,
                "assumption_robustness": 0.10,
                "choice_completeness":   0.08,
                "mechanism_defined":     0.06,
                "position_articulated":  0.05,
                "option_identified":     0.04,
                "risk_awareness":        0.02,
                "assumption_coverage":   0.01,
            }
        ),
        validation_policy=ValidationPolicy(require_evidence=False, require_assumptions=False),
    )


def _make_research(evidence_count=3, assumption_count=3, risk_count=2):
    """Minimal research object with enough populated fields to avoid zero scores."""
    m = MagicMock()
    m.run_id = "test-run"
    m.question = "test question"
    m.execution_profile = "test"
    m.assumptions = [{"statement": f"assumption {i}", "id": f"A-{i}"} for i in range(assumption_count)]
    m.risks = [{"statement": f"risk {i}", "severity": "high", "id": f"R-{i}"} for i in range(risk_count)]
    m.opportunities = [{"statement": f"opportunity {i}", "id": f"O-{i}"} for i in range(3)]
    m.recommendations = []
    m.strategic_options = []
    m.executive_confidence = {"overall_confidence": "High", "confidence_drivers": ["driver1"]}
    m.decision_analysis = {}
    m.preferred_option = {}
    m.research_object = {"id": "test-ro", "citations": [f"cite{i}" for i in range(evidence_count)]}
    return m


def _make_dim(dim_id, choices_with_complexity):
    """Build a DimensionConfig from (choice_id, title, complexity) tuples.

    complexity=None means no execution_complexity field (missing).
    """
    choice_objs = []
    for cid, ctitle, cx in choices_with_complexity:
        if cx is not None:
            choice_objs.append(ChoiceConfig(id=cid, title=ctitle, execution_complexity=cx))
        else:
            choice_objs.append(ChoiceConfig(id=cid, title=ctitle))
    return DimensionConfig(
        id=dim_id,
        title=dim_id.replace("_", " ").title(),
        description=f"Description for {dim_id}",
        required=True,
        choices=choice_objs,
    )


def _build_theory_from_choices(choices_dict):
    """Build a TheoryOfWinning with the given strategic choices (list[dict])."""
    return TheoryOfWinning(
        theory_id="TH-TEST",
        source_choice_set_id="SCS-TEST",
        recommended_option_id="opt-a",
        recommended_option_title="Option A",
        winning_position="Test winning position",
        winning_mechanism="Test winning mechanism",
        strategic_choices=choices_dict,
        success_conditions=["condition1"],
        failure_modes=[{"statement": "risk1", "severity": "high"}],
        assumptions=[{"statement": "assumption1"}],
        evidence=["cite1", "cite2", "cite3"],
        confidence="High",
    )


def _score_ef(choices_dict, n_dims=3):
    """Run execution_feasibility _raw_score and return score."""
    theory = _build_theory_from_choices(choices_dict)
    plan = StrategyPlan(
        plan_id="P",
        framework="executive",
        active_dimensions=["d1", "d2", "d3"][:n_dims],
        evaluation_model=EvaluationModel(weights={"execution_feasibility": 1.0}),
        validation_policy=ValidationPolicy(),
    )
    ev = TheoryEvaluator()
    result = ev._raw_score("execution_feasibility", theory, plan.validation_policy, n_dims)
    return result[0]


# ===========================================================================
# 1. TestExecutionComplexityPropagation
# ===========================================================================

class TestExecutionComplexityPropagation:
    """Choice generator embeds execution_complexity in choice metadata when present."""

    def test_propagates_low_complexity(self):
        dims = [
            _make_dim("d1", [("opt-a", "Option A", "low"), ("opt-b", "Option B", "high")]),
        ]
        plan = _make_plan(dims, max_candidates=1)
        sets = StrategicChoiceGenerator().build(plan, _make_research())
        choice = sets[0].choices[0]
        assert choice.metadata.get("execution_complexity") == "low"

    def test_propagates_high_complexity(self):
        dims = [
            _make_dim("d1", [("opt-a", "Option A", "high"), ("opt-b", "Option B", "low")]),
        ]
        plan = _make_plan(dims, max_candidates=2)
        sets = StrategicChoiceGenerator().build(plan, _make_research())
        # posture 1 picks opt-b → low
        choice1 = sets[1].choices[0]
        assert choice1.metadata.get("execution_complexity") == "low"

    def test_missing_complexity_absent_from_metadata(self):
        dims = [
            _make_dim("d1", [("opt-a", "Option A", None), ("opt-b", "Option B", None)]),
        ]
        plan = _make_plan(dims, max_candidates=1)
        sets = StrategicChoiceGenerator().build(plan, _make_research())
        # When ChoiceConfig has no execution_complexity, field should not be set in metadata
        assert "execution_complexity" not in sets[0].choices[0].metadata

    def test_propagation_for_all_choices_in_set(self):
        dims = [
            _make_dim("d1", [("a1", "A1", "low"), ("a2", "A2", "high")]),
            _make_dim("d2", [("b1", "B1", "medium"), ("b2", "B2", "low")]),
            _make_dim("d3", [("c1", "C1", "high"), ("c2", "C2", "medium")]),
        ]
        plan = _make_plan(dims, max_candidates=1)
        sets = StrategicChoiceGenerator().build(plan, _make_research())
        complexities = [c.metadata.get("execution_complexity") for c in sets[0].choices]
        assert complexities == ["low", "medium", "high"]

    def test_complexity_propagated_for_each_candidate(self):
        dims = [
            _make_dim("d1", [("opt-a", "A", "low"), ("opt-b", "B", "high"), ("opt-c", "C", "medium")]),
        ]
        plan = _make_plan(dims, max_candidates=3)
        sets = StrategicChoiceGenerator().build(plan, _make_research())
        assert len(sets) == 3
        complexities = [s.choices[0].metadata.get("execution_complexity") for s in sets]
        assert complexities == ["low", "high", "medium"]


# ===========================================================================
# 2. TestExecutionFeasibilityScoring
# ===========================================================================

class TestExecutionFeasibilityScoring:
    """TheoryEvaluator maps high/medium/low/missing execution_complexity correctly."""

    def test_low_complexity_scores_1_0(self):
        choices = [{"dimension": "d1", "selected_value": "opt-a", "metadata": {"execution_complexity": "low"}}]
        score = _score_ef(choices, n_dims=1)
        assert score == 1.0

    def test_medium_complexity_scores_0_75(self):
        choices = [{"dimension": "d1", "selected_value": "opt-a", "metadata": {"execution_complexity": "medium"}}]
        score = _score_ef(choices, n_dims=1)
        assert score == 0.75

    def test_high_complexity_scores_0_5(self):
        choices = [{"dimension": "d1", "selected_value": "opt-a", "metadata": {"execution_complexity": "high"}}]
        score = _score_ef(choices, n_dims=1)
        assert score == 0.5

    def test_missing_complexity_scores_0_75(self):
        choices = [{"dimension": "d1", "selected_value": "opt-a", "metadata": {}}]
        score = _score_ef(choices, n_dims=1)
        assert score == 0.75

    def test_mixed_complexity_averages_correctly(self):
        choices = [
            {"dimension": "d1", "selected_value": "a", "metadata": {"execution_complexity": "low"}},
            {"dimension": "d2", "selected_value": "b", "metadata": {"execution_complexity": "high"}},
            {"dimension": "d3", "selected_value": "c", "metadata": {"execution_complexity": "medium"}},
        ]
        score = _score_ef(choices, n_dims=3)
        # avg(1.0, 0.5, 0.75) = 2.25/3 = 0.75
        assert abs(score - 0.75) < 1e-6

    def test_case_insensitive(self):
        for val, expected in [("HIGH", 0.5), ("Low", 1.0), ("MEDIUM", 0.75)]:
            choices = [{"dimension": "d1", "selected_value": "a", "metadata": {"execution_complexity": val}}]
            score = _score_ef(choices, n_dims=1)
            assert score == expected, f"Failed for {val!r}"


# ===========================================================================
# 3. TestExecutionFeasibilityRootCause
# ===========================================================================

class TestExecutionFeasibilityRootCause:
    """Root cause: execution_feasibility defaults to 0.75 when execution_complexity is absent.

    This is the mechanism that caused all 4 theories to score identically (0.965).
    execution_feasibility is the only criterion that CAN differ per-theory (since
    theory_content is resolved after scoring), and it defaults to 0.75 for all
    choices without complexity metadata.
    """

    def _ef_scores_for_theories(self, dim_choices_per_theory):
        """Build N theories with the given (dim_id, choices_list) per theory and return ef scores."""
        # Use simple explicit theories rather than running the full generator
        ev = TheoryEvaluator()
        scores = []
        for choices_list in dim_choices_per_theory:
            theory = _build_theory_from_choices(choices_list)
            plan = StrategyPlan(
                plan_id="P",
                framework="executive",
                active_dimensions=[c["dimension"] for c in choices_list],
                evaluation_model=EvaluationModel(weights={"execution_feasibility": 1.0}),
                validation_policy=ValidationPolicy(),
            )
            result = ev._raw_score(
                "execution_feasibility", theory, plan.validation_policy,
                len(choices_list)
            )
            scores.append(result[0])
        return scores

    def test_ef_score_is_0_75_when_complexity_absent(self):
        choices = [{"dimension": "d1", "selected_value": "opt", "metadata": {}}]
        score = _score_ef(choices, n_dims=1)
        assert score == 0.75, f"Expected 0.75 (default), got {score}"

    def test_all_theories_get_same_ef_when_complexity_absent(self):
        """All 4 theories share ef=0.75 when no execution_complexity is set."""
        # Build 4 sets of choices — each picks different options but no complexity
        choices_4 = [
            [{"dimension": "d1", "selected_value": "opt-a", "metadata": {}},
             {"dimension": "d2", "selected_value": "opt-x", "metadata": {}}],
            [{"dimension": "d1", "selected_value": "opt-b", "metadata": {}},
             {"dimension": "d2", "selected_value": "opt-y", "metadata": {}}],
            [{"dimension": "d1", "selected_value": "opt-c", "metadata": {}},
             {"dimension": "d2", "selected_value": "opt-z", "metadata": {}}],
            [{"dimension": "d1", "selected_value": "opt-d", "metadata": {}},
             {"dimension": "d2", "selected_value": "opt-w", "metadata": {}}],
        ]
        ef_scores = self._ef_scores_for_theories(choices_4)
        assert all(s == 0.75 for s in ef_scores), (
            f"Expected all ef=0.75 without complexity, got {ef_scores}"
        )

    def test_saturation_when_only_ef_differs_and_all_ef_equal(self):
        """When execution_feasibility is 0.75 for all, and it's the only varying criterion,
        all theories score identically → saturation=True."""
        # Construct theories identically except for choice IDs (which don't change scores)
        def make_theory_with_ef(ef_val, theory_id):
            return TheoryOfWinning(
                theory_id=theory_id,
                source_choice_set_id=f"SCS-{theory_id}",
                recommended_option_id="opt",
                recommended_option_title="Option",
                winning_position="Position",
                winning_mechanism="Mechanism",
                # All choices have empty metadata → ef defaults to 0.75 for each
                strategic_choices=[
                    {"dimension": "d1", "selected_value": "x", "metadata": {}},
                ],
                success_conditions=["cond"],
                failure_modes=[{"statement": "risk", "severity": "high"}],
                assumptions=[{"statement": "assumption"}],
                evidence=["cite1", "cite2", "cite3"],
                confidence="High",
            )

        plan = StrategyPlan(
            plan_id="P",
            framework="executive",
            active_dimensions=["d1"],
            evaluation_model=EvaluationModel(weights={
                "execution_feasibility": 0.14,
                "strategic_fit": 0.16,
                "choice_completeness": 0.08,
                "mechanism_defined": 0.06,
                "position_articulated": 0.05,
                "option_identified": 0.04,
            }),
            validation_policy=ValidationPolicy(),
        )
        ev = TheoryEvaluator()
        theories = [make_theory_with_ef(0.75, f"TH-{i}") for i in range(4)]
        evals = [ev.build(t, plan, None, None, None) for t in theories]
        scores = [round(e.overall_score, 6) for e in evals]
        assert len(set(scores)) == 1, f"Expected saturation (identical scores), got {scores}"

        detected, _ = SaturationDetector().check(evals)
        assert detected is True


# ===========================================================================
# 4. TestScoreDifferentiationWithComplexity
# ===========================================================================

class TestScoreDifferentiationWithComplexity:
    """With varied execution_complexity, theories score differently."""

    def _build_differentiated_evals(self):
        dims = [
            _make_dim("d1", [("a", "A", "low"), ("b", "B", "high"), ("c", "C", "medium")]),
            _make_dim("d2", [("x", "X", "medium"), ("y", "Y", "low"), ("z", "Z", "high")]),
        ]
        plan = _make_plan(dims, max_candidates=3)
        research = _make_research()

        gen = StrategicChoiceGenerator()
        sets = gen.build(plan, research)
        tgen = TheoryGenerator()
        theories = [tgen.build(s, research) for s in sets]
        ev = TheoryEvaluator()
        return [ev.build(t, plan, research, None, None) for t in theories]

    def test_distinct_scores_with_complexity(self):
        evals = self._build_differentiated_evals()
        scores = [round(e.overall_score, 6) for e in evals]
        assert len(set(scores)) >= 2, f"Expected ≥2 distinct scores, got {scores}"

    def test_no_saturation_with_complexity(self):
        evals = self._build_differentiated_evals()
        detected, msg = SaturationDetector().check(evals)
        assert detected is False, f"Expected no saturation, got: {msg}"

    def test_expected_complexity_ef_scores(self):
        dims = [
            _make_dim("d1", [("a", "A", "low"), ("b", "B", "high")]),
        ]
        plan = _make_plan(dims, max_candidates=2)
        research = _make_research()
        sets = StrategicChoiceGenerator().build(plan, research)
        tgen = TheoryGenerator()
        ev = TheoryEvaluator()

        theories = [tgen.build(s, research) for s in sets]
        evals = [ev.build(t, plan, research, None, None) for t in theories]
        ef_scores = [e.criteria_scores["execution_feasibility"].score for e in evals]
        # posture 0 → "low" → 1.0, posture 1 → "high" → 0.5
        assert ef_scores[0] > ef_scores[1]


# ===========================================================================
# 5. TestWinnerMarginPositive
# ===========================================================================

class TestWinnerMarginPositive:
    """Winner score exceeds runner-up; score_margin > 0."""

    def _build_multi_eval(self, n=4):
        dims = [
            _make_dim("d1", [("a", "A", "low"), ("b", "B", "high"), ("c", "C", "medium"), ("d", "D", "low")]),
            _make_dim("d2", [("w", "W", "medium"), ("x", "X", "high"), ("y", "Y", "low"), ("z", "Z", "high")]),
        ]
        plan = _make_plan(dims, max_candidates=n)
        research = _make_research()
        sets = StrategicChoiceGenerator().build(plan, research)
        tgen = TheoryGenerator()
        ev = TheoryEvaluator()
        theories = [tgen.build(s, research) for s in sets]
        return [ev.build(t, plan, research, None, None) for t in theories]

    def test_winner_score_exceeds_runner_up(self):
        evals = self._build_multi_eval()
        scores = sorted([e.overall_score for e in evals], reverse=True)
        winner, runner_up = scores[0], scores[1]
        assert winner > runner_up, f"Expected winner ({winner:.4f}) > runner-up ({runner_up:.4f})"

    def test_margin_is_positive(self):
        evals = self._build_multi_eval()
        scores = sorted([e.overall_score for e in evals], reverse=True)
        margin = scores[0] - scores[1]
        assert margin > 0, f"Expected positive margin, got {margin}"

    def test_winner_has_lowest_complexity_average(self):
        """Theory with all-low complexity should win on execution_feasibility."""
        dims = [
            _make_dim("d1", [("low_choice", "Low Choice", "low"),
                               ("high_choice", "High Choice", "high")]),
        ]
        plan = _make_plan(dims, max_candidates=2)
        research = _make_research()
        sets = StrategicChoiceGenerator().build(plan, research)
        tgen = TheoryGenerator()
        ev = TheoryEvaluator()
        theories = [tgen.build(s, research) for s in sets]
        evals = [ev.build(t, plan, research, None, None) for t in theories]

        scores = [e.overall_score for e in evals]
        # Posture 0 picks "low_choice" → ef=1.0, posture 1 picks "high_choice" → ef=0.5
        assert scores[0] > scores[1]


# ===========================================================================
# 6. TestDistinctScoreCount
# ===========================================================================

class TestDistinctScoreCount:
    """At least 2 distinct scores across 4 theories."""

    def _build_4_evals_varied_complexity(self):
        dims = [
            _make_dim("wa", [
                ("c1", "C1", "medium"),
                ("c2", "C2", "high"),
                ("c3", "C3", "low"),
                ("c4", "C4", "low"),
            ]),
            _make_dim("wtp", [
                ("d1", "D1", "low"),
                ("d2", "D2", "medium"),
                ("d3", "D3", "medium"),
                ("d4", "D4", "high"),
            ]),
        ]
        plan = _make_plan(dims, max_candidates=4)
        research = _make_research()
        sets = StrategicChoiceGenerator().build(plan, research)
        tgen = TheoryGenerator()
        ev = TheoryEvaluator()
        theories = [tgen.build(s, research) for s in sets]
        return [ev.build(t, plan, research, None, None) for t in theories]

    def test_at_least_2_distinct_scores(self):
        evals = self._build_4_evals_varied_complexity()
        scores = [round(e.overall_score, 6) for e in evals]
        distinct = set(scores)
        assert len(distinct) >= 2, f"Expected ≥2 distinct scores, got {scores}"

    def test_4_evaluations_produced(self):
        evals = self._build_4_evals_varied_complexity()
        assert len(evals) == 4

    def test_each_evaluation_has_execution_feasibility_criterion(self):
        evals = self._build_4_evals_varied_complexity()
        for ev in evals:
            assert "execution_feasibility" in ev.criteria_scores


# ===========================================================================
# 7. TestSaturationResolvedByComplexity
# ===========================================================================

class TestSaturationResolvedByComplexity:
    """SaturationDetector returns False when execution_complexity varies."""

    def test_saturation_resolved_with_varied_complexity(self):
        dims = [
            _make_dim("d1", [("a", "A", "low"), ("b", "B", "high")]),
        ]
        plan = _make_plan(dims, max_candidates=2)
        research = _make_research()
        sets = StrategicChoiceGenerator().build(plan, research)
        tgen = TheoryGenerator()
        ev = TheoryEvaluator()
        theories = [tgen.build(s, research) for s in sets]
        evals = [ev.build(t, plan, research, None, None) for t in theories]

        detected, _ = SaturationDetector().check(evals)
        assert detected is False

    def test_saturation_resolved_4_theories(self):
        dims = [
            _make_dim("d1", [("a", "A", "medium"), ("b", "B", "high"), ("c", "C", "low"), ("d", "D", "low")]),
            _make_dim("d2", [("e", "E", "low"), ("f", "F", "medium"), ("g", "G", "medium"), ("h", "H", "high")]),
        ]
        plan = _make_plan(dims, max_candidates=4)
        research = _make_research()
        sets = StrategicChoiceGenerator().build(plan, research)
        tgen = TheoryGenerator()
        ev = TheoryEvaluator()
        theories = [tgen.build(s, research) for s in sets]
        evals = [ev.build(t, plan, research, None, None) for t in theories]

        detected, _ = SaturationDetector().check(evals)
        assert detected is False

    def test_saturation_message_reflects_distinct_scores(self):
        dims = [
            _make_dim("d1", [("a", "A", "low"), ("b", "B", "high")]),
        ]
        plan = _make_plan(dims, max_candidates=2)
        research = _make_research()
        sets = StrategicChoiceGenerator().build(plan, research)
        tgen = TheoryGenerator()
        ev = TheoryEvaluator()
        theories = [tgen.build(s, research) for s in sets]
        evals = [ev.build(t, plan, research, None, None) for t in theories]

        _, msg = SaturationDetector().check(evals)
        assert "distinct" in msg.lower()


# ===========================================================================
# 8. TestComplexitySemanticMapping
# ===========================================================================

class TestComplexitySemanticMapping:
    """execution_complexity values map to expected feasibility scores."""

    @pytest.mark.parametrize("complexity,expected", [
        ("high",   0.5),
        ("medium", 0.75),
        ("low",    1.0),
        ("",       0.75),   # unknown → 0.75 default
        ("UNKNOWN", 0.75),  # unknown → 0.75 default
    ])
    def test_complexity_to_score_mapping(self, complexity, expected):
        choices = [{"dimension": "d1", "selected_value": "opt",
                    "metadata": {"execution_complexity": complexity}}]
        score = _score_ef(choices, n_dims=1)
        assert score == expected, f"complexity={complexity!r}: expected {expected}, got {score}"

    def test_multi_choice_average_is_weighted_equally(self):
        choices = [
            {"dimension": "d1", "selected_value": "a", "metadata": {"execution_complexity": "low"}},
            {"dimension": "d2", "selected_value": "b", "metadata": {"execution_complexity": "medium"}},
        ]
        score = _score_ef(choices, n_dims=2)
        # avg(1.0, 0.75) = 0.875
        assert abs(score - 0.875) < 1e-6

    def test_all_high_complexity_scores_0_5(self):
        choices = [
            {"dimension": f"d{i}", "selected_value": "opt", "metadata": {"execution_complexity": "high"}}
            for i in range(5)
        ]
        score = _score_ef(choices, n_dims=5)
        assert score == 0.5

    def test_all_low_complexity_scores_1_0(self):
        choices = [
            {"dimension": f"d{i}", "selected_value": "opt", "metadata": {"execution_complexity": "low"}}
            for i in range(5)
        ]
        score = _score_ef(choices, n_dims=5)
        assert score == 1.0


# ===========================================================================
# 9. TestDeterminism
# ===========================================================================

class TestDeterminism:
    """Same inputs always produce the same scores (no randomness, no state)."""

    def _build_evals_once(self):
        dims = [
            _make_dim("d1", [("a", "A", "low"), ("b", "B", "high"), ("c", "C", "medium")]),
            _make_dim("d2", [("x", "X", "medium"), ("y", "Y", "low"), ("z", "Z", "high")]),
        ]
        plan = _make_plan(dims, max_candidates=3)
        research = _make_research()
        sets = StrategicChoiceGenerator().build(plan, research)
        tgen = TheoryGenerator()
        ev = TheoryEvaluator()
        theories = [tgen.build(s, research) for s in sets]
        return [ev.build(t, plan, research, None, None) for t in theories]

    def test_scores_are_identical_across_two_runs(self):
        scores_a = [round(e.overall_score, 6) for e in self._build_evals_once()]
        scores_b = [round(e.overall_score, 6) for e in self._build_evals_once()]
        assert scores_a == scores_b

    def test_ef_scores_are_identical_across_two_runs(self):
        def ef_scores():
            evals = self._build_evals_once()
            return [round(e.criteria_scores["execution_feasibility"].score, 6) for e in evals]

        assert ef_scores() == ef_scores()

    def test_winner_is_same_across_runs(self):
        def winner_score():
            return max(e.overall_score for e in self._build_evals_once())

        assert winner_score() == winner_score()


# ===========================================================================
# 10. TestMonitorEngagementRegression
# ===========================================================================

class TestMonitorEngagementRegression:
    """Full pipeline regression against the sports Monitor engagement YAML.

    Validates the PH12.2c acceptance criteria:
      - 4 choice sets, 4 theories, 4 evaluations
      - ≥2 distinct overall scores
      - winner score > runner-up score
      - score_margin > 0
      - saturation_detected = False
    """

    @pytest.fixture(scope="class")
    def engagement_run(self):
        """Load Monitor YAML and run the full coordinator pipeline."""
        import json
        import yaml as _yaml
        from functional_agents.strategy import StrategyCoordinator
        from functional_agents.strategy.strategy_config_resolver import resolve_strategy_config

        eng_path = REPO / "engagements" / "sports_strategy_monitor_v1.yaml"
        ctx_path = REPO / "outputs" / "sports_strategy_monitor_v1.context.json"

        if not eng_path.exists() or not ctx_path.exists():
            pytest.skip("Monitor engagement files not present")

        with eng_path.open() as f:
            engagement = _yaml.safe_load(f)
        raw_strategy = engagement.get("strategy", {}) or {}

        with ctx_path.open() as f:
            data = json.load(f)
        ctx = SimpleNamespace(**data)
        for attr in ("run_id", "question", "execution_profile"):
            if not hasattr(ctx, attr) or not isinstance(getattr(ctx, attr), str):
                setattr(ctx, attr, "")
        for attr in ("profiles",):
            if not hasattr(ctx, attr):
                setattr(ctx, attr, [])
        for attr in ("decision_model", "engagement", "preferred_option",
                     "research_object", "executive_confidence", "decision_analysis", "trace"):
            if not hasattr(ctx, attr):
                setattr(ctx, attr, {})
        for attr in ("strategic_options", "assumptions", "risks", "recommendations", "opportunities"):
            if not hasattr(ctx, attr):
                setattr(ctx, attr, [])

        resolved_cfg = resolve_strategy_config(raw_strategy)
        coord = StrategyCoordinator(config=resolved_cfg.resolved, raw_strategy_yaml=raw_strategy)
        coord.build(ctx)
        return coord

    def test_4_choice_sets_generated(self, engagement_run):
        assert len(engagement_run._choice_sets) == 4

    def test_4_theories_generated(self, engagement_run):
        assert len(engagement_run._theories) == 4

    def test_4_evaluations_generated(self, engagement_run):
        assert len(engagement_run._evaluations) == 4

    def test_at_least_2_distinct_scores(self, engagement_run):
        scores = [round(e.overall_score, 6) for e in engagement_run._evaluations]
        assert len(set(scores)) >= 2, f"Expected ≥2 distinct scores, got {scores}"

    def test_winner_score_exceeds_runner_up(self, engagement_run):
        sel = engagement_run._selection
        assert sel is not None
        winner_ev = next(e for e in engagement_run._evaluations if e.theory_id == sel.winner_theory_id)
        runner_ev = next(
            (e for e in engagement_run._evaluations if e.theory_id == sel.runner_up_theory_id), None
        )
        if runner_ev:
            assert winner_ev.overall_score > runner_ev.overall_score

    def test_score_margin_positive(self, engagement_run):
        sel = engagement_run._selection
        assert sel is not None
        assert sel.score_margin is not None and sel.score_margin > 0, (
            f"Expected score_margin > 0, got {sel.score_margin}"
        )

    def test_saturation_not_detected(self, engagement_run):
        sel = engagement_run._selection
        assert sel is not None
        assert sel.saturation_detected is False, (
            f"Expected saturation_detected=False, got {sel.saturation_detected}"
        )

    def test_all_choices_have_execution_complexity_in_metadata(self, engagement_run):
        for cs in engagement_run._choice_sets:
            for c in cs.choices:
                assert "execution_complexity" in c.metadata, (
                    f"Choice {c.id} missing execution_complexity in metadata"
                )

    def test_execution_feasibility_scores_differ_across_theories(self, engagement_run):
        ef_scores = [
            round(e.criteria_scores["execution_feasibility"].score, 6)
            for e in engagement_run._evaluations
            if "execution_feasibility" in e.criteria_scores
        ]
        assert len(set(ef_scores)) >= 2, f"Expected distinct ef scores, got {ef_scores}"
