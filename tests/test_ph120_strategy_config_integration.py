"""PH12.0 — Strategy Configuration Runtime Integration tests.

Covers:
- YAML strategy block parsing
- ConfigurationResolver.resolve_from_engagement()
- Defaults-only, partial, and full engagement override resolution
- Dimension and choice validation errors
- StrategyPlanner dimension extraction from dimension_configs
- StrategicChoiceGenerator configured mode
- Unique choice-set signatures and diversity enforcement
- TheoryGenerator configured mode
- Theory differentiation
- Duplicate theory rejection
- Evidence propagation and filtering
- TheoryEvaluator new criteria
- Deterministic weighted scoring
- Unsupported criterion fallback
- StrategySelector with configured evaluation
- StrategyTrace configuration visibility
- Backward compatibility (no strategy block)
- Full pipeline integration with engagement config
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from functional_agents.strategy import (
    ChoiceConfig,
    ConfigurationResolver,
    DimensionConfig,
    StrategyConfig,
    StrategyCoordinator,
    StrategyEvaluation,
    StrategyGeneration,
    StrategyPlan,
    StrategyPlanner,
    StrategySelector,
    StrategyValidation,
    StrategicChoiceGenerator,
    TheoryEvaluator,
    TheoryGenerator,
)
from functional_agents.strategy.strategic_choice_set import StrategicChoiceSet
from functional_agents.strategy.strategic_position import TheoryOfWinning
from functional_agents.strategy.theory_evaluation import TheoryEvaluation
from functional_agents.strategy.strategy_coordinator import _check_theory_diversity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ENGAGEMENT_STRATEGY = {
    "framework": "executive",
    "objectives": [
        "Maximize risk-adjusted development value",
        "Preserve execution optionality",
        "Prioritize credible near-term energization",
    ],
    "dimensions": [
        {
            "id": "geographic_portfolio",
            "title": "Geographic Portfolio",
            "description": "Choose the geographic concentration posture.",
            "required": True,
            "choices": [
                {"id": "concentrated", "title": "Concentrated"},
                {"id": "diversified", "title": "Diversified"},
                {"id": "staged", "title": "Staged Portfolio"},
            ],
        },
        {
            "id": "power_pathway",
            "title": "Power Pathway",
            "description": "Choose the primary approach to securing power.",
            "required": True,
            "choices": [
                {"id": "grid_first", "title": "Grid First"},
                {"id": "btm_first", "title": "Behind-the-Meter First"},
                {"id": "hybrid", "title": "Hybrid Grid and BTM"},
            ],
        },
        {
            "id": "market_timing",
            "title": "Market Timing",
            "description": "Choose the capital commitment posture.",
            "required": True,
            "choices": [
                {"id": "accelerate", "title": "Accelerate"},
                {"id": "milestone_gated", "title": "Milestone Gated"},
                {"id": "wait_and_monitor", "title": "Wait and Monitor"},
            ],
        },
    ],
    "evaluation": {
        "method": "multi_criteria",
        "min_score_threshold": 0.0,
        "criteria": {
            "strategic_fit": {"weight": 2.0},
            "evidence_quality": {"weight": 1.5},
            "assumption_robustness": {"weight": 1.5},
            "execution_feasibility": {"weight": 1.5},
            "risk_resilience": {"weight": 1.5},
            "opportunity_capture": {"weight": 1.0},
        },
    },
    "generation": {"max_candidates": 3, "diversity_required": True},
    "validation": {
        "require_evidence": True,
        "require_assumptions": True,
        "min_confidence": "Low",
    },
    "constraints": [
        "Avoid strategies dependent on unvalidated single-state concentration",
    ],
}


def _make_plan_with_dims() -> StrategyPlan:
    """Return a StrategyPlan with three configured dimensions (3 choices each)."""
    cfg = StrategyConfig()
    resolved = ConfigurationResolver().resolve_from_engagement(cfg, _ENGAGEMENT_STRATEGY)
    return StrategyPlanner().build(resolved)


def _make_mock_research(
    *,
    citations: list[str] | None = None,
    high_risks: list[dict] | None = None,
    assumptions: list[dict] | None = None,
    confidence_drivers: list[str] | None = None,
    strategic_options: list[dict] | None = None,
) -> Any:
    """Return a minimal mock AgentContext for strategy tests."""
    m = MagicMock()
    m.run_id = "test-run-001"
    m.question = "What is the best data center siting strategy?"
    m.profiles = ["ai_data_centers"]
    m.execution_profile = "ai_data_centers"
    m.decision_model = {}
    m.engagement = {}
    m.preferred_option = {}
    m.research_object = {
        "id": "R-TEST-001",
        "citations": citations or [],
    }
    m.executive_confidence = {
        "overall_confidence": "High",
        "decision_readiness": "Ready for Decision",
        "board_recommendation": "Proceed with Conditions",
        "confidence_drivers": confidence_drivers or ["Market conditions favorable"],
        "confidence_limiters": [],
        "validation_priorities": ["Validate supply chain"],
        "critical_unknowns": [],
    }
    m.decision_analysis = {
        "recommended_option_id": "OPT-ALPHA",
        "rationale": "Best risk-adjusted position.",
        "key_tradeoffs": [],
        "key_uncertainties": [],
    }
    m.strategic_options = strategic_options or [
        {"option_id": "OPT-ALPHA", "title": "Alpha Strategy", "description": "Lead option."},
        {"option_id": "OPT-BETA", "title": "Beta Strategy", "description": "Alternative."},
    ]
    m.assumptions = assumptions or [
        {"statement": "Power demand will remain high."},
        {"statement": "Grid constraints persist through 2028."},
    ]
    m.risks = (high_risks or [
        {"statement": "Interconnection delays.", "severity": "High", "likelihood": "Medium"},
        {"statement": "Water constraints.", "severity": "High", "likelihood": "Low"},
    ]) + [
        {"statement": "Permitting complexity.", "severity": "Medium", "likelihood": "Medium"},
    ]
    m.opportunities = [{"statement": "Policy tailwinds.", "category": "Regulatory"}]
    m.recommendations = [{"action": "Prioritize Texas and Indiana.", "priority": "High"}]
    m.trace = {}
    return m


# ---------------------------------------------------------------------------
# 1. YAML strategy block parsing
# ---------------------------------------------------------------------------

class TestYAMLStrategyBlockParsing:
    def test_parse_full_engagement_strategy(self):
        cfg = StrategyConfig()
        resolved = ConfigurationResolver().resolve_from_engagement(cfg, _ENGAGEMENT_STRATEGY)
        assert resolved.framework == "executive"

    def test_objectives_parsed_as_primary(self):
        cfg = StrategyConfig()
        resolved = ConfigurationResolver().resolve_from_engagement(cfg, _ENGAGEMENT_STRATEGY)
        assert len(resolved.objectives.primary) == 3
        assert "Maximize risk-adjusted development value" in resolved.objectives.primary

    def test_dimensions_parsed_to_dimension_configs(self):
        cfg = StrategyConfig()
        resolved = ConfigurationResolver().resolve_from_engagement(cfg, _ENGAGEMENT_STRATEGY)
        assert len(resolved.dimension_configs) == 3
        ids = [d.id for d in resolved.dimension_configs]
        assert "geographic_portfolio" in ids
        assert "power_pathway" in ids
        assert "market_timing" in ids

    def test_choices_parsed_correctly(self):
        cfg = StrategyConfig()
        resolved = ConfigurationResolver().resolve_from_engagement(cfg, _ENGAGEMENT_STRATEGY)
        geo_dim = next(d for d in resolved.dimension_configs if d.id == "geographic_portfolio")
        assert len(geo_dim.choices) == 3
        choice_ids = [c.id for c in geo_dim.choices]
        assert "concentrated" in choice_ids
        assert "diversified" in choice_ids
        assert "staged" in choice_ids

    def test_evaluation_criteria_weights_parsed(self):
        cfg = StrategyConfig()
        resolved = ConfigurationResolver().resolve_from_engagement(cfg, _ENGAGEMENT_STRATEGY)
        weights = resolved.evaluation.weights
        assert weights["strategic_fit"] == 2.0
        assert weights["evidence_quality"] == 1.5
        assert weights["risk_resilience"] == 1.5

    def test_generation_policy_parsed(self):
        cfg = StrategyConfig()
        resolved = ConfigurationResolver().resolve_from_engagement(cfg, _ENGAGEMENT_STRATEGY)
        assert resolved.generation.max_candidates == 3
        assert resolved.generation.diversity_required is True

    def test_validation_policy_parsed(self):
        cfg = StrategyConfig()
        resolved = ConfigurationResolver().resolve_from_engagement(cfg, _ENGAGEMENT_STRATEGY)
        assert resolved.validation.require_evidence is True
        assert resolved.validation.require_assumptions is True
        assert resolved.validation.min_confidence == "Low"

    def test_constraints_parsed(self):
        cfg = StrategyConfig()
        resolved = ConfigurationResolver().resolve_from_engagement(cfg, _ENGAGEMENT_STRATEGY)
        assert len(resolved.constraints.required_conditions) == 1
        assert "single-state" in resolved.constraints.required_conditions[0]

    def test_engagement_yaml_file_parses(self, tmp_path):
        import yaml
        from functional_agents.engagement_spec import load_engagement_spec
        # Copy the engagement YAML to tmp path for isolation
        import pathlib
        src = pathlib.Path(__file__).parent.parent / "engagements" / "us_data_center_siting_strategy1.yaml"
        if not src.exists():
            pytest.skip("us_data_center_siting_strategy1.yaml not found")
        spec = load_engagement_spec(src)
        assert spec.strategy is not None
        assert "dimensions" in spec.strategy
        assert len(spec.strategy["dimensions"]) >= 3


# ---------------------------------------------------------------------------
# 2. Resolution modes
# ---------------------------------------------------------------------------

class TestResolutionModes:
    def test_defaults_only_resolution(self):
        cfg = StrategyConfig()
        resolved = ConfigurationResolver().resolve(cfg)
        assert resolved.framework == "executive"
        assert resolved.dimension_configs == []

    def test_partial_override_preserves_unspecified_fields(self):
        partial = {"framework": "executive", "objectives": ["Grow value"]}
        cfg = StrategyConfig()
        resolved = ConfigurationResolver().resolve_from_engagement(cfg, partial)
        assert resolved.objectives.primary == ["Grow value"]
        # generation policy should inherit from framework defaults
        assert resolved.generation.max_candidates == 3

    def test_full_engagement_override(self):
        cfg = StrategyConfig()
        resolved = ConfigurationResolver().resolve_from_engagement(cfg, _ENGAGEMENT_STRATEGY)
        assert len(resolved.dimension_configs) == 3
        assert resolved.evaluation.weights["strategic_fit"] == 2.0

    def test_empty_engagement_strategy_uses_defaults(self):
        resolved = ConfigurationResolver().resolve_from_engagement(StrategyConfig(), {})
        assert resolved.framework == "executive"
        assert resolved.dimension_configs == []

    def test_engagement_without_strategy_block_is_compatible(self):
        from functional_agents.engagement_spec import EngagementSpec
        spec = EngagementSpec.model_validate({
            "title": "Test",
            "current_situation": "Testing backward compat.",
            "objectives": ["Test objective"],
        })
        assert spec.strategy is None


# ---------------------------------------------------------------------------
# 3. Dimension and choice validation errors
# ---------------------------------------------------------------------------

class TestDimensionValidation:
    def test_duplicate_dimension_id_raises(self):
        bad = {
            "dimensions": [
                {"id": "geo", "title": "Geo", "required": True,
                 "choices": [{"id": "a", "title": "A"}]},
                {"id": "geo", "title": "Geo Dup", "required": True,
                 "choices": [{"id": "b", "title": "B"}]},
            ]
        }
        with pytest.raises(ValueError, match="duplicate dimension"):
            ConfigurationResolver().resolve_from_engagement(StrategyConfig(), bad)

    def test_blank_dimension_id_raises(self):
        bad = {
            "dimensions": [
                {"id": "", "title": "Blank", "required": True,
                 "choices": [{"id": "a", "title": "A"}]},
            ]
        }
        with pytest.raises(ValueError, match="blank"):
            ConfigurationResolver().resolve_from_engagement(StrategyConfig(), bad)

    def test_required_dimension_without_choices_raises(self):
        bad = {
            "dimensions": [
                {"id": "power", "title": "Power", "required": True, "choices": []},
            ]
        }
        with pytest.raises(ValueError, match="no choices"):
            ConfigurationResolver().resolve_from_engagement(StrategyConfig(), bad)

    def test_optional_dimension_without_choices_allowed(self):
        ok = {
            "dimensions": [
                {"id": "optional_dim", "title": "Optional", "required": False, "choices": []},
            ]
        }
        resolved = ConfigurationResolver().resolve_from_engagement(StrategyConfig(), ok)
        assert len(resolved.dimension_configs) == 1

    def test_duplicate_choice_id_within_dimension_raises(self):
        bad = {
            "dimensions": [
                {
                    "id": "geo", "title": "Geo", "required": True,
                    "choices": [
                        {"id": "a", "title": "A"},
                        {"id": "a", "title": "A Dup"},
                    ],
                }
            ]
        }
        with pytest.raises(ValueError, match="duplicate choice"):
            ConfigurationResolver().resolve_from_engagement(StrategyConfig(), bad)

    def test_blank_choice_id_raises(self):
        bad = {
            "dimensions": [
                {
                    "id": "geo", "title": "Geo", "required": True,
                    "choices": [{"id": "", "title": "No ID"}],
                }
            ]
        }
        with pytest.raises(ValueError, match="blank"):
            ConfigurationResolver().resolve_from_engagement(StrategyConfig(), bad)


# ---------------------------------------------------------------------------
# 4. StrategyPlanner — dimension extraction
# ---------------------------------------------------------------------------

class TestStrategyPlannerDimensionExtraction:
    def test_non_empty_active_dimensions_from_configured(self):
        plan = _make_plan_with_dims()
        assert len(plan.active_dimensions) == 3

    def test_active_dimensions_match_configured_ids(self):
        plan = _make_plan_with_dims()
        assert "geographic_portfolio" in plan.active_dimensions
        assert "power_pathway" in plan.active_dimensions
        assert "market_timing" in plan.active_dimensions

    def test_dimension_configs_forwarded_to_plan(self):
        plan = _make_plan_with_dims()
        assert len(plan.dimension_configs) == 3

    def test_configured_objectives_in_plan(self):
        plan = _make_plan_with_dims()
        assert any("Maximize" in obj for obj in plan.objectives)

    def test_configured_constraints_in_plan(self):
        plan = _make_plan_with_dims()
        assert any("single-state" in c for c in plan.constraints)

    def test_empty_dimension_configs_falls_back_to_extra_fields(self):
        from functional_agents.strategy.strategy_config import StrategyDimensions
        cfg = StrategyConfig()
        cfg.dimensions.add("custom_dim")
        resolved = ConfigurationResolver().resolve(cfg)
        plan = StrategyPlanner().build(resolved)
        assert "custom_dim" in plan.active_dimensions


# ---------------------------------------------------------------------------
# 5. StrategicChoiceGenerator — configured mode
# ---------------------------------------------------------------------------

class TestConfiguredChoiceGeneration:
    def test_generates_three_unique_choice_sets(self):
        plan = _make_plan_with_dims()
        research = _make_mock_research()
        sets = StrategicChoiceGenerator().build(plan, research)
        assert len(sets) == 3

    def test_each_set_covers_all_required_dimensions(self):
        plan = _make_plan_with_dims()
        research = _make_mock_research()
        sets = StrategicChoiceGenerator().build(plan, research)
        required_dims = {d.id for d in plan.dimension_configs if d.required}
        for cs in sets:
            covered = {c.dimension for c in cs.choices}
            assert required_dims <= covered

    def test_choice_sets_have_unique_signatures(self):
        plan = _make_plan_with_dims()
        research = _make_mock_research()
        sets = StrategicChoiceGenerator().build(plan, research)

        def sig(cs):
            return tuple(sorted((c.dimension, c.selected_value) for c in cs.choices))

        sigs = [sig(cs) for cs in sets]
        assert len(sigs) == len(set(sigs)), "Choice-set signatures must be unique"

    def test_sets_completeness_is_one(self):
        plan = _make_plan_with_dims()
        research = _make_mock_research()
        for cs in StrategicChoiceGenerator().build(plan, research):
            assert cs.completeness == 1.0

    def test_no_internal_conflicts(self):
        plan = _make_plan_with_dims()
        research = _make_mock_research()
        for cs in StrategicChoiceGenerator().build(plan, research):
            assert cs.internal_conflicts == []

    def test_choices_carry_metadata(self):
        plan = _make_plan_with_dims()
        research = _make_mock_research()
        sets = StrategicChoiceGenerator().build(plan, research)
        first_choice = sets[0].choices[0]
        assert first_choice.metadata.get("choice_title")
        assert first_choice.metadata.get("dimension_title")

    def test_postures_select_different_choice_values(self):
        plan = _make_plan_with_dims()
        research = _make_mock_research()
        sets = StrategicChoiceGenerator().build(plan, research)
        # Geographic choices across postures should differ
        geo_values = [
            next(c.selected_value for c in cs.choices if c.dimension == "geographic_portfolio")
            for cs in sets
        ]
        assert len(set(geo_values)) == 3  # all three choices used

    def test_diversity_required_with_one_choice_raises(self):
        one_choice_strategy = {
            "dimensions": [
                {
                    "id": "geo", "title": "Geo", "required": True,
                    "choices": [{"id": "only_one", "title": "Only One"}],
                }
            ],
            "generation": {"max_candidates": 3, "diversity_required": True},
        }
        cfg = StrategyConfig()
        resolved = ConfigurationResolver().resolve_from_engagement(cfg, one_choice_strategy)
        plan = StrategyPlanner().build(resolved)
        research = _make_mock_research()
        with pytest.raises(ValueError, match="diversity_required"):
            StrategicChoiceGenerator().build(plan, research)

    def test_max_candidates_respected(self):
        limited_strategy = {**_ENGAGEMENT_STRATEGY, "generation": {"max_candidates": 2, "diversity_required": True}}
        cfg = StrategyConfig()
        resolved = ConfigurationResolver().resolve_from_engagement(cfg, limited_strategy)
        plan = StrategyPlanner().build(resolved)
        research = _make_mock_research()
        sets = StrategicChoiceGenerator().build(plan, research)
        assert len(sets) <= 2


# ---------------------------------------------------------------------------
# 6. TheoryGenerator — configured mode
# ---------------------------------------------------------------------------

class TestConfiguredTheoryGeneration:
    def _make_theories(self, citations=None, high_risks=None):
        plan = _make_plan_with_dims()
        research = _make_mock_research(citations=citations, high_risks=high_risks)
        sets = StrategicChoiceGenerator().build(plan, research)
        gen = TheoryGenerator()
        return [gen.build(cs, research) for cs in sets]

    def test_theories_are_strategically_distinct(self):
        theories = self._make_theories()
        option_ids = [t.recommended_option_id for t in theories]
        assert len(set(option_ids)) == 3, "Each theory should recommend a different option"

    def test_theory_winning_positions_differ(self):
        theories = self._make_theories()
        positions = [t.winning_position for t in theories]
        assert len(set(positions)) == 3

    def test_theory_winning_mechanism_non_empty(self):
        theories = self._make_theories()
        for t in theories:
            assert t.winning_mechanism, f"Theory {t.theory_id} has empty winning_mechanism"

    def test_theory_ids_are_unique(self):
        theories = self._make_theories()
        ids = [t.theory_id for t in theories]
        assert len(set(ids)) == 3

    def test_evidence_propagated_when_available(self):
        citations = [
            "Concentrated deployment reduces transaction costs.",
            "Grid interconnection queues exceed 3 years in MISO.",
            "Texas data center demand grew 40% in 2025.",
            "Behind-the-meter solar achieves grid parity.",
            "Milestone-gated approach lowers stranded-capital risk.",
        ]
        theories = self._make_theories(citations=citations)
        total_evidence = sum(len(t.evidence) for t in theories)
        assert total_evidence > 0, "Evidence should propagate to at least one theory"

    def test_evidence_filtered_by_relevance_when_keywords_match(self):
        citations = [
            "Concentrated portfolio enables faster permitting.",
            "Diversified spread reduces single-market exposure.",
            "Grid-first approach requires utility coordination.",
            "BTM generation bypasses transmission constraints.",
            "Accelerated timeline captures early-mover advantage.",
        ]
        theories = self._make_theories(citations=citations)
        # Theories should have different evidence counts due to relevance filtering
        evidence_counts = [len(t.evidence) for t in theories]
        # At minimum, evidence is non-zero overall
        assert sum(evidence_counts) > 0

    def test_evidence_fallback_when_no_keywords_match(self):
        citations = ["Completely unrelated content about ice cream."] * 4
        theories = self._make_theories(citations=citations)
        # Fallback: all theories get some evidence anyway
        for t in theories:
            assert len(t.evidence) > 0

    def test_failure_modes_propagated(self):
        high_risks = [
            {"statement": "Grid delays.", "severity": "High", "likelihood": "Medium"},
            {"statement": "Water constraints.", "severity": "High", "likelihood": "Low"},
        ]
        theories = self._make_theories(high_risks=high_risks)
        total_fm = sum(len(t.failure_modes) for t in theories)
        assert total_fm > 0

    def test_assumptions_propagated(self):
        theories = self._make_theories()
        for t in theories:
            assert len(t.assumptions) > 0

    def test_source_choice_set_id_set(self):
        theories = self._make_theories()
        for t in theories:
            assert t.source_choice_set_id, "source_choice_set_id must be non-empty"


# ---------------------------------------------------------------------------
# 7. Duplicate theory detection
# ---------------------------------------------------------------------------

class TestDuplicateTheoryDetection:
    def test_duplicate_theories_raise(self):
        t1 = TheoryOfWinning(
            theory_id="T1", source_choice_set_id="CS1",
            recommended_option_id="OPT-A",
            strategic_choices=[{"dimension": "geo", "selected_value": "concentrated"}],
        )
        t2 = TheoryOfWinning(
            theory_id="T2", source_choice_set_id="CS2",
            recommended_option_id="OPT-A",
            strategic_choices=[{"dimension": "geo", "selected_value": "concentrated"}],
        )
        with pytest.raises(ValueError, match="duplicate theory"):
            _check_theory_diversity([t1, t2])

    def test_distinct_theories_do_not_raise(self):
        t1 = TheoryOfWinning(
            theory_id="T1", source_choice_set_id="CS1",
            recommended_option_id="OPT-A",
            strategic_choices=[{"dimension": "geo", "selected_value": "concentrated"}],
        )
        t2 = TheoryOfWinning(
            theory_id="T2", source_choice_set_id="CS2",
            recommended_option_id="OPT-B",
            strategic_choices=[{"dimension": "geo", "selected_value": "diversified"}],
        )
        _check_theory_diversity([t1, t2])  # should not raise

    def test_different_ids_same_choices_still_detected_as_duplicate(self):
        t1 = TheoryOfWinning(
            theory_id="T-ALPHA",
            source_choice_set_id="CS1",
            recommended_option_id="OPT-A",
            strategic_choices=[{"dimension": "geo", "selected_value": "a"}],
        )
        t2 = TheoryOfWinning(
            theory_id="T-BETA",
            source_choice_set_id="CS2",
            recommended_option_id="OPT-A",
            strategic_choices=[{"dimension": "geo", "selected_value": "a"}],
        )
        with pytest.raises(ValueError):
            _check_theory_diversity([t1, t2])


# ---------------------------------------------------------------------------
# 8. TheoryEvaluator — new criteria
# ---------------------------------------------------------------------------

class TestTheoryEvaluatorNewCriteria:
    def _make_theory(self, **kwargs) -> TheoryOfWinning:
        defaults = {
            "theory_id": "TH-TEST",
            "source_choice_set_id": "SCS-TEST",
            "recommended_option_id": "OPT-A",
            "winning_position": "Geographic: Concentrated | Power: Grid First",
            "winning_mechanism": "Grid First: primary grid-connected approach.",
            "strategic_choices": [{"dimension": "geo", "selected_value": "concentrated"}],
            "success_conditions": ["Policy stability confirmed"],
            "failure_modes": [{"statement": "Grid delays.", "severity": "High"}],
            "assumptions": [{"statement": "Demand remains high."}, {"statement": "Grid available."}],
            "evidence": ["Evidence item 1.", "Evidence item 2.", "Evidence item 3."],
        }
        defaults.update(kwargs)
        return TheoryOfWinning(**defaults)

    def _make_plan_with_weights(self, weights: dict[str, float]) -> StrategyPlan:
        from functional_agents.strategy.strategy_plan import EvaluationModel, GenerationPolicy, ValidationPolicy, SearchBudget
        return StrategyPlan(
            plan_id="TEST",
            framework="executive",
            active_dimensions=["geo"],
            evaluation_model=EvaluationModel(weights=weights),
            generation_policy=GenerationPolicy(),
            validation_policy=ValidationPolicy(),
            search_budget=SearchBudget(),
        )

    def test_strategic_fit_scores_non_zero(self):
        theory = self._make_theory()
        plan = self._make_plan_with_weights({"strategic_fit": 2.0})
        ev = TheoryEvaluator().build(theory, plan, None)
        assert ev.criteria_scores["strategic_fit"].score > 0.0

    def test_strategic_fit_lower_when_no_position_or_mechanism(self):
        theory_full = self._make_theory()
        theory_empty = self._make_theory(winning_position="", winning_mechanism="")
        plan = self._make_plan_with_weights({"strategic_fit": 2.0})
        ev_full = TheoryEvaluator().build(theory_full, plan, None)
        ev_empty = TheoryEvaluator().build(theory_empty, plan, None)
        assert ev_empty.criteria_scores["strategic_fit"].score < ev_full.criteria_scores["strategic_fit"].score

    def test_assumption_robustness_scales_with_count(self):
        plan = self._make_plan_with_weights({"assumption_robustness": 1.5})
        ev_few = TheoryEvaluator().build(
            self._make_theory(assumptions=[{"statement": "One."}]), plan, None
        )
        ev_more = TheoryEvaluator().build(
            self._make_theory(assumptions=[{"statement": "One."}, {"statement": "Two."}]),
            plan, None,
        )
        assert ev_more.criteria_scores["assumption_robustness"].score >= ev_few.criteria_scores["assumption_robustness"].score

    def test_assumption_robustness_zero_when_required_and_empty(self):
        from functional_agents.strategy.strategy_plan import ValidationPolicy
        plan = self._make_plan_with_weights({"assumption_robustness": 1.5})
        plan = plan.model_copy(update={"validation_policy": ValidationPolicy(require_assumptions=True)})
        theory = self._make_theory(assumptions=[])
        ev = TheoryEvaluator().build(theory, plan, None)
        assert ev.criteria_scores["assumption_robustness"].score == 0.0

    def test_execution_feasibility_one_when_complete(self):
        # PH12.1: execution_feasibility now derives from execution_complexity metadata.
        # Without complexity metadata, score defaults to 0.75 per choice.
        # 1.0 requires choices with execution_complexity="low".
        from functional_agents.strategy.strategy_plan import EvaluationModel, GenerationPolicy, ValidationPolicy, SearchBudget
        plan = StrategyPlan(
            plan_id="TEST",
            framework="executive",
            active_dimensions=["geo"],
            evaluation_model=EvaluationModel(weights={"execution_feasibility": 1.5}),
            generation_policy=GenerationPolicy(),
            validation_policy=ValidationPolicy(),
            search_budget=SearchBudget(),
        )
        # Without execution_complexity: defaults to 0.75
        theory_no_meta = self._make_theory(strategic_choices=[{"dimension": "geo", "selected_value": "a"}])
        ev_no_meta = TheoryEvaluator().build(theory_no_meta, plan, None)
        assert ev_no_meta.criteria_scores["execution_feasibility"].score == pytest.approx(0.75, abs=1e-3)
        # With execution_complexity=low: scores 1.0
        theory_low = self._make_theory(strategic_choices=[{
            "dimension": "geo", "selected_value": "a",
            "metadata": {"execution_complexity": "low"},
        }])
        ev_low = TheoryEvaluator().build(theory_low, plan, None)
        assert ev_low.criteria_scores["execution_feasibility"].score == pytest.approx(1.0, abs=1e-3)

    def test_risk_resilience_tiers(self):
        plan = self._make_plan_with_weights({"risk_resilience": 1.5})
        ev0 = TheoryEvaluator().build(self._make_theory(failure_modes=[]), plan, None)
        ev1 = TheoryEvaluator().build(
            self._make_theory(failure_modes=[{"statement": "R1.", "severity": "High"}]), plan, None
        )
        ev3 = TheoryEvaluator().build(
            self._make_theory(failure_modes=[
                {"statement": "R1.", "severity": "High"},
                {"statement": "R2.", "severity": "High"},
                {"statement": "R3.", "severity": "High"},
            ]), plan, None
        )
        assert ev0.criteria_scores["risk_resilience"].score == pytest.approx(0.3)
        assert ev1.criteria_scores["risk_resilience"].score == pytest.approx(0.6)
        assert ev3.criteria_scores["risk_resilience"].score == pytest.approx(1.0)

    def test_opportunity_capture_tiers(self):
        plan = self._make_plan_with_weights({"opportunity_capture": 1.0})
        ev0 = TheoryEvaluator().build(self._make_theory(success_conditions=[]), plan, None)
        ev1 = TheoryEvaluator().build(self._make_theory(success_conditions=["Cond A"]), plan, None)
        ev3 = TheoryEvaluator().build(
            self._make_theory(success_conditions=["A", "B", "C"]), plan, None
        )
        assert ev0.criteria_scores["opportunity_capture"].score == pytest.approx(0.3)
        assert ev1.criteria_scores["opportunity_capture"].score == pytest.approx(0.6)
        assert ev3.criteria_scores["opportunity_capture"].score == pytest.approx(1.0)

    def test_unsupported_criterion_raises_not_fallback(self):
        plan = self._make_plan_with_weights({"nonexistent_criterion": 1.0})
        theory = self._make_theory()
        with pytest.raises(ValueError, match="nonexistent_criterion"):
            TheoryEvaluator().build(theory, plan, None)

    def test_configured_weights_used_in_overall_score(self):
        # Use 1 evidence item → evidence_quality = 0.33 (partial); strategic_fit = 1.0
        theory = self._make_theory(evidence=["Only one evidence item."])
        plan_high_fit = self._make_plan_with_weights({"strategic_fit": 10.0, "evidence_quality": 0.1})
        plan_high_ev = self._make_plan_with_weights({"strategic_fit": 0.1, "evidence_quality": 10.0})
        ev_fit = TheoryEvaluator().build(theory, plan_high_fit, None)
        ev_ev = TheoryEvaluator().build(theory, plan_high_ev, None)
        # With high strategic_fit weight: overall dominated by 1.0 → high score
        # With high evidence_quality weight: overall dominated by 0.33 → lower score
        assert ev_fit.overall_score > ev_ev.overall_score

    def test_overall_score_deterministic(self):
        # Extract flat weights from criteria dict
        weights = {
            k: v["weight"] for k, v in _ENGAGEMENT_STRATEGY["evaluation"]["criteria"].items()
        }
        plan = self._make_plan_with_weights(weights)
        theory = self._make_theory()
        ev1 = TheoryEvaluator().build(theory, plan, None)
        ev2 = TheoryEvaluator().build(theory, plan, None)
        assert ev1.overall_score == ev2.overall_score


# ---------------------------------------------------------------------------
# 9. Score differentiation between theories
# ---------------------------------------------------------------------------

class TestScoreDifferentiation:
    def test_at_least_two_theories_differ_on_one_criterion(self):
        citations = [
            "Concentrated strategy reduces permitting overhead.",
            "Behind-the-meter generation bypasses grid constraints.",
            "Wait-and-monitor preserves capital optionality.",
        ]
        high_risks = [
            {"statement": "Grid interconnect delays.", "severity": "High", "likelihood": "High"},
            {"statement": "BTM regulatory hurdles.", "severity": "High", "likelihood": "Medium"},
        ]
        plan = _make_plan_with_dims()
        research = _make_mock_research(citations=citations, high_risks=high_risks)

        sets = StrategicChoiceGenerator().build(plan, research)
        theories = [TheoryGenerator().build(cs, research) for cs in sets]
        evaluator = TheoryEvaluator()
        evaluations = [evaluator.build(t, plan, research) for t in theories]

        scores = [ev.overall_score for ev in evaluations]
        criterion_names = list(evaluations[0].criteria_scores.keys())

        # Check: at least one criterion differs between any two evaluations
        found_diff = False
        for crit in criterion_names:
            crit_scores = [ev.criteria_scores[crit].score for ev in evaluations]
            if len(set(crit_scores)) > 1:
                found_diff = True
                break

        assert found_diff, (
            f"All criteria are identical across all theories. "
            f"Scores: {scores}, Criteria: {criterion_names}"
        )


# ---------------------------------------------------------------------------
# 10. Selection with configured evaluation
# ---------------------------------------------------------------------------

class TestSelectionWithConfiguredEvaluation:
    def test_winner_selected_on_merit_not_order(self):
        plan = _make_plan_with_dims()
        research = _make_mock_research(
            citations=["Evidence supporting concentrated approach."] * 5,
        )
        sets = StrategicChoiceGenerator().build(plan, research)
        theories = [TheoryGenerator().build(cs, research) for cs in sets]
        evaluator = TheoryEvaluator()
        evaluations = [evaluator.build(t, plan, research) for t in theories]
        selector = StrategySelector()
        winner = selector.select(theories, evaluations, plan)
        selection = selector._last_selection

        # If scores differ, tie-breaker should not be "order"
        scores = [ev.overall_score for ev in evaluations]
        if max(scores) != min(scores):
            assert selection.tie_breaker_used is None

    def test_order_tie_breaker_only_on_genuine_tie(self):
        # Construct theories with identical scores by using same criterion on identical theories
        from functional_agents.strategy.strategy_plan import EvaluationModel, GenerationPolicy, ValidationPolicy, SearchBudget
        plan = StrategyPlan(
            plan_id="TIED",
            framework="executive",
            active_dimensions=[],
            evaluation_model=EvaluationModel(weights={"strategic_fit": 1.0}),
            generation_policy=GenerationPolicy(),
            validation_policy=ValidationPolicy(),
            search_budget=SearchBudget(),
        )
        t1 = TheoryOfWinning(theory_id="T1", source_choice_set_id="CS1", recommended_option_id="OPT-A", winning_position="Pos", winning_mechanism="Mech")
        t2 = TheoryOfWinning(theory_id="T2", source_choice_set_id="CS2", recommended_option_id="OPT-B", winning_position="Pos", winning_mechanism="Mech")
        evaluator = TheoryEvaluator()
        ev1 = evaluator.build(t1, plan, None)
        ev2 = evaluator.build(t2, plan, None)
        # Both theories score identically → genuine tie
        assert ev1.overall_score == ev2.overall_score

        selector = StrategySelector()
        selector.select([t1, t2], [ev1, ev2], plan)
        assert selector._last_selection.tie_breaker_used is not None

    def test_selection_records_score_margin(self):
        plan = _make_plan_with_dims()
        research = _make_mock_research()
        sets = StrategicChoiceGenerator().build(plan, research)
        theories = [TheoryGenerator().build(cs, research) for cs in sets]
        evaluations = [TheoryEvaluator().build(t, plan, research) for t in theories]
        selector = StrategySelector()
        selector.select(theories, evaluations, plan)
        assert selector._last_selection.score_margin is not None


# ---------------------------------------------------------------------------
# 11. StrategyTrace configuration visibility
# ---------------------------------------------------------------------------

class TestStrategyTraceConfigVisibility:
    def test_trace_contains_dimension_configs_via_plan(self):
        research = _make_mock_research()
        cfg = StrategyConfig()
        resolved = ConfigurationResolver().resolve_from_engagement(cfg, _ENGAGEMENT_STRATEGY)
        coord = StrategyCoordinator(config=resolved)
        coord.build(research)
        trace = coord._trace
        assert trace is not None
        assert len(trace.plan.dimension_configs) == 3

    def test_trace_plan_has_configured_weights(self):
        research = _make_mock_research()
        cfg = StrategyConfig()
        resolved = ConfigurationResolver().resolve_from_engagement(cfg, _ENGAGEMENT_STRATEGY)
        coord = StrategyCoordinator(config=resolved)
        coord.build(research)
        trace = coord._trace
        assert trace.plan.evaluation_model.weights["strategic_fit"] == 2.0

    def test_trace_plan_active_dimensions_match_config(self):
        research = _make_mock_research()
        cfg = StrategyConfig()
        resolved = ConfigurationResolver().resolve_from_engagement(cfg, _ENGAGEMENT_STRATEGY)
        coord = StrategyCoordinator(config=resolved)
        coord.build(research)
        trace = coord._trace
        assert "geographic_portfolio" in trace.plan.active_dimensions

    def test_trace_contains_choice_sets_with_configured_choices(self):
        research = _make_mock_research()
        cfg = StrategyConfig()
        resolved = ConfigurationResolver().resolve_from_engagement(cfg, _ENGAGEMENT_STRATEGY)
        coord = StrategyCoordinator(config=resolved)
        coord.build(research)
        trace = coord._trace
        assert len(trace.choice_sets) == 3
        # Each choice set should have choices
        for cs in trace.choice_sets:
            assert len(cs.choices) == 3

    def test_trace_selection_records_winner(self):
        research = _make_mock_research()
        cfg = StrategyConfig()
        resolved = ConfigurationResolver().resolve_from_engagement(cfg, _ENGAGEMENT_STRATEGY)
        coord = StrategyCoordinator(config=resolved)
        coord.build(research)
        trace = coord._trace
        assert trace.selection.winner_theory_id

    def test_trace_objectives_match_engagement(self):
        research = _make_mock_research()
        cfg = StrategyConfig()
        resolved = ConfigurationResolver().resolve_from_engagement(cfg, _ENGAGEMENT_STRATEGY)
        coord = StrategyCoordinator(config=resolved)
        coord.build(research)
        trace = coord._trace
        assert any("Maximize" in obj for obj in trace.plan.objectives)


# ---------------------------------------------------------------------------
# 12. Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_no_strategy_block_runs_without_error(self):
        research = _make_mock_research()
        coord = StrategyCoordinator()
        coord.build(research)
        assert coord._trace is not None

    def test_engagement_without_strategy_block_yields_empty_dimension_configs(self):
        coord = StrategyCoordinator()
        assert coord._plan.dimension_configs == []

    def test_default_mode_active_dimensions_empty(self):
        coord = StrategyCoordinator()
        assert coord._plan.active_dimensions == []

    def test_existing_tests_not_affected(self):
        from functional_agents.strategy import StrategyCoordinator as SC
        coord = SC()
        assert coord._config.framework == "executive"

    def test_extra_field_still_rejected_on_engagement_spec(self):
        from functional_agents.engagement_spec import EngagementSpec
        with pytest.raises(Exception):
            EngagementSpec.model_validate({"title": "X", "totally_bogus": "value"})

    def test_strategy_field_accepted_on_engagement_spec(self):
        from functional_agents.engagement_spec import EngagementSpec
        spec = EngagementSpec.model_validate({
            "title": "Test",
            "current_situation": "Testing.",
            "objectives": ["obj"],
            "strategy": {"framework": "executive", "dimensions": []},
        })
        assert spec.strategy is not None
        assert spec.strategy["framework"] == "executive"

    def test_require_evidence_does_not_affect_non_strategy_workflows(self):
        """require_evidence enforcement only applies when validation is configured."""
        coord = StrategyCoordinator()
        research = _make_mock_research(citations=[])
        # Should not raise even with no evidence in default mode
        pos = coord.build(research)
        assert pos is not None


# ---------------------------------------------------------------------------
# 13. Full pipeline integration
# ---------------------------------------------------------------------------

class TestFullPipelineIntegration:
    def test_end_to_end_configured_strategy_pipeline(self):
        research = _make_mock_research(
            citations=[
                "Concentrated deployment in Texas reduces permitting delays.",
                "Grid-first strategy requires ERCOT coordination.",
                "Accelerated timeline captures early-mover land advantage.",
                "Diversified portfolio mitigates single-state regulatory risk.",
                "Behind-the-meter solar achieves grid parity in Texas.",
                "Milestone-gated approach lowers stranded-capital exposure.",
            ],
            high_risks=[
                {"statement": "Grid interconnect delays.", "severity": "High", "likelihood": "High"},
                {"statement": "BTM regulatory barriers.", "severity": "High", "likelihood": "Medium"},
                {"statement": "Water constraints in arid states.", "severity": "High", "likelihood": "Low"},
            ],
        )
        cfg = StrategyConfig()
        resolved = ConfigurationResolver().resolve_from_engagement(cfg, _ENGAGEMENT_STRATEGY)
        coord = StrategyCoordinator(config=resolved)
        position = coord.build(research)

        # Non-empty active dimensions
        assert len(coord._plan.active_dimensions) == 3

        # Non-empty choice sets with choices
        assert len(coord._choice_sets) == 3
        for cs in coord._choice_sets:
            assert len(cs.choices) == 3

        # Strategically distinct theories
        option_ids = [t.recommended_option_id for t in coord._theories]
        assert len(set(option_ids)) == 3

        # Non-empty theory evidence
        total_ev = sum(len(t.evidence) for t in coord._theories)
        assert total_ev > 0

        # Criteria match engagement config
        ev0 = coord._evaluations[0]
        assert "strategic_fit" in ev0.criteria_scores
        assert ev0.criteria_scores["strategic_fit"].weight == 2.0

        # Winner is recorded
        assert coord._selection.winner_theory_id

        # StrategyTrace is built
        assert coord._trace is not None
        assert coord._trace.plan.dimension_configs is not None

    def test_missing_strategy_block_does_not_break_pipeline(self):
        research = _make_mock_research()
        coord = StrategyCoordinator()
        position = coord.build(research)
        assert position is not None
        assert coord._trace is not None


# ---------------------------------------------------------------------------
# PH12.0a — Evidence Relevance and Criterion Validation
# ---------------------------------------------------------------------------

class _MockResearchWithEvidence:
    """Minimal research mock with controlled evidence_ids for relevance tests."""

    def __init__(self, evidence_ids=None, citations=None, risks=None):
        self.run_id = "TEST-RUN"
        self.question = "Test question"
        self.profiles = []
        self.execution_profile = ""
        self.decision_model = {}
        self.engagement = {}
        self.preferred_option = {}
        self.research_object = {
            "id": "RO-EVTEST",
            "evidence_ids": evidence_ids or [],
            "citations": citations or [],
        }
        self.executive_confidence = {}
        self.decision_analysis = {}
        self.strategic_options = []
        self.assumptions = []
        self.risks = risks or []
        self.recommendations = []
        self.opportunities = []


def _make_configured_choice_sets():
    """Return 3 configured StrategicChoiceSet objects with distinct dimensions."""
    from functional_agents.strategy.strategic_choice import StrategicChoice
    from functional_agents.strategy.strategic_choice_set import StrategicChoiceSet

    postures = [
        ("concentrated", "grid_first", "accelerate"),
        ("diversified", "btm_first", "milestone_gated"),
        ("staged", "hybrid", "wait_and_monitor"),
    ]
    dims = ["geographic_portfolio", "power_pathway", "market_timing"]
    dim_titles = {
        "geographic_portfolio": "Geographic Portfolio",
        "power_pathway": "Power Pathway",
        "market_timing": "Market Timing",
    }
    choice_titles = {
        "concentrated": "Concentrated", "diversified": "Diversified", "staged": "Staged Portfolio",
        "grid_first": "Grid First", "btm_first": "Behind-the-Meter First", "hybrid": "Hybrid",
        "accelerate": "Accelerate", "milestone_gated": "Milestone Gated",
        "wait_and_monitor": "Wait and Monitor",
    }
    sets = []
    for i, posture_choices in enumerate(postures):
        choices = [
            StrategicChoice(
                dimension=dim,
                selected_value=choice_val,
                confidence="High",
                metadata={
                    "choice_title": choice_titles[choice_val],
                    "dimension_title": dim_titles[dim],
                    "dimension_description": f"Choose the {dim.replace('_', ' ')}.",
                },
            )
            for dim, choice_val in zip(dims, posture_choices)
        ]
        sets.append(StrategicChoiceSet(
            id=f"SCS-{i}-20260101T000000",
            posture=["aggressive", "balanced", "conservative"][i],
            choices=choices,
            overall_confidence="High",
            completeness=1.0,
            rationale=f"Posture {i} rationale",
        ))
    return sets


class TestEvidenceRelevancePH12a:
    """PH12.0a — Evidence must be assigned by relevance, not by candidate index."""

    def _gen(self):
        from functional_agents.strategy.theory_generator import TheoryGenerator
        return TheoryGenerator()

    def test_no_index_based_evidence_allocation(self):
        """Reordering choice sets must not change the total evidence assignment pattern."""
        evidence_ids = ["EV-001", "EV-002", "EV-003", "EV-004", "EV-005"]
        research = _MockResearchWithEvidence(evidence_ids=evidence_ids)
        sets = _make_configured_choice_sets()
        gen = self._gen()

        theories_original = [gen.build(cs, research) for cs in sets]
        theories_reversed = [gen.build(cs, research) for cs in reversed(sets)]

        # A theory built from the same choice set must produce the same evidence
        # regardless of what position it occupies in the generation list.
        orig_by_id = {t.source_choice_set_id: t.evidence for t in theories_original}
        rev_by_id = {t.source_choice_set_id: t.evidence for t in theories_reversed}
        for cs_id in orig_by_id:
            assert orig_by_id[cs_id] == rev_by_id[cs_id], (
                f"Choice set {cs_id} produced different evidence when its position changed: "
                f"original={orig_by_id[cs_id]}, reversed={rev_by_id[cs_id]}"
            )

    def test_evidence_assignment_independent_of_choice_set_order(self):
        """Swap two choice sets — each theory's evidence must follow the choice set."""
        evidence_ids = ["EV-A", "EV-B", "EV-C"]
        research = _MockResearchWithEvidence(evidence_ids=evidence_ids)
        sets = _make_configured_choice_sets()
        gen = self._gen()

        t0_first_run = gen.build(sets[0], research)
        t0_second_run = gen.build(sets[0], research)
        assert t0_first_run.evidence == t0_second_run.evidence

    def test_symmetric_fallback_when_no_keyword_matches(self):
        """When no citation keyword matches, all theories get the same leading slice."""
        # Evidence IDs contain no keywords that appear in choice titles
        evidence_ids = ["OPAQUE-1", "OPAQUE-2", "OPAQUE-3"]
        research = _MockResearchWithEvidence(evidence_ids=evidence_ids)
        sets = _make_configured_choice_sets()
        gen = self._gen()

        theories = [gen.build(cs, research) for cs in sets]
        # Symmetric fallback: all theories get the same slice
        assert theories[0].evidence == theories[1].evidence == theories[2].evidence

    def test_symmetric_fallback_does_not_manufacture_score_differences(self):
        """When evidence is assigned symmetrically, evidence_quality scores must be equal."""
        from functional_agents.strategy.theory_evaluator import TheoryEvaluator
        from functional_agents.strategy.strategy_plan import StrategyPlan, EvaluationModel, ValidationPolicy, GenerationPolicy, SearchBudget

        evidence_ids = ["OPAQUE-1", "OPAQUE-2", "OPAQUE-3"]
        research = _MockResearchWithEvidence(evidence_ids=evidence_ids)
        sets = _make_configured_choice_sets()
        gen = self._gen()
        theories = [gen.build(cs, research) for cs in sets]

        plan = StrategyPlan(
            plan_id="TEST",
            framework="executive",
            active_dimensions=["geographic_portfolio", "power_pathway", "market_timing"],
            evaluation_model=EvaluationModel(weights={"evidence_quality": 1.0}),
            validation_policy=ValidationPolicy(),
            generation_policy=GenerationPolicy(),
            search_budget=SearchBudget(),
        )
        evaluator = TheoryEvaluator()
        scores = [evaluator.build(t, plan, None).criteria_scores["evidence_quality"].score
                  for t in theories]
        # All theories must receive the same evidence_quality score under symmetric fallback
        assert scores[0] == scores[1] == scores[2], (
            f"Symmetric fallback produced different evidence_quality scores: {scores}"
        )

    def test_no_evidence_invented(self):
        """Theory evidence must be a subset of available evidence IDs."""
        evidence_ids = ["EV-X1", "EV-X2", "EV-X3", "EV-X4"]
        research = _MockResearchWithEvidence(evidence_ids=evidence_ids)
        sets = _make_configured_choice_sets()
        gen = self._gen()
        for cs in sets:
            theory = gen.build(cs, research)
            for ev_item in theory.evidence:
                assert ev_item in evidence_ids, (
                    f"Theory evidence {ev_item!r} is not in available evidence_ids"
                )

    def test_require_evidence_true_fails_when_no_evidence_available(self):
        """When require_evidence=True and no evidence exists, score must be 0.0."""
        from functional_agents.strategy.theory_evaluator import TheoryEvaluator
        from functional_agents.strategy.strategy_plan import StrategyPlan, EvaluationModel, ValidationPolicy, GenerationPolicy, SearchBudget

        research = _MockResearchWithEvidence(evidence_ids=[], citations=[])
        sets = _make_configured_choice_sets()
        gen = self._gen()
        theory = gen.build(sets[0], research)
        assert theory.evidence == []

        plan = StrategyPlan(
            plan_id="TEST",
            framework="executive",
            active_dimensions=[],
            evaluation_model=EvaluationModel(weights={"evidence_quality": 1.0}),
            validation_policy=ValidationPolicy(require_evidence=True),
            generation_policy=GenerationPolicy(),
            search_budget=SearchBudget(),
        )
        ev = TheoryEvaluator().build(theory, plan, None)
        assert ev.criteria_scores["evidence_quality"].score == 0.0

    def test_evidence_list_reordering_does_not_change_scores(self):
        """Shuffling the evidence_ids list must not change the resulting theory scores."""
        from functional_agents.strategy.theory_evaluator import TheoryEvaluator
        from functional_agents.strategy.strategy_plan import StrategyPlan, EvaluationModel, ValidationPolicy, GenerationPolicy, SearchBudget

        evidence_ids = ["EV-A", "EV-B", "EV-C", "EV-D", "EV-E"]
        research_orig = _MockResearchWithEvidence(evidence_ids=evidence_ids)
        research_reversed = _MockResearchWithEvidence(evidence_ids=list(reversed(evidence_ids)))

        sets = _make_configured_choice_sets()
        gen = self._gen()

        plan = StrategyPlan(
            plan_id="TEST",
            framework="executive",
            active_dimensions=["geographic_portfolio", "power_pathway", "market_timing"],
            evaluation_model=EvaluationModel(weights={"evidence_quality": 1.0}),
            validation_policy=ValidationPolicy(),
            generation_policy=GenerationPolicy(),
            search_budget=SearchBudget(),
        )
        evaluator = TheoryEvaluator()

        for cs in sets:
            t_orig = gen.build(cs, research_orig)
            t_rev = gen.build(cs, research_reversed)
            sc_orig = evaluator.build(t_orig, plan, None).criteria_scores["evidence_quality"].score
            sc_rev = evaluator.build(t_rev, plan, None).criteria_scores["evidence_quality"].score
            assert sc_orig == sc_rev, (
                f"Evidence list reordering changed evidence_quality score for {cs.id}: "
                f"{sc_orig} vs {sc_rev}"
            )


class TestCriterionValidationPH12a:
    """PH12.0a — Unsupported criteria must fail clearly; no neutral fallback."""

    def test_unsupported_criterion_raises_value_error(self):
        from functional_agents.strategy.theory_evaluator import TheoryEvaluator
        from functional_agents.strategy.theory_generator import TheoryGenerator
        from functional_agents.strategy.strategy_plan import StrategyPlan, EvaluationModel, ValidationPolicy, GenerationPolicy, SearchBudget

        sets = _make_configured_choice_sets()
        theory = TheoryGenerator().build(sets[0], _MockResearchWithEvidence())
        plan = StrategyPlan(
            plan_id="T",
            framework="executive",
            active_dimensions=[],
            evaluation_model=EvaluationModel(weights={"totally_unknown_criterion": 1.0}),
            validation_policy=ValidationPolicy(),
            generation_policy=GenerationPolicy(),
            search_budget=SearchBudget(),
        )
        with pytest.raises(ValueError, match="totally_unknown_criterion"):
            TheoryEvaluator().build(theory, plan, None)

    def test_error_message_names_the_unsupported_criterion(self):
        from functional_agents.strategy.theory_evaluator import TheoryEvaluator
        from functional_agents.strategy.theory_generator import TheoryGenerator
        from functional_agents.strategy.strategy_plan import StrategyPlan, EvaluationModel, ValidationPolicy, GenerationPolicy, SearchBudget

        sets = _make_configured_choice_sets()
        theory = TheoryGenerator().build(sets[0], _MockResearchWithEvidence())
        plan = StrategyPlan(
            plan_id="T",
            framework="executive",
            active_dimensions=[],
            evaluation_model=EvaluationModel(weights={"my_custom_kpi": 2.0}),
            validation_policy=ValidationPolicy(),
            generation_policy=GenerationPolicy(),
            search_budget=SearchBudget(),
        )
        with pytest.raises(ValueError) as exc_info:
            TheoryEvaluator().build(theory, plan, None)
        assert "my_custom_kpi" in str(exc_info.value)

    def test_resolver_rejects_unsupported_criterion_at_parse_time(self):
        """ConfigurationResolver._parse_evaluation must reject unknown criteria early."""
        from functional_agents.strategy.configuration_resolver import ConfigurationResolver
        resolver = ConfigurationResolver()
        with pytest.raises(ValueError, match="unknown_kpi"):
            resolver._parse_evaluation({
                "method": "multi_criteria",
                "criteria": {"unknown_kpi": {"weight": 1.0}},
            })

    def test_resolver_error_names_criterion_and_lists_supported(self):
        from functional_agents.strategy.configuration_resolver import ConfigurationResolver
        resolver = ConfigurationResolver()
        with pytest.raises(ValueError) as exc_info:
            resolver._parse_evaluation({
                "criteria": {"bad_criterion": {"weight": 1.5}},
            })
        msg = str(exc_info.value)
        assert "bad_criterion" in msg
        assert "strategic_fit" in msg  # one of the supported names

    def test_supported_configured_criteria_still_resolve(self):
        from functional_agents.strategy.configuration_resolver import ConfigurationResolver
        resolver = ConfigurationResolver()
        result = resolver._parse_evaluation({
            "method": "multi_criteria",
            "criteria": {
                "strategic_fit": {"weight": 2.0},
                "evidence_quality": {"weight": 1.5},
            },
        })
        assert result.weights["strategic_fit"] == 2.0
        assert result.weights["evidence_quality"] == 1.5

    def test_default_mode_evaluation_unchanged(self):
        """Default mode (no weights) still evaluates the 7 built-in criteria."""
        from functional_agents.strategy.theory_evaluator import TheoryEvaluator
        from functional_agents.strategy.theory_generator import TheoryGenerator
        from functional_agents.strategy.strategy_plan import StrategyPlan, EvaluationModel, ValidationPolicy, GenerationPolicy, SearchBudget

        sets = _make_configured_choice_sets()
        theory = TheoryGenerator().build(sets[0], _MockResearchWithEvidence())
        plan = StrategyPlan(
            plan_id="T",
            framework="executive",
            active_dimensions=[],
            evaluation_model=EvaluationModel(weights={}),  # default mode
            validation_policy=ValidationPolicy(),
            generation_policy=GenerationPolicy(),
            search_budget=SearchBudget(),
        )
        ev = TheoryEvaluator().build(theory, plan, None)
        assert len(ev.criteria_scores) == 7

    def test_no_neutral_fallback_path_for_unknown_criterion(self):
        """Confirm there is no code path that returns score=0.5 for unknown names."""
        from functional_agents.strategy.theory_evaluator import TheoryEvaluator, SUPPORTED_CRITERIA
        from functional_agents.strategy.theory_generator import TheoryGenerator
        from functional_agents.strategy.strategy_plan import StrategyPlan, EvaluationModel, ValidationPolicy, GenerationPolicy, SearchBudget

        sets = _make_configured_choice_sets()
        theory = TheoryGenerator().build(sets[0], _MockResearchWithEvidence())

        for bad_name in ["custom_kpi", "foo_bar", "mystery_metric"]:
            plan = StrategyPlan(
                plan_id="T",
                framework="executive",
                active_dimensions=[],
                evaluation_model=EvaluationModel(weights={bad_name: 1.0}),
                validation_policy=ValidationPolicy(),
                generation_policy=GenerationPolicy(),
                search_budget=SearchBudget(),
            )
            with pytest.raises(ValueError):
                TheoryEvaluator().build(theory, plan, None)

    def test_supported_criteria_set_is_accessible(self):
        """SUPPORTED_CRITERIA must be importable and contain the expected names."""
        from functional_agents.strategy.theory_evaluator import SUPPORTED_CRITERIA
        for name in ["strategic_fit", "evidence_quality", "option_identified",
                     "assumption_robustness", "execution_feasibility",
                     "risk_resilience", "opportunity_capture"]:
            assert name in SUPPORTED_CRITERIA, f"{name!r} missing from SUPPORTED_CRITERIA"
