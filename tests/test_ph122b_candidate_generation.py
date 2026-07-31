"""PH12.2b — Configurable Strategy Candidate Generation tests.

Covers:
- Candidate count: max_candidates=1..4 generates exactly that many choice sets
- Distinctness: all generated signatures are unique
- Dimension coverage: every candidate covers all required dimensions
- Safe cap: requested count > available distinct strategies returns maximum available
- Framework recognition: monitor_choice_cascade is recognized; no unknown-framework warning
- Compatibility: default (3-candidate) and executive-framework behaviors unchanged
- Coordinator integration: max_candidates=4 → 4 choice sets, 4 theories, 4 evaluations
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from functional_agents.strategy import (
    ChoiceConfig,
    ConfigurationResolver,
    DimensionConfig,
    StrategyConfig,
    StrategyCoordinator,
    StrategyGeneration,
    StrategyPlanner,
    StrategicChoiceGenerator,
)
from functional_agents.strategy.framework_defaults import FrameworkDefaults
from functional_agents.strategy.strategic_choice_generator import _posture_key
from functional_agents.strategy.strategy_plan import GenerationPolicy, StrategyPlan


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _dim(dim_id: str, n_choices: int) -> DimensionConfig:
    """Create a DimensionConfig with n_choices distinct choices."""
    return DimensionConfig(
        id=dim_id,
        title=dim_id.replace("_", " ").title(),
        description=f"Test dimension: {dim_id}",
        required=True,
        choices=[
            ChoiceConfig(id=f"{dim_id}_c{i}", title=f"Choice {i}", description="")
            for i in range(n_choices)
        ],
    )


def _plan(max_candidates: int, *dims: DimensionConfig, diversity_required: bool = True) -> StrategyPlan:
    """Build a StrategyPlan with configured dimensions and specified max_candidates."""
    return StrategyPlan(
        plan_id="TEST",
        framework="executive",
        active_dimensions=[d.id for d in dims],
        dimension_configs=list(dims),
        generation_policy=GenerationPolicy(
            max_candidates=max_candidates,
            diversity_required=diversity_required,
        ),
    )


def _mock_research() -> Any:
    m = MagicMock()
    m.run_id = "test-run-001"
    m.question = "What is the best strategy?"
    m.profiles = []
    m.execution_profile = "test"
    m.decision_model = {}
    m.engagement = {}
    m.preferred_option = {}
    m.research_object = {"id": "R-001", "citations": []}
    m.executive_confidence = {}
    m.decision_analysis = {}
    m.strategic_options = []
    m.assumptions = [{"statement": "Test assumption."}]
    m.risks = [{"statement": "Test risk.", "severity": "Medium", "likelihood": "Low"}]
    m.recommendations = []
    m.opportunities = []
    m.trace = {}
    return m


def _sig(cs) -> tuple:
    return tuple(sorted((c.dimension, c.selected_value) for c in cs.choices))


def _make_monitor_plan_with_dims(max_candidates: int = 4) -> StrategyPlan:
    """Build a plan matching the sports engagement Monitor configuration."""
    dims = [
        _dim("winning_aspiration", 4),
        _dim("where_to_play", 4),
        _dim("how_to_win", 5),
        _dim("must_have_capabilities", 4),
        _dim("management_systems", 4),
    ]
    return _plan(max_candidates, *dims)


# ---------------------------------------------------------------------------
# _posture_key helper
# ---------------------------------------------------------------------------

class TestPostureKeyHelper:
    def test_index_0_is_recommended(self):
        assert _posture_key(0) == "recommended"

    def test_index_1_is_alternative_a(self):
        assert _posture_key(1) == "alternative-a"

    def test_index_2_is_alternative_b(self):
        assert _posture_key(2) == "alternative-b"

    def test_index_3_is_alternative_c(self):
        assert _posture_key(3) == "alternative-c"

    def test_index_4_is_alternative_d(self):
        assert _posture_key(4) == "alternative-d"

    def test_keys_are_deterministic(self):
        # Same index always returns the same key.
        for i in range(10):
            assert _posture_key(i) == _posture_key(i)


# ---------------------------------------------------------------------------
# Candidate count tests
# ---------------------------------------------------------------------------

class TestCandidateCount:
    """Prove max_candidates=N generates exactly N choice sets."""

    def _gen(self, max_candidates: int) -> list:
        dims = [_dim("dim_a", 5), _dim("dim_b", 5), _dim("dim_c", 5)]
        plan = _plan(max_candidates, *dims)
        return StrategicChoiceGenerator().build(plan, _mock_research())

    def test_max_candidates_1(self):
        sets = self._gen(1)
        assert len(sets) == 1

    def test_max_candidates_2(self):
        sets = self._gen(2)
        assert len(sets) == 2

    def test_max_candidates_3(self):
        sets = self._gen(3)
        assert len(sets) == 3

    def test_max_candidates_4(self):
        sets = self._gen(4)
        assert len(sets) == 4

    def test_max_candidates_5(self):
        sets = self._gen(5)
        assert len(sets) == 5

    def test_monitor_config_4_candidates(self):
        plan = _make_monitor_plan_with_dims(max_candidates=4)
        sets = StrategicChoiceGenerator().build(plan, _mock_research())
        assert len(sets) == 4


# ---------------------------------------------------------------------------
# Distinctness tests
# ---------------------------------------------------------------------------

class TestDistinctness:
    """All generated choice-set signatures must be unique."""

    def test_signatures_unique_max_4(self):
        plan = _make_monitor_plan_with_dims(max_candidates=4)
        sets = StrategicChoiceGenerator().build(plan, _mock_research())
        sigs = [_sig(cs) for cs in sets]
        assert len(sigs) == len(set(sigs)), f"Duplicate signatures found: {sigs}"

    def test_signatures_unique_max_3_three_choice_dims(self):
        dims = [_dim("d1", 3), _dim("d2", 3), _dim("d3", 3)]
        plan = _plan(3, *dims)
        sets = StrategicChoiceGenerator().build(plan, _mock_research())
        sigs = [_sig(cs) for cs in sets]
        assert len(sigs) == len(set(sigs))

    def test_signatures_unique_max_2(self):
        dims = [_dim("d1", 4), _dim("d2", 4)]
        plan = _plan(2, *dims)
        sets = StrategicChoiceGenerator().build(plan, _mock_research())
        sigs = [_sig(cs) for cs in sets]
        assert len(sigs) == len(set(sigs))

    def test_no_duplicates_regardless_of_ids(self):
        """Two sets with different IDs but same choice selections are duplicates — must not appear."""
        dims = [_dim("d1", 4)]
        plan = _plan(3, *dims)
        sets = StrategicChoiceGenerator().build(plan, _mock_research())
        sigs = [_sig(cs) for cs in sets]
        assert len(sigs) == len(set(sigs))


# ---------------------------------------------------------------------------
# Dimension coverage tests
# ---------------------------------------------------------------------------

class TestDimensionCoverage:
    """Every candidate must contain every required dimension with exactly one valid choice."""

    def _required_dims(self, plan: StrategyPlan) -> set[str]:
        return {d.id for d in plan.dimension_configs if d.required}

    def _valid_choice_ids(self, dim: DimensionConfig) -> set[str]:
        return {c.id for c in dim.choices}

    def test_all_required_dimensions_present_in_every_set(self):
        plan = _make_monitor_plan_with_dims(max_candidates=4)
        sets = StrategicChoiceGenerator().build(plan, _mock_research())
        required = self._required_dims(plan)
        for cs in sets:
            covered = {c.dimension for c in cs.choices}
            assert required <= covered, (
                f"Choice set {cs.id} missing required dimensions: {required - covered}"
            )

    def test_exactly_one_choice_per_required_dimension(self):
        plan = _make_monitor_plan_with_dims(max_candidates=4)
        sets = StrategicChoiceGenerator().build(plan, _mock_research())
        required = self._required_dims(plan)
        for cs in sets:
            for dim_id in required:
                dim_choices = [c for c in cs.choices if c.dimension == dim_id]
                assert len(dim_choices) == 1, (
                    f"Choice set {cs.id} has {len(dim_choices)} choices for dimension {dim_id!r}"
                )

    def test_selected_choice_belongs_to_dimension_catalog(self):
        plan = _make_monitor_plan_with_dims(max_candidates=4)
        dim_catalogs = {d.id: self._valid_choice_ids(d) for d in plan.dimension_configs}
        sets = StrategicChoiceGenerator().build(plan, _mock_research())
        for cs in sets:
            for choice in cs.choices:
                valid = dim_catalogs.get(choice.dimension, set())
                assert choice.selected_value in valid, (
                    f"Choice {choice.selected_value!r} is not in catalog for dimension "
                    f"{choice.dimension!r}. Valid: {valid}"
                )

    def test_completeness_is_1_for_all_sets(self):
        plan = _make_monitor_plan_with_dims(max_candidates=4)
        sets = StrategicChoiceGenerator().build(plan, _mock_research())
        for cs in sets:
            assert cs.completeness == 1.0, f"Choice set {cs.id} has completeness {cs.completeness}"


# ---------------------------------------------------------------------------
# Safe-cap tests
# ---------------------------------------------------------------------------

class TestSafeCap:
    """Requested count > available distinct strategies must not generate duplicates."""

    def test_safe_cap_returns_max_available_not_duplicates(self):
        # 1 choice per dimension → only 1 unique combination exists.
        dims = [_dim("d1", 1), _dim("d2", 1)]
        plan = _plan(4, *dims, diversity_required=False)
        sets = StrategicChoiceGenerator().build(plan, _mock_research())
        assert len(sets) == 1, f"Expected 1 (cap), got {len(sets)}"
        sigs = [_sig(cs) for cs in sets]
        assert len(sigs) == len(set(sigs)), "Duplicate signatures despite safe cap"

    def test_safe_cap_emits_warning(self, caplog):
        dims = [_dim("d1", 1)]
        plan = _plan(4, *dims, diversity_required=False)
        with caplog.at_level(logging.WARNING, logger="functional_agents.strategy.strategic_choice_generator"):
            StrategicChoiceGenerator().build(plan, _mock_research())
        assert any("requested 4" in r.message.lower() or "4 candidates" in r.message.lower()
                   for r in caplog.records), (
            f"No cap warning emitted. Records: {[r.message for r in caplog.records]}"
        )

    def test_safe_cap_with_2_choices_returns_2(self):
        # 2 choices per dimension → only 2 unique combinations exist.
        dims = [_dim("d1", 2), _dim("d2", 2)]
        plan = _plan(5, *dims, diversity_required=False)
        sets = StrategicChoiceGenerator().build(plan, _mock_research())
        assert len(sets) == 2
        sigs = [_sig(cs) for cs in sets]
        assert len(sigs) == len(set(sigs))


# ---------------------------------------------------------------------------
# Framework recognition tests
# ---------------------------------------------------------------------------

class TestFrameworkRecognition:
    """monitor_choice_cascade must be recognized with no unknown-framework warning."""

    def test_monitor_choice_cascade_is_known(self):
        assert FrameworkDefaults.is_known("monitor_choice_cascade")

    def test_monitor_choice_cascade_in_known_list(self):
        known = FrameworkDefaults.known()
        assert "monitor_choice_cascade" in known

    def test_monitor_choice_cascade_get_returns_config(self):
        cfg = FrameworkDefaults.get("monitor_choice_cascade")
        assert cfg.framework == "monitor_choice_cascade"

    def test_no_unknown_framework_warning_emitted(self, caplog):
        cfg = StrategyConfig(framework="monitor_choice_cascade", version="1.0")
        with caplog.at_level(logging.WARNING, logger="functional_agents.strategy.configuration_resolver"):
            ConfigurationResolver().resolve(cfg)
        unknown_warnings = [
            r for r in caplog.records
            if "unknown framework" in r.message.lower()
        ]
        assert unknown_warnings == [], (
            f"Unknown-framework warning was emitted: {[r.message for r in unknown_warnings]}"
        )

    def test_configured_dimensions_remain_unchanged_after_resolution(self):
        monitor_strategy = {
            "framework": "monitor_choice_cascade",
            "dimensions": [
                {"id": "winning_aspiration", "title": "Winning Aspiration", "required": True,
                 "choices": [{"id": "c1", "title": "C1"}, {"id": "c2", "title": "C2"}]},
                {"id": "where_to_play", "title": "Where to Play", "required": True,
                 "choices": [{"id": "c3", "title": "C3"}, {"id": "c4", "title": "C4"}]},
            ],
            "generation": {"max_candidates": 2, "diversity_required": True},
        }
        resolved = ConfigurationResolver().resolve_from_engagement(
            StrategyConfig(), monitor_strategy
        )
        dim_ids = [d.id for d in resolved.dimension_configs]
        assert "winning_aspiration" in dim_ids
        assert "where_to_play" in dim_ids

    def test_configured_max_candidates_preserved_after_resolution(self):
        monitor_strategy = {
            "framework": "monitor_choice_cascade",
            "dimensions": [
                {"id": "dim_x", "required": True,
                 "choices": [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}]},
            ],
            "generation": {"max_candidates": 4, "diversity_required": True},
        }
        resolved = ConfigurationResolver().resolve_from_engagement(
            StrategyConfig(), monitor_strategy
        )
        assert resolved.generation.max_candidates == 4

    def test_executive_framework_still_known(self):
        assert FrameworkDefaults.is_known("executive")

    def test_unknown_framework_still_triggers_warning(self, caplog):
        cfg = StrategyConfig(framework="totally_unknown_xyz", version="1.0")
        with caplog.at_level(logging.WARNING, logger="functional_agents.strategy.configuration_resolver"):
            ConfigurationResolver().resolve(cfg)
        unknown_warnings = [
            r for r in caplog.records
            if "unknown framework" in r.message.lower()
        ]
        assert unknown_warnings, "Unknown framework should still emit a warning"


# ---------------------------------------------------------------------------
# Compatibility tests
# ---------------------------------------------------------------------------

class TestCompatibility:
    """Default and executive-framework behaviors must remain unchanged."""

    def test_default_strategy_generates_3_sets(self):
        """Default StrategyCoordinator (no config) produces 3 legacy choice sets."""
        coord = StrategyCoordinator()
        research = _mock_research()
        coord.build(research)
        assert len(coord._choice_sets) == 3

    def test_executive_framework_generates_3_sets_by_default(self):
        cfg = StrategyConfig(framework="executive", version="1.0")
        resolved = ConfigurationResolver().resolve(cfg)
        coord = StrategyCoordinator(config=resolved)
        coord.build(_mock_research())
        assert len(coord._choice_sets) == 3

    def test_three_candidate_config_generates_3(self):
        dims = [_dim("d1", 3), _dim("d2", 3), _dim("d3", 3)]
        plan = _plan(3, *dims)
        sets = StrategicChoiceGenerator().build(plan, _mock_research())
        assert len(sets) == 3

    def test_two_candidate_config_generates_2(self):
        dims = [_dim("d1", 3), _dim("d2", 3)]
        plan = _plan(2, *dims)
        sets = StrategicChoiceGenerator().build(plan, _mock_research())
        assert len(sets) == 2

    def test_legacy_mode_still_produces_3_sets(self):
        """Plan with no dimension_configs uses legacy mode → always 3 sets."""
        plan = StrategyPlan(
            plan_id="LEGACY",
            framework="executive",
            active_dimensions=["alpha", "beta"],
            generation_policy=GenerationPolicy(max_candidates=3),
        )
        sets = StrategicChoiceGenerator().build(plan, _mock_research())
        assert len(sets) == 3

    def test_first_three_posture_keys_unchanged(self):
        """Posture keys 0-2 still match legacy _POSTURE_KEYS."""
        from functional_agents.strategy.strategic_choice_generator import _POSTURE_KEYS
        assert _posture_key(0) == _POSTURE_KEYS[0]
        assert _posture_key(1) == _POSTURE_KEYS[1]
        assert _posture_key(2) == _POSTURE_KEYS[2]


# ---------------------------------------------------------------------------
# Coordinator integration tests
# ---------------------------------------------------------------------------

class TestCoordinatorIntegration:
    """max_candidates=4 → 4 choice sets, 4 theories, 4 evaluations."""

    def _make_monitor_config(self) -> StrategyConfig:
        monitor_strategy = {
            "framework": "monitor_choice_cascade",
            "dimensions": [
                {
                    "id": "winning_aspiration", "title": "Winning Aspiration", "required": True,
                    "choices": [
                        {"id": "category_leader", "title": "Category Leader"},
                        {"id": "platform_play", "title": "Platform Play"},
                        {"id": "major_events_wedge", "title": "Major Events Wedge"},
                        {"id": "focused_growth", "title": "Focused Growth"},
                    ],
                },
                {
                    "id": "where_to_play", "title": "Where to Play", "required": True,
                    "choices": [
                        {"id": "leagues_govbodies", "title": "Leagues and Governing Bodies"},
                        {"id": "teams_clubs", "title": "Teams and Clubs"},
                        {"id": "investors_tech", "title": "Investors and Tech"},
                        {"id": "multi_buyer", "title": "Multi-Buyer Portfolio"},
                    ],
                },
                {
                    "id": "how_to_win", "title": "How to Win", "required": True,
                    "choices": [
                        {"id": "advisory_led", "title": "Advisory-Led"},
                        {"id": "platform_enabled", "title": "Platform-Enabled"},
                        {"id": "governing_body_led", "title": "Governing-Body-Led"},
                        {"id": "investment_led", "title": "Investment-Led"},
                        {"id": "hybrid", "title": "Hybrid"},
                    ],
                },
                {
                    "id": "must_have_capabilities", "title": "Must-Have Capabilities", "required": True,
                    "choices": [
                        {"id": "advisory_credibility", "title": "Advisory Credibility"},
                        {"id": "data_platform", "title": "Data Platform"},
                        {"id": "major_events_infra", "title": "Major Events Infrastructure"},
                        {"id": "integrated_mdm", "title": "Integrated MDM"},
                    ],
                },
                {
                    "id": "management_systems", "title": "Management Systems", "required": True,
                    "choices": [
                        {"id": "sgo_incubation", "title": "SGO Incubation"},
                        {"id": "offer_pnl", "title": "Offer P&L"},
                        {"id": "account_rhythm", "title": "Account Rhythm"},
                        {"id": "integrated_growth", "title": "Integrated Growth"},
                    ],
                },
            ],
            "evaluation": {
                "method": "multi_criteria",
                "criteria": {
                    "strategic_fit": {"weight": 0.3},
                    "execution_feasibility": {"weight": 0.3},
                    "evidence_quality": {"weight": 0.2},
                    "risk_resilience": {"weight": 0.2},
                },
            },
            "generation": {"max_candidates": 4, "diversity_required": True},
        }
        return ConfigurationResolver().resolve_from_engagement(StrategyConfig(), monitor_strategy)

    def _mock_research(self) -> Any:
        m = MagicMock()
        m.run_id = "test-monitor-001"
        m.question = "What is the best sports strategy?"
        m.profiles = []
        m.execution_profile = ""
        m.decision_model = {}
        m.engagement = {}
        m.preferred_option = {}
        m.research_object = {"id": "R-001", "citations": []}
        m.executive_confidence = {}
        m.decision_analysis = {}
        m.strategic_options = []
        m.assumptions = [{"statement": "Market is fragmented."}]
        m.risks = [{"statement": "Competitive risk.", "severity": "High", "likelihood": "Medium"}]
        m.recommendations = []
        m.opportunities = []
        m.trace = {}
        return m

    def test_4_candidates_produce_4_choice_sets(self):
        cfg = self._make_monitor_config()
        coord = StrategyCoordinator(config=cfg)
        coord.build(self._mock_research())
        assert len(coord._choice_sets) == 4

    def test_4_choice_sets_produce_4_theories(self):
        cfg = self._make_monitor_config()
        coord = StrategyCoordinator(config=cfg)
        coord.build(self._mock_research())
        assert len(coord._theories) == 4

    def test_4_theories_produce_4_evaluations(self):
        cfg = self._make_monitor_config()
        coord = StrategyCoordinator(config=cfg)
        coord.build(self._mock_research())
        assert len(coord._evaluations) == 4

    def test_all_4_choice_sets_have_all_5_dimensions(self):
        cfg = self._make_monitor_config()
        coord = StrategyCoordinator(config=cfg)
        coord.build(self._mock_research())
        expected_dims = {
            "winning_aspiration", "where_to_play", "how_to_win",
            "must_have_capabilities", "management_systems",
        }
        for cs in coord._choice_sets:
            covered = {c.dimension for c in cs.choices}
            assert expected_dims <= covered, (
                f"Choice set {cs.id} missing dimensions: {expected_dims - covered}"
            )

    def test_all_4_choice_sets_have_unique_signatures(self):
        cfg = self._make_monitor_config()
        coord = StrategyCoordinator(config=cfg)
        coord.build(self._mock_research())
        sigs = [_sig(cs) for cs in coord._choice_sets]
        assert len(sigs) == len(set(sigs)), f"Duplicate signatures: {sigs}"

    def test_winner_is_selected(self):
        cfg = self._make_monitor_config()
        coord = StrategyCoordinator(config=cfg)
        coord.build(self._mock_research())
        assert coord._selection is not None
        assert coord._selection.winner_theory_id

    def test_trace_is_built(self):
        cfg = self._make_monitor_config()
        coord = StrategyCoordinator(config=cfg)
        coord.build(self._mock_research())
        assert coord._trace is not None
