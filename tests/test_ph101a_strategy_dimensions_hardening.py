"""PH10.1a — StrategyDimensions construction hardening tests.

Verifies that named dimensions survive every construction and
round-trip path used by the Strategy layer:

  add() → model_extra
  add() → to_dict() → from_dict()
  add() → StrategyConfig(dimensions=...) → to_dict()
  add() → StrategyConfig → ConfigurationResolver.resolve()
  add() → StrategyConfig → StrategyPlanner.active_dimensions
  add() → StrategyCoordinator → StrategicChoiceGenerator
  from_dict({"dimensions": {...}}) — existing path must be unchanged

No report, editorial, or pipeline behavior changes are tested here.
"""

from __future__ import annotations

import pytest

from functional_agents.strategy import (
    ConfigurationResolver,
    StrategicChoiceGenerator,
    StrategicChoiceSet,
    StrategyConfig,
    StrategyCoordinator,
    StrategyDimensions,
    StrategyObjectives,
    StrategyPlan,
    StrategyPlanner,
)
from functional_agents.context import AgentContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dims(*names: str) -> StrategyDimensions:
    d = StrategyDimensions()
    for name in names:
        d.add(name, f"{name} descriptor")
    return d


def _minimal_ctx() -> AgentContext:
    return AgentContext(
        question="Test question?",
        profiles=["test"],
        execution_profile="test",
        research_object={"id": "R-TEST"},
        run_id="run001",
        strategic_options=[],
        assumptions=[],
        risks=[],
        opportunities=[],
        recommendations=[],
        decision_model={"strategic_question": "Test question?"},
        executive_confidence={"overall_confidence": "Medium"},
        research_strategy={},
    )


# ---------------------------------------------------------------------------
# add() stores in model_extra
# ---------------------------------------------------------------------------

class TestAddStoresInModelExtra:
    def test_single_dimension_visible_in_model_extra(self):
        d = StrategyDimensions()
        d.add("market")
        assert "market" in d.model_extra

    def test_dimension_value_preserved(self):
        d = StrategyDimensions()
        d.add("market", "market entry strategy")
        assert d.model_extra["market"] == "market entry strategy"

    def test_none_value_preserved(self):
        d = StrategyDimensions()
        d.add("technology", None)
        assert "technology" in d.model_extra
        assert d.model_extra["technology"] is None

    def test_multiple_dimensions_all_visible(self):
        d = _dims("market", "technology", "financial")
        assert set(d.model_extra.keys()) == {"market", "technology", "financial"}

    def test_add_is_idempotent_on_same_name(self):
        d = StrategyDimensions()
        d.add("market", "v1")
        d.add("market", "v2")
        assert d.model_extra["market"] == "v2"

    def test_empty_dimensions_model_extra_is_empty(self):
        d = StrategyDimensions()
        assert d.model_extra == {} or d.model_extra is None or not d.model_extra


# ---------------------------------------------------------------------------
# Round-trip: to_dict / from_dict
# ---------------------------------------------------------------------------

class TestRoundTripToDictFromDict:
    def test_to_dict_includes_dimension(self):
        d = _dims("market")
        cfg = StrategyConfig(dimensions=d)
        d2 = cfg.to_dict()
        assert "market" in d2["dimensions"]

    def test_from_dict_restores_dimension(self):
        d = _dims("market", "technology")
        cfg = StrategyConfig(dimensions=d)
        restored = StrategyConfig.from_dict(cfg.to_dict())
        assert "market" in restored.dimensions.model_extra
        assert "technology" in restored.dimensions.model_extra

    def test_round_trip_preserves_descriptor(self):
        d = StrategyDimensions()
        d.add("risk", "risk assessment axis")
        cfg = StrategyConfig(dimensions=d)
        restored = StrategyConfig.from_dict(cfg.to_dict())
        assert restored.dimensions.model_extra["risk"] == "risk assessment axis"

    def test_round_trip_with_no_dimensions_is_stable(self):
        cfg = StrategyConfig()
        restored = StrategyConfig.from_dict(cfg.to_dict())
        assert (restored.dimensions.model_extra or {}) == {}

    def test_from_dict_path_equivalent_to_add_path(self):
        # add() path
        d = StrategyDimensions()
        d.add("market", "market entry")
        cfg_add = StrategyConfig(dimensions=d)

        # from_dict path (the prior workaround)
        cfg_dict = StrategyConfig.from_dict({
            "dimensions": {"market": "market entry"},
        })

        add_dims = set((cfg_add.dimensions.model_extra or {}).keys())
        dict_dims = set((cfg_dict.dimensions.model_extra or {}).keys())
        assert add_dims == dict_dims


# ---------------------------------------------------------------------------
# ConfigurationResolver round-trip
# ---------------------------------------------------------------------------

class TestConfigurationResolverRoundTrip:
    def test_dimensions_survive_resolver(self):
        d = _dims("market", "technology")
        cfg = StrategyConfig(dimensions=d)
        resolved = ConfigurationResolver().resolve(cfg)
        assert "market" in (resolved.dimensions.model_extra or {})
        assert "technology" in (resolved.dimensions.model_extra or {})

    def test_dimension_descriptor_survives_resolver(self):
        d = StrategyDimensions()
        d.add("financial", "capital allocation axis")
        cfg = StrategyConfig(dimensions=d)
        resolved = ConfigurationResolver().resolve(cfg)
        assert resolved.dimensions.model_extra.get("financial") == "capital allocation axis"

    def test_empty_dimensions_stable_through_resolver(self):
        cfg = StrategyConfig()
        resolved = ConfigurationResolver().resolve(cfg)
        assert (resolved.dimensions.model_extra or {}) == {}


# ---------------------------------------------------------------------------
# StrategyPlanner.active_dimensions
# ---------------------------------------------------------------------------

class TestStrategyPlannerActiveDimensions:
    def test_add_dimensions_appear_in_plan(self):
        d = _dims("market", "technology")
        cfg = StrategyConfig(dimensions=d)
        plan = StrategyPlanner().build(cfg)
        assert "market" in plan.active_dimensions
        assert "technology" in plan.active_dimensions

    def test_add_dimensions_sorted_in_plan(self):
        d = _dims("zzz", "aaa", "mmm")
        cfg = StrategyConfig(dimensions=d)
        plan = StrategyPlanner().build(cfg)
        assert plan.active_dimensions == ["aaa", "mmm", "zzz"]

    def test_no_dimensions_produces_empty_list(self):
        cfg = StrategyConfig()
        plan = StrategyPlanner().build(cfg)
        assert plan.active_dimensions == []

    def test_add_path_matches_from_dict_path(self):
        d = _dims("market", "technology")
        plan_add = StrategyPlanner().build(StrategyConfig(dimensions=d))

        plan_dict = StrategyPlanner().build(StrategyConfig.from_dict({
            "dimensions": {"market": "m", "technology": "t"},
        }))

        assert sorted(plan_add.active_dimensions) == sorted(plan_dict.active_dimensions)


# ---------------------------------------------------------------------------
# StrategicChoiceGenerator — choices produced for add() dimensions
# ---------------------------------------------------------------------------

class TestChoiceGeneratorWithAddDimensions:
    def _plan(self, *names: str) -> StrategyPlan:
        d = _dims(*names)
        cfg = StrategyConfig(dimensions=d)
        resolved = ConfigurationResolver().resolve(cfg)
        return StrategyPlanner().build(resolved)

    def _research(self):
        import types
        ns = types.SimpleNamespace()
        ns.executive_confidence = {"overall_confidence": "High"}
        ns.decision_analysis = {"recommended_option_id": "OPT-A", "rationale": "Best option"}
        ns.preferred_option = {}
        ns.assumptions = []
        return ns

    def test_one_choice_per_add_dimension(self):
        plan = self._plan("market", "technology")
        cs = StrategicChoiceGenerator().build(plan, self._research())
        assert len(cs.choices) == 2

    def test_choice_dimensions_match_add_names(self):
        plan = self._plan("market", "technology")
        cs = StrategicChoiceGenerator().build(plan, self._research())
        dims = {c.dimension for c in cs.choices}
        assert dims == {"market", "technology"}

    def test_completeness_one_with_add_dimensions(self):
        plan = self._plan("market")
        cs = StrategicChoiceGenerator().build(plan, self._research())
        assert cs.completeness == 1.0


# ---------------------------------------------------------------------------
# StrategyCoordinator full-path using add()
# ---------------------------------------------------------------------------

class TestStrategyCoordinatorWithAddDimensions:
    def test_add_dimensions_flow_through_coordinator(self):
        d = _dims("market", "technology")
        cfg = StrategyConfig(dimensions=d)
        coord = StrategyCoordinator(config=cfg)
        coord.build(_minimal_ctx())
        assert len(coord._choice_set.choices) == 2

    def test_add_dimensions_covered_in_choice_set(self):
        d = _dims("financial", "regulatory")
        cfg = StrategyConfig(dimensions=d)
        coord = StrategyCoordinator(config=cfg)
        coord.build(_minimal_ctx())
        covered = coord._choice_set.dimensions_covered()
        assert "financial" in covered
        assert "regulatory" in covered

    def test_add_path_choice_set_matches_from_dict_path(self):
        # add() path
        d = _dims("market", "technology")
        cfg_add = StrategyConfig(dimensions=d)
        coord_add = StrategyCoordinator(config=cfg_add)
        coord_add.build(_minimal_ctx())

        # from_dict path
        cfg_dict = StrategyConfig.from_dict({
            "dimensions": {"market": "m", "technology": "t"},
        })
        coord_dict = StrategyCoordinator(config=cfg_dict)
        coord_dict.build(_minimal_ctx())

        assert len(coord_add._choice_set.choices) == len(coord_dict._choice_set.choices)
        assert set(coord_add._choice_set.dimensions_covered()) == \
               set(coord_dict._choice_set.dimensions_covered())
