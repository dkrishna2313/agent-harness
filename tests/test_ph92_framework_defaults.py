"""PH9.2 — FrameworkDefaults and ConfigurationResolver merge tests.

Covers:
- FrameworkDefaults registry: get, known, is_known
- Executive framework defaults: all sections populated
- Resolver merge: baseline caller gets defaults applied
- Resolver merge: caller-specified values override defaults
- Resolver merge: neither defaults nor caller is mutated
- Unknown framework: graceful fallback
- PH9.1 validation invariants still enforced
"""

import pytest

from functional_agents.strategy import (
    ConfigurationResolver,
    FrameworkDefaults,
    StrategyConfig,
    StrategyConstraints,
    StrategyCoordinator,
    StrategyEvaluation,
    StrategyGeneration,
    StrategyObjectives,
    StrategyValidation,
)


# ---------------------------------------------------------------------------
# FrameworkDefaults — registry
# ---------------------------------------------------------------------------

class TestFrameworkDefaultsRegistry:
    def test_executive_is_known(self):
        assert FrameworkDefaults.is_known("executive") is True

    def test_unknown_framework_not_known(self):
        assert FrameworkDefaults.is_known("nonexistent") is False

    def test_known_returns_list(self):
        known = FrameworkDefaults.known()
        assert isinstance(known, list)
        assert "executive" in known

    def test_get_executive_returns_config(self):
        cfg = FrameworkDefaults.get("executive")
        assert isinstance(cfg, StrategyConfig)
        assert cfg.framework == "executive"

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy framework"):
            FrameworkDefaults.get("board")

    def test_get_unknown_error_mentions_available(self):
        with pytest.raises(ValueError, match="executive"):
            FrameworkDefaults.get("board")


# ---------------------------------------------------------------------------
# FrameworkDefaults — executive content
# ---------------------------------------------------------------------------

class TestExecutiveFrameworkContent:
    def setup_method(self):
        self.cfg = FrameworkDefaults.get("executive")

    def test_version(self):
        assert self.cfg.version == "1.0"

    def test_framework(self):
        assert self.cfg.framework == "executive"

    def test_objectives_primary_populated(self):
        assert len(self.cfg.objectives.primary) >= 1

    def test_objectives_secondary_populated(self):
        assert len(self.cfg.objectives.secondary) >= 1

    def test_evaluation_method(self):
        assert self.cfg.evaluation.method == "multi_criteria"

    def test_evaluation_min_score_threshold(self):
        assert self.cfg.evaluation.min_score_threshold >= 0.0

    def test_generation_max_candidates(self):
        assert self.cfg.generation.max_candidates >= 1

    def test_generation_diversity_required(self):
        assert isinstance(self.cfg.generation.diversity_required, bool)

    def test_constraints_are_empty_lists(self):
        assert self.cfg.constraints.excluded_options == []
        assert self.cfg.constraints.required_conditions == []

    def test_validation_defaults(self):
        assert self.cfg.validation.require_evidence is False
        assert self.cfg.validation.require_assumptions is False


# ---------------------------------------------------------------------------
# ConfigurationResolver — merge: defaults applied
# ---------------------------------------------------------------------------

class TestConfigurationResolverMerge:
    def test_baseline_caller_gets_executive_objectives(self):
        """A plain StrategyConfig() has empty objectives; defaults fill them in."""
        caller = StrategyConfig()  # baseline — objectives.primary == []
        resolved = ConfigurationResolver().resolve(caller)
        exec_defaults = FrameworkDefaults.get("executive")
        assert resolved.objectives.primary == exec_defaults.objectives.primary

    def test_baseline_caller_gets_executive_objectives_secondary(self):
        caller = StrategyConfig()
        resolved = ConfigurationResolver().resolve(caller)
        exec_defaults = FrameworkDefaults.get("executive")
        assert resolved.objectives.secondary == exec_defaults.objectives.secondary

    def test_baseline_caller_gets_default_evaluation(self):
        caller = StrategyConfig()
        resolved = ConfigurationResolver().resolve(caller)
        exec_defaults = FrameworkDefaults.get("executive")
        assert resolved.evaluation.method == exec_defaults.evaluation.method

    def test_baseline_caller_gets_default_generation(self):
        caller = StrategyConfig()
        resolved = ConfigurationResolver().resolve(caller)
        exec_defaults = FrameworkDefaults.get("executive")
        assert resolved.generation.max_candidates == exec_defaults.generation.max_candidates

    def test_resolved_framework_is_executive(self):
        caller = StrategyConfig()
        resolved = ConfigurationResolver().resolve(caller)
        assert resolved.framework == "executive"


# ---------------------------------------------------------------------------
# ConfigurationResolver — merge: caller values preserved
# ---------------------------------------------------------------------------

class TestConfigurationResolverCallerWins:
    def test_caller_objectives_override_defaults(self):
        caller = StrategyConfig(
            objectives=StrategyObjectives(primary=["custom objective"])
        )
        resolved = ConfigurationResolver().resolve(caller)
        assert resolved.objectives.primary == ["custom objective"]

    def test_caller_max_candidates_overrides_default(self):
        caller = StrategyConfig(generation=StrategyGeneration(max_candidates=7))
        resolved = ConfigurationResolver().resolve(caller)
        assert resolved.generation.max_candidates == 7

    def test_caller_evaluation_method_overrides_default(self):
        caller = StrategyConfig(evaluation=StrategyEvaluation(method="scoring"))
        resolved = ConfigurationResolver().resolve(caller)
        assert resolved.evaluation.method == "scoring"

    def test_caller_constraints_override_defaults(self):
        caller = StrategyConfig(
            constraints=StrategyConstraints(excluded_options=["OPT-Z"])
        )
        resolved = ConfigurationResolver().resolve(caller)
        assert resolved.constraints.excluded_options == ["OPT-Z"]

    def test_caller_validation_overrides_defaults(self):
        caller = StrategyConfig(
            validation=StrategyValidation(require_evidence=True, min_confidence="High")
        )
        resolved = ConfigurationResolver().resolve(caller)
        assert resolved.validation.require_evidence is True
        assert resolved.validation.min_confidence == "High"

    def test_caller_version_preserved(self):
        caller = StrategyConfig(version="2.0")
        resolved = ConfigurationResolver().resolve(caller)
        assert resolved.version == "2.0"


# ---------------------------------------------------------------------------
# ConfigurationResolver — immutability
# ---------------------------------------------------------------------------

class TestConfigurationResolverMergeImmutability:
    def test_caller_not_mutated(self):
        caller = StrategyConfig()
        original_primary = list(caller.objectives.primary)
        ConfigurationResolver().resolve(caller)
        assert caller.objectives.primary == original_primary

    def test_framework_defaults_not_mutated(self):
        defaults_before = FrameworkDefaults.get("executive").objectives.primary[:]
        caller = StrategyConfig(objectives=StrategyObjectives(primary=["override"]))
        ConfigurationResolver().resolve(caller)
        defaults_after = FrameworkDefaults.get("executive").objectives.primary
        assert defaults_after == defaults_before

    def test_resolved_is_new_object(self):
        caller = StrategyConfig()
        resolved = ConfigurationResolver().resolve(caller)
        assert resolved is not caller

    def test_multiple_resolves_independent(self):
        caller = StrategyConfig()
        r1 = ConfigurationResolver().resolve(caller)
        r2 = ConfigurationResolver().resolve(caller)
        assert r1 is not r2
        assert r1.objectives.primary == r2.objectives.primary


# ---------------------------------------------------------------------------
# ConfigurationResolver — unknown framework fallback
# ---------------------------------------------------------------------------

class TestConfigurationResolverUnknownFramework:
    def test_unknown_framework_resolves_without_error(self):
        caller = StrategyConfig(framework="board")
        # Should not raise — falls back to caller's config as-is
        resolved = ConfigurationResolver().resolve(caller)
        assert resolved.framework == "board"

    def test_unknown_framework_preserves_all_caller_values(self):
        caller = StrategyConfig(
            framework="board",
            objectives=StrategyObjectives(primary=["win market"]),
        )
        resolved = ConfigurationResolver().resolve(caller)
        assert resolved.objectives.primary == ["win market"]


# ---------------------------------------------------------------------------
# PH9.1 validation invariants still hold
# ---------------------------------------------------------------------------

class TestValidationInvariantsPreserved:
    def test_empty_version_still_raises(self):
        with pytest.raises(ValueError, match="version"):
            ConfigurationResolver().resolve(StrategyConfig(version=""))

    def test_empty_framework_still_raises(self):
        with pytest.raises(ValueError, match="framework"):
            ConfigurationResolver().resolve(StrategyConfig(framework=""))

    def test_zero_max_candidates_still_raises(self):
        with pytest.raises(ValueError, match="max_candidates"):
            ConfigurationResolver().resolve(
                StrategyConfig(generation=StrategyGeneration(max_candidates=0))
            )

    def test_negative_threshold_still_raises(self):
        with pytest.raises(ValueError, match="min_score_threshold"):
            ConfigurationResolver().resolve(
                StrategyConfig(evaluation=StrategyEvaluation(min_score_threshold=-1.0))
            )


# ---------------------------------------------------------------------------
# StrategyCoordinator — picks up merged config
# ---------------------------------------------------------------------------

class TestStrategyCoordinatorWithDefaults:
    def test_default_coordinator_gets_executive_objectives(self):
        coord = StrategyCoordinator()
        exec_defaults = FrameworkDefaults.get("executive")
        assert coord._config.objectives.primary == exec_defaults.objectives.primary

    def test_custom_objectives_preserved_through_coordinator(self):
        cfg = StrategyConfig(objectives=StrategyObjectives(primary=["custom"]))
        coord = StrategyCoordinator(config=cfg)
        assert coord._config.objectives.primary == ["custom"]
