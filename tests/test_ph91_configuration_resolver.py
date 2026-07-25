"""PH9.1 — ConfigurationResolver unit tests.

Covers:
- Pass-through: resolved config values equal input values
- Immutability: resolved instance is a distinct object
- Defensive validation: hard invariants raise ValueError
- StrategyCoordinator: routes through ConfigurationResolver
- StrategyCoordinator: default config resolves without error
"""

import pytest

from functional_agents.strategy import (
    ConfigurationResolver,
    StrategyConfig,
    StrategyCoordinator,
    StrategyEvaluation,
    StrategyGeneration,
    StrategyObjectives,
)


# ---------------------------------------------------------------------------
# ConfigurationResolver — pass-through
# ---------------------------------------------------------------------------

class TestConfigurationResolverPassThrough:
    def test_default_config_resolves(self):
        cfg = StrategyConfig()
        resolved = ConfigurationResolver().resolve(cfg)
        assert resolved.version == cfg.version
        assert resolved.framework == cfg.framework

    def test_all_top_level_fields_preserved(self):
        cfg = StrategyConfig(
            version="2.0",
            framework="board",
            objectives=StrategyObjectives(primary=["win"], secondary=["survive"]),
            evaluation=StrategyEvaluation(method="scoring", min_score_threshold=0.5),
            generation=StrategyGeneration(max_candidates=5, diversity_required=False),
        )
        resolved = ConfigurationResolver().resolve(cfg)

        assert resolved.version == "2.0"
        assert resolved.framework == "board"
        assert resolved.objectives.primary == ["win"]
        assert resolved.objectives.secondary == ["survive"]
        assert resolved.evaluation.method == "scoring"
        assert resolved.evaluation.min_score_threshold == 0.5
        assert resolved.generation.max_candidates == 5
        assert resolved.generation.diversity_required is False

    def test_constraints_preserved(self):
        from functional_agents.strategy import StrategyConstraints
        cfg = StrategyConfig(
            constraints=StrategyConstraints(
                excluded_options=["OPT-Z"],
                required_conditions=["market_ready"],
            )
        )
        resolved = ConfigurationResolver().resolve(cfg)
        assert resolved.constraints.excluded_options == ["OPT-Z"]
        assert resolved.constraints.required_conditions == ["market_ready"]

    def test_validation_fields_preserved(self):
        from functional_agents.strategy import StrategyValidation
        cfg = StrategyConfig(
            validation=StrategyValidation(
                require_evidence=True,
                min_confidence="High",
                require_assumptions=True,
            )
        )
        resolved = ConfigurationResolver().resolve(cfg)
        assert resolved.validation.require_evidence is True
        assert resolved.validation.min_confidence == "High"
        assert resolved.validation.require_assumptions is True

    def test_metadata_preserved(self):
        from functional_agents.strategy import StrategyMetadata
        cfg = StrategyConfig(
            metadata=StrategyMetadata(
                author="D. Krishna",
                engagement_id="ENG-001",
                notes="test run",
            )
        )
        resolved = ConfigurationResolver().resolve(cfg)
        assert resolved.metadata.author == "D. Krishna"
        assert resolved.metadata.engagement_id == "ENG-001"
        assert resolved.metadata.notes == "test run"


# ---------------------------------------------------------------------------
# ConfigurationResolver — immutability
# ---------------------------------------------------------------------------

class TestConfigurationResolverImmutability:
    def test_resolved_is_new_object(self):
        cfg = StrategyConfig()
        resolved = ConfigurationResolver().resolve(cfg)
        assert resolved is not cfg

    def test_caller_object_not_mutated(self):
        cfg = StrategyConfig(framework="executive")
        original_framework = cfg.framework
        ConfigurationResolver().resolve(cfg)
        assert cfg.framework == original_framework

    def test_resolved_objectives_is_new_object(self):
        cfg = StrategyConfig(objectives=StrategyObjectives(primary=["win"]))
        resolved = ConfigurationResolver().resolve(cfg)
        assert resolved.objectives is not cfg.objectives

    def test_multiple_resolves_are_independent(self):
        cfg = StrategyConfig(framework="executive")
        r1 = ConfigurationResolver().resolve(cfg)
        r2 = ConfigurationResolver().resolve(cfg)
        assert r1 is not r2
        assert r1.framework == r2.framework


# ---------------------------------------------------------------------------
# ConfigurationResolver — defensive validation
# ---------------------------------------------------------------------------

class TestConfigurationResolverValidation:
    def test_empty_version_raises(self):
        cfg = StrategyConfig(version="")
        with pytest.raises(ValueError, match="version"):
            ConfigurationResolver().resolve(cfg)

    def test_empty_framework_raises(self):
        cfg = StrategyConfig(framework="")
        with pytest.raises(ValueError, match="framework"):
            ConfigurationResolver().resolve(cfg)

    def test_zero_max_candidates_raises(self):
        cfg = StrategyConfig(generation=StrategyGeneration(max_candidates=0))
        with pytest.raises(ValueError, match="max_candidates"):
            ConfigurationResolver().resolve(cfg)

    def test_negative_max_candidates_raises(self):
        cfg = StrategyConfig(generation=StrategyGeneration(max_candidates=-1))
        with pytest.raises(ValueError, match="max_candidates"):
            ConfigurationResolver().resolve(cfg)

    def test_negative_min_score_threshold_raises(self):
        cfg = StrategyConfig(evaluation=StrategyEvaluation(min_score_threshold=-0.1))
        with pytest.raises(ValueError, match="min_score_threshold"):
            ConfigurationResolver().resolve(cfg)

    def test_valid_custom_config_does_not_raise(self):
        cfg = StrategyConfig(
            version="1.0",
            framework="board",
            generation=StrategyGeneration(max_candidates=1),
            evaluation=StrategyEvaluation(min_score_threshold=0.0),
        )
        resolved = ConfigurationResolver().resolve(cfg)
        assert resolved.framework == "board"


# ---------------------------------------------------------------------------
# StrategyCoordinator — routes through ConfigurationResolver
# ---------------------------------------------------------------------------

class TestStrategyCoordinatorUsesResolver:
    def test_default_coordinator_has_resolved_config(self):
        coord = StrategyCoordinator()
        assert isinstance(coord._config, StrategyConfig)
        assert coord._config.framework == "executive"
        assert coord._config.version == "1.0"

    def test_custom_config_is_resolved(self):
        cfg = StrategyConfig(framework="board")
        coord = StrategyCoordinator(config=cfg)
        assert coord._config.framework == "board"

    def test_coordinator_config_is_independent_of_input(self):
        cfg = StrategyConfig(framework="executive")
        coord = StrategyCoordinator(config=cfg)
        assert coord._config is not cfg

    def test_invalid_config_raises_at_construction(self):
        cfg = StrategyConfig(framework="")
        with pytest.raises(ValueError, match="framework"):
            StrategyCoordinator(config=cfg)

    def test_none_config_resolves_to_default(self):
        coord = StrategyCoordinator(config=None)
        assert coord._config.framework == "executive"
        assert coord._config.generation.max_candidates == 3
