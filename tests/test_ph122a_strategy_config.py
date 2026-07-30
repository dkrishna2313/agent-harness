"""Tests for PH12.2a strategy configuration models and resolver.

Covers all requirements from the PH12.2a spec §38:
  - StrategyEvaluationConfig (weight policies, criterion bounds)
  - StrategyConstraintConfig (violation policies, penalty bounds)
  - StrategyMappingConfig (authority, confidence, posture weights)
  - StrategyAlignmentConfig (margin, confidence gate, status strings)
  - StrategyHomogenizationConfig (ascending thresholds, dimension names)
  - StrategyDiagnosticsConfig (locked severities)
  - StrategyConfig fingerprint stability
  - ResolvedStrategyConfig (defaults, provenance, backward-compat migrations)
  - CLI integration (validate / show commands)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from functional_agents.strategy.strategy_config import (
    ContentConfig,
    StrategyCoverageConfig,
    StrategyCriterionConfig,
    StrategyAlignmentConfig,
    StrategyConfig,
    StrategyConstraintConfig,
    StrategyContentConfidenceConfig,
    StrategyDiagnosticsConfig,
    StrategyDiscriminationConfig,
    StrategyEvaluationConfig,
    StrategyFallbackConfig,
    StrategyHomogenizationConfig,
    StrategyMappingConfig,
    StrategyMappingConfidenceConfig,
    StrategyRelationshipPriorityConfig,
    StrategyReportingConfig,
)
from functional_agents.strategy.strategy_config_resolver import (
    ResolvedStrategyConfig,
    resolve_strategy_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WORKTREE = Path(__file__).parent.parent  # .../ph5x-exec-lang-hotfix/
_VALID_ENGAGEMENT = str(_WORKTREE / "engagements" / "ENG-002_go_no_go.yaml")
_INVALID_ENGAGEMENT = str(_WORKTREE / "engagements" / "invalid_strategy_config_ph122a_test.yaml")


# ---------------------------------------------------------------------------
# TestEvaluationConfig
# ---------------------------------------------------------------------------

class TestEvaluationConfig:
    """Tests for StrategyEvaluationConfig weight policies and criterion bounds."""

    def test_default_weight_policy_is_strict(self):
        """Default weight_policy is 'strict'."""
        cfg = StrategyEvaluationConfig()
        assert cfg.weight_policy == "strict"

    def test_strict_valid_weights_sum_to_one(self):
        """Strict policy accepts criteria whose enabled weights sum to 1.0."""
        cfg = StrategyEvaluationConfig(
            weight_policy="strict",
            criteria={
                "geographic": StrategyCriterionConfig(weight=0.40),
                "power": StrategyCriterionConfig(weight=0.35),
                "timing": StrategyCriterionConfig(weight=0.25),
            },
        )
        total = sum(c.weight for c in cfg.criteria.values() if c.enabled)
        assert abs(total - 1.0) <= 0.001

    def test_strict_invalid_weight_sum_fails(self):
        """Strict policy rejects criteria whose enabled weights do not sum to 1.0."""
        with pytest.raises(ValidationError, match="sum to"):
            StrategyEvaluationConfig(
                weight_policy="strict",
                criteria={
                    "a": StrategyCriterionConfig(weight=0.5),
                    "b": StrategyCriterionConfig(weight=0.3),
                },
            )

    def test_normalize_policy_normalizes_weights(self):
        """Normalize policy accepts unequal weights; manually normalized sum equals 1.0."""
        cfg = StrategyEvaluationConfig(
            weight_policy="normalize",
            criteria={
                "a": StrategyCriterionConfig(weight=0.40),
                "b": StrategyCriterionConfig(weight=0.30),
                "c": StrategyCriterionConfig(weight=0.20),
            },
        )
        # Weights don't sum to 1.0 (0.90), but normalize policy accepts them.
        weights = [c.weight for c in cfg.criteria.values() if c.enabled]
        total = sum(weights)
        normalized_sum = sum(w / total for w in weights)
        assert abs(normalized_sum - 1.0) < 1e-9

    def test_disabled_criteria_excluded_from_sum(self):
        """Disabled criteria are excluded from the strict sum-to-1 check."""
        cfg = StrategyEvaluationConfig(
            weight_policy="strict",
            criteria={
                "a": StrategyCriterionConfig(weight=0.6, enabled=True),
                "b": StrategyCriterionConfig(weight=0.4, enabled=True),
                "c": StrategyCriterionConfig(weight=0.9, enabled=False),  # excluded
            },
        )
        assert cfg.criteria["c"].enabled is False

    def test_weight_out_of_range_fails(self):
        """Criterion weight > 1.0 is rejected."""
        with pytest.raises(ValidationError):
            StrategyCriterionConfig(weight=1.5)

    def test_weight_negative_fails(self):
        """Criterion weight < 0.0 is rejected."""
        with pytest.raises(ValidationError):
            StrategyCriterionConfig(weight=-0.1)

    def test_all_zero_normalize_fails(self):
        """Normalize policy rejects all-zero enabled weights (would cause division-by-zero)."""
        with pytest.raises(ValidationError, match="weight > 0"):
            StrategyEvaluationConfig(
                weight_policy="normalize",
                criteria={
                    "a": StrategyCriterionConfig(weight=0.0),
                    "b": StrategyCriterionConfig(weight=0.0),
                },
            )

    def test_score_precision_must_be_positive(self):
        """score_precision <= 0 is rejected."""
        with pytest.raises(ValidationError):
            StrategyEvaluationConfig(score_precision=0)

    def test_thresholds_bounded(self):
        """Fraction-type thresholds reject values outside [0.0, 1.0]."""
        with pytest.raises(ValidationError):
            StrategyEvaluationConfig(saturation_threshold=1.5)
        with pytest.raises(ValidationError):
            StrategyEvaluationConfig(minimum_winner_margin=-0.01)


# ---------------------------------------------------------------------------
# TestConstraintConfig
# ---------------------------------------------------------------------------

class TestConstraintConfig:
    """Tests for StrategyConstraintConfig violation policies and penalty bounds."""

    def test_valid_violation_policies(self):
        """All three violation policies are accepted."""
        for policy in ("penalize", "disqualify", "warn"):
            cfg = StrategyConstraintConfig(required_violation_policy=policy)
            assert cfg.required_violation_policy == policy

    def test_invalid_violation_policy_fails(self):
        """An unknown violation policy is rejected."""
        with pytest.raises(ValidationError, match="penalize"):
            StrategyConstraintConfig(required_violation_policy="ignore")

    def test_penalty_bounds(self):
        """Penalty values must be in [0.0, 1.0]."""
        with pytest.raises(ValidationError):
            StrategyConstraintConfig(default_required_penalty=1.5)
        with pytest.raises(ValidationError):
            StrategyConstraintConfig(default_preferred_penalty=-0.1)

    def test_maximum_total_penalty_bounded(self):
        """maximum_total_penalty must be in [0.0, 1.0]."""
        with pytest.raises(ValidationError):
            StrategyConstraintConfig(maximum_total_penalty=1.1)
        cfg = StrategyConstraintConfig(maximum_total_penalty=0.50)
        assert cfg.maximum_total_penalty == 0.50


# ---------------------------------------------------------------------------
# TestMappingConfig
# ---------------------------------------------------------------------------

class TestMappingConfig:
    """Tests for StrategyMappingConfig authority, confidence, and posture weights."""

    def test_default_enabled(self):
        """Option mapping is enabled by default."""
        cfg = StrategyMappingConfig()
        assert cfg.enabled is True

    def test_authority_single_pass_default_true(self):
        """Authority single_pass defaults to True."""
        cfg = StrategyMappingConfig()
        assert cfg.authority.single_pass is True

    def test_allow_content_resolution_remap_default_false(self):
        """Authority allow_content_resolution_remap defaults to False."""
        cfg = StrategyMappingConfig()
        assert cfg.authority.allow_content_resolution_remap is False

    def test_high_threshold_gte_minimum_threshold(self):
        """high_score_threshold must be >= minimum_authoritative_score."""
        with pytest.raises(ValidationError, match="must be >="):
            StrategyMappingConfidenceConfig(
                minimum_authoritative_score=0.50,
                high_score_threshold=0.30,  # less than minimum
            )
        # Valid: equal thresholds are allowed
        cfg = StrategyMappingConfidenceConfig(
            minimum_authoritative_score=0.20,
            high_score_threshold=0.20,
        )
        assert cfg.high_score_threshold >= cfg.minimum_authoritative_score

    def test_hard_penalty_gte_soft_penalty(self):
        """geographic_hard penalty must be >= geographic_soft penalty."""
        with pytest.raises(ValidationError, match="must be >="):
            StrategyMappingConfig(
                contradiction_penalties={
                    "geographic_hard": 0.05,
                    "geographic_soft": 0.20,  # soft > hard — invalid
                }
            )

    def test_unresolved_policy_enum(self):
        """Only the three recognised unresolved_policy values are accepted."""
        for policy in ("preserve_upstream", "report_unresolved", "fail"):
            cfg = StrategyMappingConfig(unresolved_policy=policy)
            assert cfg.unresolved_policy == policy
        with pytest.raises(ValidationError):
            StrategyMappingConfig(unresolved_policy="skip")


# ---------------------------------------------------------------------------
# TestAlignmentConfig
# ---------------------------------------------------------------------------

class TestAlignmentConfig:
    """Tests for StrategyAlignmentConfig margin, confidence gate, and status assignments."""

    def test_challenge_margin_bounded(self):
        """minimum_challenge_margin must be in [0.0, 1.0]."""
        with pytest.raises(ValidationError):
            StrategyAlignmentConfig(minimum_challenge_margin=1.5)
        cfg = StrategyAlignmentConfig(minimum_challenge_margin=0.10)
        assert cfg.minimum_challenge_margin == 0.10

    def test_valid_confidence_levels(self):
        """'Low', 'Medium', and 'High' are all valid confidence levels."""
        for level in ("Low", "Medium", "High"):
            cfg = StrategyAlignmentConfig(minimum_challenge_confidence=level)
            assert cfg.minimum_challenge_confidence == level

    def test_invalid_confidence_level_fails(self):
        """An unrecognised confidence level is rejected."""
        with pytest.raises(ValidationError):
            StrategyAlignmentConfig(minimum_challenge_confidence="Very High")

    def test_valid_status_values(self):
        """The four recognised status strings are all accepted."""
        for status in ("confirmed", "challenged", "refined", "unresolved"):
            cfg = StrategyAlignmentConfig(same_option_high_confidence_status=status)
            assert cfg.same_option_high_confidence_status == status

    def test_invalid_status_fails(self):
        """An unrecognised status string is rejected."""
        with pytest.raises(ValidationError):
            StrategyAlignmentConfig(same_option_high_confidence_status="uncertain")

    def test_valid_unresolved_authority(self):
        """All three unresolved_authority values are accepted."""
        for authority in ("upstream_preferred", "strategy_selected", "none"):
            cfg = StrategyAlignmentConfig(unresolved_authority=authority)
            assert cfg.unresolved_authority == authority


# ---------------------------------------------------------------------------
# TestHomogenizationConfig
# ---------------------------------------------------------------------------

class TestHomogenizationConfig:
    """Tests for StrategyHomogenizationConfig threshold ordering and dimension validation."""

    def test_thresholds_ascending_order(self):
        """partial <= substantial <= full must hold."""
        cfg = StrategyHomogenizationConfig(
            partial_threshold=0.70,
            substantial_threshold=0.85,
            full_threshold=0.99,
        )
        assert cfg.partial_threshold <= cfg.substantial_threshold <= cfg.full_threshold

    def test_invalid_threshold_order_fails(self):
        """Thresholds in wrong order are rejected."""
        with pytest.raises(ValidationError, match="must satisfy"):
            StrategyHomogenizationConfig(
                partial_threshold=0.90,
                substantial_threshold=0.80,  # < partial — invalid
                full_threshold=0.99,
            )

    def test_all_in_unit_interval(self):
        """Threshold values must be in [0.0, 1.0]."""
        with pytest.raises(ValidationError):
            StrategyHomogenizationConfig(partial_threshold=1.5)

    def test_material_dimensions_validated(self):
        """Valid dimension names from the allowed set are accepted."""
        cfg = StrategyHomogenizationConfig(
            material_dimensions=["assumptions", "risks", "opportunities"]
        )
        assert "assumptions" in cfg.material_dimensions

    def test_invalid_dimension_fails(self):
        """Unknown dimension names are rejected."""
        with pytest.raises(ValidationError, match="Invalid material_dimensions"):
            StrategyHomogenizationConfig(material_dimensions=["unknowndim"])


# ---------------------------------------------------------------------------
# TestDiagnosticsConfig
# ---------------------------------------------------------------------------

class TestDiagnosticsConfig:
    """Tests for StrategyDiagnosticsConfig locked severities and valid values."""

    def test_valid_severities(self):
        """All four severity levels are accepted on free fields."""
        for severity in ("ignore", "info", "warning", "error"):
            cfg = StrategyDiagnosticsConfig(low_mapping_confidence=severity)
            assert cfg.low_mapping_confidence == severity

    def test_invalid_severity_fails(self):
        """Unknown severity strings are rejected."""
        with pytest.raises(ValidationError):
            StrategyDiagnosticsConfig(low_mapping_confidence="critical")

    def test_mapping_mismatch_cannot_be_below_error(self):
        """mapping_mismatch is locked to 'error'; setting it to 'warning' raises."""
        with pytest.raises(ValidationError, match="mapping_mismatch"):
            StrategyDiagnosticsConfig(mapping_mismatch="warning")

    def test_invalid_weight_sum_cannot_be_below_error(self):
        """invalid_weight_sum is locked to 'error'; setting it to 'warning' raises."""
        with pytest.raises(ValidationError, match="invalid_weight_sum"):
            StrategyDiagnosticsConfig(invalid_weight_sum="warning")


# ---------------------------------------------------------------------------
# TestStrategyConfigFingerprint
# ---------------------------------------------------------------------------

class TestStrategyConfigFingerprint:
    """Tests for StrategyConfig.compute_fingerprint() stability and format."""

    def test_fingerprint_is_hex_string(self):
        """compute_fingerprint returns a hex string."""
        cfg = StrategyConfig()
        fp = cfg.compute_fingerprint()
        # Valid hex: all chars in 0-9a-f
        assert all(c in "0123456789abcdef" for c in fp)

    def test_fingerprint_stable_across_equivalent_configs(self):
        """Two identically-constructed StrategyConfig instances produce the same fingerprint."""
        cfg_a = StrategyConfig()
        cfg_b = StrategyConfig()
        assert cfg_a.compute_fingerprint() == cfg_b.compute_fingerprint()

    def test_fingerprint_changes_on_policy_change(self):
        """A change in a policy value changes the fingerprint."""
        cfg_a = StrategyConfig()
        cfg_b = StrategyConfig(
            alignment_config=StrategyAlignmentConfig(minimum_challenge_margin=0.25)
        )
        assert cfg_a.compute_fingerprint() != cfg_b.compute_fingerprint()

    def test_fingerprint_16_chars(self):
        """Fingerprint is exactly 16 hexadecimal characters."""
        cfg = StrategyConfig()
        fp = cfg.compute_fingerprint()
        assert len(fp) == 16


# ---------------------------------------------------------------------------
# TestResolveStrategyConfig
# ---------------------------------------------------------------------------

class TestResolveStrategyConfig:
    """Tests for resolve_strategy_config() provenance, migrations, and source tagging."""

    def test_none_input_returns_all_defaults(self):
        """None input produces a fully-defaulted ResolvedStrategyConfig."""
        result = resolve_strategy_config(None)
        assert isinstance(result, ResolvedStrategyConfig)
        assert result.resolved is not None

    def test_empty_dict_returns_defaults(self):
        """Empty dict is equivalent to None — all fields use defaults."""
        result = resolve_strategy_config({})
        assert isinstance(result, ResolvedStrategyConfig)

    def test_source_defaults_only_when_no_input(self):
        """Source tag is 'defaults_only' when no YAML block is supplied."""
        result = resolve_strategy_config(None)
        assert result.source == "defaults_only"
        result2 = resolve_strategy_config({})
        assert result2.source == "defaults_only"

    def test_source_engagement_yaml_when_provided(self):
        """Source tag is 'engagement_yaml' when a non-empty dict is supplied."""
        result = resolve_strategy_config({"framework": "executive"})
        assert result.source == "engagement_yaml"

    def test_defaults_applied_list_populated(self):
        """defaults_applied lists fields that came entirely from defaults."""
        result = resolve_strategy_config(None)
        # When no input, all policy fields are defaults
        assert len(result.defaults_applied) > 0
        assert "evaluation_config" in result.defaults_applied
        assert "alignment_config" in result.defaults_applied

    def test_raw_config_unchanged_by_resolution(self):
        """The raw dict stored in ResolvedStrategyConfig matches the input exactly."""
        raw = {"framework": "executive", "enabled": True}
        result = resolve_strategy_config(raw)
        assert result.raw == raw

    def test_resolved_config_complete(self):
        """The resolved config is a fully-populated StrategyConfig instance."""
        result = resolve_strategy_config(None)
        cfg = result.resolved
        assert isinstance(cfg, StrategyConfig)
        assert cfg.enabled is True
        assert cfg.config_version == "ph12.2a-v1"

    def test_fingerprint_present_and_stable(self):
        """Fingerprint is present and deterministic for the same input."""
        result_a = resolve_strategy_config(None)
        result_b = resolve_strategy_config(None)
        assert result_a.fingerprint != ""
        assert result_a.fingerprint == result_b.fingerprint

    def test_backward_compat_old_alignment_policy_translates(self):
        """Old alignment_policy dict is migrated to alignment_config."""
        raw = {
            "alignment_policy": {
                "minimum_challenge_margin": 0.15,
                "minimum_mapping_confidence": "High",
            }
        }
        result = resolve_strategy_config(raw)
        # Migration should be reflected in deprecations
        assert any("alignment_policy" in d for d in result.deprecations)
        # The resolved alignment_config should carry the migrated margin
        assert result.resolved.alignment_config.minimum_challenge_margin == 0.15
        # minimum_mapping_confidence → minimum_challenge_confidence
        assert result.resolved.alignment_config.minimum_challenge_confidence == "High"


# ---------------------------------------------------------------------------
# TestBackwardCompat
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """Tests that PH12.2a additions do not break existing usage patterns."""

    def test_strategy_config_without_new_fields_validates(self):
        """StrategyConfig constructed with only pre-PH12.2a fields validates correctly."""
        cfg = StrategyConfig(
            framework="executive",
            version="1.0",
        )
        assert cfg.framework == "executive"
        assert cfg.enabled is True

    def test_old_content_config_still_works(self):
        """ContentConfig with only pre-PH12.2a fields validates correctly."""
        cfg = ContentConfig(
            minimum_relevance_score=0.20,
            maximum_assumptions_per_theory=5,
            allow_symmetric_fallback=True,
            minimum_content_coverage=0.50,
        )
        assert cfg.minimum_relevance_score == 0.20

    def test_full_strategy_config_defaults_reproduce_ph122b_behavior(self):
        """Default StrategyConfig preserves PH12.2b content config thresholds exactly."""
        cfg = StrategyConfig()
        content = cfg.content
        # PH12.2b defaults
        assert content.minimum_discrimination_score == 0.20
        assert content.partial_homogenization_threshold == 0.75
        assert content.full_homogenization_threshold == 0.95
        assert content.maximum_identical_dimensions == 2
        assert content.allow_symmetric_fallback is True
        assert content.minimum_relevance_score == 0.20
        assert content.minimum_content_coverage == 0.50
        # PH12.2a addition
        assert content.minimum_distinctive_coverage == 0.20


# ---------------------------------------------------------------------------
# TestCLIIntegration
# ---------------------------------------------------------------------------

class TestCLIIntegration:
    """Integration tests for the 'strategy config' CLI sub-commands.

    Uses subprocess to drive the actual CLI so we exercise the full stack.
    """

    def test_validate_command_exits_zero_for_valid_config(self):
        """'strategy config validate' exits 0 for a well-formed engagement YAML."""
        result = subprocess.run(
            [
                "python3", "-m", "functional_agents.cli",
                "strategy", "config", "validate",
                "--engagement", _VALID_ENGAGEMENT,
            ],
            capture_output=True,
            text=True,
            cwd=str(_WORKTREE),
        )
        assert result.returncode == 0, (
            f"Expected exit 0, got {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "valid" in result.stdout.lower()

    def test_validate_command_exits_nonzero_for_invalid_config(self):
        """'strategy config validate' exits non-zero for a structurally invalid YAML."""
        result = subprocess.run(
            [
                "python3", "-m", "functional_agents.cli",
                "strategy", "config", "validate",
                "--engagement", _INVALID_ENGAGEMENT,
            ],
            capture_output=True,
            text=True,
            cwd=str(_WORKTREE),
        )
        assert result.returncode != 0, (
            f"Expected non-zero exit, got {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_show_command_produces_output(self):
        """'strategy config show' produces non-empty text output."""
        result = subprocess.run(
            [
                "python3", "-m", "functional_agents.cli",
                "strategy", "config", "show",
                "--engagement", _VALID_ENGAGEMENT,
            ],
            capture_output=True,
            text=True,
            cwd=str(_WORKTREE),
        )
        assert result.returncode == 0, (
            f"Expected exit 0.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert len(result.stdout.strip()) > 0

    def test_show_json_format(self):
        """'strategy config show --format json' emits valid JSON with expected keys."""
        result = subprocess.run(
            [
                "python3", "-m", "functional_agents.cli",
                "strategy", "config", "show",
                "--format", "json",
                "--engagement", _VALID_ENGAGEMENT,
            ],
            capture_output=True,
            text=True,
            cwd=str(_WORKTREE),
        )
        assert result.returncode == 0, (
            f"Expected exit 0.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        parsed = json.loads(result.stdout)
        assert "fingerprint" in parsed
        assert "resolved" in parsed
        assert "config_version" in parsed
        assert "source" in parsed


class TestYAMLDiscovery:
    """§25 — StrategyCoordinator correctly receives raw_strategy_yaml from engagement YAML."""

    def test_coordinator_raw_yaml_from_engagement(self):
        """Coordinator built via the orchestrator pattern returns source=engagement_yaml."""
        import yaml
        from functional_agents.engagement_spec import load_engagement_spec
        from functional_agents.strategy import StrategyCoordinator, StrategyConfig, ConfigurationResolver
        from functional_agents.strategy.strategy_config_resolver import resolve_strategy_config

        spec = load_engagement_spec(str(_WORKTREE / "engagements" / "us_data_center_siting_strategy1.yaml"))
        strategy_raw = getattr(spec, "strategy", None)

        assert strategy_raw is not None, "engagement_spec.strategy must be non-None"
        assert isinstance(strategy_raw, dict), "engagement_spec.strategy must be a dict"
        assert len(strategy_raw) > 0, "engagement_spec.strategy must be non-empty"

        strategy_config = StrategyConfig()
        if strategy_raw:
            strategy_config = ConfigurationResolver().resolve_from_engagement(strategy_config, strategy_raw)

        sc = StrategyCoordinator(config=strategy_config, raw_strategy_yaml=strategy_raw or {})

        assert len(sc._raw_strategy_yaml) > 0, "_raw_strategy_yaml must be non-empty"

        resolved = resolve_strategy_config(sc._raw_strategy_yaml)
        assert resolved.source == "engagement_yaml", (
            f"Expected source=engagement_yaml, got source={resolved.source!r}. "
            "This means raw_strategy_yaml threading is broken."
        )
        assert len(resolved.fingerprint) == 16, (
            f"Expected 16-char fingerprint, got {resolved.fingerprint!r} ({len(resolved.fingerprint)} chars)."
        )
        assert len(resolved.defaults_applied) <= 10, (
            f"Expected ≤10 defaults, got {len(resolved.defaults_applied)}: {resolved.defaults_applied}"
        )

    def test_coordinator_defaults_only_without_engagement(self):
        """Coordinator built without raw_strategy_yaml returns source=defaults_only."""
        from functional_agents.strategy import StrategyCoordinator
        from functional_agents.strategy.strategy_config_resolver import resolve_strategy_config

        sc = StrategyCoordinator()
        resolved = resolve_strategy_config(sc._raw_strategy_yaml)
        assert resolved.source == "defaults_only"


class TestVersionGate:
    """Version gate — unknown config_version values emit a warning."""

    def test_known_version_no_warning(self):
        """Known config_version produces no version warning."""
        from functional_agents.strategy.strategy_config_resolver import resolve_strategy_config
        result = resolve_strategy_config({"config_version": "ph12.2a-v1", "enabled": True})
        version_warns = [w for w in result.warnings if "not recognised" in w]
        assert version_warns == []

    def test_unknown_version_emits_warning(self):
        """Unknown config_version produces a warning in the warnings list."""
        from functional_agents.strategy.strategy_config_resolver import resolve_strategy_config
        result = resolve_strategy_config({"config_version": "ph99.0-unknown", "enabled": True})
        version_warns = [w for w in result.warnings if "not recognised" in w]
        assert len(version_warns) == 1
        assert "ph99.0-unknown" in version_warns[0]

    def test_no_version_no_warning(self):
        """Missing config_version (defaults path) produces no version warning."""
        from functional_agents.strategy.strategy_config_resolver import resolve_strategy_config
        result = resolve_strategy_config({"enabled": True})
        version_warns = [w for w in result.warnings if "not recognised" in w]
        assert version_warns == []


# ---------------------------------------------------------------------------
# PH12.2a — Canonical snapshot, normalized weights, and fingerprint consistency
# ---------------------------------------------------------------------------

_PROD_ENGAGEMENT = str(_WORKTREE / "engagements" / "us_data_center_siting_strategy1.yaml")


def _prod_resolved():
    from functional_agents.engagement_spec import load_engagement_spec
    from functional_agents.strategy.strategy_config_resolver import resolve_strategy_config
    spec = load_engagement_spec(_PROD_ENGAGEMENT)
    return resolve_strategy_config(spec.strategy)


class TestCanonicalSnapshot:
    """Canonical snapshot structure, normalized weights, and fingerprint consistency."""

    def test_all_sections_present_and_populated(self):
        """Every required section in canonical_snapshot is a non-empty dict."""
        r = _prod_resolved()
        snap = r.canonical_snapshot
        for section in ("evaluation", "constraints", "mapping", "alignment", "content", "reporting", "diagnostics"):
            val = snap.get(section)
            assert isinstance(val, dict), f"section {section!r} is not a dict: {val!r}"
            assert val, f"section {section!r} is empty"

    def test_no_section_is_null(self):
        """No resolved section is null when using production engagement YAML."""
        r = _prod_resolved()
        for section, val in r.canonical_snapshot.items():
            assert val is not None, f"section {section!r} is None"

    def test_normalized_weights_sum_to_one(self):
        """resolved_weight values for enabled criteria sum to 1.0."""
        r = _prod_resolved()
        criteria = r.canonical_snapshot["evaluation"]["criteria"]
        weights = [v["resolved_weight"] for v in criteria.values() if v.get("enabled", True)]
        assert weights, "No enabled criteria found"
        assert abs(sum(weights) - 1.0) <= 0.001, f"Weights sum to {sum(weights)}, expected 1.0"

    def test_configured_and_resolved_weight_present(self):
        """Each criterion exposes both configured_weight and resolved_weight."""
        r = _prod_resolved()
        criteria = r.canonical_snapshot["evaluation"]["criteria"]
        for name, crit in criteria.items():
            assert "configured_weight" in crit, f"criterion {name!r} missing configured_weight"
            assert "resolved_weight" in crit, f"criterion {name!r} missing resolved_weight"

    def test_mapping_section_populated_without_yaml(self):
        """mapping section is populated even when omitted from YAML."""
        from functional_agents.strategy.strategy_config_resolver import resolve_strategy_config
        r = resolve_strategy_config({"config_version": "ph12.2a-v1"})
        mapping = r.canonical_snapshot.get("mapping")
        assert isinstance(mapping, dict), f"mapping should be dict, got {mapping!r}"
        assert mapping, "mapping should be non-empty (filled from defaults)"

    def test_alignment_section_populated_without_yaml(self):
        """alignment section is populated even when omitted from YAML."""
        from functional_agents.strategy.strategy_config_resolver import resolve_strategy_config
        r = resolve_strategy_config({"config_version": "ph12.2a-v1"})
        alignment = r.canonical_snapshot.get("alignment")
        assert isinstance(alignment, dict), f"alignment should be dict, got {alignment!r}"
        assert alignment, "alignment should be non-empty (filled from defaults)"

    def test_reporting_section_populated_without_yaml(self):
        """reporting section is populated even when omitted from YAML."""
        from functional_agents.strategy.strategy_config_resolver import resolve_strategy_config
        r = resolve_strategy_config({"config_version": "ph12.2a-v1"})
        reporting = r.canonical_snapshot.get("reporting")
        assert isinstance(reporting, dict), f"reporting should be dict, got {reporting!r}"
        assert reporting, "reporting should be non-empty (filled from defaults)"

    def test_diagnostics_section_populated_without_yaml(self):
        """diagnostics section is populated even when omitted from YAML."""
        from functional_agents.strategy.strategy_config_resolver import resolve_strategy_config
        r = resolve_strategy_config({"config_version": "ph12.2a-v1"})
        diagnostics = r.canonical_snapshot.get("diagnostics")
        assert isinstance(diagnostics, dict), f"diagnostics should be dict, got {diagnostics!r}"
        assert diagnostics, "diagnostics should be non-empty (filled from defaults)"

    def test_fingerprint_consistent_across_calls(self):
        """Same engagement YAML always produces the same fingerprint."""
        r1 = _prod_resolved()
        r2 = _prod_resolved()
        assert r1.fingerprint == r2.fingerprint, "Fingerprint is not deterministic"

    def test_cli_and_resolver_fingerprint_match(self):
        """CLI config show --format json fingerprint matches resolve_strategy_config() fingerprint."""
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "functional_agents.cli", "strategy", "config", "show",
             "--format", "json", "--engagement", _PROD_ENGAGEMENT],
            capture_output=True, text=True, cwd=str(_WORKTREE),
        )
        assert result.returncode == 0, f"CLI error: {result.stderr}"
        cli_out = json.loads(result.stdout)
        r = _prod_resolved()
        assert cli_out["fingerprint"] == r.fingerprint, (
            f"CLI fingerprint {cli_out['fingerprint']!r} != resolver fingerprint {r.fingerprint!r}"
        )

    def test_cli_resolved_matches_canonical_snapshot(self):
        """CLI --format json resolved dict is structurally equal to canonical_snapshot."""
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "functional_agents.cli", "strategy", "config", "show",
             "--format", "json", "--engagement", _PROD_ENGAGEMENT],
            capture_output=True, text=True, cwd=str(_WORKTREE),
        )
        assert result.returncode == 0
        cli_resolved = json.loads(result.stdout)["resolved"]
        r = _prod_resolved()
        # Keys must match
        assert set(cli_resolved.keys()) == set(r.canonical_snapshot.keys()), (
            f"CLI resolved keys {sorted(cli_resolved)} != canonical_snapshot keys {sorted(r.canonical_snapshot)}"
        )


class TestAlignmentRegression:
    """Alignment evaluator unit tests for PH12.2a policy compliance."""

    def _make_ctx(self, preferred_option_id: str, strategic_options: list | None = None):
        """Build a minimal AgentContext-like namespace for alignment testing."""
        from types import SimpleNamespace
        return SimpleNamespace(
            preferred_option={"option_id": preferred_option_id},
            strategic_options=strategic_options or [],
        )

    def _make_theory(self, theory_id: str = "TH-SCS-1"):
        from functional_agents.strategy.strategic_position import TheoryOfWinning
        return TheoryOfWinning(
            theory_id=theory_id,
            source_choice_set_id="SCS-1",
            recommended_option_id="OPT-B",
            recommended_option_title="Option B",
        )

    def _make_mapping(self, mapped_option_id: str, confidence: str = "Medium"):
        from functional_agents.strategy.alignment import OptionMapping
        return OptionMapping(
            mapped_option_id=mapped_option_id,
            mapping_score=0.45,
            mapping_confidence=confidence,
            mapping_rationale="test",
        )

    def _make_selection(self, margin: float = 0.15):
        from functional_agents.strategy.strategy_selector import StrategySelection
        return StrategySelection(
            winner_theory_id="TH-SCS-1",
            winner_score=0.72,
            runner_up_theory_id="TH-SCS-2",
            runner_up_score=0.72 - margin,
            score_margin=margin,
            tie_breaker_used=None,
        )

    def test_same_preferred_and_mapped_medium_confidence_yields_refined(self):
        """preferred==mapped==OPT-B + Medium confidence → alignment_status=refined (not challenged)."""
        from functional_agents.strategy.alignment_evaluator import AlignmentEvaluator
        from functional_agents.strategy.strategy_config import AlignmentPolicy

        ap = AlignmentPolicy(minimum_challenge_margin=0.10, minimum_mapping_confidence="Medium")
        ctx = self._make_ctx("OPT-B")
        theory = self._make_theory()
        mapping = self._make_mapping("OPT-B", "Medium")
        selection = self._make_selection(margin=0.15)

        result = AlignmentEvaluator().evaluate(theory, mapping, selection, ctx, policy=ap)
        assert result.status in {"refined", "confirmed"}, (
            f"Expected refined/confirmed, got {result.status!r}. "
            f"preferred={result.preferred_option_id!r} mapped={result.mapped_option_id!r}"
        )

    def test_different_option_with_high_margin_yields_challenged(self):
        """preferred=OPT-A, mapped=OPT-B, margin=0.20 → challenged."""
        from functional_agents.strategy.alignment_evaluator import AlignmentEvaluator
        from functional_agents.strategy.strategy_config import AlignmentPolicy

        ap = AlignmentPolicy(minimum_challenge_margin=0.10, minimum_mapping_confidence="Medium")
        ctx = self._make_ctx("OPT-A")
        theory = self._make_theory()
        mapping = self._make_mapping("OPT-B", "High")
        selection = self._make_selection(margin=0.20)

        result = AlignmentEvaluator().evaluate(theory, mapping, selection, ctx, policy=ap)
        assert result.status == "challenged", (
            f"Expected challenged, got {result.status!r}"
        )

    def test_resolved_alignment_policy_reaches_evaluator(self):
        """AlignmentPolicy built from resolved alignment_config reaches AlignmentEvaluator correctly."""
        from functional_agents.strategy.strategy_config_resolver import resolve_strategy_config
        from functional_agents.strategy.alignment_evaluator import AlignmentEvaluator
        from functional_agents.strategy.strategy_config import AlignmentPolicy

        r = resolve_strategy_config({
            "config_version": "ph12.2a-v1",
            "alignment_config": {
                "minimum_challenge_margin": 0.05,
                "minimum_challenge_confidence": "Low",
            },
        })
        _align_cfg = r.resolved.alignment_config
        ap = AlignmentPolicy(
            minimum_challenge_margin=_align_cfg.minimum_challenge_margin,
            minimum_mapping_confidence=_align_cfg.minimum_challenge_confidence,
        )
        assert ap.minimum_challenge_margin == 0.05
        assert ap.minimum_mapping_confidence == "Low"

        ctx = self._make_ctx("OPT-B")
        theory = self._make_theory()
        mapping = self._make_mapping("OPT-B", "Low")
        selection = self._make_selection(margin=0.10)

        result = AlignmentEvaluator().evaluate(theory, mapping, selection, ctx, policy=ap)
        assert result.status in {"refined", "confirmed"}, (
            f"Low-confidence medium-margin same-option should be refined/confirmed, got {result.status!r}"
        )

    def test_resolved_mapping_policy_reaches_option_mapper(self):
        """OptionMapper constructed with mapping_config uses configured thresholds."""
        from functional_agents.strategy.strategy_config_resolver import resolve_strategy_config
        from functional_agents.strategy.option_mapper import OptionMapper

        r = resolve_strategy_config({"config_version": "ph12.2a-v1"})
        mapper = OptionMapper(mapping_config=r.resolved.mapping_config)

        assert mapper._high_threshold == r.resolved.mapping_config.confidence.high_score_threshold
        assert mapper._med_threshold == r.resolved.mapping_config.confidence.minimum_authoritative_score
