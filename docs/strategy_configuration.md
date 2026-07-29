# Strategy Configuration (PH12.2a)

## Overview

The Strategy Layer accepts an optional `strategy:` block in engagement YAML files. When absent, all defaults reproduce the PH12.2b production baseline: winner = diversified/BTM-first, mapped option = OPT-B, alignment = refined.

PH12.2a adds a validated, typed configuration surface for all major Strategy policies. The configuration is resolved at runtime, snapshotted into `StrategyTrace`, and identified by a deterministic SHA-256 fingerprint. It does not yet drive scoring behavior (that is planned for a future phase); it makes configuration explicit and auditable.

---

## Configuration Hierarchy

Resolution order (later entries win):

1. Hard-coded defaults in Pydantic model field definitions
2. Fields absent from the engagement YAML (filled from defaults)
3. Fields present in the engagement YAML's `strategy:` block
4. Deprecated-field migrations applied before validation (backward compat)

The result is a `ResolvedStrategyConfig` containing:

- `resolved` — fully-validated `StrategyConfig`
- `defaults_applied` — list of top-level keys taken from defaults
- `deprecations` — migration warnings for renamed fields
- `warnings` — non-fatal issues
- `fingerprint` — first 16 hex chars of SHA-256 of canonical JSON
- `source` — `"engagement_yaml"` or `"defaults_only"`

---

## How to Configure

Add a `strategy:` block to your engagement YAML. Only the fields you set override defaults; all others remain at defaults.

```yaml
strategy:
  enabled: true
  config_version: ph12.2a-v1

  # Evaluation policy — normalized weights for the six scoring criteria.
  # weight_policy: strict requires enabled weights to sum to 1.0 ± 0.001.
  evaluation_config:
    weight_policy: strict
    criteria:
      evidence_quality:
        weight: 0.20
        enabled: true
      assumption_robustness:
        weight: 0.20
        enabled: true
      risk_resilience:
        weight: 0.20
        enabled: true
      opportunity_capture:
        weight: 0.15
        enabled: true
      strategic_fit:
        weight: 0.15
        enabled: true
      execution_feasibility:
        weight: 0.10
        enabled: true
    minimum_winner_margin: 0.05
    saturation_threshold: 0.95

  # Option-mapping policy.
  mapping_config:
    enabled: true
    unresolved_policy: preserve_upstream
    authority:
      single_pass: true
      fail_on_mapping_mismatch: true
      allow_content_resolution_remap: false

  # Alignment policy: challenge margin and confidence gate.
  alignment_config:
    enabled: true
    minimum_challenge_margin: 0.10
    minimum_challenge_confidence: Medium
    unresolved_authority: upstream_preferred

  # Theory content assignment policy.
  content:
    minimum_relevance_score: 0.20
    minimum_discrimination_score: 0.20
    maximum_assumptions_per_theory: 5
    maximum_risks_per_theory: 5
    maximum_opportunities_per_theory: 5
    maximum_recommendations_per_theory: 5
    maximum_evidence_per_theory: 12
    allow_symmetric_fallback: true
    minimum_content_coverage: 0.50
    partial_homogenization_threshold: 0.75
    full_homogenization_threshold: 0.99

  # Diagnostic severity policy.
  diagnostics_config:
    unknown_reference: error
    mapping_mismatch: error
    invalid_weight_sum: error
    low_mapping_confidence: warning
    partial_homogenization: warning
```

### Coexistence with existing strategy blocks

Engagement YAMLs that already have `evaluation:`, `dimensions:`, `objectives:`, and `constraints:` blocks (old format used by `ConfigurationResolver`) should add the new PH12.2a blocks as separate keys. The old and new blocks coexist:

- Old `evaluation:` block drives `ConfigurationResolver` (used for active scoring).
- New `evaluation_config:` block is captured in `StrategyTrace` (not yet active for scoring).

---

## Key Fields

### Root fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `enabled` | bool | `true` | Disable to skip strategy layer entirely |
| `config_version` | str | `ph12.2a-v1` | Recorded in trace for auditability |

### evaluation_config

| Field | Type | Default | Range / Values |
|---|---|---|---|
| `weight_policy` | str | `strict` | `strict` or `normalize` |
| `criteria` | dict | `{}` | Keys: criterion name; values: `{weight, enabled}` |
| `minimum_winner_margin` | float | `0.05` | `[0.0, 1.0]` |
| `saturation_threshold` | float | `0.95` | `[0.0, 1.0]` |
| `score_precision` | int | `6` | positive integer |

When `weight_policy=strict`, all enabled criteria weights must sum to `1.0 ± 0.001`. When `weight_policy=normalize`, weights are scaled at runtime.

### mapping_config

| Field | Type | Default | Notes |
|---|---|---|---|
| `enabled` | bool | `true` | |
| `unresolved_policy` | str | `preserve_upstream` | `preserve_upstream`, `report_unresolved`, `fail` |
| `authority.single_pass` | bool | `true` | Exactly one mapping pass; no remapping |
| `authority.fail_on_mapping_mismatch` | bool | `true` | Raise on selection/content mismatch |
| `authority.allow_content_resolution_remap` | bool | `false` | Block post-selection remapping |

### alignment_config

| Field | Type | Default | Values |
|---|---|---|---|
| `enabled` | bool | `true` | |
| `minimum_challenge_margin` | float | `0.10` | `[0.0, 1.0]` |
| `minimum_challenge_confidence` | str | `Medium` | `Low`, `Medium`, `High` |
| `unresolved_authority` | str | `upstream_preferred` | `upstream_preferred`, `strategy_preferred`, `abstain` |

### content

| Field | Type | Default | Notes |
|---|---|---|---|
| `minimum_relevance_score` | float | `0.20` | `[0.0, 1.0]` |
| `minimum_discrimination_score` | float | `0.20` | `[0.0, 1.0]` |
| `maximum_*_per_theory` | int | varies | Max items per content type per theory |
| `allow_symmetric_fallback` | bool | `true` | Allow shared-pool fallback |
| `minimum_content_coverage` | float | `0.50` | `[0.0, 1.0]` |
| `partial_homogenization_threshold` | float | `0.75` | `[0.0, 1.0]`, must be <= `full_homogenization_threshold` |
| `full_homogenization_threshold` | float | `0.99` | `[0.0, 1.0]` |

### diagnostics_config

| Field | Default | Locked? |
|---|---|---|
| `mapping_mismatch` | `error` | Yes — cannot be lowered |
| `invalid_weight_sum` | `error` | Yes — cannot be lowered |
| `unknown_reference` | `error` | No |
| `low_mapping_confidence` | `warning` | No |
| `partial_homogenization` | `warning` | No |

Valid severity values: `ignore`, `info`, `warning`, `error`.

---

## Safe vs Non-Configurable Behavior

**Safe to configure** (engagement-YAML-level):
- Criterion weights and enabled flags
- Minimum winner margin and saturation threshold
- Content volume limits (max items per theory)
- Content coverage and homogenization thresholds
- Alignment margin and confidence gate
- Non-locked diagnostic severity levels

**Not user-configurable** (enforced by Pydantic validators):
- `mapping_mismatch` severity — locked to `error`
- `invalid_weight_sum` severity — locked to `error`
- `authority.single_pass` semantics (enforced structurally, not just by default)

**Not yet active** (captured in trace, not driving scoring):
- `evaluation_config` criteria weights — current scoring uses `ConfigurationResolver` with the old `evaluation.criteria` block
- `mapping_config` posture weights and contradiction penalties
- `alignment_config` status strings

---

## CLI Commands

### Validate an engagement's strategy configuration

```bash
python3 -m functional_agents.cli strategy config validate \
  --engagement engagements/us_data_center_siting_strategy1.yaml
```

Exits 0 when validation passes. Exits 1 on `ValidationError` with per-field error details.

**Note:** Engagement YAMLs that use list-form `objectives:`, `dimensions:`, and `constraints:` (old format) will show validation errors for those fields. These are pre-existing structural mismatches — the old list format is handled by `ConfigurationResolver`, not `StrategyConfig.from_yaml_dict`. Only the new PH12.2a policy blocks (`evaluation_config`, `mapping_config`, `alignment_config`, `content`, `diagnostics_config`) are fully validated.

### Show the resolved configuration

```bash
# Human-readable summary
python3 -m functional_agents.cli strategy config show \
  --engagement engagements/us_data_center_siting_strategy1.yaml

# Full JSON dump of resolved config
python3 -m functional_agents.cli strategy config show \
  --engagement engagements/us_data_center_siting_strategy1.yaml \
  --format json
```

The text summary reports: version, fingerprint, source, enabled status, evaluation criteria, mapping policy, alignment policy, content policy, homogenization thresholds, and any defaults applied or deprecation warnings.

---

## Legacy Compatibility

Three backward-compatibility migration paths are applied before validation:

| Old field | New field | Notes |
|---|---|---|
| `alignment_policy.minimum_challenge_margin` | `alignment_config.minimum_challenge_margin` | |
| `alignment_policy.minimum_mapping_confidence` | `alignment_config.minimum_challenge_confidence` | |
| `scoring_policy.constraint_violation_penalty` | `constraint_config.default_required_penalty` | |
| `scoring_policy.partial_constraint_penalty` | `constraint_config.default_preferred_penalty` | |
| `evaluation.weights` dict | `evaluation_config.criteria` | Converts raw floats to `StrategyCriterionConfig` entries |

Migrations fire only when the target key is absent (they do not override explicit values). Each migration is reported in `ResolvedStrategyConfig.deprecations`.

---

## Fingerprinting

Every resolved configuration produces a deterministic 16-character hex fingerprint:

```
SHA-256(canonical_json(resolved_config))[:16]
```

Where `canonical_json` uses `sort_keys=True` and compact separators. The fingerprint is:

- Stored in `StrategyTrace.strategy_config_fingerprint`
- Included in `strategy config show` text output
- Stable across Python versions for the same logical configuration
- Differs when any field value changes, including those taken from defaults

The fingerprint is suitable for:
- Confirming two runs used identical configuration
- Cache invalidation keyed on configuration state
- Audit logging

---

## Schema Catalogue

The full field catalogue (37 entries) is exported to:

```
functional_agents/strategy/strategy_config_schema.json
```

Generate or refresh it:

```python
from functional_agents.strategy.strategy_config_resolver import export_config_schema
import json
schema = export_config_schema()
with open("functional_agents/strategy/strategy_config_schema.json", "w") as f:
    json.dump(schema, f, indent=2)
```

Each entry in `schema["fields"]` contains: `path`, `title`, `description`, `type`, `default`, optional `minimum`/`maximum`/`allowed_values`, `required`, `category`, and `unsafe_to_hide`.
