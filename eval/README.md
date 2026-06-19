# Evaluation Dataset — Research Harness

Gold-standard benchmark questions and contradiction test cases for the
AI-infrastructure / SMR research harness.  Every dataset entry is manually
curated to serve as a regression anchor: if a harness change causes a
previously-correct answer to regress, the benchmark catches it.

---

## Directory Structure

```
eval/
├── nvidia/          # AI infrastructure questions (NVIDIA GB200 / data centre)
├── smr/             # Small modular reactor questions
├── contradictions/  # Contradiction detection test cases
└── README.md        # This file
```

---

## Dataset Summary

| Domain       | Files | Difficulty breakdown |
|---|---|---|
| NVIDIA       | 12    | 3 easy · 6 medium · 3 hard |
| SMR          | 11    | 3 easy · 6 medium · 2 hard |
| Contradiction | 11   | 2 true-contradiction · 7 no-contradiction · 2 known-limitation |

Total: **34 entries** (23 Q&A questions + 11 contradiction test cases).

---

## Question File Format

Each question is a YAML file with a stable `question_id`.

```yaml
question_id: NVIDIA_001
domain: nvidia                         # "nvidia" | "smr"
difficulty: easy                       # "easy" | "medium" | "hard"

question: >
  What is the rack-level DC power requirement of the NVIDIA GB200 NVL72?

must_include:                          # answer must contain at least one of these
  - 120 kW
  - rack

acceptable_alternatives:              # correct but less-precise answers
  - "132 kW"

must_not_include:                      # any match here fails the answer
  - air cooled only

expected_topics:                       # expected contradiction/evidence categories
  - power
  - rack architecture

evaluation_tags:                       # machine-readable scoring hints
  - numeric_answer
  - factual

notes: >
  Human-curated rationale: what a correct answer looks like, what context
  to expect, and what failure modes to watch for.
```

### Difficulty Levels

| Level  | Description |
|---|---|
| easy   | Single-fact retrieval; answer is directly stated in sources |
| medium | Requires synthesis across ≥2 claims or comparison between systems |
| hard   | Requires multi-factor reasoning; no single source contains the full answer |

---

## Contradiction File Format

```yaml
contradiction_id: CONTRA_001
domain: smr
expected_result: contradiction         # "contradiction" | "no_contradiction"
severity: high                         # "high" | "medium" | "low" | null
category: duration_conflict            # see categories below

claim_a: >
  Reactor Alpha construction duration is 24 to 36 months.

claim_b: >
  Reactor Alpha construction schedule is estimated at 7 to 12 years.

entity: Reactor Alpha                  # shared entity, or entity_a/entity_b for different entities
scope_a: unit
scope_b: unit
metric_a: construction_duration_months
metric_b: construction_duration_months

why_contradiction: >                   # or why_no_contradiction
  Both claims describe the same entity, same scope, same metric.
  Ranges [24, 36] and [84, 144] months do not overlap.

suppression_should_not_fire: true      # or suppression_should_fire: true
expected_suppression_reason: null      # "scope_mismatch" | "milestone_progression" | "entity_mismatch"
```

### Contradiction Categories

| Category | Description |
|---|---|
| `duration_conflict` | Two numeric duration claims for the same entity do not overlap |
| `numeric_conflict` | Two numeric values for the same metric differ > 20% |
| `scope_mismatch` | Claims measure different physical scales (rack vs component, unit vs fleet) |
| `milestone_progression` | Year values represent sequential lifecycle milestones, not conflicting dates |
| `entity_mismatch` | Claims name different specific entities (NVL72 vs NVL36) |
| `rate_vs_target` | Numeric conflict between a rate (per year) and a cumulative target |
| `context_mismatch` | Same metric but different contexts (NOAK vs FOAK); known engine limitation |
| `geographic_scope_mismatch` | Claims about different geopolitical jurisdictions |

---

## Evaluation Philosophy

### Precision over recall

A missed contradiction is less harmful than a false positive.  False positives
erode trust in the harness output; a researcher who sees 15 spurious
"contradictions" will start ignoring the section entirely.

The benchmark therefore contains more `no_contradiction` test cases than
`contradiction` test cases — the harder problem is preventing false flags.

### Same entity + same scope + same metric + material divergence

A contradiction is only warranted when ALL four conditions hold:

1. **Same entity**: claims describe the same named thing.
2. **Same scope**: claims measure at the same physical scale.
3. **Same metric**: claims describe the same quantity.
4. **Material divergence**: values are irreconcilably different (> 20% for numeric;
   mutually exclusive for categorical).

Each CONTRA_NNN file documents which conditions do or do not hold.

### Known limitations

Some cases (CONTRA_007) document situations where the engine currently produces
a false positive — typically because context extraction (NOAK vs FOAK; Russia vs
OECD) is beyond the current scope of the pattern-based classifier.  These are
included to prevent regression and to guide future work.

---

## Scoring Concepts (for future automated evaluation)

The following scoring dimensions are anticipated.  No automated scoring is
implemented yet.

### Q&A scoring

| Dimension | Description |
|---|---|
| `must_include_hit_rate` | Fraction of `must_include` terms present in the answer |
| `must_not_include_violation` | 1 if any `must_not_include` term is present |
| `topic_coverage` | Fraction of `expected_topics` addressed by extracted evidence |
| `evidence_count` | Number of evidence items supporting the answer |
| `contradiction_false_positive_rate` | Contradictions flagged / expected contradictions |

### Contradiction scoring

| Dimension | Description |
|---|---|
| `true_positive_rate` | Fraction of `expected_result: contradiction` cases detected |
| `false_positive_rate` | Fraction of `expected_result: no_contradiction` cases incorrectly flagged |
| `suppression_precision` | Fraction of suppressions with correct `expected_suppression_reason` |

The target operating point is:
- True positive rate ≥ 80% (don't miss real contradictions)
- False positive rate ≤ 10% (don't flag valid claims)

---

## Future Expansion

When adding new benchmark entries, follow these guidelines:

1. **Use a stable sequential ID** (`NVIDIA_013`, `SMR_012`, `CONTRA_012`).
2. **Write the `notes` field first** — if you cannot articulate why the expected
   answer is correct, the question is not ready.
3. **Include at least one `must_not_include` term** — this documents failure modes
   explicitly.
4. **For contradiction cases, always document `why_no_contradiction`** even for
   `expected_result: contradiction` entries (explain what the engine must see).
5. **Mark known limitations honestly** — use `known_limitation:` to flag cases
   where the engine is expected to fail until a future improvement.

### Planned future domains

- **Wind power / grid storage**: capacity factor, storage duration, LCOE
- **Carbon capture**: capture rate, cost per tonne, geological storage capacity
- **Hydrogen production**: electrolyser efficiency, green vs blue hydrogen costs
