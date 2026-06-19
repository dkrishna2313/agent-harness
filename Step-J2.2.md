# J2.2 – Automated Evaluation Runner

J2.1 created a gold evaluation dataset.

The harness now needs an automated evaluation system that can execute all benchmark questions and produce quantitative scores.

Goal:

```text
Change
↓
Run evaluation suite
↓
Compare scores
↓
Determine whether the system improved
```

This is the foundation for future regression testing.

---

# Objective

Create an automated evaluation runner.

Input:

```text
eval/
```

Output:

```text
evaluation_report.json
evaluation_report.md
```

The evaluation runner should execute all benchmark questions and score the responses.

---

# J2.2.1 Evaluation Runner

Create:

```text
research_agent/evaluation/
```

and:

```python
EvaluationRunner
```

The runner should:

1. Load all YAML files under:

```text
eval/nvidia/
eval/smr/
```

2. Execute each question through the harness.

3. Capture:

```text
answer
evidence
citations
contradictions
trace
```

4. Score the response.

---

# J2.2.2 Scoring

For each evaluation question:

Score:

```yaml
must_include:
```

Example:

```yaml
must_include:
  - 120 kW
  - liquid cooling
  - rack-scale system
```

Score:

```text
fact_coverage_score
```

Example:

```text
3/3 = 100%
2/3 = 67%
```

---

# J2.2.3 Must-Not-Include

Evaluate:

```yaml
must_not_include:
```

Example:

```yaml
must_not_include:
  - 30 kW
  - air cooled only
```

Score:

```text
hallucination_penalty
```

Any prohibited fact should reduce score.

---

# J2.2.4 Citation Coverage

Measure:

```text
response contains evidence citations
```

Score:

```text
citation_score
```

---

# J2.2.5 Domain Aggregation

Aggregate scores by:

```text
NVIDIA
SMR
```

Example:

```json
{
  "nvidia": {
    "questions": 12,
    "coverage_score": 0.91,
    "citation_score": 0.95
  },
  "smr": {
    "questions": 12,
    "coverage_score": 0.88,
    "citation_score": 0.92
  }
}
```

---

# J2.2.6 Contradiction Regression

Load:

```text
eval/contradictions/
```

Execute contradiction tests.

Expected:

```yaml
expected_result: contradiction
```

or:

```yaml
expected_result: no_contradiction
```

Score:

```text
contradiction_accuracy
```

---

# J2.2.7 Reports

Generate:

```text
outputs/evaluation_report.json
outputs/evaluation_report.md
```

Include:

```text
overall score
domain scores
contradiction score
failed tests
coverage metrics
```

---

# J2.2.8 CLI

Create:

```bash
python3 -m research_agent.eval_runner
```

Options:

```bash
--eval-dir ./eval
--profile smr
--profile ai_data_centers
--web-search
```

Example:

```bash
python3 -m research_agent.eval_runner \
  --eval-dir ./eval \
  --web-search
```

---

# Acceptance Criteria

Running:

```bash
python3 -m research_agent.eval_runner \
  --eval-dir ./eval \
  --web-search
```

produces:

```text
evaluation_report.json
evaluation_report.md
```

with:

```text
overall score
NVIDIA score
SMR score
contradiction score
```

and identifies any failed benchmark questions.

---

# Non-Goals

Do not:

- modify retrieval
- modify contradiction logic
- modify evidence extraction
- redesign the harness

Only build the evaluation framework.

---

# Deliverables

Provide:

1. Files created
2. CLI usage
3. Sample evaluation_report.json
4. Sample evaluation_report.md
5. Example failed benchmark output
6. Any future enhancements for J2.3