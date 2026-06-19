"""Inject synthetic regressions into an evaluation report for J2.3a validation.

Usage:
    python3 scripts/inject_regression.py

Reads:  outputs/evaluation_report_regression_test.json
Writes: outputs/evaluation_report_regression_test.json (in-place)

Modifications applied:
    summary.overall_score          -= 0.10
    summary.fact_coverage_score    -= 0.10
    summary.qa_questions_passed    -= 3
"""

import json
from pathlib import Path

TARGET = Path("outputs/evaluation_report_regression_test.json")

data = json.loads(TARGET.read_text(encoding="utf-8"))
s = data["summary"]

s["overall_score"]       = round(s["overall_score"]       - 0.10, 6)
s["fact_coverage_score"] = round(s["fact_coverage_score"] - 0.10, 6)
s["qa_questions_passed"] = s["qa_questions_passed"] - 3

TARGET.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"Injected regressions into {TARGET}")
print(f"  overall_score:       {s['overall_score'] + 0.10:.4f} → {s['overall_score']:.4f}")
print(f"  fact_coverage_score: {s['fact_coverage_score'] + 0.10:.4f} → {s['fact_coverage_score']:.4f}")
print(f"  qa_questions_passed: {s['qa_questions_passed'] + 3} → {s['qa_questions_passed']}")
