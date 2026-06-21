"""Citation audit – diagnostic command to identify citation-score gaps (read-only)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Must match the scorer in scorer.py exactly
_CITATION_RE = re.compile(r"\[Source:\s*[^,\]]+,\s*Evidence:\s*E\d{3}\]")


def load_qa_results(report_path: Path, trace_path: Path | None) -> list[dict[str, Any]]:
    """Merge qa_results from report and (optionally) trace.

    The report has all scoring fields but no actual_answer.
    The trace has actual_answer.  We merge by question_id so the combined
    record contains everything we need.
    """
    report = json.loads(report_path.read_text())
    report_rows: dict[str, dict] = {
        r["question_id"]: dict(r)
        for r in report.get("qa_results", [])
    }

    if trace_path is not None and trace_path.exists():
        trace = json.loads(trace_path.read_text())
        for row in trace.get("qa_results", []):
            qid = row.get("question_id", "")
            if qid in report_rows:
                # Merge: keep report values for scoring fields, add actual_answer
                report_rows[qid].setdefault("actual_answer", row.get("actual_answer", ""))
            else:
                # Question only in trace — include it
                report_rows[qid] = dict(row)

    return list(report_rows.values())


def _detect_citation_patterns(text: str) -> list[str]:
    """Return all citation marker matches found in *text*."""
    return _CITATION_RE.findall(text)


def _citation_debug(row: dict[str, Any]) -> dict[str, Any]:
    """Build a citation_debug block for one question record."""
    actual = row.get("actual_answer", "")
    patterns = _detect_citation_patterns(actual)
    return {
        "evidence_count": row.get("evidence_count", 0),
        "answer_length": len(actual),
        "answer_is_truncated": len(actual) >= 500,  # scorer truncates at 500 chars
        "citation_patterns_detected": patterns,
        "citation_count_detected": len(patterns),
        "citation_count_reported": row.get("citation_count", 0),
        "discrepancy": row.get("citation_count", 0) != len(patterns),
    }


def run_citation_audit(
    report_path: Path,
    trace_path: Path | None,
    threshold: float,
) -> int:
    """Run citation audit and print results.  Returns exit code (0 = clean)."""
    rows = load_qa_results(report_path, trace_path)

    # Filter to questions below threshold, sort worst-first
    flagged = [r for r in rows if r.get("citation_score", 1.0) < threshold]
    flagged.sort(key=lambda r: r.get("citation_score", 1.0))

    if not flagged:
        print("No citation coverage issues found.")
        return 0

    # ------------------------------------------------------------------ table
    print("Citation Audit")
    print("==============")
    print()

    col_w = {"qid": 14, "domain": 8, "passed": 8, "score": 16, "count": 16, "overall": 14}
    header = (
        f"{'Question ID':<{col_w['qid']}} "
        f"{'Domain':<{col_w['domain']}} "
        f"{'Passed':<{col_w['passed']}} "
        f"{'Citation Score':<{col_w['score']}} "
        f"{'Citation Count':<{col_w['count']}} "
        f"{'Overall Score'}"
    )
    print(header)
    print()

    for r in flagged:
        passed_str = str(r.get("passed", "")).lower()
        print(
            f"{r.get('question_id', ''):<{col_w['qid']}} "
            f"{r.get('domain', ''):<{col_w['domain']}} "
            f"{passed_str:<{col_w['passed']}} "
            f"{r.get('citation_score', 0.0):<{col_w['score']}.2f} "
            f"{r.get('citation_count', 0):<{col_w['count']}} "
            f"{r.get('overall_score', 0.0):.2f}"
        )

    # --------------------------------------------------------------- detail
    separator = "-" * 50
    for r in flagged:
        debug = _citation_debug(r)
        actual = r.get("actual_answer", "")

        print()
        print(separator)
        print(f"Question: {r.get('question_id', '')}")
        print(f"Citation Score: {r.get('citation_score', 0.0):.2f}")
        print(f"Citation Count: {r.get('citation_count', 0)}")
        print(f"Overall Score: {r.get('overall_score', 0.0):.2f}")
        print(f"Passed: {str(r.get('passed', '')).lower()}")

        print()
        print("Citation Debug:")
        print(f"  Evidence items available : {debug['evidence_count']}")
        print(f"  Answer length (stored)   : {debug['answer_length']} chars"
              + (" [TRUNCATED — scorer uses full text]" if debug["answer_is_truncated"] else ""))
        print(f"  Citation regex applied to: stored actual_answer (500-char window)")
        patterns = debug["citation_patterns_detected"]
        if patterns:
            print(f"  Detected citation patterns ({len(patterns)}):")
            for p in patterns[:5]:
                print(f"    {p}")
            if len(patterns) > 5:
                print(f"    ... and {len(patterns) - 5} more")
        else:
            print("  Detected citation patterns: NONE")
            print("  Expected format: [Source: <filename>, Evidence: E001]")
        if debug["discrepancy"]:
            print(f"  *** DISCREPANCY: reported={debug['citation_count_reported']}  "
                  f"detected_in_stored_window={debug['citation_count_detected']}")
            print("  NOTE: scorer runs regex on full answer_text; stored actual_answer")
            print("        is truncated to 500 chars — regex results may differ.")

        if actual:
            print()
            print("Actual Answer (stored 500-char window):")
            print(actual)
        print(separator)

    return 0
