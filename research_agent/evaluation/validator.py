"""Benchmark definition validation (J2.2a.1, J2.2a.5).

Validates Q&A questions and contradiction cases for internal consistency
before any harness evaluation runs.  Invalid definitions are benchmark
errors — they should never be counted as harness failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .loader import QAQuestion, ContradictionCase


@dataclass(frozen=True)
class ValidationError:
    """One validation problem found in a benchmark definition."""

    item_id: str          # question_id or contradiction_id
    source_file: str
    code: str             # machine-readable error code
    message: str          # human-readable description
    benchmark_error: bool = True   # always True; distinguishes from harness errors


@dataclass
class ValidationReport:
    """Aggregate result from validating all benchmark files."""

    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate_benchmark(
    qa_questions: list[QAQuestion],
    contradiction_cases: list[ContradictionCase],
) -> ValidationReport:
    """Validate all benchmark items and return a ``ValidationReport``."""

    report = ValidationReport()

    # Duplicate ID checks
    _check_duplicate_qa_ids(qa_questions, report)
    _check_duplicate_contra_ids(contradiction_cases, report)

    # Per-item validation
    for q in qa_questions:
        _validate_qa_question(q, report)

    for c in contradiction_cases:
        _validate_contradiction_case(c, report)

    return report


# ---------------------------------------------------------------------------
# Q&A validation
# ---------------------------------------------------------------------------

_REQUIRED_QA_FIELDS = ("question_id", "domain", "question")


def _validate_qa_question(q: QAQuestion, report: ValidationReport) -> None:
    qid = q.question_id

    # Required fields
    for field_name in _REQUIRED_QA_FIELDS:
        if not getattr(q, field_name, "").strip():
            report.errors.append(ValidationError(
                item_id=qid,
                source_file=q.source_file,
                code="missing_required_field",
                message=f"Field '{field_name}' is empty or missing.",
            ))

    # Domain must be known
    if q.domain not in ("nvidia", "smr", "unknown"):
        report.warnings.append(ValidationError(
            item_id=qid,
            source_file=q.source_file,
            code="unknown_domain",
            message=f"Domain {q.domain!r} is not one of the known domains (nvidia, smr).",
        ))

    # must_include / must_not_include conflict detection (J2.2a.1)
    _check_include_conflicts(q, report)

    # must_include should not be empty (warn)
    if not q.must_include and not q.acceptable_alternatives:
        report.warnings.append(ValidationError(
            item_id=qid,
            source_file=q.source_file,
            code="no_must_include",
            message="Neither must_include nor acceptable_alternatives defined; question cannot be scored.",
        ))

    # Overly short prohibited terms (likely to false-positive)
    for term in q.must_not_include:
        if len(term.strip()) <= 2:  # noqa: PLR2004
            report.warnings.append(ValidationError(
                item_id=qid,
                source_file=q.source_file,
                code="short_prohibited_term",
                message=(
                    f"must_not_include term {term!r} is very short (<= 2 chars); "
                    "it may appear in valid answers as a substring."
                ),
            ))


def _check_include_conflicts(q: QAQuestion, report: ValidationReport) -> None:
    """Flag must_not_include terms that are substrings of must_include terms (J2.2a.1).

    A genuine conflict exists only when the PROHIBITED term is a substring of the
    REQUIRED term — because then any answer containing the required term will
    *also* contain the prohibited term, making both rules unsatisfiable at once.

    The reverse direction (required term appears inside a longer prohibited phrase)
    is NOT a conflict: the prohibited term would only fire if the full phrase is
    present, which is independent of whether the required term appears elsewhere.
    """

    for prohibited in q.must_not_include:
        p_lower = prohibited.lower().strip()
        if not p_lower:
            continue
        for required in q.must_include:
            r_lower = required.lower().strip()
            # Genuine conflict: prohibited is a substring of required
            # (every hit of 'required' would also trigger 'prohibited')
            if p_lower in r_lower and p_lower != r_lower:
                report.errors.append(ValidationError(
                    item_id=q.question_id,
                    source_file=q.source_file,
                    code="must_include_conflict",
                    message=(
                        f"must_not_include term {prohibited!r} is a substring of "
                        f"must_include term {required!r}. Any answer satisfying the "
                        f"must_include requirement will also trigger the prohibition."
                    ),
                ))
            # Exact match: same term required and prohibited
            elif p_lower == r_lower:
                report.errors.append(ValidationError(
                    item_id=q.question_id,
                    source_file=q.source_file,
                    code="must_include_conflict",
                    message=(
                        f"Term {prohibited!r} appears in both must_include and "
                        f"must_not_include — unsatisfiable."
                    ),
                ))

    # Also check alternatives (warning only)
    for prohibited in q.must_not_include:
        p_lower = prohibited.lower().strip()
        if not p_lower:
            continue
        for alt in q.acceptable_alternatives:
            a_lower = alt.lower().strip()
            if p_lower in a_lower and p_lower != a_lower:
                report.warnings.append(ValidationError(
                    item_id=q.question_id,
                    source_file=q.source_file,
                    code="alternative_conflict",
                    message=(
                        f"must_not_include term {prohibited!r} is a substring of "
                        f"acceptable_alternative {alt!r}."
                    ),
                ))


def _check_duplicate_qa_ids(questions: list[QAQuestion], report: ValidationReport) -> None:
    seen: dict[str, str] = {}
    for q in questions:
        if q.question_id in seen:
            report.errors.append(ValidationError(
                item_id=q.question_id,
                source_file=q.source_file,
                code="duplicate_question_id",
                message=(
                    f"Duplicate question_id {q.question_id!r}. "
                    f"First seen in {seen[q.question_id]}."
                ),
            ))
        else:
            seen[q.question_id] = q.source_file


# ---------------------------------------------------------------------------
# Contradiction validation
# ---------------------------------------------------------------------------

def _validate_contradiction_case(c: ContradictionCase, report: ValidationReport) -> None:
    cid = c.contradiction_id

    # Required fields
    for field_name in ("contradiction_id", "expected_result", "claim_a", "claim_b"):
        if not getattr(c, field_name, "").strip():
            report.errors.append(ValidationError(
                item_id=cid,
                source_file=c.source_file,
                code="missing_required_field",
                message=f"Field '{field_name}' is empty or missing.",
            ))

    # expected_result must be valid
    if c.expected_result not in ("contradiction", "no_contradiction"):
        report.errors.append(ValidationError(
            item_id=cid,
            source_file=c.source_file,
            code="invalid_expected_result",
            message=(
                f"expected_result must be 'contradiction' or 'no_contradiction', "
                f"got {c.expected_result!r}."
            ),
        ))

    # Contradictory suppression flags
    if c.suppression_should_fire and c.suppression_should_not_fire:
        report.errors.append(ValidationError(
            item_id=cid,
            source_file=c.source_file,
            code="conflicting_suppression_flags",
            message="Both suppression_should_fire and suppression_should_not_fire are true.",
        ))

    # suppression_should_fire=True but no expected reason
    if c.suppression_should_fire and not c.expected_suppression_reason:
        report.warnings.append(ValidationError(
            item_id=cid,
            source_file=c.source_file,
            code="missing_suppression_reason",
            message="suppression_should_fire=true but expected_suppression_reason is not set.",
        ))

    # contradiction expected but suppression should fire (logically inconsistent)
    if c.expected_result == "contradiction" and c.suppression_should_fire:
        report.errors.append(ValidationError(
            item_id=cid,
            source_file=c.source_file,
            code="contradiction_with_suppression",
            message=(
                "expected_result=contradiction but suppression_should_fire=true. "
                "A suppressed comparison cannot produce a contradiction."
            ),
        ))

    # Warn about known_limitation cases (they will always appear as failures)
    if c.known_limitation:
        report.warnings.append(ValidationError(
            item_id=cid,
            source_file=c.source_file,
            code="known_limitation",
            message=(
                "This case is marked known_limitation=true. "
                "It will be excluded from pass/fail counts and reported separately."
            ),
        ))


def _check_duplicate_contra_ids(cases: list[ContradictionCase], report: ValidationReport) -> None:
    seen: dict[str, str] = {}
    for c in cases:
        if c.contradiction_id in seen:
            report.errors.append(ValidationError(
                item_id=c.contradiction_id,
                source_file=c.source_file,
                code="duplicate_contradiction_id",
                message=(
                    f"Duplicate contradiction_id {c.contradiction_id!r}. "
                    f"First seen in {seen[c.contradiction_id]}."
                ),
            ))
        else:
            seen[c.contradiction_id] = c.source_file
