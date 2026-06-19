"""Scoring functions for Q&A and contradiction benchmark results."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..schemas import ResearchMemo, SuppressedComparison
from ..schemas import Contradiction
from .loader import QAQuestion, ContradictionCase
from .semantic_matcher import SemanticMatch, compute_match_stats, score_term_coverage
from .prohibited_term_checker import (
    ProhibitedTermResult,
    build_prohibition_stats,
    check_all_prohibited_terms,
)

_CITATION_RE = re.compile(r"\[Source:\s*[^,\]]+,\s*Evidence:\s*E\d{3}\]")


@dataclass
class QAScore:
    """Scores for one Q&A benchmark question."""

    question_id: str
    domain: str
    difficulty: str
    question: str

    # Fact coverage: fraction of must_include terms found in the answer
    must_include_hits: int = 0
    must_include_total: int = 0

    # Hallucination: 1 if any must_not_include term is present
    must_not_include_violations: list[str] = field(default_factory=list)

    # Citation coverage: does the memo contain evidence citations?
    citation_count: int = 0

    # Evidence breadth
    evidence_count: int = 0

    # Derived scores (set after construction)
    fact_coverage_score: float = 0.0     # 0-1
    hallucination_penalty: float = 0.0   # 0 or 1 (1 = bad)
    citation_score: float = 0.0          # 0-1 (1 = has citations)
    overall_score: float = 0.0           # 0-1 composite

    passed: bool = False
    fail_reasons: list[str] = field(default_factory=list)

    # J2.2a.2 — failure diagnostics
    missing_facts: list[str] = field(default_factory=list)       # must_include terms not found (semantic)
    unexpected_facts: list[str] = field(default_factory=list)    # must_not_include terms found
    actual_answer: str = ""                                       # truncated answer text for inspection
    benchmark_error: bool = False                                 # True = definition problem, not harness
    benchmark_error_reason: str = ""

    # J3.1a — semantic match details
    semantic_matches: list[dict] = field(default_factory=list)   # per-term SemanticMatch.to_dict()
    exact_matches_found: int = 0
    semantic_matches_found: int = 0

    # J3.1c — context-aware prohibited term audit
    prohibited_term_audit: list[dict] = field(default_factory=list)  # per-term ProhibitedTermResult.to_dict()
    context_allowed_count: int = 0   # terms present but exempted by context

    # J3.2 — retrieval diversity (from memo.metadata)
    retrieval_diversity: dict = field(default_factory=dict)


@dataclass
class ContradictionScore:
    """Score for one contradiction benchmark case."""

    contradiction_id: str
    domain: str
    expected_result: str          # "contradiction" | "no_contradiction"
    actual_result: str            # "contradiction" | "no_contradiction"
    correct: bool = False
    suppression_fired: bool = False
    expected_suppression_reason: str | None = None
    actual_suppression_reasons: list[str] = field(default_factory=list)
    suppression_correct: bool = True
    notes: str = ""
    known_limitation: bool = False

    # J2.2a.3 — contradiction failure diagnostics
    evidence_a: str = ""
    evidence_b: str = ""
    entity_a: str = ""
    entity_b: str = ""
    scope_a: str = ""
    scope_b: str = ""
    metric_a: str = ""
    metric_b: str = ""
    suppression_details: list[dict] = field(default_factory=list)


def score_qa_response(question: QAQuestion, memo: ResearchMemo) -> QAScore:
    """Score a harness response against the benchmark question criteria."""

    score = QAScore(
        question_id=question.question_id,
        domain=question.domain,
        difficulty=question.difficulty,
        question=question.question,
        must_include_total=len(question.must_include),
    )

    # Collect the answer text to search through
    answer_text = _collect_answer_text(memo)
    answer_lower = answer_text.lower()

    # J3.1a: must_include semantic scoring (synonym expansion + token overlap)
    # acceptable_alternatives are tried as extra synonyms for every term.
    # must_not_include stays exact (J3.1a.6).
    term_matches: list[SemanticMatch] = score_term_coverage(
        question.must_include,
        answer_text,
        alternatives=question.acceptable_alternatives,
        threshold=0.80,
    )
    score.must_include_hits = sum(1 for m in term_matches if m.matched)
    _stats = compute_match_stats(term_matches)
    score.semantic_matches = [m.to_dict() for m in term_matches]
    score.exact_matches_found = _stats["exact_matches_found"]
    score.semantic_matches_found = _stats["semantic_matches_found"]

    score.fact_coverage_score = (
        score.must_include_hits / score.must_include_total
        if score.must_include_total > 0
        else 1.0   # no requirements = full credit
    )

    # J3.1c: must_not_include context-aware scoring
    # Terms appearing in a negating or contrastive sentence are CONTEXT_ALLOWED
    # (no penalty).  Only HARD_PROHIBITED terms (no exempting context) are penalized.
    prohibited_results = check_all_prohibited_terms(
        question.must_not_include, answer_text
    )
    score.prohibited_term_audit = [r.to_dict() for r in prohibited_results]
    for result in prohibited_results:
        if result.penalty_applied:
            score.must_not_include_violations.append(result.term)
    pstats = build_prohibition_stats(prohibited_results)
    score.context_allowed_count = pstats["context_allowed"]
    score.hallucination_penalty = 1.0 if score.must_not_include_violations else 0.0

    # J2.2.4: citation scoring
    score.citation_count = len(_CITATION_RE.findall(answer_text))
    score.citation_score = 1.0 if score.citation_count > 0 else 0.0

    # Evidence breadth
    score.evidence_count = len(memo.evidence or memo.source_notes)

    # Composite score: coverage × (1 − penalty) × citation weight
    # Citation bonus: +10% when present, but never inflates a zero-coverage score
    citation_weight = 0.10
    base_weight = 1.0 - citation_weight
    score.overall_score = round(
        score.fact_coverage_score * (1.0 - score.hallucination_penalty) * base_weight
        + score.citation_score * citation_weight,
        4,
    )

    # Pass/fail
    score.fail_reasons = _collect_fail_reasons(score, question)
    score.passed = len(score.fail_reasons) == 0

    # J2.2a.2 / J3.1a — failure diagnostics using semantic match results
    score.missing_facts = [m.expected for m in term_matches if not m.matched]
    score.unexpected_facts = list(score.must_not_include_violations)
    score.actual_answer = answer_text[:500]  # first 500 chars for inspection

    # J3.2 — retrieval diversity from memo metadata
    score.retrieval_diversity = memo.metadata.get("retrieval_diversity", {})

    return score


def score_contradiction_result(
    case: ContradictionCase,
    contradictions: list[Contradiction],
    suppressed: list[SuppressedComparison],
    enriched_items: list | None = None,
) -> ContradictionScore:
    """Score a contradiction detection result against a benchmark case."""

    actual = "contradiction" if contradictions else "no_contradiction"
    correct = actual == case.expected_result

    suppression_fired = len(suppressed) > 0
    actual_reasons = sorted({s.reason for s in suppressed})

    # Suppression correctness
    suppression_correct = True
    if case.suppression_should_fire and not suppression_fired:
        suppression_correct = False
    if case.suppression_should_not_fire and suppression_fired:
        suppression_correct = False
    if (
        case.expected_suppression_reason
        and suppression_fired
        and case.expected_suppression_reason not in actual_reasons
    ):
        suppression_correct = False

    notes_parts = []
    if not correct:
        notes_parts.append(
            f"Expected {case.expected_result}, got {actual}. "
            f"contradictions={len(contradictions)}, suppressed={len(suppressed)}"
        )
    if not suppression_correct:
        notes_parts.append(
            f"Suppression mismatch: "
            f"expected_reason={case.expected_suppression_reason!r}, "
            f"actual_reasons={actual_reasons!r}, "
            f"should_fire={case.suppression_should_fire}, "
            f"fired={suppression_fired}"
        )

    # J2.2a.3 — enrich diagnostics from enriched evidence items
    item_a = enriched_items[0] if enriched_items else None
    item_b = enriched_items[1] if enriched_items and len(enriched_items) > 1 else None

    # Prefer enriched entity/scope; fall back to case definition
    ea = item_a.entity if item_a else (case.entity_a or case.entity)
    eb = item_b.entity if item_b else (case.entity_b or case.entity)
    sa = item_a.scope if item_a else case.scope_a
    sb = item_b.scope if item_b else case.scope_b

    # Extract metric types from detected contradictions or suppressed comparisons
    ma, mb = "", ""
    if contradictions:
        ma = contradictions[0].metric_type_a
        mb = contradictions[0].metric_type_b
    elif suppressed:
        ma = suppressed[0].metric_a
        mb = suppressed[0].metric_b

    sup_details = [
        {
            "reason": s.reason,
            "scope_a": s.scope_a,
            "scope_b": s.scope_b,
            "metric_a": s.metric_a,
            "metric_b": s.metric_b,
            "detail": s.detail,
        }
        for s in suppressed
    ]

    return ContradictionScore(
        contradiction_id=case.contradiction_id,
        domain=case.domain,
        expected_result=case.expected_result,
        actual_result=actual,
        correct=correct,
        suppression_fired=suppression_fired,
        expected_suppression_reason=case.expected_suppression_reason,
        actual_suppression_reasons=actual_reasons,
        suppression_correct=suppression_correct,
        notes="; ".join(notes_parts),
        known_limitation=case.known_limitation,
        evidence_a=case.claim_a.strip(),
        evidence_b=case.claim_b.strip(),
        entity_a=ea,
        entity_b=eb,
        scope_a=sa,
        scope_b=sb,
        metric_a=ma,
        metric_b=mb,
        suppression_details=sup_details,
    )


def _collect_answer_text(memo: ResearchMemo) -> str:
    """Flatten all user-visible answer sections into one searchable string."""

    parts = [
        memo.executive_summary,
        *memo.confirmed_facts,
        *memo.inferences,
        *memo.power_implications,
        *memo.cooling_implications,
        *memo.networking_implications,
        *memo.rack_architecture_implications,
        *memo.open_questions,
    ]
    return " ".join(p for p in parts if p)


def _collect_fail_reasons(score: QAScore, question: QAQuestion) -> list[str]:
    reasons: list[str] = []

    if score.must_include_total > 0 and score.fact_coverage_score < 0.5:
        hits = score.must_include_hits
        total = score.must_include_total
        reasons.append(
            f"Low fact coverage: {hits}/{total} must_include terms found "
            f"({score.fact_coverage_score:.0%})"
        )

    for term in score.must_not_include_violations:
        reasons.append(f"Prohibited term found: {term!r}")

    return reasons
