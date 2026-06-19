"""EvaluationRunner: execute benchmark questions and contradiction tests."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ..agent import DcPowerAgent
from ..contradiction import detect_contradictions, enrich_evidence_items
from ..loaders import load_sources
from ..profile import DomainProfile
from ..research_object import (
    create_research_object,
    infer_profile_from_domain,
    update_research_object,
    write_research_object,
)
from ..schemas import EvidenceItem, SuppressedComparison
from .loader import QAQuestion, ContradictionCase
from .scorer import QAScore, ContradictionScore, score_qa_response, score_contradiction_result
from .validator import validate_benchmark

LOGGER = logging.getLogger(__name__)


@dataclass
class EvaluationRun:
    """Results from one complete evaluation pass."""

    qa_scores: list[QAScore] = field(default_factory=list)
    contradiction_scores: list[ContradictionScore] = field(default_factory=list)

    # Validation errors (benchmark definition problems, not harness failures)
    validation_errors: list = field(default_factory=list)

    # Aggregate metrics (computed after all runs, excluding known_limitation items)
    overall_score: float = 0.0
    fact_coverage_score: float = 0.0
    hallucination_rate: float = 0.0
    citation_score: float = 0.0
    contradiction_accuracy: float = 0.0

    # Domain breakdown
    domain_scores: dict[str, dict] = field(default_factory=dict)

    # Failed tests
    failed_qa: list[QAScore] = field(default_factory=list)
    failed_contradictions: list[ContradictionScore] = field(default_factory=list)


class EvaluationRunner:
    """Run all benchmark questions through the harness and produce scored results.

    Parameters
    ----------
    agent:
        A configured ``DcPowerAgent`` used for Q&A questions.
        When *None*, a mock agent (no LLM calls) is used.
    sources_dir:
        Directory of source documents passed to the agent.
    profile:
        Optional domain profile; forwarded to the agent and contradiction engine.
    """

    def __init__(
        self,
        agent: DcPowerAgent | None = None,
        *,
        sources_dir: str | Path = "sources",
        profile: DomainProfile | None = None,
        ro_out_dir: str | Path | None = None,
    ) -> None:
        self._agent = agent or DcPowerAgent(profile=profile)
        self._sources_dir = Path(sources_dir)
        self._profile = profile
        self._ro_out_dir: Path | None = Path(ro_out_dir) if ro_out_dir else None

    def run(
        self,
        qa_questions: Sequence[QAQuestion],
        contradiction_cases: Sequence[ContradictionCase],
    ) -> EvaluationRun:
        """Execute all benchmark items and return an ``EvaluationRun``."""

        result = EvaluationRun()

        # Validate benchmark definitions first
        validation = validate_benchmark(list(qa_questions), list(contradiction_cases))
        result.validation_errors = validation.errors + validation.warnings
        if validation.errors:
            LOGGER.warning(
                "Benchmark has %d definition error(s) — these will not count as harness failures.",
                validation.error_count,
            )

        # Load source documents once
        collection = load_sources(self._sources_dir)
        documents = collection.documents
        if collection.errors:
            for err in collection.errors:
                LOGGER.warning("Source load error: %s — %s", err.path.name, err.message)

        # Q&A questions
        total = len(qa_questions)
        for idx, question in enumerate(qa_questions, start=1):
            LOGGER.info(
                "[%d/%d] Running %s: %s",
                idx,
                total,
                question.question_id,
                question.question[:60],
            )
            try:
                memo = self._agent.analyze(question.question, documents)
                qa_score = score_qa_response(question, memo)
                # J4.5 – write per-question research object
                if self._ro_out_dir is not None:
                    _write_qa_research_object(
                        question=question,
                        memo=memo,
                        profile=self._profile,
                        out_dir=self._ro_out_dir,
                    )
            except Exception as exc:  # pragma: no cover
                LOGGER.error("Error running %s: %s", question.question_id, exc)
                qa_score = _failed_qa_score(question, str(exc))
            result.qa_scores.append(qa_score)

        # Contradiction cases
        total_c = len(contradiction_cases)
        for idx, case in enumerate(contradiction_cases, start=1):
            LOGGER.info(
                "[%d/%d] Contradiction test %s",
                idx,
                total_c,
                case.contradiction_id,
            )
            try:
                c_score = self._run_contradiction_case(case)
            except Exception as exc:  # pragma: no cover
                LOGGER.error("Error running %s: %s", case.contradiction_id, exc)
                c_score = ContradictionScore(
                    contradiction_id=case.contradiction_id,
                    domain=case.domain,
                    expected_result=case.expected_result,
                    actual_result="error",
                    correct=False,
                    notes=f"Runner error: {exc}",
                )
            result.contradiction_scores.append(c_score)

        _compute_aggregates(result)
        return result

    def _run_contradiction_case(self, case: ContradictionCase) -> ContradictionScore:
        """Build synthetic evidence items from the case's claims and run detection."""

        items = [
            EvidenceItem(
                evidence_id="CA",
                claim=case.claim_a,
                source_document="benchmark",
                evidence_snippet=case.claim_a[:200],
                category="other",
                relevance="benchmark",
                confidence="high",
                entity=case.entity_a or case.entity,
                scope=case.scope_a,
            ),
            EvidenceItem(
                evidence_id="CB",
                claim=case.claim_b,
                source_document="benchmark",
                evidence_snippet=case.claim_b[:200],
                category="other",
                relevance="benchmark",
                confidence="high",
                entity=case.entity_b or case.entity,
                scope=case.scope_b,
            ),
        ]

        enriched = enrich_evidence_items(items)
        suppressed: list[SuppressedComparison] = []
        contradictions = detect_contradictions(
            enriched,
            {},
            profile=self._profile,
            out_suppressed=suppressed,
        )

        return score_contradiction_result(case, contradictions, suppressed, enriched_items=enriched)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _failed_qa_score(question: QAQuestion, error: str) -> QAScore:
    from .scorer import QAScore
    s = QAScore(
        question_id=question.question_id,
        domain=question.domain,
        difficulty=question.difficulty,
        question=question.question,
        must_include_total=len(question.must_include),
    )
    s.fail_reasons = [f"Runner error: {error}"]
    s.passed = False
    return s


def _compute_aggregates(result: EvaluationRun) -> None:
    """Compute aggregate metrics and populate domain breakdowns in-place.

    known_limitation cases are excluded from accuracy counts so they don't
    artificially drag down the overall score.
    """

    # Q&A aggregates (benchmark_error items excluded from counts)
    qa = [s for s in result.qa_scores if not s.benchmark_error]
    if qa:
        result.fact_coverage_score = round(
            sum(s.fact_coverage_score for s in qa) / len(qa), 4
        )
        result.hallucination_rate = round(
            sum(1 for s in qa if s.hallucination_penalty > 0) / len(qa), 4
        )
        result.citation_score = round(
            sum(s.citation_score for s in qa) / len(qa), 4
        )

    # Contradiction accuracy — exclude known_limitation cases
    contra_scoreable = [s for s in result.contradiction_scores if not s.known_limitation]
    if contra_scoreable:
        result.contradiction_accuracy = round(
            sum(1 for s in contra_scoreable if s.correct) / len(contra_scoreable), 4
        )

    # Overall (equal weight: fact_coverage + citation + contradiction_accuracy)
    components = []
    if qa:
        components.append(result.fact_coverage_score * (1.0 - result.hallucination_rate))
        components.append(result.citation_score)
    if contra_scoreable:
        components.append(result.contradiction_accuracy)
    result.overall_score = round(sum(components) / len(components), 4) if components else 0.0

    # Domain breakdown
    all_domains = sorted({s.domain for s in qa} | {s.domain for s in contra_scoreable})
    for domain in all_domains:
        domain_qa = [s for s in qa if s.domain == domain]
        domain_c = [s for s in contra_scoreable if s.domain == domain]

        entry: dict = {"questions": len(domain_qa), "contradiction_tests": len(domain_c)}

        if domain_qa:
            entry["fact_coverage_score"] = round(
                sum(s.fact_coverage_score for s in domain_qa) / len(domain_qa), 4
            )
            entry["citation_score"] = round(
                sum(s.citation_score for s in domain_qa) / len(domain_qa), 4
            )
            entry["hallucination_rate"] = round(
                sum(1 for s in domain_qa if s.hallucination_penalty > 0) / len(domain_qa), 4
            )
            entry["pass_rate"] = round(
                sum(1 for s in domain_qa if s.passed) / len(domain_qa), 4
            )

        if domain_c:
            entry["contradiction_accuracy"] = round(
                sum(1 for s in domain_c if s.correct) / len(domain_c), 4
            )

        result.domain_scores[domain] = entry

    # Failed items (known_limitation excluded from failures)
    result.failed_qa = [s for s in qa if not s.passed]
    result.failed_contradictions = [
        s for s in contra_scoreable
        if not s.correct or not s.suppression_correct
    ]


def _write_qa_research_object(
    *,
    question: "QAQuestion",
    memo: "ResearchMemo",
    profile: "DomainProfile | None",
    out_dir: Path,
) -> None:
    """Create, update, and persist a research object for one Q&A benchmark question.

    The research object profile is always derived from the question's domain/ID
    (benchmark_domain_mapping), not from the agent's execution profile. The agent
    profile controls retrieval and analysis; the research object records WHICH domain
    this question belongs to. Using the agent profile here was the source of the
    ai_data_centers bleed-over into SMR research objects (J4.5e fix).
    """
    profile_name = infer_profile_from_domain(question.domain, question.question_id)
    mapping_rule = (
        f"question_id.startswith('SMR_')" if (question.question_id or "").startswith("SMR_")
        else f"question_id.startswith('NVIDIA_')" if (question.question_id or "").startswith("NVIDIA_")
        else f"domain='{question.domain}'"
    )
    LOGGER.debug(
        "[research_object] question_id=%s  resolved_profile=%s  mapping_rule=%s",
        question.question_id, profile_name, mapping_rule,
    )
    ro = create_research_object(
        question=question.question,
        profile_name=profile_name,
        profile_source="benchmark_domain_mapping",
        sources_dir=None,
        web_search=False,
        mock_mode=False,
        question_id=question.question_id,
    )
    ro = update_research_object(ro, memo=memo, question_id=question.question_id)
    try:
        write_research_object(ro, out_dir=out_dir)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Could not write research object for %s: %s", question.question_id, exc)
