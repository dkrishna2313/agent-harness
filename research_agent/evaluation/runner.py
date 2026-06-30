"""EvaluationRunner: execute benchmark questions and contradiction tests."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

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
from .agent_scorer import AgentScores, score_agents, aggregate_agent_scores
from .validator import validate_benchmark

LOGGER = logging.getLogger(__name__)

DEFAULT_WORKERS = 1


@dataclass
class BenchmarkPerfRecord:
    """Per-question timing and token data collected during a benchmark run."""

    question_id: str
    total_ms: float
    analysis_ms: float
    retrieval_ms: float
    reranking_ms: float
    llm_ms: float
    scoring_ms: float
    report_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    llm_calls: int


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

    # J5.7 — per-question agent evaluation scores
    agent_scores: list[AgentScores] = field(default_factory=list)

    # J8.9a — per-question benchmark performance records
    perf_records: list[BenchmarkPerfRecord] = field(default_factory=list)

    # J5.7 / J6.6 / J6.6a — aggregate agent scores (computed in _compute_aggregates)
    planner_score: float = 0.0
    evidence_score: float = 0.0
    qa_agent_score: float = 0.0
    report_score: float = 0.0
    recommendation_score: float = 0.0
    recommendation_dimension_summary: dict = field(default_factory=dict)


class EvaluationRunner:
    """Run all benchmark questions through the harness and produce scored results.

    Parameters
    ----------
    agent:
        A configured ``DcPowerAgent`` used as the template for Q&A runs.
        Its constructor args (client, top_evidence, top_chunks, profile) are
        copied so each parallel worker gets its own fresh instance with no
        shared mutable state (e.g. client.call_traces).
    sources_dir:
        Directory of source documents passed to the agent.
    profile:
        Optional domain profile; forwarded to the agent and contradiction engine.
    workers:
        Number of parallel workers for Q&A and contradiction runs.
        1 = sequential (default, safe for all environments).
        >1 = parallel — each worker gets its own DcPowerAgent instance.
        Recommended: 3-5 for live LLM runs (stay within API rate limits).
    """

    def __init__(
        self,
        agent: DcPowerAgent | None = None,
        *,
        sources_dir: str | Path = "sources",
        profile: DomainProfile | None = None,
        ro_out_dir: str | Path | None = None,
        workers: int = DEFAULT_WORKERS,
    ) -> None:
        self._agent = agent or DcPowerAgent(profile=profile)
        self._sources_dir = Path(sources_dir)
        self._profile = profile
        self._ro_out_dir: Path | None = Path(ro_out_dir) if ro_out_dir else None
        self._workers = max(1, workers)

    def _make_agent(self) -> DcPowerAgent:
        """Create a fresh DcPowerAgent with the same config as the template.

        Each parallel worker gets its own instance so client.call_traces
        and any other per-instance state is never shared across threads.
        """
        from ..claude_client import ClaudeClient, MockClaudeClient
        template = self._agent
        is_mock = getattr(template.client, "is_mock", False)
        if is_mock:
            client: Any = MockClaudeClient()
        else:
            client = ClaudeClient(
                model=getattr(template.client, "model", None),
                api_key=getattr(template.client, "api_key", None),
                use_extraction_cache=True,
            )
        return DcPowerAgent(
            client=client,
            top_evidence=template.top_evidence,
            top_chunks=template.top_chunks,
            profile=template.profile,
        )

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

        # Q&A questions — parallel when workers > 1
        total = len(qa_questions)
        if self._workers > 1:
            LOGGER.info("Running %d Q&A questions with %d workers", total, self._workers)
        result.qa_scores, result.agent_scores, result.perf_records = self._run_qa_parallel(
            list(qa_questions), documents, total
        )

        # Contradiction cases — parallel when workers > 1
        total_c = len(contradiction_cases)
        if self._workers > 1 and total_c:
            LOGGER.info("Running %d contradiction cases with %d workers", total_c, self._workers)
        result.contradiction_scores = self._run_contradiction_parallel(
            list(contradiction_cases), total_c
        )

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

    def _run_one_qa(
        self,
        idx: int,
        total: int,
        question: QAQuestion,
        documents: list,
    ) -> tuple[int, QAScore, AgentScores, BenchmarkPerfRecord]:
        """Run a single Q&A question. Returns (original_index, score, agent_score, perf)."""
        LOGGER.info(
            "[%d/%d] Running %s: %s",
            idx + 1, total, question.question_id, question.question[:60],
        )
        t_question_start = time.monotonic()
        perf: BenchmarkPerfRecord | None = None
        try:
            agent = self._make_agent()
            client = agent.client
            traces_start = len(client.call_traces)

            # --- Analysis phase (retrieval + LLM) ---
            t_analysis = time.monotonic()
            memo = agent.analyze(question.question, documents)
            analysis_ms = (time.monotonic() - t_analysis) * 1000

            # Derive LLM stats from call_traces accumulated during analyze()
            agent_traces = client.call_traces[traces_start:]
            llm_ms = sum(t.duration_ms for t in agent_traces)
            prompt_tokens = sum(t.token_usage.get("input_tokens", 0) for t in agent_traces)
            completion_tokens = sum(t.token_usage.get("output_tokens", 0) for t in agent_traces)
            total_tokens = prompt_tokens + completion_tokens
            llm_call_count = len(agent_traces)
            # Retrieval ≈ everything in analyze() that isn't LLM time
            retrieval_ms = max(0.0, analysis_ms - llm_ms)
            reranking_ms = 0.0  # not applicable in legacy DcPowerAgent path

            # --- Scoring / evaluation phase ---
            t_scoring = time.monotonic()
            qa_score = score_qa_response(question, memo)
            agent_score = score_agents(question.question_id, question.domain, memo, qa_score)
            scoring_ms = (time.monotonic() - t_scoring) * 1000

            # --- Report generation phase ---
            t_report = time.monotonic()
            if self._ro_out_dir is not None:
                _write_qa_research_object(
                    question=question,
                    memo=memo,
                    profile=self._profile,
                    out_dir=self._ro_out_dir,
                )
            report_ms = (time.monotonic() - t_report) * 1000

            total_ms = (time.monotonic() - t_question_start) * 1000
            perf = BenchmarkPerfRecord(
                question_id=question.question_id,
                total_ms=total_ms,
                analysis_ms=analysis_ms,
                retrieval_ms=retrieval_ms,
                reranking_ms=reranking_ms,
                llm_ms=llm_ms,
                scoring_ms=scoring_ms,
                report_ms=report_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                llm_calls=llm_call_count,
            )
        except Exception as exc:
            LOGGER.error("Error running %s: %s", question.question_id, exc)
            qa_score = _failed_qa_score(question, str(exc))
            agent_score = AgentScores(question_id=question.question_id, domain=question.domain)
            total_ms = (time.monotonic() - t_question_start) * 1000
            perf = BenchmarkPerfRecord(
                question_id=question.question_id,
                total_ms=total_ms,
                analysis_ms=0.0,
                retrieval_ms=0.0,
                reranking_ms=0.0,
                llm_ms=0.0,
                scoring_ms=0.0,
                report_ms=0.0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                llm_calls=0,
            )
        return idx, qa_score, agent_score, perf

    def _run_qa_parallel(
        self,
        questions: list[QAQuestion],
        documents: list,
        total: int,
    ) -> tuple[list[QAScore], list[AgentScores], list[BenchmarkPerfRecord]]:
        """Run Q&A questions in parallel, preserving original ordering."""
        qa_slots: list[QAScore | None] = [None] * len(questions)
        agent_slots: list[AgentScores | None] = [None] * len(questions)
        perf_slots: list[BenchmarkPerfRecord | None] = [None] * len(questions)
        if self._workers == 1:
            for idx, question in enumerate(questions):
                _, qa_score, agent_score, perf = self._run_one_qa(idx, total, question, documents)
                qa_slots[idx] = qa_score
                agent_slots[idx] = agent_score
                perf_slots[idx] = perf
        else:
            with ThreadPoolExecutor(max_workers=self._workers) as pool:
                futures = {
                    pool.submit(self._run_one_qa, idx, total, q, documents): idx
                    for idx, q in enumerate(questions)
                }
                for future in as_completed(futures):
                    idx, qa_score, agent_score, perf = future.result()
                    qa_slots[idx] = qa_score
                    agent_slots[idx] = agent_score
                    perf_slots[idx] = perf
        return (
            [s for s in qa_slots if s is not None],
            [s for s in agent_slots if s is not None],
            [p for p in perf_slots if p is not None],
        )

    def _run_one_contradiction(
        self,
        idx: int,
        total: int,
        case: ContradictionCase,
    ) -> tuple[int, ContradictionScore]:
        """Run a single contradiction case. Returns (original_index, score)."""
        LOGGER.info("[%d/%d] Contradiction test %s", idx + 1, total, case.contradiction_id)
        try:
            score = self._run_contradiction_case(case)
        except Exception as exc:
            LOGGER.error("Error running %s: %s", case.contradiction_id, exc)
            score = ContradictionScore(
                contradiction_id=case.contradiction_id,
                domain=case.domain,
                expected_result=case.expected_result,
                actual_result="error",
                correct=False,
                notes=f"Runner error: {exc}",
            )
        return idx, score

    def _run_contradiction_parallel(
        self,
        cases: list[ContradictionCase],
        total: int,
    ) -> list[ContradictionScore]:
        """Run contradiction cases in parallel, preserving original ordering."""
        scores: list[ContradictionScore | None] = [None] * len(cases)
        if self._workers == 1:
            for idx, case in enumerate(cases):
                _, score = self._run_one_contradiction(idx, total, case)
                scores[idx] = score
        else:
            with ThreadPoolExecutor(max_workers=self._workers) as pool:
                futures = {
                    pool.submit(self._run_one_contradiction, idx, total, case): idx
                    for idx, case in enumerate(cases)
                }
                for future in as_completed(futures):
                    idx, score = future.result()
                    scores[idx] = score
        return [s for s in scores if s is not None]


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

    # J5.7 — aggregate agent scores
    agg = aggregate_agent_scores(result.agent_scores)
    result.planner_score = agg["planner_score"]
    result.evidence_score = agg["evidence_score"]
    result.qa_agent_score = agg["qa_score"]
    result.report_score = agg["report_score"]
    result.recommendation_score = agg["recommendation_score"]
    result.recommendation_dimension_summary = agg.get("recommendation_dimension_summary", {})


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
        # J7.6a – write_latest=False: benchmark runs must never overwrite
        # latest_research_object.json, which belongs to the interactive pipeline.
        write_research_object(ro, out_dir=out_dir, write_latest=False)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Could not write research object for %s: %s", question.question_id, exc)
