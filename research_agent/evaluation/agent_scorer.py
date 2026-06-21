"""Per-agent evaluation scoring (J5.7).

Derives per-agent quality proxies from ResearchMemo + QAScore produced by the
benchmark pipeline.  Scores are structural/ratio-based and profile-agnostic —
they work for SMR, AI Data Centers, Transmission, and future profiles without
changing evaluation logic.

Planner  — investigation area breadth and coverage
Evidence — evidence count, quality, and source diversity
QA       — gap detection and contradiction awareness
Report   — citation score and confirmed-fact richness
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..schemas import ResearchMemo
    from .scorer import QAScore

# Normalisation constants
_EVIDENCE_COUNT_GOOD = 20   # ≥ this → full count-component credit
_FACTS_COUNT_GOOD = 10      # ≥ this → full fact-richness credit
_GAPS_COUNT_GOOD = 5        # ≥ this → full gap-detection credit
_AREA_MIN = 3               # minimum investigation areas expected from a good plan


@dataclass
class AgentScores:
    """Per-agent quality scores for one Q&A benchmark question."""

    question_id: str
    domain: str

    # ── Per-agent composite scores (0.0–1.0) ──────────────────────────────
    planner_score: float = 0.0
    evidence_score: float = 0.0
    qa_score: float = 0.0
    report_score: float = 0.0

    # ── Planner detail ─────────────────────────────────────────────────────
    investigation_area_count: int = 0
    investigation_areas_covered: int = 0

    # ── Evidence detail ────────────────────────────────────────────────────
    evidence_count: int = 0
    high_quality_evidence: int = 0   # overall_score ≥ 4
    source_diversity: int = 0         # unique source documents

    # ── QA detail ─────────────────────────────────────────────────────────
    gaps_identified: int = 0
    contradictions_found: int = 0

    # ── Report detail ─────────────────────────────────────────────────────
    citation_count: int = 0
    confirmed_facts: int = 0
    report_citation_score: float = 0.0


def score_agents(
    question_id: str,
    domain: str,
    memo: "ResearchMemo",
    qa_score: "QAScore",
) -> AgentScores:
    """Derive per-agent scores from a ResearchMemo and a QAScore.

    All inputs are produced by the benchmark pipeline's DcPowerAgent.analyze()
    call — no functional agent pipeline required.
    """
    scores = AgentScores(question_id=question_id, domain=domain)

    # ── Planner ────────────────────────────────────────────────────────────
    # proxy: investigation area breadth and coverage from coverage_matrix
    coverage_areas = memo.coverage_matrix or []
    area_count = len(coverage_areas)
    areas_covered = sum(
        1 for ca in coverage_areas
        if ca.coverage_level != "none"
    )

    scores.investigation_area_count = area_count
    scores.investigation_areas_covered = areas_covered

    if area_count > 0:
        coverage_ratio = areas_covered / area_count
        breadth_score = min(1.0, area_count / _AREA_MIN)
        scores.planner_score = round(coverage_ratio * 0.7 + breadth_score * 0.3, 3)
    else:
        scores.planner_score = 0.5  # neutral — no coverage matrix available

    # ── Evidence ───────────────────────────────────────────────────────────
    evidence_items = memo.source_notes or memo.evidence or []
    ev_count = len(evidence_items)

    high_quality = sum(
        1 for e in evidence_items
        if getattr(e, "overall_score", 0) >= 4
    )
    sources = {getattr(e, "source_document", "") for e in evidence_items}
    sources.discard("")

    scores.evidence_count = ev_count
    scores.high_quality_evidence = high_quality
    scores.source_diversity = len(sources)

    count_score = min(1.0, ev_count / _EVIDENCE_COUNT_GOOD)
    quality_ratio = (high_quality / ev_count) if ev_count > 0 else 0.0
    scores.evidence_score = round(count_score * 0.5 + quality_ratio * 0.5, 3)

    # ── QA ────────────────────────────────────────────────────────────────
    gaps = memo.research_gaps or []
    contradictions = memo.contradictions or []

    scores.gaps_identified = len(gaps)
    scores.contradictions_found = len(contradictions)

    gap_score = min(1.0, len(gaps) / _GAPS_COUNT_GOOD)
    scores.qa_score = round(gap_score, 3)

    # ── Report ────────────────────────────────────────────────────────────
    confirmed = memo.confirmed_facts or []

    scores.citation_count = qa_score.citation_count
    scores.report_citation_score = qa_score.citation_score
    scores.confirmed_facts = len(confirmed)

    fact_score = min(1.0, len(confirmed) / _FACTS_COUNT_GOOD)
    scores.report_score = round(
        qa_score.citation_score * 0.5 + fact_score * 0.5, 3
    )

    return scores


def aggregate_agent_scores(all_scores: list[AgentScores]) -> dict[str, float]:
    """Compute mean per-agent scores across all questions."""
    if not all_scores:
        return {
            "planner_score": 0.0,
            "evidence_score": 0.0,
            "qa_score": 0.0,
            "report_score": 0.0,
        }
    n = len(all_scores)
    return {
        "planner_score": round(sum(s.planner_score for s in all_scores) / n, 4),
        "evidence_score": round(sum(s.evidence_score for s in all_scores) / n, 4),
        "qa_score": round(sum(s.qa_score for s in all_scores) / n, 4),
        "report_score": round(sum(s.report_score for s in all_scores) / n, 4),
    }
