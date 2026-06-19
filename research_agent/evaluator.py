"""Warning-mode quality checks for generated research memos."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .schemas import EvaluationWarning, EvidenceItem, ResearchMemo, SourceDocument

if TYPE_CHECKING:
    from .profile import DomainProfile

_BASE_REQUIRED_LIST_FIELDS = {
    "confirmed_facts": "Confirmed Facts",
    "inferences": "Inferences",
    "open_questions": "Open Questions",
}

_QUESTION_TOPIC_TERMS = {
    "power": ("power", "electrical", "voltage", "utility", "grid", "ups", "bbu", "busway", "pdu"),
    "cooling": ("cooling", "liquid", "thermal", "cdu", "heat", "chilled water"),
    "networking": (
        "networking",
        "network",
        "nvlink",
        "infiniband",
        "ethernet",
        "spectrum",
        "connectx",
    ),
    "rack architecture": ("rack", "nvl72", "rack-scale", "tray", "shelf", "cabinet"),
    "backup/resiliency": ("backup", "battery", "bbu", "ups", "resiliency", "redundancy"),
    "operations": ("operations", "commissioning", "maintenance", "monitoring"),
}

_TOPIC_SECTION_CHECKS = {
    "power": ("power_implications", "Power Implications", "missing_power_implications", "missing_power_citations"),
    "cooling": (
        "cooling_implications",
        "Cooling Implications",
        "missing_cooling_implications",
        "missing_cooling_citations",
    ),
    "networking": (
        "networking_implications",
        "Networking Implications",
        "missing_networking_implications",
        "missing_networking_citations",
    ),
    "rack architecture": (
        "rack_architecture_implications",
        "Rack Architecture Implications",
        "missing_rack_architecture_implications",
        "missing_rack_architecture_citations",
    ),
}

_CITATION_RE = re.compile(r"\[Source:\s*.+?,\s*Evidence:\s*(E\d{3})\]")


def evaluate_memo(
    memo: ResearchMemo,
    documents: Sequence[SourceDocument],
    *,
    mock_llm: bool = False,
    profile: "DomainProfile | None" = None,
) -> list[EvaluationWarning]:
    """Return non-fatal warnings about memo completeness and source support."""

    warnings: list[EvaluationWarning] = []

    if mock_llm:
        warnings.append(
            EvaluationWarning(
                code="mock_llm",
                message="Memo was generated with the mock Claude client; domain synthesis is provisional.",
            )
        )

    if not documents:
        warnings.append(
            EvaluationWarning(
                code="no_sources",
                message="No supported source documents were loaded.",
            )
        )
        warnings.append(
            EvaluationWarning(
                code="zero_documents_loaded",
                message="Zero source documents were loaded.",
            )
        )

    if len(documents) < 3:
        warnings.append(
            EvaluationWarning(
                code="few_documents_loaded",
                message="Fewer than 3 source documents were loaded.",
            )
        )

    if any(document.char_count == 0 for document in documents):
        warnings.append(
            EvaluationWarning(
                code="empty_document_text",
                message="One or more loaded documents have zero extracted characters.",
            )
        )

    if not memo.executive_summary.strip():
        warnings.append(
            EvaluationWarning(
                code="missing_executive_summary",
                message="Executive Summary is empty.",
            )
        )

    for field_name, section_name in _BASE_REQUIRED_LIST_FIELDS.items():
        value = getattr(memo, field_name)
        if not value:
            warnings.append(
                EvaluationWarning(
                    code=f"missing_{field_name}",
                    message=f"{section_name} section has no entries.",
                )
            )

    evidence = _evidence_items(memo)

    if not evidence:
        warnings.append(
            EvaluationWarning(
                code="no_evidence",
                message="No evidence items were produced for Source Notes.",
            )
        )
    if len(evidence) < 10:
        warnings.append(
            EvaluationWarning(
                code="low_evidence_count",
                message="Fewer than 10 total evidence items were produced.",
            )
        )
    high_quality_count = sum(1 for item in evidence if item.overall_score >= 3.5)
    if high_quality_count < 10:
        warnings.append(
            EvaluationWarning(
                code="low_high_quality_evidence_count",
                message="Fewer than 10 high-quality evidence items are available.",
            )
        )

    chunks_per_doc = memo.metadata.get("chunks_per_document", {})
    if chunks_per_doc:
        for doc in documents:
            if chunks_per_doc.get(doc.path.name, -1) == 0:
                warnings.append(EvaluationWarning(
                    code="zero_chunks_for_document",
                    message=f"Document '{doc.path.name}' produced zero chunks.",
                ))

    evidence_per_chunk = memo.metadata.get("evidence_per_chunk", {})
    if evidence_per_chunk:
        zero = [cid for cid, cnt in evidence_per_chunk.items() if cnt == 0]
        if zero:
            warnings.append(EvaluationWarning(
                code="zero_evidence_for_chunk",
                message=f"{len(zero)} chunk(s) produced zero evidence items.",
            ))

    evidence_source_count = len(
        {item.source_document for item in evidence if item.source_document.strip()}
    )
    if evidence_source_count < 3:
        warnings.append(
            EvaluationWarning(
                code="insufficient_source_evidence",
                message="Fewer than 3 source documents have evidence items.",
            )
        )

    if any(not item.evidence_snippet.strip() for item in evidence):
        warnings.append(
            EvaluationWarning(
                code="empty_evidence_snippet",
                message="One or more evidence items have empty snippets.",
            )
        )

    question_topics = classify_question_topics(memo.question, profile)
    warnings.extend(_topic_section_warnings(memo, question_topics, profile))
    warnings.extend(_citation_warnings(memo, evidence, question_topics, profile))

    # Informational chunk retrieval metrics (not warnings)
    chunk_count = memo.metadata.get("chunk_count", 0)
    chunks_selected = memo.metadata.get("chunks_selected", 0)
    if chunk_count:
        retrieval_ratio = round(chunks_selected / chunk_count, 4) if chunk_count else 0.0
        warnings.append(EvaluationWarning(
            code="retrieval_metrics",
            message=(
                f"Chunk retrieval: {chunks_selected}/{chunk_count} chunks selected "
                f"(retrieval_ratio={retrieval_ratio:.2%})."
            ),
            severity="info",
        ))

    # Informational coverage matrix metrics (not warnings)
    coverage_matrix = memo.metadata.get("coverage_matrix", [])
    if coverage_matrix:
        strong = [a["topic"] for a in coverage_matrix if a.get("coverage_level") == "strong"]
        moderate = [a["topic"] for a in coverage_matrix if a.get("coverage_level") == "moderate"]
        weak = [a["topic"] for a in coverage_matrix if a.get("coverage_level") == "weak"]
        uncovered = [a["topic"] for a in coverage_matrix if a.get("coverage_level") == "none"]
        parts = []
        if strong:
            parts.append(f"strong=[{', '.join(strong)}]")
        if moderate:
            parts.append(f"moderate=[{', '.join(moderate)}]")
        if weak:
            parts.append(f"weak=[{', '.join(weak)}]")
        if uncovered:
            parts.append(f"uncovered=[{', '.join(uncovered)}]")
        warnings.append(EvaluationWarning(
            code="coverage_matrix_metrics",
            message=f"Coverage matrix: {'; '.join(parts) if parts else 'no topics assessed'}.",
            severity="info",
        ))

    # Informational research gap metrics (not warnings)
    research_gaps = memo.metadata.get("research_gaps", [])
    if research_gaps:
        gap_count = len(research_gaps)
        high_priority_gap_count = sum(
            1 for g in research_gaps if g.get("priority") == "high"
        )
        warnings.append(EvaluationWarning(
            code="research_gap_metrics",
            message=(
                f"Research gaps: {gap_count} total, "
                f"{high_priority_gap_count} high-priority."
            ),
            severity="info",
        ))

    # High-severity contradiction warning (non-blocking)
    contradictions = memo.metadata.get("contradictions", [])
    high_severity = [c for c in contradictions if c.get("severity") == "high"]
    if high_severity:
        ids = ", ".join(
            f"{c.get('evidence_a_id', '?')} vs {c.get('evidence_b_id', '?')}"
            for c in high_severity[:3]
        )
        warnings.append(EvaluationWarning(
            code="high_severity_contradiction",
            message=(
                f"{len(high_severity)} high-severity contradiction(s) detected in evidence "
                f"({ids})."
            ),
        ))

    return warnings


def classify_question_topics(
    question: str,
    profile: "DomainProfile | None" = None,
) -> set[str]:
    """Classify question topics using the domain profile when supplied.

    When *profile* is ``None`` the legacy hard-coded ``_QUESTION_TOPIC_TERMS``
    mapping is used so that existing callers without a profile continue to work
    identically.
    """
    if profile is not None:
        # Use evaluator-specific topic terms when available, otherwise fall back to profile's topic_keywords
        evaluator_terms = profile.get_evaluator_topic_terms()
        normalized = question.lower()
        return {
            topic
            for topic, terms in evaluator_terms.items()
            if any(term in normalized for term in terms)
        }

    # Legacy fallback: hard-coded AI data-center topic terms
    normalized = question.lower()
    topics: set[str] = set()
    for topic, terms in _QUESTION_TOPIC_TERMS.items():
        if any(term in normalized for term in terms):
            topics.add(topic)
    return topics


def _evidence_items(memo: ResearchMemo) -> list[EvidenceItem]:
    return memo.source_notes or memo.evidence


def _topic_section_warnings(
    memo: ResearchMemo,
    question_topics: set[str],
    profile: "DomainProfile | None" = None,
) -> list[EvaluationWarning]:
    section_checks = (
        profile.topic_section_checks if profile is not None and profile.topic_section_checks
        else _TOPIC_SECTION_CHECKS
    )
    warnings: list[EvaluationWarning] = []
    for topic in sorted(question_topics):
        if topic not in section_checks:
            continue
        field_name, section_name, missing_code, _citation_code = section_checks[topic]
        if not getattr(memo, field_name, None):
            warnings.append(
                EvaluationWarning(
                    code=missing_code,
                    message=f"{section_name} section is required by the question topic '{topic}' but is empty.",
                )
            )
    return warnings


def _citation_warnings(
    memo: ResearchMemo,
    evidence: list[EvidenceItem],
    question_topics: set[str],
    profile: "DomainProfile | None" = None,
) -> list[EvaluationWarning]:
    section_checks = (
        profile.topic_section_checks if profile is not None and profile.topic_section_checks
        else _TOPIC_SECTION_CHECKS
    )
    warnings: list[EvaluationWarning] = []
    citation_checks = [("confirmed_facts", "Confirmed Facts", "missing_confirmed_fact_citations")]
    for topic in sorted(question_topics):
        if topic not in section_checks:
            continue
        field_name, section_name, _missing_code, citation_code = section_checks[topic]
        citation_checks.append((field_name, section_name, citation_code))

    for field_name, section_name, code in citation_checks:
        values = getattr(memo, field_name)
        if values and not _section_has_citation(values):
            warnings.append(
                EvaluationWarning(
                    code=code,
                    message=f"{section_name} section has no source citations.",
                )
            )

    known_ids = {item.evidence_id for item in evidence if item.evidence_id}
    referenced_ids = set()
    for field_name in [
        "confirmed_facts",
        "power_implications",
        "cooling_implications",
        "networking_implications",
        "rack_architecture_implications",
    ]:
        referenced_ids.update(_section_citation_ids(getattr(memo, field_name)))

    unknown_ids = sorted(referenced_ids - known_ids)
    if unknown_ids:
        warnings.append(
            EvaluationWarning(
                code="unknown_evidence_citation",
                message="Citations reference unknown evidence IDs: " + ", ".join(unknown_ids) + ".",
            )
        )

    return warnings


def _section_has_citation(items: list[str]) -> bool:
    return any(_CITATION_RE.search(item) for item in items)


def _section_citation_ids(items: list[str]) -> set[str]:
    ids: set[str] = set()
    for item in items:
        ids.update(_CITATION_RE.findall(item))
    return ids
