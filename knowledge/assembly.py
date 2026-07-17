"""Evidence Assembly Completeness — PH5.5d.

Evaluates whether the retrieved evidence set adequately covers the active
research questions and subquestions.

Design:
- No retrieval behavior changes.
- No Evidence schema changes.
- No stochastic reasoning — all scores are deterministic arithmetic on
  evidence counts and validated contradiction lists.
- Gaps are surfaced explicitly; unknown coverage is labelled UNKNOWN rather
  than assumed complete or partial.
- The module is a pure computation layer: it reads existing mapping data
  produced by EvidenceAgent and emits structured assessments.

Thresholds:
    COMPLETE   ≥ _COMPLETE_THRESHOLD evidence items assigned to the subquestion
    PARTIAL    ≥ 1 item (weak or moderate coverage — corroboration present but
               insufficient for high confidence)
    INCOMPLETE = 0 items
    UNKNOWN    = no subquestions defined (completeness cannot be assessed)
"""

from __future__ import annotations

from .models import AssemblyCompleteness, CompletenessStatus, SubquestionCompleteness

# Evidence count at which a subquestion is considered fully covered.
# Must match _STRONG in functional_agents/evidence_agent.py for consistency.
_COMPLETE_THRESHOLD = 4

# ---------------------------------------------------------------------------
# Internal helpers — all pure functions, no I/O
# ---------------------------------------------------------------------------


def _status(evidence_count: int) -> CompletenessStatus:
    """Map an evidence count to a CompletenessStatus value.

    COMPLETE  ≥ _COMPLETE_THRESHOLD items.
    PARTIAL   ≥ 1 item.
    INCOMPLETE = 0 items.
    """
    if evidence_count >= _COMPLETE_THRESHOLD:
        return "COMPLETE"
    if evidence_count >= 1:
        return "PARTIAL"
    return "INCOMPLETE"


def _coverage_fraction(evidence_count: int) -> float:
    """Linear coverage from 0.0 to 1.0, saturating at _COMPLETE_THRESHOLD."""
    return min(1.0, evidence_count / _COMPLETE_THRESHOLD)


def _score(evidence_count: int, contradicting_count: int) -> float:
    """Deterministic completeness score in [0.0, 1.0].

    Starts from the coverage fraction and applies a contradiction penalty
    proportional to how many assigned items are implicated in contradictions.
    """
    fraction = _coverage_fraction(evidence_count)
    if contradicting_count > 0 and evidence_count > 0:
        penalty = min(1.0, contradicting_count / evidence_count)
        fraction = round(fraction * (1.0 - penalty), 4)
    return round(fraction, 4)


def _gap_notes(evidence_count: int, contradicting_count: int) -> list[str]:
    """Return explicit gap descriptions for a subquestion.

    Only produces notes that are directly derivable from the counts —
    nothing is inferred from statement content or retrieved documents.
    """
    notes: list[str] = []
    if evidence_count == 0:
        notes.append("No evidence retrieved for this subquestion.")
    elif evidence_count == 1:
        notes.append(
            "Single evidence item — insufficient for independent corroboration."
        )
    elif evidence_count < _COMPLETE_THRESHOLD:
        shortfall = _COMPLETE_THRESHOLD - evidence_count
        notes.append(
            f"Partial coverage ({evidence_count} item(s)); "
            f"{shortfall} more would reach COMPLETE."
        )
    if contradicting_count > 0:
        notes.append(
            f"{contradicting_count} contradicting evidence item(s) detected — "
            "manual review recommended."
        )
    return notes


def _overall_status(statuses: list[CompletenessStatus]) -> CompletenessStatus:
    """Aggregate per-subquestion statuses into an overall assessment.

    Rule (weakest-wins):
        All COMPLETE                → COMPLETE
        Any COMPLETE or PARTIAL     → PARTIAL
        All INCOMPLETE              → INCOMPLETE
        No subquestions (empty)     → UNKNOWN
    """
    if not statuses:
        return "UNKNOWN"
    if all(s == "COMPLETE" for s in statuses):
        return "COMPLETE"
    if any(s in ("COMPLETE", "PARTIAL") for s in statuses):
        return "PARTIAL"
    return "INCOMPLETE"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_subquestion(
    subquestion_text: str,
    evidence_ids: list[str],
    *,
    subquestion_id: str | None = None,
    research_question_id: str | None = None,
    contradicting_ids: list[str] | None = None,
) -> SubquestionCompleteness:
    """Produce a completeness assessment for one subquestion.

    Parameters
    ----------
    subquestion_text:
        The natural-language subquestion text.
    evidence_ids:
        IDs of evidence items mapped to this subquestion.
    subquestion_id:
        Optional structured identifier (e.g. "SQ-3"). None when not available.
    research_question_id:
        Optional identifier for the parent research question.
    contradicting_ids:
        IDs of evidence items involved in validated contradictions that
        implicate this subquestion's evidence.  Pass an empty list (the
        default) when contradiction detection has not run.
    """
    contradicting_ids = list(contradicting_ids or [])
    ev_count = len(evidence_ids)
    contra_count = len(contradicting_ids)

    return SubquestionCompleteness(
        research_question_id=research_question_id,
        subquestion_id=subquestion_id,
        subquestion_text=subquestion_text,
        evidence_count=ev_count,
        supporting_evidence_count=ev_count,  # all assigned items are supporting
        contradicting_evidence_count=contra_count,
        missing_area_count=0,              # area gaps are reported at the top level
        coverage_fraction=_coverage_fraction(ev_count),
        completeness_score=_score(ev_count, contra_count),
        completeness_status=_status(ev_count),
        gap_notes=_gap_notes(ev_count, contra_count),
        evidence_ids=list(evidence_ids),
    )


def assess_assembly_completeness(
    question: str,
    subquestions: list[str],
    evidence_by_subquestion: dict[str, list[str]],
    *,
    investigation_areas: list[str] | None = None,
    evidence_by_area: dict[str, list[str]] | None = None,
    validated_contradictions: list[dict] | None = None,
    research_question_id: str | None = None,
    total_retrieved: int | None = None,
) -> AssemblyCompleteness:
    """Build a deterministic assembly completeness assessment.

    Uses existing evidence mapping data — no new retrieval or inference.

    Parameters
    ----------
    question:
        The primary research question.
    subquestions:
        Ordered list of subquestion texts from the Planner.
    evidence_by_subquestion:
        Mapping produced by _map_evidence_to_subquestions() in EvidenceAgent.
        Keys include subquestion texts and the special "_unmapped" key.
    investigation_areas:
        Investigation areas from the Planner (optional).
    evidence_by_area:
        Mapping produced by _map_evidence_to_areas() in EvidenceAgent.
    validated_contradictions:
        List of contradiction dicts with "evidence_id_a" / "evidence_id_b"
        keys.  Pass [] when contradiction detection has not run.
    research_question_id:
        Optional structured identifier for the research question.
    total_retrieved:
        Total candidate count from the retriever.  When provided, used as
        total_evidence_count; otherwise computed from evidence_by_subquestion.
    """
    investigation_areas = investigation_areas or []
    evidence_by_area = evidence_by_area or {}
    validated_contradictions = validated_contradictions or []

    # Build contradiction index: evidence_id → set of IDs it contradicts
    contra_index: dict[str, set[str]] = {}
    for c in validated_contradictions:
        a = c.get("evidence_id_a", "")
        b = c.get("evidence_id_b", "")
        if a and b:
            contra_index.setdefault(a, set()).add(b)
            contra_index.setdefault(b, set()).add(a)

    # No subquestions → UNKNOWN (cannot assess completeness)
    if not subquestions:
        all_ids: set[str] = set()
        for ids in evidence_by_subquestion.values():
            all_ids.update(ids)
        return AssemblyCompleteness(
            question=question,
            research_question_id=research_question_id,
            total_subquestions=0,
            covered_subquestions=0,
            total_evidence_count=total_retrieved if total_retrieved is not None else len(all_ids),
            missing_area_count=sum(
                1 for a in investigation_areas if not evidence_by_area.get(a)
            ),
            overall_completeness_score=0.0,
            overall_completeness_status="UNKNOWN",
            subquestion_assessments=[],
            gap_summary=["No subquestions defined — completeness cannot be assessed."],
        )

    # Per-subquestion assessments
    sq_assessments: list[SubquestionCompleteness] = []
    assigned_ids: set[str] = set()

    for sq in subquestions:
        ev_ids = evidence_by_subquestion.get(sq, [])
        assigned_ids.update(ev_ids)

        # Contradicting items: any item in ev_ids that appears in contra_index
        contra_ids: list[str] = [
            cid
            for eid in ev_ids
            for cid in contra_index.get(eid, set())
        ]

        sq_assessments.append(
            assess_subquestion(
                sq,
                ev_ids,
                research_question_id=research_question_id,
                contradicting_ids=list(set(contra_ids)),
            )
        )

    # Aggregate metrics
    statuses = [a.completeness_status for a in sq_assessments]
    covered = sum(1 for s in statuses if s in ("COMPLETE", "PARTIAL"))
    avg_score = round(
        sum(a.completeness_score for a in sq_assessments) / max(1, len(sq_assessments)),
        4,
    )

    # Missing investigation areas
    missing_areas = [a for a in investigation_areas if not evidence_by_area.get(a)]
    missing_area_count = len(missing_areas)

    # Gap summary (top-level, human-readable)
    gap_summary: list[str] = []
    incomplete_sqs = [a for a in sq_assessments if a.completeness_status == "INCOMPLETE"]
    partial_sqs = [a for a in sq_assessments if a.completeness_status == "PARTIAL"]
    if incomplete_sqs:
        gap_summary.append(
            f"{len(incomplete_sqs)} subquestion(s) have no evidence coverage."
        )
    if partial_sqs:
        gap_summary.append(
            f"{len(partial_sqs)} subquestion(s) have partial coverage only."
        )
    if missing_areas:
        area_list = ", ".join(missing_areas[:3])
        suffix = "..." if len(missing_areas) > 3 else ""
        gap_summary.append(
            f"{missing_area_count} investigation area(s) uncovered: {area_list}{suffix}."
        )
    if not gap_summary:
        gap_summary.append("All subquestions and investigation areas have adequate coverage.")

    return AssemblyCompleteness(
        question=question,
        research_question_id=research_question_id,
        total_subquestions=len(subquestions),
        covered_subquestions=covered,
        total_evidence_count=(
            total_retrieved if total_retrieved is not None else len(assigned_ids)
        ),
        missing_area_count=missing_area_count,
        overall_completeness_score=avg_score,
        overall_completeness_status=_overall_status(statuses),
        subquestion_assessments=sq_assessments,
        gap_summary=gap_summary,
    )
