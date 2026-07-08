"""IterationPlanAgent — deterministic iteration planning (J12.2).

Runs immediately after ExecutiveConfidenceAgent, before MultiProfileAgent.
Converts research-gap diagnostics and decision-quality signals into a
structured plan for the next research iteration.

Inputs (read-only):
  context.executive_confidence, context.assumptions, context.recommendations,
  context.risks, context.strategic_options, context.decision_analysis,
  context.research_gap_analysis, context.research_object (fallbacks)

Writes to:
  context.iteration_plan
  context.research_object["iteration_plan"]
  context.trace["_iteration_plan"]

No LLM calls. Does not modify any upstream artifact.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext, NextAction

LOGGER = logging.getLogger(__name__)

_MAX_TASKS = 10

# Regex patterns for structured artifact ID extraction (J12.2a)
_A_RE = re.compile(r'(?<![A-Za-z])A-\d+')    # assumption IDs: A-001
_RSK_RE = re.compile(r'\bRSK-\d+')            # risk IDs: RSK-001
_REC_RE = re.compile(r'\bREC-\d+')            # recommendation IDs: REC-001
_OPT_RE = re.compile(r'\bOPT-[A-Z]\b')        # option IDs: OPT-A


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _priority_score(gain: str, urgency: str) -> int:
    if gain == "HIGH" and urgency == "HIGH":
        return 1
    if gain == "HIGH":
        return 2
    if gain == "MEDIUM" and urgency == "HIGH":
        return 3
    if gain == "MEDIUM":
        return 4
    return 5


def _sort_tasks(tasks: list[dict]) -> list[dict]:
    """Stable-sort by priority bucket; within each bucket source order is preserved."""
    return sorted(
        enumerate(tasks),
        key=lambda pair: _priority_score(
            pair[1].get("expected_confidence_gain", "LOW"),
            pair[1].get("urgency", "LOW"),
        ),
    )  # type: ignore[return-value]
    # The enumeration wraps are stripped below; callers must unwrap.


def _sorted_tasks(tasks: list[dict]) -> list[dict]:
    """Return tasks sorted by priority, preserving within-bucket order."""
    indexed = list(enumerate(tasks))
    indexed.sort(key=lambda pair: (
        _priority_score(
            pair[1].get("expected_confidence_gain", "LOW"),
            pair[1].get("urgency", "LOW"),
        ),
        pair[0],  # original index for stable within-bucket ordering
    ))
    return [t for _, t in indexed]


def _assign_task_ids(tasks: list[dict]) -> list[dict]:
    for i, t in enumerate(tasks, start=1):
        t["task_id"] = f"IRT-{i:03d}"
    return tasks


def _extract_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        return (
            item.get("action") or item.get("priority") or
            item.get("description") or item.get("text") or
            item.get("title") or item.get("summary") or ""
        ).strip()
    return ""


def _extract_ids_from_text(text: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """Extract structured artifact IDs from a text string (J12.2a).

    Returns (assumption_ids, risk_ids, recommendation_ids, option_ids).
    Each list is sorted and deduplicated.
    """
    return (
        sorted(set(_A_RE.findall(text))),
        sorted(set(_RSK_RE.findall(text))),
        sorted(set(_REC_RE.findall(text))),
        sorted(set(_OPT_RE.findall(text))),
    )


def _enrich_task_ids(task: dict) -> dict:
    """Merge IDs found in task text fields into the structured ID lists (J12.2a).

    Ensures linkage completeness: any structured ID (A-NNN, RSK-NNN, REC-NNN,
    OPT-X) appearing in any task text field is also present in the corresponding
    related_*_ids list. Deduplicates and sorts each list.
    """
    parts: list[str] = [
        task.get("task_title") or "",
        task.get("research_objective") or "",
        task.get("why_it_matters") or "",
    ]
    for s in task.get("evidence_needed") or []:
        if s:
            parts.append(s)
    for s in task.get("suggested_queries") or []:
        if s:
            parts.append(s)
    combined = " ".join(parts)
    a_ids, r_ids, rec_ids, o_ids = _extract_ids_from_text(combined)
    task["related_assumption_ids"] = sorted(
        set((task.get("related_assumption_ids") or []) + a_ids)
    )
    task["related_risk_ids"] = sorted(
        set((task.get("related_risk_ids") or []) + r_ids)
    )
    task["related_recommendation_ids"] = sorted(
        set((task.get("related_recommendation_ids") or []) + rec_ids)
    )
    task["related_option_ids"] = sorted(
        set((task.get("related_option_ids") or []) + o_ids)
    )
    return task


def _validate_plan(
    plan: dict,
    *,
    known_assumption_ids: set[str] | None = None,
    known_risk_ids: set[str] | None = None,
    known_option_ids: set[str] | None = None,
) -> list[str]:
    """Validate the iteration_plan artifact (J12.2a).

    Returns a list of warning strings. Empty list means no issues.
    Structural violations are always checked; cross-reference warnings
    require the corresponding known_* sets to be provided.
    """
    warnings_out: list[str] = []
    tasks = plan.get("priority_research_tasks") or []

    # Task IDs unique
    seen_task_ids: set[str] = set()
    for t in tasks:
        tid = t.get("task_id") or ""
        if tid and tid in seen_task_ids:
            warnings_out.append(f"Duplicate task_id: {tid!r}")
        elif tid:
            seen_task_ids.add(tid)

    # No duplicate tasks (by normalized title)
    seen_titles: set[str] = set()
    for t in tasks:
        norm = _normalize_text(t.get("task_title") or "")
        if norm and norm in seen_titles:
            warnings_out.append(f"{t.get('task_id', '?')}: duplicate task title")
        elif norm:
            seen_titles.add(norm)

    for t in tasks:
        tid = t.get("task_id") or "?"

        if not (t.get("task_title") or "").strip():
            warnings_out.append(f"{tid}: empty task_title")
        if not (t.get("research_objective") or "").strip():
            warnings_out.append(f"{tid}: empty research_objective")
        gain = t.get("expected_confidence_gain")
        if gain not in ("HIGH", "MEDIUM", "LOW"):
            warnings_out.append(f"{tid}: invalid expected_confidence_gain={gain!r}")
        urgency = t.get("urgency")
        if urgency not in ("HIGH", "MEDIUM", "LOW"):
            warnings_out.append(f"{tid}: invalid urgency={urgency!r}")

        # Structured IDs unique within task
        for field in (
            "related_assumption_ids",
            "related_risk_ids",
            "related_recommendation_ids",
            "related_option_ids",
        ):
            ids_list = t.get(field) or []
            if len(set(ids_list)) < len(ids_list):
                warnings_out.append(f"{tid}: duplicate IDs in {field}")

        # Cross-reference: linked IDs should exist in source collections
        if known_assumption_ids is not None:
            for aid in (t.get("related_assumption_ids") or []):
                if aid not in known_assumption_ids:
                    warnings_out.append(
                        f"{tid}: assumption {aid!r} not found in input assumptions"
                    )
        if known_risk_ids is not None:
            for rid in (t.get("related_risk_ids") or []):
                if rid not in known_risk_ids:
                    warnings_out.append(
                        f"{tid}: risk {rid!r} not found in input risks"
                    )
        if known_option_ids is not None:
            for oid in (t.get("related_option_ids") or []):
                if oid not in known_option_ids:
                    warnings_out.append(
                        f"{tid}: option {oid!r} not found in input strategic_options"
                    )

    return warnings_out


# ---------------------------------------------------------------------------
# Heuristic 1 — Validation priorities (from ExecutiveConfidenceAgent)
# ---------------------------------------------------------------------------

def _tasks_from_validation_priorities(
    exec_conf: dict,
    seen_text: set,
) -> list[dict]:
    """Convert exec_conf.validation_priorities into HIGH/HIGH IRT tasks."""
    priorities = exec_conf.get("validation_priorities") or []
    tasks: list[dict] = []
    for item in priorities:
        text = _extract_text(item)
        if not text:
            continue
        norm = _normalize_text(text)
        if norm in seen_text:
            continue
        seen_text.add(norm)
        assume_id = ""
        if isinstance(item, dict):
            assume_id = (
                item.get("assumption_id") or item.get("related_assumption_id") or ""
            )
        tasks.append({
            "source_type": "executive_confidence",
            "source_id": "validation_priority",
            "task_title": text[:120],
            "research_objective": text,
            "why_it_matters": (
                "Identified as a decision-critical validation priority by ExecutiveConfidenceAgent."
            ),
            "expected_confidence_gain": "HIGH",
            "urgency": "HIGH",
            "evidence_needed": [f"Evidence validating: {text[:100]}"],
            "suggested_queries": [text[:100]],
            "related_assumption_ids": [assume_id] if assume_id else [],
            "related_risk_ids": [],
            "related_recommendation_ids": [],
            "related_option_ids": [],
        })
    return tasks


# ---------------------------------------------------------------------------
# Heuristic 2 — Critical unknowns (dedup against validation priorities)
# ---------------------------------------------------------------------------

def _tasks_from_critical_unknowns(
    exec_conf: dict,
    seen_text: set,
) -> list[dict]:
    """Convert exec_conf.critical_unknowns into HIGH/MEDIUM IRT tasks, skipping VP duplicates."""
    unknowns = exec_conf.get("critical_unknowns") or []
    tasks: list[dict] = []
    for item in unknowns:
        text = _extract_text(item)
        if not text:
            continue
        norm = _normalize_text(text)
        if norm in seen_text:
            continue
        seen_text.add(norm)
        tasks.append({
            "source_type": "executive_confidence",
            "source_id": "critical_unknown",
            "task_title": text[:120],
            "research_objective": text,
            "why_it_matters": (
                "Identified as a critical unknown by ExecutiveConfidenceAgent; "
                "resolution directly improves decision readiness."
            ),
            "expected_confidence_gain": "HIGH",
            "urgency": "MEDIUM",
            "evidence_needed": [f"Research resolving unknown: {text[:100]}"],
            "suggested_queries": [text[:100]],
            "related_assumption_ids": [],
            "related_risk_ids": [],
            "related_recommendation_ids": [],
            "related_option_ids": [],
        })
    return tasks


# ---------------------------------------------------------------------------
# Heuristic 3 — Low-confidence Critical assumptions → HIGH gain tasks
# ---------------------------------------------------------------------------

def _tasks_from_low_confidence_assumptions(
    assumptions: list[dict],
    seen_assumption_ids: set,
) -> list[dict]:
    """Generate HIGH-gain tasks for Critical assumptions with Low confidence."""
    tasks: list[dict] = []
    for a in assumptions:
        if (a.get("importance") or "").strip() != "Critical":
            continue
        if (a.get("confidence") or "").strip() != "Low":
            continue
        aid = (a.get("assumption_id") or "").strip()
        if aid and aid in seen_assumption_ids:
            continue
        if aid:
            seen_assumption_ids.add(aid)
        statement = (a.get("statement") or a.get("title") or aid or "").strip()
        no_evidence = (
            (a.get("evidence_support") or "").strip() == "None" or
            len(a.get("evidence_ids") or []) == 0
        )
        urgency = "HIGH" if no_evidence else "MEDIUM"
        tasks.append({
            "source_type": "assumption",
            "source_id": aid or "unknown",
            "task_title": f"Validate critical assumption: {statement[:80]}",
            "research_objective": (
                f"Gather empirical evidence to increase confidence in assumption "
                f"{aid}: {statement}"
            ),
            "why_it_matters": (
                f"Assumption {aid} is Critical with Low confidence"
                + (" and no supporting evidence" if no_evidence else "")
                + ". Decision quality depends on resolving this uncertainty."
            ),
            "expected_confidence_gain": "HIGH",
            "urgency": urgency,
            "evidence_needed": [f"Empirical data validating: {statement[:100]}"],
            "suggested_queries": [
                statement[:80],
                f"{aid} validation evidence" if aid else statement[:60],
            ],
            "related_assumption_ids": [aid] if aid else [],
            "related_risk_ids": [],
            "related_recommendation_ids": list(a.get("supported_recommendation_ids") or []),
            "related_option_ids": [],
        })
    return tasks


# ---------------------------------------------------------------------------
# Heuristic 4 — Critical/Important assumptions with no evidence → MEDIUM gain
# ---------------------------------------------------------------------------

def _tasks_from_no_evidence_assumptions(
    assumptions: list[dict],
    seen_assumption_ids: set,
) -> list[dict]:
    """Generate MEDIUM-gain tasks for Critical/Important assumptions lacking evidence."""
    tasks: list[dict] = []
    for a in assumptions:
        imp = (a.get("importance") or "").strip()
        if imp not in ("Critical", "Important"):
            continue
        no_evidence = (
            (a.get("evidence_support") or "").strip() == "None" or
            len(a.get("evidence_ids") or []) == 0
        )
        if not no_evidence:
            continue
        aid = (a.get("assumption_id") or "").strip()
        if aid and aid in seen_assumption_ids:
            continue
        if aid:
            seen_assumption_ids.add(aid)
        statement = (a.get("statement") or a.get("title") or aid or "").strip()
        urgency = "HIGH" if imp == "Critical" else "MEDIUM"
        tasks.append({
            "source_type": "assumption",
            "source_id": aid or "unknown",
            "task_title": f"Provide evidence for {imp.lower()} assumption: {statement[:70]}",
            "research_objective": (
                f"Locate or generate at least one piece of supporting evidence for "
                f"assumption {aid}: {statement}"
            ),
            "why_it_matters": (
                f"Assumption {aid} ({imp}) has no supporting evidence. "
                "Without evidence this assumption remains unsubstantiated and "
                "exposes the recommendation to challenge."
            ),
            "expected_confidence_gain": "MEDIUM",
            "urgency": urgency,
            "evidence_needed": [f"At least one supporting source for: {statement[:100]}"],
            "suggested_queries": [
                statement[:80],
                f"{imp.lower()} assumption evidence {aid}" if aid else statement[:60],
            ],
            "related_assumption_ids": [aid] if aid else [],
            "related_risk_ids": [],
            "related_recommendation_ids": list(a.get("supported_recommendation_ids") or []),
            "related_option_ids": [],
        })
    return tasks


# ---------------------------------------------------------------------------
# Heuristic 5 — High-severity risks with weak/no evidence → MEDIUM gain
# ---------------------------------------------------------------------------

def _tasks_from_high_severity_risks(
    risks: list[dict],
    seen_risk_ids: set,
) -> list[dict]:
    """Generate MEDIUM-gain tasks for High-severity risks lacking strong evidence."""
    tasks: list[dict] = []
    for r in risks:
        if (r.get("severity") or "").strip() != "High":
            continue
        weak = (
            (r.get("evidence_support") or "").strip() in ("Weak", "None") or
            len(r.get("evidence_ids") or []) == 0
        )
        if not weak:
            continue
        rid = (r.get("risk_id") or "").strip()
        if rid and rid in seen_risk_ids:
            continue
        if rid:
            seen_risk_ids.add(rid)
        title = (r.get("title") or r.get("description") or rid or "").strip()
        no_evidence = (
            (r.get("evidence_support") or "").strip() == "None" or
            len(r.get("evidence_ids") or []) == 0
        )
        urgency = "HIGH" if no_evidence else "MEDIUM"
        tasks.append({
            "source_type": "risk",
            "source_id": rid or "unknown",
            "task_title": f"Assess high-severity risk: {title[:80]}",
            "research_objective": (
                f"Gather evidence to assess the probability and impact of risk {rid}: {title}"
            ),
            "why_it_matters": (
                f"Risk {rid} is rated High severity"
                + (" with no supporting evidence" if no_evidence else " with only weak evidence")
                + ". Unverified high-severity risks undermine decision credibility."
            ),
            "expected_confidence_gain": "MEDIUM",
            "urgency": urgency,
            "evidence_needed": [
                f"Probability assessment for: {title[:100]}",
                f"Mitigation pathways for: {title[:100]}",
            ],
            "suggested_queries": [
                f"{title[:80]} risk probability",
                f"{title[:80]} mitigation evidence",
            ],
            "related_assumption_ids": list(r.get("related_assumption_ids") or []),
            "related_risk_ids": [rid] if rid else [],
            "related_recommendation_ids": [],
            "related_option_ids": [],
        })
    return tasks


# ---------------------------------------------------------------------------
# Heuristic 6 — Recommended option fragility
# ---------------------------------------------------------------------------

def _tasks_from_recommended_option(
    decision_analysis: dict,
    strategic_options: list[dict],
    assumptions_by_id: dict,
    seen_option_ids: set,
) -> list[dict]:
    """Generate a HIGH/HIGH task when the recommended option has weak assumption dependencies."""
    rec_id = (decision_analysis.get("recommended_option_id") or "").strip()
    if not rec_id:
        for o in strategic_options:
            if o.get("recommended"):
                rec_id = (o.get("option_id") or "").strip()
                break
    if not rec_id:
        return []
    if rec_id in seen_option_ids:
        return []

    rec_option = next(
        (o for o in strategic_options if (o.get("option_id") or "").strip() == rec_id),
        None,
    )
    if rec_option is None:
        return []

    dep_ids: list[str] = list(
        rec_option.get("supporting_assumption_ids") or
        rec_option.get("assumption_ids") or
        []
    )
    weak_deps: list[str] = []
    for aid in dep_ids:
        a = assumptions_by_id.get(aid)
        if a is None:
            continue
        if (
            len(a.get("evidence_ids") or []) == 0 or
            (a.get("confidence") or "").strip() == "Low"
        ):
            weak_deps.append(aid)

    if not weak_deps:
        return []

    seen_option_ids.add(rec_id)
    option_title = (rec_option.get("title") or rec_option.get("name") or rec_id).strip()
    return [{
        "source_type": "strategic_option",
        "source_id": rec_id,
        "task_title": f"Strengthen evidence base for recommended option {rec_id}",
        "research_objective": (
            f"Validate the supporting assumptions for recommended option {rec_id} "
            f"({option_title}) that currently have low confidence or no evidence: "
            f"{', '.join(weak_deps)}"
        ),
        "why_it_matters": (
            f"Recommended option {rec_id} depends on assumption(s) "
            f"{', '.join(weak_deps)} that are weakly supported. "
            "Fragile foundations expose the recommendation to reversal."
        ),
        "expected_confidence_gain": "HIGH",
        "urgency": "HIGH",
        "evidence_needed": [
            f"Validated evidence for each of: {', '.join(weak_deps[:3])}",
        ],
        "suggested_queries": [
            f"{option_title} supporting evidence",
            f"{', '.join(weak_deps[:2])} validation" if weak_deps else option_title,
        ],
        "related_assumption_ids": weak_deps,
        "related_risk_ids": [],
        "related_recommendation_ids": [],
        "related_option_ids": [rec_id],
    }]


# ---------------------------------------------------------------------------
# Heuristic 7 — Research gap followups (lowest priority)
# ---------------------------------------------------------------------------

def _tasks_from_research_gap_followups(
    research_gap_analysis: dict,
    seen_text: set,
) -> list[dict]:
    """Convert research_gap_analysis.recommended_followups into LOW-priority IRT tasks."""
    followups = research_gap_analysis.get("recommended_followups") or []
    tasks: list[dict] = []
    for item in followups:
        text = _extract_text(item)
        if not text:
            continue
        norm = _normalize_text(text)
        if norm in seen_text:
            continue
        seen_text.add(norm)
        tasks.append({
            "source_type": "research_gap",
            "source_id": "research_gap_followup",
            "task_title": text[:120],
            "research_objective": text,
            "why_it_matters": (
                "Flagged as a recommended followup by ResearchGapAgent; "
                "completing it reduces overall research coverage gaps."
            ),
            "expected_confidence_gain": "LOW",
            "urgency": "LOW",
            "evidence_needed": [f"Coverage data for: {text[:100]}"],
            "suggested_queries": [text[:100]],
            "related_assumption_ids": [],
            "related_risk_ids": [],
            "related_recommendation_ids": [],
            "related_option_ids": [],
        })
    return tasks


# ---------------------------------------------------------------------------
# Plan-level computations
# ---------------------------------------------------------------------------

def _iteration_needed(
    exec_conf: dict,
    assumptions: list[dict],
    strategic_options: list[dict],
    decision_analysis: dict,
) -> tuple[bool, str]:
    reasons: list[str] = []

    overall = (exec_conf.get("overall_confidence") or "").strip()
    if overall in ("Low", "Medium"):
        reasons.append(f"Executive confidence is {overall}.")

    readiness = (exec_conf.get("decision_readiness") or "").lower()
    if "needs additional validation" in readiness or "not ready" in readiness:
        reasons.append("Decision readiness indicates additional validation is needed.")

    vps = exec_conf.get("validation_priorities") or []
    if vps:
        label = "priority" if len(vps) == 1 else "priorities"
        reasons.append(f"{len(vps)} validation {label} remain unresolved.")

    cus = exec_conf.get("critical_unknowns") or []
    if cus:
        label = "unknown" if len(cus) == 1 else "unknowns"
        reasons.append(f"{len(cus)} critical {label} identified.")

    for a in assumptions:
        if (
            (a.get("importance") or "").strip() == "Critical" and
            (a.get("confidence") or "").strip() == "Low"
        ):
            reasons.append("At least one Critical assumption has Low confidence.")
            break

    for a in assumptions:
        imp = (a.get("importance") or "").strip()
        if imp in ("Critical", "Important") and (
            (a.get("evidence_support") or "").strip() == "None" or
            len(a.get("evidence_ids") or []) == 0
        ):
            reasons.append(
                "At least one Critical or Important assumption has no supporting evidence."
            )
            break

    # Check recommended option dependency on weak assumptions
    rec_id = (decision_analysis.get("recommended_option_id") or "").strip()
    if not rec_id:
        for o in strategic_options:
            if o.get("recommended"):
                rec_id = (o.get("option_id") or "").strip()
                break
    if rec_id:
        rec_opt = next(
            (o for o in strategic_options if (o.get("option_id") or "").strip() == rec_id),
            None,
        )
        if rec_opt:
            deps = (
                rec_opt.get("supporting_assumption_ids") or
                rec_opt.get("assumption_ids") or []
            )
            aid_to_a = {
                a.get("assumption_id"): a
                for a in assumptions
                if a.get("assumption_id")
            }
            for aid in deps:
                dep_a = aid_to_a.get(aid)
                if dep_a and (
                    len(dep_a.get("evidence_ids") or []) == 0 or
                    (dep_a.get("confidence") or "").strip() == "Low"
                ):
                    reasons.append(
                        f"Recommended option {rec_id} depends on weakly supported assumptions."
                    )
                    break

    if reasons:
        return True, " ".join(reasons)
    return False, (
        "All critical assumptions are adequately supported and executive confidence is High."
    )


def _stop_conditions(
    exec_conf: dict,
    assumptions: list[dict],
    decision_analysis: dict,
    strategic_options: list[dict],
) -> list[str]:
    conditions: list[str] = []

    overall = (exec_conf.get("overall_confidence") or "").strip()
    if overall in ("Low", "Medium"):
        conditions.append(f"Executive confidence improves from {overall} to High.")

    low_conf_critical = [
        a.get("assumption_id")
        for a in assumptions
        if (
            (a.get("importance") or "").strip() == "Critical" and
            (a.get("confidence") or "").strip() == "Low" and
            a.get("assumption_id")
        )
    ]
    if low_conf_critical:
        ids = ", ".join(low_conf_critical)
        conditions.append(
            f"Critical assumption(s) {ids} achieve at least Medium confidence."
        )

    no_ev = [
        a.get("assumption_id")
        for a in assumptions
        if (
            (a.get("importance") or "").strip() in ("Critical", "Important") and
            (
                (a.get("evidence_support") or "").strip() == "None" or
                len(a.get("evidence_ids") or []) == 0
            ) and
            a.get("assumption_id")
        )
    ]
    if no_ev:
        ids = ", ".join(no_ev)
        conditions.append(
            f"Assumption(s) {ids} obtain at least Weak supporting evidence."
        )

    vps = exec_conf.get("validation_priorities") or []
    if vps:
        conditions.append(
            "All validation priorities identified by ExecutiveConfidenceAgent are resolved."
        )

    rec_id = (decision_analysis.get("recommended_option_id") or "").strip()
    if not rec_id:
        for o in strategic_options:
            if o.get("recommended"):
                rec_id = (o.get("option_id") or "").strip()
                break
    if rec_id:
        conditions.append(
            f"Recommended option {rec_id} no longer depends on assumptions "
            "with no supporting evidence."
        )

    if not conditions:
        conditions.append(
            "Current research coverage and decision confidence are sufficient for commitment."
        )

    return conditions


def _expected_confidence_after_completion(
    exec_conf: dict,
    assumptions: list[dict],
    tasks: list[dict],
) -> str:
    overall = (exec_conf.get("overall_confidence") or "").strip()

    task_assumption_ids: set[str] = set()
    for t in tasks:
        for aid in (t.get("related_assumption_ids") or []):
            if aid:
                task_assumption_ids.add(aid)

    low_conf_critical = [
        a for a in assumptions
        if (
            (a.get("importance") or "").strip() == "Critical" and
            (a.get("confidence") or "").strip() == "Low"
        )
    ]
    no_ev_important = [
        a for a in assumptions
        if (
            (a.get("importance") or "").strip() in ("Critical", "Important") and
            (
                (a.get("evidence_support") or "").strip() == "None" or
                len(a.get("evidence_ids") or []) == 0
            )
        )
    ]

    low_conf_addressed = all(
        (a.get("assumption_id") or "") in task_assumption_ids
        for a in low_conf_critical
    )
    no_ev_addressed = all(
        (a.get("assumption_id") or "") in task_assumption_ids
        for a in no_ev_important
    )

    if overall == "High":
        return "High"
    if overall == "Low" and low_conf_addressed:
        return "Medium"
    if overall == "Medium" and low_conf_addressed and no_ev_addressed:
        return "High"
    return "Unknown"


def _plan_confidence(
    exec_conf: dict,
    assumptions: list[dict],
    risks: list[dict],
    strategic_options: list[dict],
    tasks: list[dict],
) -> float:
    score = 0.3
    if exec_conf:
        score += 0.3
    if assumptions:
        score += 0.15
    if risks:
        score += 0.1
    if strategic_options:
        score += 0.05
    if tasks:
        score += 0.1
    return round(min(1.0, score), 2)


# ---------------------------------------------------------------------------
# Main computation (pure; no context dependency; fully testable)
# ---------------------------------------------------------------------------

def compute_iteration_plan(
    *,
    exec_conf: dict,
    assumptions: list[dict],
    recommendations: list[dict],
    risks: list[dict],
    strategic_options: list[dict],
    decision_analysis: dict,
    research_gap_analysis: dict,
    max_tasks: int = _MAX_TASKS,
) -> dict:
    """Produce the full iteration_plan artifact from decision-quality diagnostics.

    All heuristics are deterministic pure-Python. No LLM calls.
    """
    seen_text: set[str] = set()
    seen_assumption_ids: set[str] = set()
    seen_risk_ids: set[str] = set()
    seen_option_ids: set[str] = set()

    assumptions_by_id: dict[str, dict] = {
        a.get("assumption_id"): a
        for a in assumptions
        if a.get("assumption_id")
    }

    tasks: list[dict] = []

    # Heuristic 1 — Executive validation priorities (HIGH/HIGH)
    tasks.extend(_tasks_from_validation_priorities(exec_conf, seen_text))

    # Heuristic 2 — Critical unknowns (HIGH/MEDIUM, dedup against H1)
    tasks.extend(_tasks_from_critical_unknowns(exec_conf, seen_text))

    # Heuristic 3 — Low-confidence Critical assumptions (HIGH gain)
    tasks.extend(_tasks_from_low_confidence_assumptions(assumptions, seen_assumption_ids))

    # Heuristic 4 — No-evidence Critical/Important assumptions (MEDIUM gain)
    tasks.extend(_tasks_from_no_evidence_assumptions(assumptions, seen_assumption_ids))

    # Heuristic 5 — High-severity weak-evidence risks (MEDIUM gain)
    tasks.extend(_tasks_from_high_severity_risks(risks, seen_risk_ids))

    # Heuristic 6 — Recommended option fragility (HIGH/HIGH)
    tasks.extend(
        _tasks_from_recommended_option(
            decision_analysis, strategic_options, assumptions_by_id, seen_option_ids
        )
    )

    # Heuristic 7 — Research gap followups (LOW gain, dedup by text)
    tasks.extend(_tasks_from_research_gap_followups(research_gap_analysis, seen_text))

    tasks = _sorted_tasks(tasks)[:max_tasks]
    tasks = [_enrich_task_ids(t) for t in tasks]   # J12.2a: linkage completeness
    tasks = _assign_task_ids(tasks)

    needed, reason = _iteration_needed(
        exec_conf, assumptions, strategic_options, decision_analysis
    )
    stop_conds = _stop_conditions(exec_conf, assumptions, decision_analysis, strategic_options)
    exp_conf = _expected_confidence_after_completion(exec_conf, assumptions, tasks)
    confidence = _plan_confidence(exec_conf, assumptions, risks, strategic_options, tasks)

    plan: dict[str, Any] = {
        "iteration_needed": needed,
        "iteration_reason": reason,
        "priority_research_tasks": tasks,
        "stop_conditions": stop_conds,
        "expected_confidence_after_completion": exp_conf,
        "plan_confidence": confidence,
    }

    # J12.2a: deterministic validation
    known_assumption_ids: set[str] = set(assumptions_by_id.keys())
    known_risk_ids: set[str] = {
        (r.get("risk_id") or "").strip() for r in risks if r.get("risk_id")
    } - {""}
    known_option_ids: set[str] = {
        (o.get("option_id") or "").strip() for o in strategic_options if o.get("option_id")
    } - {""}
    plan["validation_warnings"] = _validate_plan(
        plan,
        known_assumption_ids=known_assumption_ids,
        known_risk_ids=known_risk_ids,
        known_option_ids=known_option_ids,
    )

    return plan


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class IterationPlanAgent(FunctionalAgent):
    """Deterministic iteration planning (J12.2).

    Converts ExecutiveConfidence validation priorities, critical unknowns,
    low-confidence assumptions, high-severity risks, recommended-option
    fragility, and ResearchGap followups into a ranked list of IRT tasks.
    """

    def _execute(self, context: AgentContext) -> AgentContext:
        from research_agent.log import PROGRESS

        ro = context.research_object or {}

        exec_conf = (
            context.executive_confidence or
            ro.get("executive_confidence") or
            {}
        )
        assumptions = (
            context.assumptions or
            ro.get("strategic_assumptions") or
            []
        )
        recommendations = (
            context.recommendations or
            ro.get("recommendations") or
            []
        )
        risks = (
            context.risks or
            ro.get("strategic_risks") or
            ro.get("risks") or
            []
        )
        strategic_options = (
            context.strategic_options or
            ro.get("strategic_options") or
            []
        )
        decision_analysis = (
            context.decision_analysis or
            ro.get("decision_analysis") or
            {}
        )
        research_gap_analysis = (
            context.research_gap_analysis or
            ro.get("research_gap_analysis") or
            {}
        )

        plan = compute_iteration_plan(
            exec_conf=exec_conf,
            assumptions=assumptions,
            recommendations=recommendations,
            risks=risks,
            strategic_options=strategic_options,
            decision_analysis=decision_analysis,
            research_gap_analysis=research_gap_analysis,
        )

        tasks = plan["priority_research_tasks"]
        high_priority_tasks = sum(
            1 for t in tasks
            if t.get("expected_confidence_gain") == "HIGH" and t.get("urgency") == "HIGH"
        )
        critical_assumptions_count = sum(
            1 for a in assumptions
            if (a.get("importance") or "").strip() == "Critical" and
               (a.get("confidence") or "").strip() == "Low"
        )
        validation_priorities_count = len(exec_conf.get("validation_priorities") or [])
        critical_unknowns_count = len(exec_conf.get("critical_unknowns") or [])
        exp_conf_after = plan["expected_confidence_after_completion"]
        warn_count = len(plan.get("validation_warnings") or [])

        context.iteration_plan = plan
        ro["iteration_plan"] = plan
        context.trace["_iteration_plan"] = {
            "iteration_needed": plan["iteration_needed"],
            "task_count": len(tasks),
            "plan_confidence": plan["plan_confidence"],
            "high_priority_tasks": high_priority_tasks,
            "critical_assumptions": critical_assumptions_count,
            "validation_priorities": validation_priorities_count,
            "critical_unknowns": critical_unknowns_count,
            "expected_confidence_after_completion": exp_conf_after,
            "validation_warning_count": warn_count,
        }

        if warn_count:
            LOGGER.warning(
                "[IterationPlanAgent] %d validation warning(s): %s",
                warn_count,
                "; ".join(plan["validation_warnings"][:3]),
            )

        LOGGER.log(
            PROGRESS,
            "[IterationPlanAgent] iteration_needed=%s  tasks=%d  high_priority=%d  "
            "plan_confidence=%.2f",
            plan["iteration_needed"],
            len(tasks),
            high_priority_tasks,
            plan["plan_confidence"],
        )

        self._record(
            context,
            status="success",
            summary=(
                f"Iteration plan: needed={plan['iteration_needed']}  "
                f"tasks={len(tasks)}  "
                f"confidence={plan['plan_confidence']}"
            ),
            next_action=NextAction.CONTINUE,
            iteration_needed=plan["iteration_needed"],
            task_count=len(tasks),
            plan_confidence=plan["plan_confidence"],
            high_priority_tasks=high_priority_tasks,
            critical_assumptions=critical_assumptions_count,
            validation_priorities=validation_priorities_count,
            critical_unknowns=critical_unknowns_count,
            expected_confidence_after_completion=exp_conf_after,
            validation_warning_count=warn_count,
        )
        return context

    def _extract_outputs(self, context: AgentContext) -> dict:
        return {"iteration_plan": context.iteration_plan}
