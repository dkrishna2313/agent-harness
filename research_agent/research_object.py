"""Research Object – durable representation of a research request (J4.5).

Each CLI run or benchmark question produces a ResearchObject that captures
the full lifecycle: question → retrieval → evidence → findings → answer.

Objects are written to outputs/research_objects/<research_id>.json and
mirrored to outputs/latest_research_object.json.

Profile assignment (J4.5c) follows a single canonical flow:
  CLI with --profile      → profile_source = "cli_argument"
  Benchmark domain rule   → profile_source = "benchmark_mapping"
  CLI without --profile   → profile_source = "unset"
The profile field is frozen at creation; update_research_object never
re-derives or overwrites it.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import ResearchMemo


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

def _new_research_id(question_id: str | None = None) -> str:
    """Return a stable research object ID."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if question_id:
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", question_id)[:32]
        return f"R-{safe}"
    return f"R-{ts}"


# ---------------------------------------------------------------------------
# Domain → profile mapping  (J4.5c.3)
# ---------------------------------------------------------------------------

_DOMAIN_TO_PROFILE: dict[str, str] = {
    "nvidia": "ai_data_centers",
    "ai_dc": "ai_data_centers",
    "smr": "smr",
    "nuclear": "smr",
}


def _classify_research_type(question: str) -> str:
    """Heuristic question classification — no LLM required.

    PlannerAgent overrides this with an LLM-generated value in functional_agents runs.
    Benchmark runs use this heuristic so ROs are never left with research_type=null.
    """
    q = question.lower()
    if any(w in q for w in ("compare", " vs ", "versus", "difference between", "contrast")):
        return "COMPARISON"
    if any(w in q for w in ("why ", "how does", "how do", "explain", "what causes", "reason for")):
        return "EXPLANATION"
    if any(
        q.startswith(p)
        for p in ("what is ", "what are ", "how many ", "how much ", "list ", "name ", "when ")
    ) or any(w in q for w in ("output capacity", "power output", "how many", "how much", "what is the")):
        return "FACT_LOOKUP"
    return "RESEARCH"


def _heuristic_subquestions(question: str, research_type: str) -> list[str]:
    """Generate basic subquestions from the question text (no LLM).

    PlannerAgent replaces these with LLM-generated subquestions in functional_agents runs.
    """
    if research_type == "FACT_LOOKUP":
        return [
            f"What is the specific answer to: {question}",
            "What sources contain this information?",
            "Are there any caveats or conditions on this fact?",
        ]
    if research_type == "COMPARISON":
        return [
            f"What are the key dimensions for comparing in: {question}",
            "What are the strengths of each option?",
            "What are the limitations of each option?",
            "Which option is preferred under what conditions?",
        ]
    if research_type == "EXPLANATION":
        return [
            f"What is the mechanism behind: {question}",
            "What are the key contributing factors?",
            "What evidence supports this explanation?",
            "Are there alternative explanations?",
        ]
    # RESEARCH
    return [
        f"What are the key facts relevant to: {question}",
        "What evidence exists in the available sources?",
        "What are the main constraints or limitations?",
        "What are the practical implications?",
        "What gaps remain in the available evidence?",
    ]


def _heuristic_investigation_areas(question: str, research_type: str) -> list[str]:
    """Generate basic investigation areas from the question text (no LLM)."""
    q = question.lower()
    areas = ["Overview"]
    for keyword, area in [
        ("power", "Power"),
        ("cool", "Cooling"),
        ("cost", "Economics"),
        ("econom", "Economics"),
        ("deploy", "Deployment Timeline"),
        ("grid", "Grid Integration"),
        ("regulat", "Regulation"),
        ("safety", "Safety"),
        ("nuclear", "Nuclear Technology"),
        ("smr", "SMR Technology"),
        ("data center", "Data Center Requirements"),
        ("nvidia", "NVIDIA Technology"),
        ("gpu", "GPU Architecture"),
    ]:
        if keyword in q and area not in areas:
            areas.append(area)
    areas += ["Key Findings", "Open Questions"]
    return areas


def infer_profile_from_domain(domain: str | None, question_id: str | None = None) -> str | None:
    """Return the canonical profile for a benchmark domain or question_id.

    Uses explicit prefix matching (J4.5d.1) — no dict-key fallbacks.
    Used only by callers that set the profile at creation time (EvaluationRunner).
    Not called inside update_research_object.
    """
    # Explicit question_id prefix rules take priority (most specific)
    if question_id:
        if question_id.startswith("SMR_"):
            return "smr"
        if question_id.startswith("NVIDIA_"):
            return "ai_data_centers"
    # Domain string fallback (for future domains)
    if domain:
        key = domain.lower()
        if key in _DOMAIN_TO_PROFILE:
            return _DOMAIN_TO_PROFILE[key]
    return None


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def create_research_object(
    *,
    question: str,
    profile_name: str | None = None,
    profile_names: list[str] | None = None,
    profile_source: str = "unset",
    sources_dir: str | Path | None = None,
    web_search: bool = False,
    mock_mode: bool = False,
    model_name: str | None = None,
    question_id: str | None = None,
    # J7.0a – strategic engagement linkage (nullable, backward-compatible)
    engagement_id: str | None = None,
    decision_model_id: str | None = None,
) -> dict[str, Any]:
    """Create a new research object at the start of a run.

    Parameters
    ----------
    profile_name:
        The primary/execution profile.  Must come from the caller's execution
        context — never inferred inside this function.
    profile_names:
        All profiles loaded for this run (J6.1a multi-profile model).
        Defaults to [profile_name] when not provided.
    profile_source:
        Where the profile value originated: "cli_argument", "benchmark_mapping",
        or "unset".
    """
    now = datetime.now(timezone.utc).isoformat()
    research_id = _new_research_id(question_id)
    _rtype = _classify_research_type(question)
    # Multi-profile model (J6.1a): profiles[] is the authoritative list.
    _profiles: list[str] = (
        profile_names if profile_names
        else ([profile_name] if profile_name else [])
    )
    _primary = _profiles[0] if _profiles else profile_name
    return {
        "research_id": research_id,
        "created_at": now,
        "updated_at": now,
        "status": "running",
        # Core research intent
        "question": question,
        # Multi-profile model (J6.1a)
        "profiles": _profiles,
        "primary_profile": _primary,
        # Legacy singular field retained for backward compatibility
        "profile": profile_name,
        "profile_source": profile_source,
        # Run configuration
        "run_config": {
            "sources_dir": str(sources_dir) if sources_dir else None,
            "web_search": web_search,
            "mock_mode": mock_mode,
            "model_name": model_name,
        },
        # Populated during/after run — heuristic baseline; PlannerAgent overrides in functional runs
        "research_type": _rtype,
        "subquestions": _heuristic_subquestions(question, _rtype),
        "investigation_areas": _heuristic_investigation_areas(question, _rtype),
        "retrieval_plan": {},
        "evidence_ids": [],
        "evidence_topics": {},
        "findings": [],
        "contradictions": [],
        "research_gaps": [],
        "outputs": {},
        "summary": {
            "evidence_count": 0,
            "citation_count": 0,
            "contradictions_found": 0,
            "research_gaps_found": 0,
            "coverage_score": 0.0,
        },
        # J4.5.9 – future-proofing fields
        "owner": None,
        "assigned_agent": None,
        "review_status": "unreviewed",
        "parent_research_id": None,
        "child_research_ids": [],
        # J7.0a – strategic engagement linkage (null when engagement layer not used)
        "engagement_id": engagement_id,
        "decision_model_id": decision_model_id,
    }


def update_research_object(
    obj: dict[str, Any],
    *,
    memo: ResearchMemo,
    output_path: str | Path | None = None,
    trace_path: str | Path | None = None,
    question_id: str | None = None,
) -> dict[str, Any]:
    """Return an updated copy of *obj* populated from a completed run memo.

    The ``profile`` field is never modified here — it is frozen at creation
    time by the caller (J4.5c.2).  profile_validation reflects the frozen value.
    """
    evidence_items = memo.source_notes or memo.evidence
    contradictions = memo.metadata.get("contradictions", [])
    research_gaps = memo.metadata.get("research_gaps", [])
    coverage_matrix = memo.metadata.get("coverage_matrix", [])

    # Coverage score from coverage matrix
    level_weights = {"strong": 1.0, "moderate": 0.6, "weak": 0.2, "none": 0.0}
    if coverage_matrix:
        scores = [level_weights.get(a.get("coverage_level", "none"), 0.0) for a in coverage_matrix]
        coverage_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    else:
        coverage_score = 0.0

    from collections import Counter
    topic_counts: Counter[str] = Counter(
        topic for item in evidence_items for topic in (item.topics or [])
    )

    findings = [
        f.split(" [Source:")[0].strip()
        for f in (memo.confirmed_facts or [])[:5]
        if f
    ]

    retrieval_plan = {
        k: memo.metadata.get(k)
        for k in ("retrieval_queries", "retrieval_plan_stats", "web_search")
        if memo.metadata.get(k) is not None
    }
    retrieval_plan["source_count"] = memo.metadata.get(
        "documents_loaded", len({item.source_document for item in evidence_items})
    )

    # J4.5c.2 – profile is frozen at creation; validation confirms it is set.
    # No re-inference here.  expected_profile == actual_profile by design when
    # the caller (CLI / EvaluationRunner) sets it correctly.
    recorded_profile = obj.get("profile")
    profile_source = obj.get("profile_source", "unset")
    profile_valid = recorded_profile is not None
    profile_validation = {
        "expected_profile": recorded_profile,
        "actual_profile": recorded_profile,
        "source": profile_source,
        "valid": profile_valid,
    }

    updated: dict[str, Any] = {
        **obj,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "profile_validation": profile_validation,
        "evidence_ids": [item.evidence_id for item in evidence_items if item.evidence_id],
        "evidence_topics": dict(topic_counts),
        "findings": findings,
        "contradictions": [
            {
                "contradiction_id": c.get("contradiction_id", ""),
                "severity": c.get("severity", ""),
                "topic": c.get("topic", ""),
            }
            for c in contradictions[:10]
        ],
        "research_gaps": [
            {
                "gap_id": g.get("gap_id", ""),
                "priority": g.get("priority", ""),
                "topic": g.get("topic", ""),
                "description": g.get("description", "")[:120],
            }
            for g in research_gaps[:10]
        ],
        "retrieval_plan": retrieval_plan,
        "outputs": {},
        "summary": {
            "evidence_count": len(evidence_items),
            "citation_count": memo.metadata.get("citation_count", len(evidence_items)),
            "contradictions_found": len(contradictions),
            "research_gaps_found": len(research_gaps),
            "coverage_score": coverage_score,
        },
    }

    if output_path:
        updated["outputs"]["memo_path"] = str(output_path)
    if trace_path:
        updated["outputs"]["trace_path"] = str(trace_path)
    if question_id:
        updated["outputs"]["question_id"] = question_id

    return updated


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

_RO_DIR = Path("outputs/research_objects")
_LATEST_PATH = Path("outputs/latest_research_object.json")


def write_research_object(
    obj: dict[str, Any],
    *,
    out_dir: str | Path | None = None,
    write_latest: bool = True,
) -> Path:
    """Write the research object to disk.

    write_latest=True (default): also updates latest_research_object.json.
    Pass write_latest=False for benchmark / simple-CLI runs so they do not
    overwrite the canonical latest produced by an interactive functional-pipeline run.
    Mirrors the write_latest semantics of write_decision_model().
    """
    base = Path(out_dir) if out_dir else _RO_DIR.parent
    ro_dir = base / "research_objects"
    ro_dir.mkdir(parents=True, exist_ok=True)

    research_id = obj["research_id"]
    ro_path = ro_dir / f"{research_id}.json"
    ro_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    if write_latest:
        latest_path = base / "latest_research_object.json"
        latest_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    return ro_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = ("research_id", "question", "profile", "status", "created_at", "summary")


def validate_research_object(obj: dict[str, Any]) -> list[str]:
    """Return a list of validation error strings (empty = valid)."""
    errors: list[str] = []
    for field in _REQUIRED_FIELDS:
        if obj.get(field) is None or obj.get(field) == "":
            errors.append(f"missing required field: {field!r}")
    status = obj.get("status", "")
    if status not in ("running", "completed", "failed"):
        errors.append(f"invalid status: {status!r}")
    if not isinstance(obj.get("summary") or {}, dict):
        errors.append("summary must be a dict")
    return errors


# ---------------------------------------------------------------------------
# Trace stub
# ---------------------------------------------------------------------------

def research_object_trace_stub(obj: dict[str, Any], ro_path: Path) -> dict[str, Any]:
    """Return the compact trace fragment (J4.5.5 / J4.5a.6 / J4.5b.5 / J4.5c.7 / J6.1a)."""
    errors = validate_research_object(obj)
    pv = obj.get("profile_validation", {})
    profile = obj.get("profile")
    profile_source = obj.get("profile_source", "unset")
    profile_valid = pv.get("valid", profile is not None)
    profiles = obj.get("profiles", [profile] if profile else [])
    primary_profile = obj.get("primary_profile", profile)
    decision_model = obj.get("decision_model")
    return {
        "research_id": obj["research_id"],
        "path": str(ro_path),
        "status": obj.get("status", "completed"),
        # Multi-profile model (J6.1a)
        "profiles": profiles,
        "primary_profile": primary_profile,
        # Legacy singular field
        "profile": profile,
        "profile_valid": profile_valid,
        "profile_resolution": {
            "question_id": obj.get("outputs", {}).get("question_id"),
            "source": profile_source,
            "resolved_profile": profile,
            "valid": profile_valid,
        },
        "validation_status": "valid" if not errors else "invalid",
        "validation_errors": errors,
        # J6.1a — research object validation block
        "research_object_validation": {
            "decision_model_present": decision_model is not None,
            "profiles_valid": bool(profiles),
        },
    }
