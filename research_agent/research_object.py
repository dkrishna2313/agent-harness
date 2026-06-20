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
    profile_source: str = "unset",
    sources_dir: str | Path | None = None,
    web_search: bool = False,
    mock_mode: bool = False,
    model_name: str | None = None,
    question_id: str | None = None,
) -> dict[str, Any]:
    """Create a new research object at the start of a run.

    Parameters
    ----------
    profile_name:
        The profile to record.  Must come from the caller's execution context —
        never inferred inside this function.
    profile_source:
        Where the profile value originated: "cli_argument", "benchmark_mapping",
        or "unset".
    """
    now = datetime.now(timezone.utc).isoformat()
    research_id = _new_research_id(question_id)
    return {
        "research_id": research_id,
        "created_at": now,
        "updated_at": now,
        "status": "running",
        # Core research intent
        "question": question,
        "profile": profile_name,
        "profile_source": profile_source,
        # Run configuration
        "run_config": {
            "sources_dir": str(sources_dir) if sources_dir else None,
            "web_search": web_search,
            "mock_mode": mock_mode,
            "model_name": model_name,
        },
        # Populated during/after run
        "research_type": None,
        "subquestions": [],
        "investigation_areas": [],
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
) -> Path:
    """Write the research object to disk and update latest_research_object.json."""
    base = Path(out_dir) if out_dir else _RO_DIR.parent
    ro_dir = base / "research_objects"
    ro_dir.mkdir(parents=True, exist_ok=True)

    research_id = obj["research_id"]
    ro_path = ro_dir / f"{research_id}.json"
    ro_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

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
    """Return the compact trace fragment (J4.5.5 / J4.5a.6 / J4.5b.5 / J4.5c.7)."""
    errors = validate_research_object(obj)
    pv = obj.get("profile_validation", {})
    profile = obj.get("profile")
    profile_source = obj.get("profile_source", "unset")
    profile_valid = pv.get("valid", profile is not None)
    return {
        "research_id": obj["research_id"],
        "path": str(ro_path),
        "status": obj.get("status", "completed"),
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
    }
