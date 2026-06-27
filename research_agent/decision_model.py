"""Decision Model v2 – canonical representation of the decision being made (J7.0b).

Sits between a Strategic Engagement and one or more Research Objects:

    Strategic Engagement
        ↓
    Decision Model v2  ←  describes WHAT is being decided, WHY, and HOW success is judged
        ↓
    Research Object(s)
        ↓
    Executive Decision Report  (future milestone)

The Decision Model is produced by ProblemFramingAgent in goal-driven runs, or
auto-created as a minimal "general" instance in question-driven/benchmark runs
for backward compatibility.

Objects are written to outputs/decision_models/<decision_model_id>.json and
mirrored to outputs/latest_decision_model.json.

Fields intentionally left for future milestones (not present here):
  - assumptions
  - risks
  - opportunities
  - confidence_levels
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

DecisionType = Literal[
    "investment",       # Should we invest / acquire / divest?
    "policy",           # Should we adopt / change a policy or regulation?
    "strategic_plan",   # What strategy should we pursue?
    "risk_assessment",  # What risks do we face and how do we mitigate them?
    "market_entry",     # Should we / how do we enter a new market?
    "regulatory",       # How do we respond to a regulatory change?
    "research",         # Pure knowledge / understanding objective (no action yet)
    "general",          # Default for question-driven / backward-compat runs
]


class DecisionCriterion(BaseModel):
    """A single criterion used to judge whether the decision is correct."""
    name: str
    description: str
    weight: str = "medium"          # "high" | "medium" | "low" — extensible later


class DecisionObjective(BaseModel):
    """A specific, measurable objective the decision must achieve."""
    objective: str
    rationale: str = ""


# ---------------------------------------------------------------------------
# DecisionAssumption (J7.1)
# ---------------------------------------------------------------------------

AssumptionCategory = Literal[
    "Technology", "Market", "Economics", "Regulation", "Policy",
    "Supply Chain", "Competition", "Customer", "Execution", "Geopolitics",
    "Environment", "Infrastructure", "Finance", "Other",
]

AssumptionImportance = Literal["Critical", "Important", "Supporting"]
AssumptionEvidenceSupport = Literal["Strong", "Moderate", "Weak", "None"]
AssumptionConfidence = Literal["High", "Medium", "Low"]
AssumptionStatus = Literal["Active", "Validated", "Invalidated"]


class DecisionAssumption(BaseModel):
    """A first-class strategic assumption in the Decision Model (J7.1).

    Represents something that must be true for a recommendation to remain valid.
    Populated by AssumptionAgent after Challenge and before Recommendation.
    """

    assumption_id: str
    statement: str                              # what must be true
    category: AssumptionCategory = "Other"
    importance: AssumptionImportance = "Important"
    evidence_support: AssumptionEvidenceSupport = "Moderate"
    confidence: AssumptionConfidence = "Medium"
    rationale: str = ""                         # why this assumption matters
    evidence_ids: list[str] = Field(default_factory=list)
    supported_recommendation_ids: list[str] = Field(default_factory=list)
    status: AssumptionStatus = "Active"
    conflicts_with: list[str] = Field(default_factory=list)  # other assumption_ids


class DecisionModel(BaseModel):
    """Decision Model v2 — canonical object describing the decision (J7.0b).

    Required fields capture the decision intent.
    Optional lists provide structure that guides research and evaluation.
    All list fields default to empty so callers can populate incrementally.
    """

    # --- Identity -----------------------------------------------------------
    decision_model_id: str
    created_at: str

    # --- Linkage (nullable, backward-compatible) ----------------------------
    engagement_id: str | None = None

    # --- Decision intent (required) ----------------------------------------
    strategic_question: str              # the core question being answered
    decision_type: DecisionType = "general"

    # --- Decision structure (populated by ProblemFramingAgent or caller) ----
    objectives: list[DecisionObjective] = Field(default_factory=list)
    decision_criteria: list[DecisionCriterion] = Field(default_factory=list)
    investigation_areas: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)

    # --- Quality bar --------------------------------------------------------
    required_confidence: str = "medium"   # "high" | "medium" | "low"

    # --- Strategic Assumptions (J7.1) — populated by AssumptionAgent --------
    strategic_assumptions: list[DecisionAssumption] = Field(default_factory=list)

    # --- Source traceability ------------------------------------------------
    source: str = "auto"    # "problem_framing_agent" | "question_driven" | "auto"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

def _new_dm_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"DM-{ts}"


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def create_decision_model(
    *,
    strategic_question: str,
    decision_type: DecisionType = "general",
    engagement_id: str | None = None,
    objectives: list[dict] | None = None,
    decision_criteria: list[dict] | None = None,
    investigation_areas: list[str] | None = None,
    alternatives: list[str] | None = None,
    constraints: list[str] | None = None,
    out_of_scope: list[str] | None = None,
    required_confidence: str = "medium",
    source: str = "auto",
) -> DecisionModel:
    """Create a new DecisionModel with a generated ID and timestamp."""
    return DecisionModel(
        decision_model_id=_new_dm_id(),
        created_at=datetime.now(timezone.utc).isoformat(),
        engagement_id=engagement_id,
        strategic_question=strategic_question,
        decision_type=decision_type,
        objectives=[DecisionObjective(**o) if isinstance(o, dict) else o for o in (objectives or [])],
        decision_criteria=[DecisionCriterion(**c) if isinstance(c, dict) else c for c in (decision_criteria or [])],
        investigation_areas=investigation_areas or [],
        alternatives=alternatives or [],
        constraints=constraints or [],
        out_of_scope=out_of_scope or [],
        required_confidence=required_confidence,
        source=source,
    )


def from_question(
    question: str,
    *,
    engagement_id: str | None = None,
) -> DecisionModel:
    """Create a minimal DecisionModel from a bare research question.

    Used by CLI and benchmark runs for backward compatibility.  No LLM call;
    no behavioural change to any downstream agent.
    """
    return create_decision_model(
        strategic_question=question,
        decision_type="general",
        engagement_id=engagement_id,
        source="question_driven",
    )


def from_framing_payload(
    payload: Any,
    *,
    strategic_question: str,
    engagement_id: str | None = None,
) -> DecisionModel:
    """Build a DecisionModel v2 from a DecisionModelPayload (ProblemFramingAgent output).

    Maps the existing v1 fields into the v2 schema without any information loss.
    """
    # payload is a DecisionModelPayload Pydantic model
    objectives = [
        DecisionObjective(objective=payload.objective, rationale="")
    ]
    criteria = [
        DecisionCriterion(name=u, description=u, weight="high")
        for u in (payload.critical_uncertainties or [])
    ]
    investigation_areas = list(payload.decision_areas or [])

    return create_decision_model(
        strategic_question=strategic_question,
        decision_type="research",
        engagement_id=engagement_id,
        objectives=[o.model_dump() for o in objectives],
        decision_criteria=[c.model_dump() for c in criteria],
        investigation_areas=investigation_areas,
        required_confidence="medium",
        source="problem_framing_agent",
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

_DM_DIR = Path("outputs/decision_models")
_LATEST_PATH = Path("outputs/latest_decision_model.json")


def write_decision_model(
    dm: DecisionModel,
    base: Path = Path("outputs"),
    *,
    write_latest: bool = True,
) -> Path:
    """Persist the decision model to disk.

    write_latest=True (default): also updates latest_decision_model.json.
    Pass write_latest=False for minimal auto-created DMs (simple CLI /
    benchmark path) so they do not overwrite a richer functional-pipeline DM.
    """
    dm_dir = base / "decision_models"
    dm_dir.mkdir(parents=True, exist_ok=True)
    path = dm_dir / f"{dm.decision_model_id}.json"
    data = dm.to_dict()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    if write_latest:
        latest = base / "latest_decision_model.json"
        latest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_decision_model(decision_model_id: str, base: Path = Path("outputs")) -> DecisionModel:
    """Load a persisted decision model by ID."""
    path = base / "decision_models" / f"{decision_model_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return DecisionModel.model_validate(data)
