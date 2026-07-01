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

from pydantic import BaseModel, Field, field_validator


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


# ---------------------------------------------------------------------------
# StrategicRisk (J7.3)
# ---------------------------------------------------------------------------

RiskCategory = Literal[
    "Technology", "Market", "Economics", "Regulation", "Policy",
    "Supply Chain", "Competition", "Customer", "Execution", "Geopolitics",
    "Environment", "Infrastructure", "Finance", "Other",
]

RiskSeverity = Literal["High", "Medium", "Low"]
RiskLikelihood = Literal["High", "Medium", "Low"]
RiskEvidenceSupport = Literal["Strong", "Moderate", "Weak", "None"]
RiskConfidence = Literal["High", "Medium", "Low"]
RiskStatus = Literal["Active", "Mitigated", "Retired"]


class StrategicRisk(BaseModel):
    """A first-class strategic risk in the Decision Model (J7.3).

    Describes what could cause a strategic assumption to fail, and which
    recommendations would be affected as a consequence.
    Populated by RiskAgent after Recommendation Linkage.
    """

    risk_id: str                                      # e.g. "RSK-001"
    statement: str                                    # what could go wrong
    category: RiskCategory = "Other"
    severity: RiskSeverity = "Medium"
    likelihood: RiskLikelihood = "Medium"
    evidence_support: RiskEvidenceSupport = "Moderate"
    confidence: RiskConfidence = "Medium"
    rationale: str = ""                               # why this risk matters
    related_assumption_ids: list[str] = Field(default_factory=list)
    affected_recommendation_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    mitigation_notes: str = ""
    status: RiskStatus = "Active"


# ---------------------------------------------------------------------------
# StrategicOpportunity (J7.4)
# ---------------------------------------------------------------------------

OpportunityCategory = Literal[
    "Technology", "Market", "Economics", "Regulation", "Policy",
    "Supply Chain", "Competition", "Customer", "Execution", "Geopolitics",
    "Environment", "Infrastructure", "Finance", "Other",
]

OpportunityImpact = Literal["High", "Medium", "Low"]
OpportunityLikelihood = Literal["High", "Medium", "Low"]
OpportunityEvidenceSupport = Literal["Strong", "Moderate", "Weak", "None"]
OpportunityConfidence = Literal["High", "Medium", "Low"]
OpportunityStatus = Literal["Active", "Realized", "Expired"]


class StrategicOpportunity(BaseModel):
    """A first-class strategic opportunity in the Decision Model (J7.4).

    Describes additional value that could be created when a strategic assumption
    proves more favourable than expected — the positive counterpart to StrategicRisk.
    Populated by OpportunityAgent after RiskAgent.
    """

    opportunity_id: str                                      # e.g. "OPP-001"
    statement: str                                           # what upside becomes possible
    category: OpportunityCategory = "Other"
    impact: OpportunityImpact = "Medium"
    likelihood: OpportunityLikelihood = "Medium"
    evidence_support: OpportunityEvidenceSupport = "Moderate"
    confidence: OpportunityConfidence = "Medium"
    rationale: str = ""                                      # why this opportunity matters
    related_assumption_ids: list[str] = Field(default_factory=list)
    enabled_recommendation_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    exploitation_notes: str = ""
    status: OpportunityStatus = "Active"

    @field_validator("category", mode="before")
    @classmethod
    def coerce_category(cls, v: object) -> object:
        _valid: set[str] = {
            "Technology", "Market", "Economics", "Regulation", "Policy",
            "Supply Chain", "Competition", "Customer", "Execution", "Geopolitics",
            "Environment", "Infrastructure", "Finance", "Other",
        }
        return v if v in _valid else "Other"


# ---------------------------------------------------------------------------
# StrategicOption (J7.5)
# ---------------------------------------------------------------------------

ImplementationComplexity = Literal["Low", "Medium", "High"]
TimeHorizon = Literal["Near-term", "Medium-term", "Long-term"]
CapitalIntensity = Literal["Low", "Medium", "High"]
OptionConfidence = Literal["High", "Medium", "Low"]


class StrategicOption(BaseModel):
    """A first-class strategic option synthesising the full J7 reasoning graph (J7.5).

    Represents a coherent, actionable course of action derived from assumptions,
    risks, opportunities, and recommendations. Exactly one option per run has
    recommended=True — the preferred course of action.
    Populated by StrategicOptionAgent after OpportunityAgent.
    """

    option_id: str                                         # e.g. "OPT-A"
    title: str                                             # short descriptive name
    description: str                                       # what this option entails
    strategic_objective: str                               # what it is trying to achieve
    expected_outcomes: list[str] = Field(default_factory=list)
    supporting_assumption_ids: list[str] = Field(default_factory=list)
    associated_risk_ids: list[str] = Field(default_factory=list)
    associated_opportunity_ids: list[str] = Field(default_factory=list)
    supporting_recommendation_ids: list[str] = Field(default_factory=list)
    advantages: list[str] = Field(default_factory=list)
    disadvantages: list[str] = Field(default_factory=list)
    implementation_complexity: ImplementationComplexity = "Medium"
    estimated_time_horizon: TimeHorizon = "Medium-term"
    capital_intensity: CapitalIntensity = "Medium"
    confidence: OptionConfidence = "Medium"
    recommended: bool = False
    rationale: str = ""                                    # why this option is (or is not) preferred


# ---------------------------------------------------------------------------
# DecisionAnalysis (J7.6)
# ---------------------------------------------------------------------------

ScoreRating = Literal["Very High", "High", "Medium", "Low", "Very Low"]
AnalysisConfidence = Literal["High", "Medium", "Low"]


class DecisionMatrixEntry(BaseModel):
    """Per-option row in the decision matrix (J7.6)."""

    option_id: str
    strategic_fit: ScoreRating = "Medium"
    implementation_risk: ScoreRating = "Medium"
    execution_complexity: ScoreRating = "Medium"
    capital_requirement: ScoreRating = "Medium"
    expected_return: ScoreRating = "Medium"
    time_to_value: ScoreRating = "Medium"
    dependency_strength: ScoreRating = "Medium"
    assumption_strength: ScoreRating = "Medium"
    risk_exposure: ScoreRating = "Medium"
    opportunity_capture: ScoreRating = "Medium"
    overall_score: ScoreRating = "Medium"
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)

    @field_validator(
        "strategic_fit", "implementation_risk", "execution_complexity",
        "capital_requirement", "expected_return", "time_to_value",
        "dependency_strength", "assumption_strength", "risk_exposure",
        "opportunity_capture", "overall_score",
        mode="before",
    )
    @classmethod
    def coerce_score(cls, v: object) -> object:
        _valid = {"Very High", "High", "Medium", "Low", "Very Low"}
        return v if v in _valid else "Medium"


class DecisionAnalysis(BaseModel):
    """Explicit comparison of Strategic Options using the full J7 reasoning graph (J7.6).

    DecisionAnalysis is an explanation, not a recommendation generator.
    It answers 'Why is Option B preferred over Option A?' using the existing graph.
    Populated by DecisionAnalysisAgent after StrategicOptionAgent.
    """

    analysis_id: str                                         # e.g. "DA-20260627-120000"
    recommended_option_id: str                               # must match exactly one StrategicOption
    executive_summary: str                                   # 2-4 sentence plain-English summary
    comparison_dimensions: list[str] = Field(default_factory=list)   # dimensions used in the matrix
    option_rankings: list[str] = Field(default_factory=list)          # option_ids ordered best→worst
    decision_matrix: list[DecisionMatrixEntry] = Field(default_factory=list)
    key_tradeoffs: list[str] = Field(default_factory=list)            # explicit tradeoff statements
    key_uncertainties: list[str] = Field(default_factory=list)        # uncertainties that could shift the choice
    sensitivity_analysis: str = ""                           # which assumption failures would change the winner
    confidence_summary: str = ""                             # overall confidence and limiting factors
    rationale: str = ""                                      # full explanation of why the recommended option wins
    confidence: AnalysisConfidence = "Medium"


# ---------------------------------------------------------------------------
# ExecutiveConfidence (J7.7)
# ---------------------------------------------------------------------------

OverallConfidence = Literal["High", "Medium", "Low"]
DecisionReadiness = Literal["Ready for Decision", "Needs Additional Validation", "Not Ready"]
BoardRecommendation = Literal[
    "Proceed", "Proceed with Conditions", "Delay Pending Evidence", "Reject"
]


class ExecutiveConfidence(BaseModel):
    """Synthesis over the completed Decision Graph — answers 'Should we act now?' (J7.7).

    Not a recommendation generator. Evaluates the existing graph to produce an
    executive-level approval signal and due-diligence checklist.
    Populated by ExecutiveConfidenceAgent after DecisionAnalysisAgent.
    """

    confidence_id: str                                      # e.g. "EC-20260628-120000"
    overall_confidence: OverallConfidence = "Medium"
    decision_readiness: DecisionReadiness = "Needs Additional Validation"
    board_recommendation: BoardRecommendation = "Proceed with Conditions"
    confidence_rationale: str = ""                          # 2-4 sentence plain-English rationale
    confidence_drivers: list[str] = Field(default_factory=list)      # factors that raise confidence
    confidence_limiters: list[str] = Field(default_factory=list)     # factors that lower confidence
    critical_unknowns: list[str] = Field(default_factory=list)       # unknowns that must resolve first
    validation_priorities: list[str] = Field(default_factory=list)   # due diligence checklist
    confidence_if_assumptions_hold: str = ""  # confidence level if all Critical assumptions hold
    confidence_if_assumptions_fail: str = ""  # confidence level if key Critical assumptions fail
    decision_horizon: str = ""                # when a decision must be made (e.g. "Q3 2026")
    last_updated: str = ""


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

    # --- Decision Architecture (J9.2) — executive decision framing derived by
    #     ProblemFramingAgent (statement, scope, streams, board decisions, …).
    #     Stored as a dict to stay decoupled from the functional_agents layer.
    decision_architecture: dict = Field(default_factory=dict)

    # --- Quality bar --------------------------------------------------------
    required_confidence: str = "medium"   # "high" | "medium" | "low"

    # --- Strategic Assumptions (J7.1) — populated by AssumptionAgent --------
    strategic_assumptions: list[DecisionAssumption] = Field(default_factory=list)

    # --- Strategic Risks (J7.3) — populated by RiskAgent --------------------
    strategic_risks: list[StrategicRisk] = Field(default_factory=list)

    # --- Strategic Opportunities (J7.4) — populated by OpportunityAgent -----
    strategic_opportunities: list[StrategicOpportunity] = Field(default_factory=list)

    # --- Strategic Options (J7.5) — populated by StrategicOptionAgent -------
    strategic_options: list[StrategicOption] = Field(default_factory=list)

    # --- Decision Analysis (J7.6) — populated by DecisionAnalysisAgent ------
    decision_analysis: DecisionAnalysis | None = None

    # --- Executive Confidence (J7.7) — populated by ExecutiveConfidenceAgent -
    executive_confidence: ExecutiveConfidence | None = None  # J7.7

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
    decision_architecture: dict | None = None,
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
        decision_architecture=decision_architecture or {},
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
    decision_architecture: dict | None = None,
) -> DecisionModel:
    """Build a DecisionModel v2 from a DecisionModelPayload (ProblemFramingAgent output).

    Maps the existing v1 fields into the v2 schema without any information loss.
    J9.2: optionally carries the derived Decision Architecture dict.
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
        decision_architecture=decision_architecture or {},
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
