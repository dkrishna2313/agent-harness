"""EditorialBrief — canonical handoff object for the Editorial Platform (PH6.2a).

Contains structured executive knowledge only. No prose, no markdown, no formatting.
All string fields are stored at full length; truncation is a writer/renderer concern.

Serialisation: use editorial_brief.to_dict() → JSON-safe dict.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@dataclass
class BriefMetadata:
    """Identity and provenance of this EditorialBrief."""

    brief_id: str
    created_at: str
    pipeline_run_id: str         # AgentContext.run_id
    decision_model_id: str       # links back to the DecisionModel that drove the run
    engagement_id: str           # empty string if no engagement
    question: str                # primary research/strategic question
    profiles: list[str] = field(default_factory=list)
    execution_profile: str = ""


# ---------------------------------------------------------------------------
# Executive Summary
# ---------------------------------------------------------------------------

@dataclass
class ExecutiveSummarySection:
    """Board-level decision inputs — no rendered paragraph, structured facts only."""

    recommended_option_id: str
    recommended_option_title: str
    board_recommendation: str    # canonical enum string, e.g. "Proceed with Conditions"
    decision_readiness: str      # e.g. "Needs Additional Validation"
    overall_confidence: str      # e.g. "Low"
    why_this_option: str         # full rationale from decision_analysis (no truncation)
    executive_summary_prose: str # decision_analysis.executive_summary (2-4 sentences)
    key_conditions: list[str] = field(default_factory=list)   # confidence_limiters
    critical_unknowns: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Decision Analysis
# ---------------------------------------------------------------------------

@dataclass
class DecisionAnalysisSection:
    """Structured output of DecisionAnalysisAgent — explicit option comparison."""

    analysis_id: str
    recommended_option_id: str
    executive_summary: str           # 2-4 sentence plain-English summary
    comparison_dimensions: list[str] = field(default_factory=list)
    option_rankings: list[str] = field(default_factory=list)   # ordered option_ids
    key_tradeoffs: list[str] = field(default_factory=list)
    key_uncertainties: list[str] = field(default_factory=list)
    sensitivity_analysis: str = ""
    confidence_summary: str = ""
    rationale: str = ""              # full explanation of why the recommended option wins
    decision_matrix: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Strategic Options
# ---------------------------------------------------------------------------

@dataclass
class StrategicOptionEntry:
    """A single strategic option from StrategicOptionAgent."""

    option_id: str
    title: str
    description: str
    strategic_objective: str
    expected_outcomes: list[str] = field(default_factory=list)
    advantages: list[str] = field(default_factory=list)
    disadvantages: list[str] = field(default_factory=list)
    implementation_complexity: str = ""
    estimated_time_horizon: str = ""
    capital_intensity: str = ""
    confidence: str = ""
    recommended: bool = False
    rationale: str = ""
    supporting_assumption_ids: list[str] = field(default_factory=list)
    associated_risk_ids: list[str] = field(default_factory=list)
    associated_opportunity_ids: list[str] = field(default_factory=list)
    supporting_recommendation_ids: list[str] = field(default_factory=list)


@dataclass
class StrategicOptionsSection:
    """All strategic options evaluated in this pipeline run."""

    options: list[StrategicOptionEntry] = field(default_factory=list)
    option_rankings: list[str] = field(default_factory=list)  # ordered option_ids from DecisionAnalysis


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

@dataclass
class RecommendationEntry:
    """A single actionable recommendation from RecommendationAgent."""

    recommendation_id: str
    title: str
    summary: str
    time_horizon: str            # e.g. "near_term"
    priority: str                # e.g. "high"
    supported_assumption_ids: list[str] = field(default_factory=list)
    affected_risk_ids: list[str] = field(default_factory=list)


@dataclass
class RecommendationsSection:
    """All recommendations, ordered as produced by the pipeline."""

    recommendations: list[RecommendationEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Risks
# ---------------------------------------------------------------------------

@dataclass
class RiskEntry:
    """A single risk from RiskAgent."""

    risk_id: str
    statement: str
    severity: str
    likelihood: str
    mitigation_notes: str = ""
    related_assumption_ids: list[str] = field(default_factory=list)
    affected_recommendation_ids: list[str] = field(default_factory=list)


@dataclass
class RisksSection:
    """All risks, ordered as produced by the pipeline."""

    risks: list[RiskEntry] = field(default_factory=list)
    top_risk_id: str = ""   # risk_id of the highest-severity risk (for editorial emphasis)


# ---------------------------------------------------------------------------
# Opportunities
# ---------------------------------------------------------------------------

@dataclass
class OpportunityEntry:
    """A single opportunity from OpportunityAgent."""

    opportunity_id: str
    statement: str
    category: str
    likelihood: str
    impact: str
    rationale: str = ""
    related_assumption_ids: list[str] = field(default_factory=list)
    enabled_recommendation_ids: list[str] = field(default_factory=list)


@dataclass
class OpportunitiesSection:
    """All opportunities, ordered as produced by the pipeline."""

    opportunities: list[OpportunityEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Assumptions
# ---------------------------------------------------------------------------

@dataclass
class AssumptionEntry:
    """A single assumption from AssumptionAgent."""

    assumption_id: str
    statement: str
    importance: str       # e.g. "Critical", "Important"
    confidence: str       # e.g. "High", "Medium", "Low"
    evidence_support: str
    supported_recommendation_ids: list[str] = field(default_factory=list)


@dataclass
class AssumptionsSection:
    """All assumptions, Critical entries first."""

    assumptions: list[AssumptionEntry] = field(default_factory=list)
    critical_count: int = 0


# ---------------------------------------------------------------------------
# Executive Confidence
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceSection:
    """Structured output of ExecutiveConfidenceAgent."""

    confidence_id: str
    overall_confidence: str
    decision_readiness: str
    board_recommendation: str
    confidence_rationale: str = ""
    confidence_drivers: list[str] = field(default_factory=list)
    confidence_limiters: list[str] = field(default_factory=list)
    critical_unknowns: list[str] = field(default_factory=list)
    confidence_if_assumptions_hold: str = ""
    confidence_if_assumptions_fail: str = ""
    decision_horizon: str = ""


# ---------------------------------------------------------------------------
# Validation Priorities
# ---------------------------------------------------------------------------

@dataclass
class ValidationPrioritiesSection:
    """Due-diligence checklist from ExecutiveConfidenceAgent."""

    priorities: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Appendix / Supporting Evidence
# ---------------------------------------------------------------------------

@dataclass
class AppendixSection:
    """Evidence provenance and citation data."""

    total_evidence_items: int = 0
    citation_count: int = 0
    profiles: list[str] = field(default_factory=list)
    evidence_topics: dict[str, int] = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# EditorialBrief — canonical handoff object
# ---------------------------------------------------------------------------

@dataclass
class EditorialBrief:
    """Structured executive knowledge produced by EditorialCoordinator.

    Contains no prose, no markdown, no formatting instructions.
    All string fields are at full length — truncation is a writer concern.
    Read-only from the perspective of all editorial writers.
    """

    metadata: BriefMetadata
    executive_summary: ExecutiveSummarySection
    decision_analysis: DecisionAnalysisSection
    strategic_options: StrategicOptionsSection
    recommendations: RecommendationsSection
    strategic_assumptions: AssumptionsSection
    strategic_risks: RisksSection
    strategic_opportunities: OpportunitiesSection
    executive_confidence: ConfidenceSection
    validation_priorities: ValidationPrioritiesSection
    appendix: AppendixSection

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
