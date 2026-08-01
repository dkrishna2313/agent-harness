"""EditorialBrief — canonical handoff object for the Editorial Platform (PH6.2b).

Contains structured executive knowledge only. No prose, no markdown, no formatting.
Prose fields (rationale paragraphs, summary sentences, sensitivity analysis) are
excluded — writers compose those from the structured inputs here.

Serialisation: editorial_brief.to_dict() → JSON-safe dict.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .strategy_narrative import StrategyNarrative


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

@dataclass
class SectionProvenance:
    """Source object IDs for every section — supports traceability and debugging."""

    decision_model_id: str = ""
    research_object_id: str = ""
    analysis_id: str = ""
    confidence_id: str = ""
    risk_ids: list[str] = field(default_factory=list)
    opportunity_ids: list[str] = field(default_factory=list)
    recommendation_ids: list[str] = field(default_factory=list)
    assumption_ids: list[str] = field(default_factory=list)
    option_ids: list[str] = field(default_factory=list)


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
    """Structured inputs for the board-level decision opener.

    No prose. A writer composes the opening paragraph from these fields.
    """

    recommended_option_id: str
    recommended_option_title: str
    board_recommendation: str           # canonical enum, e.g. "Proceed with Conditions"
    decision_readiness: str             # e.g. "Needs Additional Validation"
    overall_confidence: str             # e.g. "Low"
    key_conditions: list[str] = field(default_factory=list)     # confidence_limiters
    critical_unknowns: list[str] = field(default_factory=list)
    supporting_recommendation_ids: list[str] = field(default_factory=list)
    provenance: SectionProvenance = field(default_factory=SectionProvenance)


# ---------------------------------------------------------------------------
# Decision Analysis
# ---------------------------------------------------------------------------

@dataclass
class DecisionAnalysisSection:
    """Structured inputs for the option comparison narrative.

    Prose fields (executive_summary paragraph, rationale, sensitivity_analysis,
    confidence_summary) are excluded — writers compose those from the structured
    data here.
    """

    analysis_id: str
    recommended_option_id: str
    comparison_dimensions: list[str] = field(default_factory=list)
    option_rankings: list[str] = field(default_factory=list)    # ordered option_ids, best→worst
    key_tradeoffs: list[str] = field(default_factory=list)
    key_uncertainties: list[str] = field(default_factory=list)
    decision_matrix: list[dict[str, Any]] = field(default_factory=list)
    provenance: SectionProvenance = field(default_factory=SectionProvenance)


# ---------------------------------------------------------------------------
# Strategic Options
# ---------------------------------------------------------------------------

@dataclass
class StrategicOptionEntry:
    """A single strategic option — structured inputs only, no rationale paragraph."""

    option_id: str
    title: str
    description: str              # definitional: what this option entails
    strategic_objective: str      # what it aims to achieve
    expected_outcomes: list[str] = field(default_factory=list)
    advantages: list[str] = field(default_factory=list)
    disadvantages: list[str] = field(default_factory=list)
    implementation_complexity: str = ""   # "Low" | "Medium" | "High"
    estimated_time_horizon: str = ""      # "Near-term" | "Medium-term" | "Long-term"
    capital_intensity: str = ""           # "Low" | "Medium" | "High"
    confidence: str = ""                  # "High" | "Medium" | "Low"
    recommended: bool = False
    supporting_assumption_ids: list[str] = field(default_factory=list)
    associated_risk_ids: list[str] = field(default_factory=list)
    associated_opportunity_ids: list[str] = field(default_factory=list)
    supporting_recommendation_ids: list[str] = field(default_factory=list)


@dataclass
class StrategicOptionsSection:
    """All strategic options evaluated in this pipeline run.

    Ranking order is authoritative in DecisionAnalysisSection.option_rankings.
    """

    options: list[StrategicOptionEntry] = field(default_factory=list)
    provenance: SectionProvenance = field(default_factory=SectionProvenance)


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

@dataclass
class RecommendationEntry:
    """A single actionable recommendation."""

    recommendation_id: str
    title: str
    summary: str          # brief factual statement of the action (not a prose paragraph)
    time_horizon: str     # e.g. "near_term"
    priority: str         # e.g. "high"
    supported_assumption_ids: list[str] = field(default_factory=list)
    affected_risk_ids: list[str] = field(default_factory=list)


@dataclass
class RecommendationsSection:
    """All recommendations, ordered as produced by the pipeline."""

    recommendations: list[RecommendationEntry] = field(default_factory=list)
    provenance: SectionProvenance = field(default_factory=SectionProvenance)


# ---------------------------------------------------------------------------
# Risks
# ---------------------------------------------------------------------------

@dataclass
class RiskEntry:
    """A single risk."""

    risk_id: str
    statement: str
    severity: str         # "High" | "Medium" | "Low"
    likelihood: str       # "High" | "Medium" | "Low"
    mitigation_notes: str = ""
    related_assumption_ids: list[str] = field(default_factory=list)
    affected_recommendation_ids: list[str] = field(default_factory=list)


@dataclass
class RisksSection:
    """All risks. top_risk_id identifies the highest-severity risk."""

    risks: list[RiskEntry] = field(default_factory=list)
    top_risk_id: str = ""
    provenance: SectionProvenance = field(default_factory=SectionProvenance)


# ---------------------------------------------------------------------------
# Opportunities
# ---------------------------------------------------------------------------

@dataclass
class OpportunityEntry:
    """A single opportunity — structured inputs only, no rationale paragraph."""

    opportunity_id: str
    statement: str
    category: str
    likelihood: str
    impact: str
    related_assumption_ids: list[str] = field(default_factory=list)
    enabled_recommendation_ids: list[str] = field(default_factory=list)


@dataclass
class OpportunitiesSection:
    """All opportunities, ordered as produced by the pipeline."""

    opportunities: list[OpportunityEntry] = field(default_factory=list)
    provenance: SectionProvenance = field(default_factory=SectionProvenance)


# ---------------------------------------------------------------------------
# Assumptions
# ---------------------------------------------------------------------------

@dataclass
class AssumptionEntry:
    """A single assumption."""

    assumption_id: str
    statement: str
    importance: str       # "Critical" | "Important" | "Informational"
    confidence: str       # "High" | "Medium" | "Low"
    evidence_support: str
    supported_recommendation_ids: list[str] = field(default_factory=list)


@dataclass
class AssumptionsSection:
    """All assumptions. Critical entries should appear first."""

    assumptions: list[AssumptionEntry] = field(default_factory=list)
    critical_count: int = 0
    provenance: SectionProvenance = field(default_factory=SectionProvenance)


# ---------------------------------------------------------------------------
# Executive Confidence
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceSection:
    """Structured inputs from ExecutiveConfidenceAgent.

    confidence_rationale paragraph is excluded — writers compose from the
    structured signals here (drivers, limiters, unknowns, ratings).
    """

    confidence_id: str
    overall_confidence: str       # "High" | "Medium" | "Low"
    decision_readiness: str       # "Ready for Decision" | "Needs Additional Validation" | "Not Ready"
    board_recommendation: str     # canonical enum, e.g. "Proceed with Conditions"
    confidence_drivers: list[str] = field(default_factory=list)
    confidence_limiters: list[str] = field(default_factory=list)
    critical_unknowns: list[str] = field(default_factory=list)
    confidence_if_assumptions_hold: str = ""  # rating, e.g. "High"
    confidence_if_assumptions_fail: str = ""  # rating, e.g. "Low"
    decision_horizon: str = ""
    provenance: SectionProvenance = field(default_factory=SectionProvenance)


# ---------------------------------------------------------------------------
# Validation Priorities
# ---------------------------------------------------------------------------

@dataclass
class ValidationPrioritiesSection:
    """Due-diligence checklist from ExecutiveConfidenceAgent."""

    priorities: list[str] = field(default_factory=list)
    provenance: SectionProvenance = field(default_factory=SectionProvenance)


# ---------------------------------------------------------------------------
# Appendix / Supporting Evidence
# ---------------------------------------------------------------------------

@dataclass
class AppendixSection:
    """Evidence provenance and citation data."""

    research_object_id: str = ""
    total_evidence_items: int = 0
    citation_count: int = 0
    profiles: list[str] = field(default_factory=list)
    evidence_topics: dict[str, int] = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)
    provenance: SectionProvenance = field(default_factory=SectionProvenance)


# ---------------------------------------------------------------------------
# EditorialBrief — canonical handoff object
# ---------------------------------------------------------------------------

@dataclass
class EditorialBrief:
    """Structured executive knowledge produced by EditorialCoordinator.

    Contains no prose, no markdown, no formatting instructions.
    All string fields are stored at full length.
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
    strategy_narrative: "StrategyNarrative | None" = field(default=None)

    # PH12.2f — top-level strategy summary fields (optional, backward compatible)
    # Populated by EditorialCoordinator.build() when strategy_narrative is present.
    strategic_direction: str = field(default="")   # winning_position executive statement
    core_thesis: str = field(default="")           # winning_mechanism (the "how")
    recommended_option: str = field(default="")    # mapped_option_id
    mapped_option_title: str = field(default="")   # human-readable option name
    alignment: str = field(default="")             # alignment_status
    execution_implications: list[str] = field(default_factory=list)
    strategy_provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        # strategy_narrative is a Pydantic model — serialize it explicitly
        sn = self.strategy_narrative
        d["strategy_narrative"] = sn.model_dump(mode="json") if sn is not None else None
        return d
