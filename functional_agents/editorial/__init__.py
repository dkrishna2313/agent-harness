"""Editorial Platform (PH6.3).

Provides the canonical chain between executive reasoning and editorial artifacts:

    EditorialBrief          -- structured executive knowledge, no formatting
    EditorialManuscript     -- editorial scaffold; writers populate in PH6.4+
    EditorialCoordinator    -- AgentContext → EditorialBrief → EditorialManuscript

The editorial package never calls an LLM, never mutates AgentContext reasoning
fields, and never produces rendered prose or markdown.
"""

from __future__ import annotations

from .editorial_brief import (
    EditorialBrief,
    BriefMetadata,
    SectionProvenance,
    ExecutiveSummarySection,
    DecisionAnalysisSection,
    StrategicOptionsSection,
    StrategicOptionEntry,
    RecommendationsSection,
    RecommendationEntry,
    RisksSection,
    RiskEntry,
    OpportunitiesSection,
    OpportunityEntry,
    AssumptionsSection,
    AssumptionEntry,
    ConfidenceSection,
    ValidationPrioritiesSection,
    AppendixSection,
)
from .editorial_manuscript import (
    EditorialManuscript,
    ManuscriptMetadata,
    ManuscriptProvenance,
    ManuscriptSection,
    ExecutiveSummaryManuscriptSection,
    DecisionAnalysisManuscriptSection,
    RecommendationManuscriptSection,
    RiskManuscriptSection,
    OpportunityManuscriptSection,
    ConfidenceManuscriptSection,
    AppendixManuscriptSection,
)
from .editorial_coordinator import EditorialCoordinator

__all__ = [
    # Brief
    "EditorialBrief",
    "BriefMetadata",
    "SectionProvenance",
    "ExecutiveSummarySection",
    "DecisionAnalysisSection",
    "StrategicOptionsSection",
    "StrategicOptionEntry",
    "RecommendationsSection",
    "RecommendationEntry",
    "RisksSection",
    "RiskEntry",
    "OpportunitiesSection",
    "OpportunityEntry",
    "AssumptionsSection",
    "AssumptionEntry",
    "ConfidenceSection",
    "ValidationPrioritiesSection",
    "AppendixSection",
    # Manuscript
    "EditorialManuscript",
    "ManuscriptMetadata",
    "ManuscriptProvenance",
    "ManuscriptSection",
    "ExecutiveSummaryManuscriptSection",
    "DecisionAnalysisManuscriptSection",
    "RecommendationManuscriptSection",
    "RiskManuscriptSection",
    "OpportunityManuscriptSection",
    "ConfidenceManuscriptSection",
    "AppendixManuscriptSection",
    # Coordinator
    "EditorialCoordinator",
]
