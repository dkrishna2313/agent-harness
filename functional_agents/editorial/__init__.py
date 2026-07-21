"""Editorial Platform (PH6.4).

Provides the canonical chain between executive reasoning and editorial artifacts:

    EditorialBrief              -- structured executive knowledge, no formatting
    EditorialManuscript         -- editorial scaffold populated by writers
    EditorialCoordinator        -- AgentContext → EditorialBrief → EditorialManuscript
    ExecutiveSummaryWriter      -- populates EditorialManuscript.executive_summary (PH6.4)

The editorial package never mutates AgentContext reasoning fields and never
produces rendered markdown, DOCX, or PPTX.
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
from .executive_summary_writer import ExecutiveSummaryWriter

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
    # Writers (PH6.4+)
    "ExecutiveSummaryWriter",
]
