"""Editorial Platform (PH6.5).

Provides the canonical chain between executive reasoning and editorial artifacts:

    EditorialBrief              -- structured executive knowledge, no formatting
    EditorialManuscript         -- editorial scaffold populated by writers
    EditorialCoordinator        -- AgentContext → EditorialBrief → EditorialManuscript
    EditorialWriter             -- abstract base class for all writers (PH6.5)
    ExecutiveSummaryWriter      -- populates EditorialManuscript.executive_summary (PH6.4)
    DecisionAnalysisWriter      -- populates EditorialManuscript.decision_analysis (PH6.5)

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
from .editorial_writer import EditorialWriter
from .executive_summary_writer import ExecutiveSummaryWriter
from .decision_analysis_writer import DecisionAnalysisWriter

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
    # Abstract base
    "EditorialWriter",
    # Writers (PH6.4+)
    "ExecutiveSummaryWriter",
    "DecisionAnalysisWriter",
]
