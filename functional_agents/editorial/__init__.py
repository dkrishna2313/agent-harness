"""Editorial Platform (PH7) — complete.

Provides the canonical chain between executive reasoning and editorial artifacts:

    EditorialBrief              -- structured executive knowledge, no formatting
    EditorialManuscript         -- editorial scaffold populated by writers
    EditorialCoordinator        -- AgentContext → EditorialBrief → EditorialManuscript
    EditorialValidationError    -- raised when registry completeness contract is violated
    EditorialWriter             -- abstract base class for all writers (PH6.5)
    ExecutiveSummaryWriter      -- populates EditorialManuscript.executive_summary (PH6.4)
    DecisionAnalysisWriter      -- populates EditorialManuscript.decision_analysis (PH6.5)
    RecommendationWriter        -- populates EditorialManuscript.recommendations (PH6.6)
    RiskWriter                  -- populates EditorialManuscript.strategic_risks (PH6.7)
    OpportunityWriter           -- populates EditorialManuscript.strategic_opportunities (PH6.8)
    ConfidenceWriter            -- populates EditorialManuscript.executive_confidence (PH6.9)
    AppendixWriter              -- populates EditorialManuscript.appendix (PH6.10)
    MarkdownRenderer            -- EditorialManuscript → Markdown report string (PH7)

The editorial package never mutates AgentContext reasoning fields.
MarkdownRenderer is the sole authoritative source of Markdown output when PH7 is active.
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
from .editorial_coordinator import EditorialCoordinator, EditorialValidationError
from .editorial_writer import EditorialWriter
from .executive_summary_writer import ExecutiveSummaryWriter
from .decision_analysis_writer import DecisionAnalysisWriter
from .recommendation_writer import RecommendationWriter
from .risk_writer import RiskWriter
from .opportunity_writer import OpportunityWriter
from .confidence_writer import ConfidenceWriter
from .appendix_writer import AppendixWriter
from .markdown_renderer import MarkdownRenderer

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
    # Coordinator + validation
    "EditorialCoordinator",
    "EditorialValidationError",
    # Abstract base
    "EditorialWriter",
    # Writers (PH6.4–PH6.10)
    "ExecutiveSummaryWriter",
    "DecisionAnalysisWriter",
    "RecommendationWriter",
    "RiskWriter",
    "OpportunityWriter",
    "ConfidenceWriter",
    "AppendixWriter",
    # Renderer (PH7)
    "MarkdownRenderer",
]
