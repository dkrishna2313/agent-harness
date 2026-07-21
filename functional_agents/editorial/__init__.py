"""Editorial Platform (PH6.2a).

Provides the canonical handoff boundary between executive reasoning and
executive communication:

    EditorialBrief          -- structured executive knowledge, no formatting
    EditorialCoordinator    -- maps AgentContext → EditorialBrief + persistence

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
from .editorial_coordinator import EditorialCoordinator

__all__ = [
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
    "EditorialCoordinator",
]
