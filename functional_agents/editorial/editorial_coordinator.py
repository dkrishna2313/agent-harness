"""EditorialCoordinator — maps AgentContext → EditorialBrief (PH6.2a).

Responsibilities:
  - Consume AgentContext after the reasoning pipeline completes
  - Produce an EditorialBrief containing structured executive knowledge
  - Persist latest_editorial_brief.json alongside decision_model and research_object

Rules:
  - No summarisation
  - No writing or prose generation
  - No markdown, no report generation, no presentation generation
  - All strings stored at full length; no truncation
  - Never mutates AgentContext
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .editorial_brief import (
    AppendixSection,
    AssumptionEntry,
    AssumptionsSection,
    BriefMetadata,
    ConfidenceSection,
    DecisionAnalysisSection,
    EditorialBrief,
    ExecutiveSummarySection,
    OpportunityEntry,
    OpportunitiesSection,
    RecommendationEntry,
    RecommendationsSection,
    RiskEntry,
    RisksSection,
    StrategicOptionEntry,
    StrategicOptionsSection,
    ValidationPrioritiesSection,
)

if TYPE_CHECKING:
    from ..context import AgentContext

LOGGER = logging.getLogger(__name__)

_BRIEF_DIR = Path("outputs/editorial_briefs")
_LATEST_PATH = Path("outputs/latest_editorial_brief.json")


class EditorialCoordinator:
    """Maps a completed AgentContext to an EditorialBrief and persists it."""

    def build(self, ctx: "AgentContext") -> EditorialBrief:
        """Produce an EditorialBrief from a completed AgentContext.

        Does not mutate ctx. Does not call an LLM. Does not generate prose.
        """
        return EditorialBrief(
            metadata=self._build_metadata(ctx),
            executive_summary=self._build_executive_summary(ctx),
            decision_analysis=self._build_decision_analysis(ctx),
            strategic_options=self._build_strategic_options(ctx),
            recommendations=self._build_recommendations(ctx),
            strategic_assumptions=self._build_assumptions(ctx),
            strategic_risks=self._build_risks(ctx),
            strategic_opportunities=self._build_opportunities(ctx),
            executive_confidence=self._build_confidence(ctx),
            validation_priorities=self._build_validation_priorities(ctx),
            appendix=self._build_appendix(ctx),
        )

    def persist(
        self,
        brief: EditorialBrief,
        base: Path = Path("outputs"),
        *,
        write_latest: bool = True,
    ) -> Path:
        """Persist the EditorialBrief to disk.

        Writes to outputs/editorial_briefs/{brief_id}.json.
        When write_latest=True (default), also updates latest_editorial_brief.json.
        Returns the path of the versioned file.
        """
        brief_dir = base / "editorial_briefs"
        brief_dir.mkdir(parents=True, exist_ok=True)

        data = brief.to_dict()
        path = brief_dir / f"{brief.metadata.brief_id}.json"
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

        if write_latest:
            latest = base / "latest_editorial_brief.json"
            latest.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

        return path

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_metadata(self, ctx: "AgentContext") -> BriefMetadata:
        dm = ctx.decision_model or {}
        engagement = ctx.engagement or {}
        created_at = datetime.now(timezone.utc).isoformat()
        brief_id = f"EB-{created_at[:10].replace('-', '')}-{(ctx.run_id or 'unknown')[:8]}"
        return BriefMetadata(
            brief_id=brief_id,
            created_at=created_at,
            pipeline_run_id=ctx.run_id or "",
            decision_model_id=dm.get("decision_model_id", ""),
            engagement_id=engagement.get("engagement_id", ""),
            question=ctx.question or "",
            profiles=list(ctx.profiles or []),
            execution_profile=ctx.execution_profile or "",
        )

    def _build_executive_summary(self, ctx: "AgentContext") -> ExecutiveSummarySection:
        da = ctx.decision_analysis or {}
        ec = ctx.executive_confidence or {}
        preferred = ctx.preferred_option or {}

        recommended_id = (
            preferred.get("option_id")
            or da.get("recommended_option_id")
            or ""
        )
        recommended_title = preferred.get("title", "")
        if not recommended_title:
            # Fall back to looking up the title in strategic_options
            for opt in (ctx.strategic_options or []):
                if opt.get("option_id") == recommended_id:
                    recommended_title = opt.get("title", "")
                    break

        return ExecutiveSummarySection(
            recommended_option_id=recommended_id,
            recommended_option_title=recommended_title,
            board_recommendation=ec.get("board_recommendation", ""),
            decision_readiness=ec.get("decision_readiness", ""),
            overall_confidence=ec.get("overall_confidence", ""),
            why_this_option=da.get("rationale", ""),
            executive_summary_prose=da.get("executive_summary", ""),
            key_conditions=list(ec.get("confidence_limiters", [])),
            critical_unknowns=list(ec.get("critical_unknowns", [])),
        )

    def _build_decision_analysis(self, ctx: "AgentContext") -> DecisionAnalysisSection:
        da = ctx.decision_analysis or {}
        return DecisionAnalysisSection(
            analysis_id=da.get("analysis_id", ""),
            recommended_option_id=da.get("recommended_option_id", ""),
            executive_summary=da.get("executive_summary", ""),
            comparison_dimensions=list(da.get("comparison_dimensions", [])),
            option_rankings=list(da.get("option_rankings", [])),
            key_tradeoffs=list(da.get("key_tradeoffs", [])),
            key_uncertainties=list(da.get("key_uncertainties", [])),
            sensitivity_analysis=da.get("sensitivity_analysis", ""),
            confidence_summary=da.get("confidence_summary", ""),
            rationale=da.get("rationale", ""),
            decision_matrix=list(da.get("decision_matrix", [])),
        )

    def _build_strategic_options(self, ctx: "AgentContext") -> StrategicOptionsSection:
        da = ctx.decision_analysis or {}
        rankings = list(da.get("option_rankings", []))
        entries = [
            StrategicOptionEntry(
                option_id=opt.get("option_id", ""),
                title=opt.get("title", ""),
                description=opt.get("description", ""),
                strategic_objective=opt.get("strategic_objective", ""),
                expected_outcomes=list(opt.get("expected_outcomes", [])),
                advantages=list(opt.get("advantages", [])),
                disadvantages=list(opt.get("disadvantages", [])),
                implementation_complexity=opt.get("implementation_complexity", ""),
                estimated_time_horizon=opt.get("estimated_time_horizon", ""),
                capital_intensity=opt.get("capital_intensity", ""),
                confidence=opt.get("confidence", ""),
                recommended=bool(opt.get("recommended", False)),
                rationale=opt.get("rationale", ""),
                supporting_assumption_ids=list(opt.get("supporting_assumption_ids", [])),
                associated_risk_ids=list(opt.get("associated_risk_ids", [])),
                associated_opportunity_ids=list(opt.get("associated_opportunity_ids", [])),
                supporting_recommendation_ids=list(opt.get("supporting_recommendation_ids", [])),
            )
            for opt in (ctx.strategic_options or [])
        ]
        return StrategicOptionsSection(options=entries, option_rankings=rankings)

    def _build_recommendations(self, ctx: "AgentContext") -> RecommendationsSection:
        entries = [
            RecommendationEntry(
                recommendation_id=rec.get("recommendation_id", ""),
                title=rec.get("title", ""),
                summary=rec.get("summary", ""),
                time_horizon=rec.get("time_horizon", ""),
                priority=rec.get("priority", ""),
                supported_assumption_ids=list(rec.get("supported_assumption_ids", [])),
                affected_risk_ids=list(rec.get("affected_risk_ids", [])),
            )
            for rec in (ctx.recommendations or [])
        ]
        return RecommendationsSection(recommendations=entries)

    def _build_assumptions(self, ctx: "AgentContext") -> AssumptionsSection:
        entries = [
            AssumptionEntry(
                assumption_id=a.get("assumption_id", ""),
                statement=a.get("statement", ""),
                importance=a.get("importance", ""),
                confidence=a.get("confidence", ""),
                evidence_support=a.get("evidence_support", ""),
                supported_recommendation_ids=list(a.get("supported_recommendation_ids", [])),
            )
            for a in (ctx.assumptions or [])
        ]
        critical_count = sum(1 for e in entries if e.importance == "Critical")
        return AssumptionsSection(assumptions=entries, critical_count=critical_count)

    def _build_risks(self, ctx: "AgentContext") -> RisksSection:
        _severity_order = {"High": 0, "Medium": 1, "Low": 2}
        risks = ctx.risks or []
        entries = [
            RiskEntry(
                risk_id=r.get("risk_id", ""),
                statement=r.get("statement", ""),
                severity=r.get("severity", ""),
                likelihood=r.get("likelihood", ""),
                mitigation_notes=r.get("mitigation_notes", ""),
                related_assumption_ids=list(r.get("related_assumption_ids", [])),
                affected_recommendation_ids=list(r.get("affected_recommendation_ids", [])),
            )
            for r in risks
        ]
        top_risk_id = ""
        if entries:
            top = min(entries, key=lambda e: _severity_order.get(e.severity, 99))
            top_risk_id = top.risk_id
        return RisksSection(risks=entries, top_risk_id=top_risk_id)

    def _build_opportunities(self, ctx: "AgentContext") -> OpportunitiesSection:
        entries = [
            OpportunityEntry(
                opportunity_id=o.get("opportunity_id", ""),
                statement=o.get("statement", ""),
                category=o.get("category", ""),
                likelihood=o.get("likelihood", ""),
                impact=o.get("impact", ""),
                rationale=o.get("rationale", ""),
                related_assumption_ids=list(o.get("related_assumption_ids", [])),
                enabled_recommendation_ids=list(o.get("enabled_recommendation_ids", [])),
            )
            for o in (ctx.opportunities or [])
        ]
        return OpportunitiesSection(opportunities=entries)

    def _build_confidence(self, ctx: "AgentContext") -> ConfidenceSection:
        ec = ctx.executive_confidence or {}
        return ConfidenceSection(
            confidence_id=ec.get("confidence_id", ""),
            overall_confidence=ec.get("overall_confidence", ""),
            decision_readiness=ec.get("decision_readiness", ""),
            board_recommendation=ec.get("board_recommendation", ""),
            confidence_rationale=ec.get("confidence_rationale", ""),
            confidence_drivers=list(ec.get("confidence_drivers", [])),
            confidence_limiters=list(ec.get("confidence_limiters", [])),
            critical_unknowns=list(ec.get("critical_unknowns", [])),
            confidence_if_assumptions_hold=ec.get("confidence_if_assumptions_hold", ""),
            confidence_if_assumptions_fail=ec.get("confidence_if_assumptions_fail", ""),
            decision_horizon=ec.get("decision_horizon", ""),
        )

    def _build_validation_priorities(self, ctx: "AgentContext") -> ValidationPrioritiesSection:
        ec = ctx.executive_confidence or {}
        return ValidationPrioritiesSection(
            priorities=list(ec.get("validation_priorities", [])),
        )

    def _build_appendix(self, ctx: "AgentContext") -> AppendixSection:
        ro = ctx.research_object or {}
        ev_summary = ro.get("evidence_summary", {}) or {}
        citations_raw: list[Any] = ro.get("citations", []) or []
        citations: list[str] = [
            c if isinstance(c, str) else str(c.get("text", c.get("citation", "")))
            for c in citations_raw
        ]
        return AppendixSection(
            total_evidence_items=int(ev_summary.get("total_evidence_items", 0)),
            citation_count=int(ev_summary.get("citation_count", len(citations))),
            profiles=list(ctx.profiles or []),
            evidence_topics=dict(ro.get("evidence_topics", {})),
            citations=citations,
        )
