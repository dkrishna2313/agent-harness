"""EditorialCoordinator — maps AgentContext → EditorialBrief → EditorialManuscript (PH6.3+).

Responsibilities:
  - Consume AgentContext after the reasoning pipeline completes
  - Produce an EditorialBrief containing structured executive knowledge
  - Produce an EditorialManuscript scaffold from the EditorialBrief
  - Run the authoritative writer registry (run_writers) with completeness validation
  - Persist latest_editorial_brief.json and latest_editorial_manuscript.json

Rules:
  - No summarisation, no prose generation, no LLM calls
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
    SectionProvenance,
    StrategicOptionEntry,
    StrategicOptionsSection,
    ValidationPrioritiesSection,
)

from .editorial_manuscript import (
    AppendixManuscriptSection,
    ConfidenceManuscriptSection,
    DecisionAnalysisManuscriptSection,
    EditorialManuscript,
    ExecutiveSummaryManuscriptSection,
    ManuscriptMetadata,
    ManuscriptProvenance,
    OpportunityManuscriptSection,
    RecommendationManuscriptSection,
    RiskManuscriptSection,
)

if TYPE_CHECKING:
    from ..context import AgentContext

LOGGER = logging.getLogger(__name__)

_LATEST_BRIEF_PATH = Path("outputs/latest_editorial_brief.json")
_LATEST_MANUSCRIPT_PATH = Path("outputs/latest_editorial_manuscript.json")

# All manuscript section attribute names — defines the completeness contract.
_ALL_MANUSCRIPT_SECTIONS = [
    "executive_summary",
    "decision_analysis",
    "recommendations",
    "strategic_risks",
    "strategic_opportunities",
    "executive_confidence",
    "appendix",
]


class EditorialValidationError(Exception):
    """Raised when the writer registry violates the manuscript completeness contract."""


def _section_populated(section: Any) -> bool:
    """Return True if a ManuscriptSection has non-empty paragraphs or tables."""
    if section is None:
        return False
    return bool(getattr(section, "paragraphs", None) or getattr(section, "tables", None))


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
            for opt in (ctx.strategic_options or []):
                if opt.get("option_id") == recommended_id:
                    recommended_title = opt.get("title", "")
                    break

        rec_ids = [r.get("recommendation_id", "") for r in (ctx.recommendations or [])]

        return ExecutiveSummarySection(
            recommended_option_id=recommended_id,
            recommended_option_title=recommended_title,
            board_recommendation=ec.get("board_recommendation", ""),
            decision_readiness=ec.get("decision_readiness", ""),
            overall_confidence=ec.get("overall_confidence", ""),
            key_conditions=list(ec.get("confidence_limiters", [])),
            critical_unknowns=list(ec.get("critical_unknowns", [])),
            supporting_recommendation_ids=rec_ids,
            provenance=SectionProvenance(
                decision_model_id=(ctx.decision_model or {}).get("decision_model_id", ""),
                analysis_id=da.get("analysis_id", ""),
                confidence_id=ec.get("confidence_id", ""),
                recommendation_ids=rec_ids,
                option_ids=[recommended_id] if recommended_id else [],
            ),
        )

    def _build_decision_analysis(self, ctx: "AgentContext") -> DecisionAnalysisSection:
        da = ctx.decision_analysis or {}
        option_ids = [o.get("option_id", "") for o in (ctx.strategic_options or [])]
        return DecisionAnalysisSection(
            analysis_id=da.get("analysis_id", ""),
            recommended_option_id=da.get("recommended_option_id", ""),
            comparison_dimensions=list(da.get("comparison_dimensions", [])),
            option_rankings=list(da.get("option_rankings", [])),
            key_tradeoffs=list(da.get("key_tradeoffs", [])),
            key_uncertainties=list(da.get("key_uncertainties", [])),
            decision_matrix=list(da.get("decision_matrix", [])),
            provenance=SectionProvenance(
                decision_model_id=(ctx.decision_model or {}).get("decision_model_id", ""),
                analysis_id=da.get("analysis_id", ""),
                option_ids=option_ids,
            ),
        )

    def _build_strategic_options(self, ctx: "AgentContext") -> StrategicOptionsSection:
        option_ids = [o.get("option_id", "") for o in (ctx.strategic_options or [])]
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
                supporting_assumption_ids=list(opt.get("supporting_assumption_ids", [])),
                associated_risk_ids=list(opt.get("associated_risk_ids", [])),
                associated_opportunity_ids=list(opt.get("associated_opportunity_ids", [])),
                supporting_recommendation_ids=list(opt.get("supporting_recommendation_ids", [])),
            )
            for opt in (ctx.strategic_options or [])
        ]
        return StrategicOptionsSection(
            options=entries,
            provenance=SectionProvenance(
                decision_model_id=(ctx.decision_model or {}).get("decision_model_id", ""),
                option_ids=option_ids,
                assumption_ids=[a.get("assumption_id", "") for a in (ctx.assumptions or [])],
                risk_ids=[r.get("risk_id", "") for r in (ctx.risks or [])],
                opportunity_ids=[o.get("opportunity_id", "") for o in (ctx.opportunities or [])],
            ),
        )

    def _build_recommendations(self, ctx: "AgentContext") -> RecommendationsSection:
        rec_ids = [r.get("recommendation_id", "") for r in (ctx.recommendations or [])]
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
        return RecommendationsSection(
            recommendations=entries,
            provenance=SectionProvenance(
                decision_model_id=(ctx.decision_model or {}).get("decision_model_id", ""),
                recommendation_ids=rec_ids,
                assumption_ids=[a.get("assumption_id", "") for a in (ctx.assumptions or [])],
                risk_ids=[r.get("risk_id", "") for r in (ctx.risks or [])],
            ),
        )

    def _build_assumptions(self, ctx: "AgentContext") -> AssumptionsSection:
        assumption_ids = [a.get("assumption_id", "") for a in (ctx.assumptions or [])]
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
        return AssumptionsSection(
            assumptions=entries,
            critical_count=critical_count,
            provenance=SectionProvenance(
                decision_model_id=(ctx.decision_model or {}).get("decision_model_id", ""),
                assumption_ids=assumption_ids,
                recommendation_ids=[r.get("recommendation_id", "") for r in (ctx.recommendations or [])],
            ),
        )

    def _build_risks(self, ctx: "AgentContext") -> RisksSection:
        _severity_order = {"High": 0, "Medium": 1, "Low": 2}
        risk_ids = [r.get("risk_id", "") for r in (ctx.risks or [])]
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
            for r in (ctx.risks or [])
        ]
        top_risk_id = ""
        if entries:
            top = min(entries, key=lambda e: _severity_order.get(e.severity, 99))
            top_risk_id = top.risk_id
        return RisksSection(
            risks=entries,
            top_risk_id=top_risk_id,
            provenance=SectionProvenance(
                decision_model_id=(ctx.decision_model or {}).get("decision_model_id", ""),
                risk_ids=risk_ids,
                assumption_ids=[a.get("assumption_id", "") for a in (ctx.assumptions or [])],
                recommendation_ids=[r.get("recommendation_id", "") for r in (ctx.recommendations or [])],
            ),
        )

    def _build_opportunities(self, ctx: "AgentContext") -> OpportunitiesSection:
        opp_ids = [o.get("opportunity_id", "") for o in (ctx.opportunities or [])]
        entries = [
            OpportunityEntry(
                opportunity_id=o.get("opportunity_id", ""),
                statement=o.get("statement", ""),
                category=o.get("category", ""),
                likelihood=o.get("likelihood", ""),
                impact=o.get("impact", ""),
                related_assumption_ids=list(o.get("related_assumption_ids", [])),
                enabled_recommendation_ids=list(o.get("enabled_recommendation_ids", [])),
            )
            for o in (ctx.opportunities or [])
        ]
        return OpportunitiesSection(
            opportunities=entries,
            provenance=SectionProvenance(
                decision_model_id=(ctx.decision_model or {}).get("decision_model_id", ""),
                opportunity_ids=opp_ids,
                assumption_ids=[a.get("assumption_id", "") for a in (ctx.assumptions or [])],
                recommendation_ids=[r.get("recommendation_id", "") for r in (ctx.recommendations or [])],
            ),
        )

    def _build_confidence(self, ctx: "AgentContext") -> ConfidenceSection:
        ec = ctx.executive_confidence or {}
        return ConfidenceSection(
            confidence_id=ec.get("confidence_id", ""),
            overall_confidence=ec.get("overall_confidence", ""),
            decision_readiness=ec.get("decision_readiness", ""),
            board_recommendation=ec.get("board_recommendation", ""),
            confidence_drivers=list(ec.get("confidence_drivers", [])),
            confidence_limiters=list(ec.get("confidence_limiters", [])),
            critical_unknowns=list(ec.get("critical_unknowns", [])),
            confidence_if_assumptions_hold=ec.get("confidence_if_assumptions_hold", ""),
            confidence_if_assumptions_fail=ec.get("confidence_if_assumptions_fail", ""),
            decision_horizon=ec.get("decision_horizon", ""),
            provenance=SectionProvenance(
                decision_model_id=(ctx.decision_model or {}).get("decision_model_id", ""),
                confidence_id=ec.get("confidence_id", ""),
                assumption_ids=[a.get("assumption_id", "") for a in (ctx.assumptions or [])],
                risk_ids=[r.get("risk_id", "") for r in (ctx.risks or [])],
            ),
        )

    def _build_validation_priorities(self, ctx: "AgentContext") -> ValidationPrioritiesSection:
        ec = ctx.executive_confidence or {}
        return ValidationPrioritiesSection(
            priorities=list(ec.get("validation_priorities", [])),
            provenance=SectionProvenance(
                confidence_id=ec.get("confidence_id", ""),
                decision_model_id=(ctx.decision_model or {}).get("decision_model_id", ""),
            ),
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
            research_object_id=ro.get("research_id", ""),
            total_evidence_items=int(ev_summary.get("total_evidence_items", 0)),
            citation_count=int(ev_summary.get("citation_count", len(citations))),
            profiles=list(ctx.profiles or []),
            evidence_topics=dict(ro.get("evidence_topics", {})),
            citations=citations,
            provenance=SectionProvenance(
                research_object_id=ro.get("research_id", ""),
                decision_model_id=(ctx.decision_model or {}).get("decision_model_id", ""),
            ),
        )

    # ------------------------------------------------------------------
    # Manuscript scaffold (PH6.3)
    # ------------------------------------------------------------------

    def build_manuscript(self, brief: EditorialBrief) -> EditorialManuscript:
        """Produce an empty EditorialManuscript scaffold from an EditorialBrief.

        Does not generate prose. Content fields (paragraphs, bullet_groups,
        tables, figures) are all empty — writer agents populate them in PH6.4+.
        """
        meta = brief.metadata
        created_at = datetime.now(timezone.utc).isoformat()
        manuscript_id = f"EM-{created_at[:10].replace('-', '')}-{(meta.pipeline_run_id or 'unknown')[:8]}"

        return EditorialManuscript(
            metadata=ManuscriptMetadata(
                manuscript_id=manuscript_id,
                created_at=created_at,
                brief_id=meta.brief_id,
                pipeline_run_id=meta.pipeline_run_id,
                decision_model_id=meta.decision_model_id,
                research_object_id=brief.appendix.research_object_id,
                question=meta.question,
                profiles=list(meta.profiles),
                execution_profile=meta.execution_profile,
            ),
            executive_summary=ExecutiveSummaryManuscriptSection(
                title="Executive Summary",
                subtitle="",
                source_section_ids=["executive_summary"],
                provenance=self._manuscript_provenance(brief, "executive_summary"),
            ),
            decision_analysis=DecisionAnalysisManuscriptSection(
                title="Strategic Analysis",
                subtitle="",
                source_section_ids=["decision_analysis", "strategic_options"],
                provenance=self._manuscript_provenance(brief, "decision_analysis"),
            ),
            recommendations=RecommendationManuscriptSection(
                title="Immediate Actions",
                subtitle="",
                source_section_ids=["recommendations"],
                provenance=self._manuscript_provenance(brief, "recommendations"),
            ),
            strategic_risks=RiskManuscriptSection(
                title="Key Risks",
                subtitle="",
                source_section_ids=["strategic_risks"],
                provenance=self._manuscript_provenance(brief, "strategic_risks"),
            ),
            strategic_opportunities=OpportunityManuscriptSection(
                title="Strategic Opportunities",
                subtitle="",
                source_section_ids=["strategic_opportunities"],
                provenance=self._manuscript_provenance(brief, "strategic_opportunities"),
            ),
            executive_confidence=ConfidenceManuscriptSection(
                title="Decision Readiness",
                subtitle="",
                source_section_ids=["executive_confidence", "validation_priorities"],
                provenance=self._manuscript_provenance(brief, "executive_confidence"),
            ),
            appendix=AppendixManuscriptSection(
                title="Supporting Evidence",
                subtitle="",
                source_section_ids=["appendix"],
                provenance=self._manuscript_provenance(brief, "appendix"),
            ),
        )

    def run_writers(
        self,
        brief: EditorialBrief,
        manuscript: EditorialManuscript,
        client: Any | None = None,
    ) -> EditorialManuscript:
        """Run the ordered writer registry against brief and manuscript.

        Writers execute in registration order. After all writers run, the
        coordinator validates the completeness contract:
          - No duplicate section_name across the registry.
          - Every registered section is populated (paragraphs or tables non-empty).
          - No populated section lacks a registered owner.

        Raises EditorialValidationError on any violation.
        """
        from .executive_summary_writer import ExecutiveSummaryWriter
        from .decision_analysis_writer import DecisionAnalysisWriter
        from .recommendation_writer import RecommendationWriter
        from .risk_writer import RiskWriter
        from .opportunity_writer import OpportunityWriter
        from .confidence_writer import ConfidenceWriter
        from .appendix_writer import AppendixWriter

        _registry = [
            ExecutiveSummaryWriter(client=client),
            DecisionAnalysisWriter(client=client),
            RecommendationWriter(client=client),
            RiskWriter(client=client),
            OpportunityWriter(client=client),
            ConfidenceWriter(client=client),
            AppendixWriter(client=client),
        ]

        # Pre-flight: verify no duplicate section_names and all writers declare one.
        seen: dict[str, str] = {}  # section_name → writer class name
        for writer in _registry:
            sn = getattr(writer, "section_name", None)
            if not sn:
                raise EditorialValidationError(
                    f"{type(writer).__name__} does not declare section_name"
                )
            if sn in seen:
                raise EditorialValidationError(
                    f"Section '{sn}' claimed by both {seen[sn]} and {type(writer).__name__}"
                )
            if not hasattr(manuscript, sn):
                raise EditorialValidationError(
                    f"{type(writer).__name__}.section_name='{sn}' has no matching manuscript attribute"
                )
            seen[sn] = type(writer).__name__

        # Execute writers in order.
        for writer in _registry:
            writer.write(brief, manuscript)

        # Post-flight: verify every registered section is populated.
        unpopulated = [
            sn for sn in seen
            if not _section_populated(getattr(manuscript, sn))
        ]
        if unpopulated:
            raise EditorialValidationError(
                f"Sections not populated after writers ran: {unpopulated}"
            )

        # Post-flight: verify no populated section lacks a registered owner.
        ownerless = [
            attr for attr in _ALL_MANUSCRIPT_SECTIONS
            if attr not in seen and _section_populated(getattr(manuscript, attr, None))
        ]
        if ownerless:
            raise EditorialValidationError(
                f"Sections populated without a registered writer: {ownerless}"
            )

        return manuscript

    def persist_manuscript(
        self,
        manuscript: EditorialManuscript,
        base: Path = Path("outputs"),
        *,
        write_latest: bool = True,
    ) -> Path:
        """Persist the EditorialManuscript to disk.

        Writes to outputs/editorial_manuscripts/{manuscript_id}.json.
        When write_latest=True (default), also updates latest_editorial_manuscript.json.
        Returns the path of the versioned file.
        """
        ms_dir = base / "editorial_manuscripts"
        ms_dir.mkdir(parents=True, exist_ok=True)

        data = manuscript.to_dict()
        path = ms_dir / f"{manuscript.metadata.manuscript_id}.json"
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

        if write_latest:
            latest = base / "latest_editorial_manuscript.json"
            latest.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

        return path

    def _manuscript_provenance(self, brief: EditorialBrief, section_key: str) -> ManuscriptProvenance:
        """Build a ManuscriptProvenance from the corresponding brief section's provenance."""
        section = getattr(brief, section_key, None)
        bp = getattr(section, "provenance", None) if section else None

        return ManuscriptProvenance(
            brief_id=brief.metadata.brief_id,
            brief_section_key=section_key,
            decision_model_id=brief.metadata.decision_model_id or (bp.decision_model_id if bp else ""),
            research_object_id=brief.appendix.research_object_id,
            analysis_id=getattr(bp, "analysis_id", "") if bp else "",
            confidence_id=getattr(bp, "confidence_id", "") if bp else "",
            risk_ids=list(getattr(bp, "risk_ids", [])) if bp else [],
            opportunity_ids=list(getattr(bp, "opportunity_ids", [])) if bp else [],
            recommendation_ids=list(getattr(bp, "recommendation_ids", [])) if bp else [],
            assumption_ids=list(getattr(bp, "assumption_ids", [])) if bp else [],
            option_ids=list(getattr(bp, "option_ids", [])) if bp else [],
        )
