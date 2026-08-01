"""EditorialCoordinator — maps StrategicPosition → EditorialBrief → EditorialManuscript (PH8).

Responsibilities:
  - Consume a StrategicPosition produced by the Strategy Layer
  - Produce an EditorialBrief containing structured executive knowledge
  - Produce an EditorialManuscript scaffold from the EditorialBrief
  - Run the authoritative writer registry (run_writers) with completeness validation
  - Persist latest_editorial_brief.json and latest_editorial_manuscript.json

Rules:
  - No summarisation, no prose generation, no LLM calls
  - No markdown, no report generation, no presentation generation
  - All strings stored at full length; no truncation
  - Never mutates the input object
  - Communication Layer reads from StrategicPosition, not AgentContext

PH8 backward compatibility: build() accepts AgentContext as a fallback.
When AgentContext is passed, StrategyCoordinator converts it to a
StrategicPosition first, then the normal path runs.
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
    StrategyManuscriptSection,
)
from .strategy_narrative import build_strategy_narrative

if TYPE_CHECKING:
    from ..context import AgentContext
    from ..strategy import StrategicPosition, StrategyTrace

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
    """Maps a StrategicPosition to an EditorialBrief and persists it (PH8)."""

    def build(
        self,
        position: "StrategicPosition | AgentContext",
        *,
        strategy_trace: "StrategyTrace | None" = None,
    ) -> EditorialBrief:
        """Produce an EditorialBrief from a StrategicPosition.

        PH8 canonical path: accepts StrategicPosition.
        Backward-compatible path: accepts AgentContext, converts via
        StrategyCoordinator first so all code follows the same path.

        When strategy_trace is provided, the brief's strategy_narrative field
        is populated from the trace for downstream strategy section rendering.
        When strategy_trace is None, strategy_narrative is None and the
        existing report path is preserved unchanged.

        Does not mutate the input. Does not call an LLM. Does not generate prose.
        """
        from ..strategy import StrategyCoordinator, StrategicPosition as _SP
        if not isinstance(position, _SP):
            # Backward compat — AgentContext passed directly (tests, legacy callers)
            position = StrategyCoordinator().build(position)  # type: ignore[arg-type]

        narrative = (
            build_strategy_narrative(strategy_trace)
            if strategy_trace is not None
            else None
        )

        # PH12.2f — build shared strategy output view for top-level brief fields
        from ..strategy.strategy_output_view import build_strategy_output_view
        strategy_view = build_strategy_output_view(
            narrative,
            strategic_options=list(position.strategic_options or []),
        )

        # PH12.2f — extract top-level strategy summary fields
        strategic_direction = ""
        core_thesis = ""
        recommended_option = ""
        mapped_option_title_val = ""
        alignment_val = ""
        execution_implications: list[str] = []
        strategy_provenance: dict = {}

        if strategy_view is not None:
            strategic_direction = strategy_view.strategic_position
            core_thesis = strategy_view.strategic_mechanism
            recommended_option = strategy_view.mapped_option_id
            mapped_option_title_val = strategy_view.mapped_option_title
            alignment_val = strategy_view.alignment_status
            execution_implications = list(strategy_view.execution_implications)
            strategy_provenance = {
                "strategic_position_id": strategy_view.strategic_position_id,
                "winning_theory_id": strategy_view.winning_theory_id,
                "mapped_option_id": strategy_view.mapped_option_id,
                "alignment_status": strategy_view.alignment_status,
                "framework": strategy_view.framework,
                "trace_id": strategy_view.trace_id,
                "strategy_config_fingerprint": strategy_view.strategy_config_fingerprint,
            }

        return EditorialBrief(
            metadata=self._build_metadata(position),
            executive_summary=self._build_executive_summary(position),
            decision_analysis=self._build_decision_analysis(position),
            strategic_options=self._build_strategic_options(position),
            recommendations=self._build_recommendations(position),
            strategic_assumptions=self._build_assumptions(position),
            strategic_risks=self._build_risks(position),
            strategic_opportunities=self._build_opportunities(position),
            executive_confidence=self._build_confidence(position),
            validation_priorities=self._build_validation_priorities(position),
            appendix=self._build_appendix(position),
            strategy_narrative=narrative,
            # PH12.2f — top-level strategy fields
            strategic_direction=strategic_direction,
            core_thesis=core_thesis,
            recommended_option=recommended_option,
            mapped_option_title=mapped_option_title_val,
            alignment=alignment_val,
            execution_implications=execution_implications,
            strategy_provenance=strategy_provenance,
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

    def _build_metadata(self, position: "StrategicPosition") -> BriefMetadata:
        dm = position.decision_model or {}
        engagement = position.engagement or {}
        created_at = datetime.now(timezone.utc).isoformat()
        brief_id = f"EB-{created_at[:10].replace('-', '')}-{(position.run_id or 'unknown')[:8]}"
        return BriefMetadata(
            brief_id=brief_id,
            created_at=created_at,
            pipeline_run_id=position.run_id or "",
            decision_model_id=dm.get("decision_model_id", ""),
            engagement_id=engagement.get("engagement_id", ""),
            question=position.question or "",
            profiles=list(position.profiles or []),
            execution_profile=position.execution_profile or "",
        )

    def _build_executive_summary(self, position: "StrategicPosition") -> ExecutiveSummarySection:
        da = position.decision_analysis or {}
        ec = position.executive_confidence or {}
        preferred = position.preferred_option or {}

        recommended_id = (
            preferred.get("option_id")
            or da.get("recommended_option_id")
            or ""
        )
        recommended_title = preferred.get("title", "")
        if not recommended_title:
            for opt in (position.strategic_options or []):
                if opt.get("option_id") == recommended_id:
                    recommended_title = opt.get("title", "")
                    break

        rec_ids = [r.get("recommendation_id", "") for r in (position.recommendations or [])]

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
                decision_model_id=(position.decision_model or {}).get("decision_model_id", ""),
                analysis_id=da.get("analysis_id", ""),
                confidence_id=ec.get("confidence_id", ""),
                recommendation_ids=rec_ids,
                option_ids=[recommended_id] if recommended_id else [],
            ),
        )

    def _build_decision_analysis(self, position: "StrategicPosition") -> DecisionAnalysisSection:
        da = position.decision_analysis or {}
        option_ids = [o.get("option_id", "") for o in (position.strategic_options or [])]
        return DecisionAnalysisSection(
            analysis_id=da.get("analysis_id", ""),
            recommended_option_id=da.get("recommended_option_id", ""),
            comparison_dimensions=list(da.get("comparison_dimensions", [])),
            option_rankings=list(da.get("option_rankings", [])),
            key_tradeoffs=list(da.get("key_tradeoffs", [])),
            key_uncertainties=list(da.get("key_uncertainties", [])),
            decision_matrix=list(da.get("decision_matrix", [])),
            provenance=SectionProvenance(
                decision_model_id=(position.decision_model or {}).get("decision_model_id", ""),
                analysis_id=da.get("analysis_id", ""),
                option_ids=option_ids,
            ),
        )

    def _build_strategic_options(self, position: "StrategicPosition") -> StrategicOptionsSection:
        option_ids = [o.get("option_id", "") for o in (position.strategic_options or [])]
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
            for opt in (position.strategic_options or [])
        ]
        return StrategicOptionsSection(
            options=entries,
            provenance=SectionProvenance(
                decision_model_id=(position.decision_model or {}).get("decision_model_id", ""),
                option_ids=option_ids,
                assumption_ids=[a.get("assumption_id", "") for a in (position.assumptions or [])],
                risk_ids=[r.get("risk_id", "") for r in (position.risks or [])],
                opportunity_ids=[o.get("opportunity_id", "") for o in (position.opportunities or [])],
            ),
        )

    def _build_recommendations(self, position: "StrategicPosition") -> RecommendationsSection:
        rec_ids = [r.get("recommendation_id", "") for r in (position.recommendations or [])]
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
            for rec in (position.recommendations or [])
        ]
        return RecommendationsSection(
            recommendations=entries,
            provenance=SectionProvenance(
                decision_model_id=(position.decision_model or {}).get("decision_model_id", ""),
                recommendation_ids=rec_ids,
                assumption_ids=[a.get("assumption_id", "") for a in (position.assumptions or [])],
                risk_ids=[r.get("risk_id", "") for r in (position.risks or [])],
            ),
        )

    def _build_assumptions(self, position: "StrategicPosition") -> AssumptionsSection:
        assumption_ids = [a.get("assumption_id", "") for a in (position.assumptions or [])]
        entries = [
            AssumptionEntry(
                assumption_id=a.get("assumption_id", ""),
                statement=a.get("statement", ""),
                importance=a.get("importance", ""),
                confidence=a.get("confidence", ""),
                evidence_support=a.get("evidence_support", ""),
                supported_recommendation_ids=list(a.get("supported_recommendation_ids", [])),
            )
            for a in (position.assumptions or [])
        ]
        critical_count = sum(1 for e in entries if e.importance == "Critical")
        return AssumptionsSection(
            assumptions=entries,
            critical_count=critical_count,
            provenance=SectionProvenance(
                decision_model_id=(position.decision_model or {}).get("decision_model_id", ""),
                assumption_ids=assumption_ids,
                recommendation_ids=[r.get("recommendation_id", "") for r in (position.recommendations or [])],
            ),
        )

    def _build_risks(self, position: "StrategicPosition") -> RisksSection:
        _severity_order = {"High": 0, "Medium": 1, "Low": 2}
        risk_ids = [r.get("risk_id", "") for r in (position.risks or [])]
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
            for r in (position.risks or [])
        ]
        top_risk_id = ""
        if entries:
            top = min(entries, key=lambda e: _severity_order.get(e.severity, 99))
            top_risk_id = top.risk_id
        return RisksSection(
            risks=entries,
            top_risk_id=top_risk_id,
            provenance=SectionProvenance(
                decision_model_id=(position.decision_model or {}).get("decision_model_id", ""),
                risk_ids=risk_ids,
                assumption_ids=[a.get("assumption_id", "") for a in (position.assumptions or [])],
                recommendation_ids=[r.get("recommendation_id", "") for r in (position.recommendations or [])],
            ),
        )

    def _build_opportunities(self, position: "StrategicPosition") -> OpportunitiesSection:
        opp_ids = [o.get("opportunity_id", "") for o in (position.opportunities or [])]
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
            for o in (position.opportunities or [])
        ]
        return OpportunitiesSection(
            opportunities=entries,
            provenance=SectionProvenance(
                decision_model_id=(position.decision_model or {}).get("decision_model_id", ""),
                opportunity_ids=opp_ids,
                assumption_ids=[a.get("assumption_id", "") for a in (position.assumptions or [])],
                recommendation_ids=[r.get("recommendation_id", "") for r in (position.recommendations or [])],
            ),
        )

    def _build_confidence(self, position: "StrategicPosition") -> ConfidenceSection:
        ec = position.executive_confidence or {}
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
                decision_model_id=(position.decision_model or {}).get("decision_model_id", ""),
                confidence_id=ec.get("confidence_id", ""),
                assumption_ids=[a.get("assumption_id", "") for a in (position.assumptions or [])],
                risk_ids=[r.get("risk_id", "") for r in (position.risks or [])],
            ),
        )

    def _build_validation_priorities(self, position: "StrategicPosition") -> ValidationPrioritiesSection:
        ec = position.executive_confidence or {}
        return ValidationPrioritiesSection(
            priorities=list(ec.get("validation_priorities", [])),
            provenance=SectionProvenance(
                confidence_id=ec.get("confidence_id", ""),
                decision_model_id=(position.decision_model or {}).get("decision_model_id", ""),
            ),
        )

    def _build_appendix(self, position: "StrategicPosition") -> AppendixSection:
        ro = position.research_object or {}
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
            profiles=list(position.profiles or []),
            evidence_topics=dict(ro.get("evidence_topics", {})),
            citations=citations,
            provenance=SectionProvenance(
                research_object_id=ro.get("research_id", ""),
                decision_model_id=(position.decision_model or {}).get("decision_model_id", ""),
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
            strategic_direction=StrategyManuscriptSection(
                title="Strategic Direction",
                subtitle="",
                source_section_ids=["strategy_narrative"],
                provenance=ManuscriptProvenance(
                    brief_id=meta.brief_id,
                    brief_section_key="strategy_narrative",
                    decision_model_id=meta.decision_model_id,
                    research_object_id=brief.appendix.research_object_id,
                ),
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
        from .strategy_writer import StrategyWriter

        _registry = [
            ExecutiveSummaryWriter(client=client),
            DecisionAnalysisWriter(client=client),
            RecommendationWriter(client=client),
            RiskWriter(client=client),
            OpportunityWriter(client=client),
            ConfidenceWriter(client=client),
            AppendixWriter(client=client),
            StrategyWriter(client=client),
        ]

        # Pre-flight: verify no duplicate section_names and all writers declare one.
        seen: dict[str, str] = {}  # section_name → writer class name
        optional_sections: set[str] = set()  # sections declared optional by their writer
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
            if getattr(writer, "optional", False):
                optional_sections.add(sn)

        # Execute writers in order.
        for writer in _registry:
            writer.write(brief, manuscript)

        # Post-flight: verify every non-optional registered section is populated.
        unpopulated = [
            sn for sn in seen
            if not _section_populated(getattr(manuscript, sn))
            and sn not in optional_sections
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
