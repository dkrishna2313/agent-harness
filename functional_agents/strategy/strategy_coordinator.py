"""StrategyCoordinator — maps AgentContext → StrategicPosition (PH8/PH9).

Responsibilities:
  - Consume AgentContext after the reasoning pipeline completes
  - Consume optional StrategyConfig (PH9.0)
  - Produce a StrategicPosition containing the selected strategy
  - Persist the position for diagnostics

Rules:
  - No LLM calls
  - No prose generation
  - No markdown or report generation
  - Never mutates AgentContext
  - Pure extraction and structuring of existing reasoning outputs

PH8: StrategyCoordinator is a pass-through — it structures AgentContext
reasoning outputs into the canonical StrategicPosition.
PH9.0: Accepts an optional StrategyConfig. If omitted, a default instance
is constructed. Behavior is unchanged — config is carried but not yet applied.
Future phases will use the config to drive StrategyPlan, TheoryGenerator,
and TheoryEvaluator.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .strategic_position import (
    StrategicExecution,
    StrategicJustification,
    StrategicPosition,
    StrategicRecommendation,
    TheoryOfWinning,
)
from .configuration_resolver import ConfigurationResolver
from .strategic_choice_generator import StrategicChoiceGenerator
from .strategy_config import StrategyConfig
from .strategy_planner import StrategyPlanner
from .theory_generator import TheoryGenerator

if TYPE_CHECKING:
    from ..context import AgentContext

LOGGER = logging.getLogger(__name__)


class StrategyCoordinator:
    """Maps a completed AgentContext to a StrategicPosition.

    PH9.0: Accepts an optional StrategyConfig.
    PH9.1: Routes the config through ConfigurationResolver.
    PH9.3: Passes the resolved config through StrategyPlanner to produce
    a StrategyPlan.
    PH10.2: Invokes StrategicChoiceGenerator to produce three diverse
    StrategicChoiceSets (one per posture). Stored as _choice_sets.
    PH10.3: Invokes TheoryGenerator for each choice set to produce three
    TheoryOfWinning objects. Stored as _theories. Neither _choice_sets nor
    _theories yet influence StrategicPosition construction.
    """

    def __init__(self, config: StrategyConfig | None = None) -> None:
        raw = config if config is not None else StrategyConfig()
        self._config = ConfigurationResolver().resolve(raw)
        self._plan = StrategyPlanner().build(self._config)
        self._choice_sets: list = []  # set in build(); empty until first call
        self._theories: list = []     # set in build(); empty until first call

    def build(self, ctx: "AgentContext") -> StrategicPosition:
        """Produce a StrategicPosition from a completed AgentContext.

        PH10.3 runtime:
          StrategyPlan → StrategicChoiceGenerator → list[StrategicChoiceSet]
          → TheoryGenerator (one per set) → list[TheoryOfWinning]
          → (existing StrategicPosition construction, unchanged)

        Does not mutate ctx. Does not call an LLM. Does not generate prose.
        _choice_sets and _theories are stored but do not yet influence
        the returned StrategicPosition.
        """
        # PH10.2: generate three diverse StrategicChoiceSets (one per posture)
        self._choice_sets = StrategicChoiceGenerator().build(self._plan, ctx)
        # PH10.3: generate one TheoryOfWinning per StrategicChoiceSet
        gen = TheoryGenerator()
        self._theories = [gen.build(cs, ctx) for cs in self._choice_sets]
        created_at = datetime.now(timezone.utc).isoformat()
        position_id = f"SP-{created_at[:10].replace('-', '')}-{(ctx.run_id or 'unknown')[:8]}"

        ec = ctx.executive_confidence or {}
        da = ctx.decision_analysis or {}
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

        theory = self._build_theory_of_winning(
            ctx, recommended_id, recommended_title, ec, da
        )
        recommendation = self._build_recommendation(
            recommended_id, recommended_title, ec
        )
        justification = StrategicJustification(
            decision_analysis=da,
            strategic_options=list(ctx.strategic_options or []),
            assumptions=list(ctx.assumptions or []),
            risks=list(ctx.risks or []),
            opportunities=list(ctx.opportunities or []),
        )
        execution = StrategicExecution(
            recommendations=list(ctx.recommendations or []),
            validation_priorities=list(ec.get("validation_priorities", [])),
        )

        return StrategicPosition(
            position_id=position_id,
            created_at=created_at,

            # Provenance
            run_id=ctx.run_id or "",
            question=ctx.question or "",
            profiles=list(ctx.profiles or []),
            execution_profile=ctx.execution_profile or "",
            decision_model=dict(ctx.decision_model or {}),
            engagement=dict(ctx.engagement or {}),
            preferred_option=dict(ctx.preferred_option or {}),
            research_object=dict(ctx.research_object or {}),

            # Raw reasoning outputs (consumed by EditorialCoordinator._build_*)
            decision_analysis=da,
            executive_confidence=ec,
            strategic_options=list(ctx.strategic_options or []),
            assumptions=list(ctx.assumptions or []),
            risks=list(ctx.risks or []),
            opportunities=list(ctx.opportunities or []),
            recommendations=list(ctx.recommendations or []),

            # Canonical spec structure
            theory_of_winning=theory,
            recommendation=recommendation,
            justification=justification,
            execution=execution,
        )

    def persist(
        self,
        position: StrategicPosition,
        base: Path = Path("outputs"),
        *,
        write_latest: bool = True,
    ) -> Path:
        """Persist the StrategicPosition to disk.

        Writes to outputs/strategic_positions/{position_id}.json.
        When write_latest=True (default), also updates
        latest_strategic_position.json.
        Returns the path of the versioned file.
        """
        out_dir = base / "strategic_positions"
        out_dir.mkdir(parents=True, exist_ok=True)

        data = position.to_dict()
        path = out_dir / f"{position.position_id}.json"
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

        if write_latest:
            latest = base / "latest_strategic_position.json"
            latest.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

        return path

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_theory_of_winning(
        self,
        ctx: "AgentContext",
        recommended_id: str,
        recommended_title: str,
        ec: dict[str, Any],
        da: dict[str, Any],
    ) -> TheoryOfWinning:
        # Winning position: rationale from decision_analysis or first key tradeoff
        key_tradeoffs = da.get("key_tradeoffs", [])
        winning_position = (
            da.get("rationale", "")
            or (key_tradeoffs[0] if key_tradeoffs else "")
        )

        # Winning mechanism: description of the recommended option
        winning_mechanism = ""
        for opt in (ctx.strategic_options or []):
            if opt.get("option_id") == recommended_id:
                winning_mechanism = opt.get("description", "")
                break

        # Success conditions: confidence drivers
        success_conditions = list(ec.get("confidence_drivers", []))

        # Failure modes: high-severity risks
        failure_modes = [
            r for r in (ctx.risks or [])
            if r.get("severity", "").lower() == "high"
        ]

        # Evidence: citations from research_object
        ro = ctx.research_object or {}
        citations_raw = ro.get("citations", []) or []
        evidence = [
            c if isinstance(c, str) else str(c.get("text", c.get("citation", "")))
            for c in citations_raw[:10]
        ]

        return TheoryOfWinning(
            recommended_option_id=recommended_id,
            recommended_option_title=recommended_title,
            winning_position=winning_position,
            winning_mechanism=winning_mechanism,
            strategic_choices=list(ctx.strategic_options or []),
            success_conditions=success_conditions,
            failure_modes=failure_modes,
            assumptions=list(ctx.assumptions or []),
            evidence=evidence,
            confidence=ec.get("overall_confidence", ""),
        )

    def _build_recommendation(
        self,
        recommended_id: str,
        recommended_title: str,
        ec: dict[str, Any],
    ) -> StrategicRecommendation:
        return StrategicRecommendation(
            recommended_option_id=recommended_id,
            recommended_option_title=recommended_title,
            board_recommendation=ec.get("board_recommendation", ""),
            decision_readiness=ec.get("decision_readiness", ""),
            overall_confidence=ec.get("overall_confidence", ""),
            key_conditions=list(ec.get("confidence_limiters", [])),
            critical_unknowns=list(ec.get("critical_unknowns", [])),
        )
