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
PH10.6: StrategySelector selects the winning TheoryOfWinning from evaluated
theories. StrategicPosition.theory_of_winning now comes from the selected
theory rather than the legacy _build_theory_of_winning() path.
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
from .alignment_evaluator import AlignmentEvaluator
from .configuration_resolver import ConfigurationResolver
from .constraint_evaluator import ConstraintEvaluator
from .option_mapper import OptionMapper
from .saturation_detector import SaturationDetector
from .strategic_choice_generator import StrategicChoiceGenerator
from .strategy_config import AlignmentPolicy, StrategyConfig
from .strategy_config_resolver import resolve_strategy_config
from .strategy_planner import StrategyPlanner
from .strategy_selector import StrategySelection, StrategySelector
from .strategy_lineage import build_strategy_lineage  # PH11.2
from .strategy_trace import StrategyTrace
from .content_graph import ContentGraph                   # PH12.2
from .content_resolver import ContentResolver              # PH12.2
from .content_differentiation import compute_differentiation  # PH12.2
from .discrimination_calculator import enrich_with_discrimination  # PH12.2b
from .theory_evaluator import TheoryEvaluator
from .theory_generator import TheoryGenerator

if TYPE_CHECKING:
    from ..context import AgentContext

LOGGER = logging.getLogger(__name__)


def _theory_strategic_signature(theory: "TheoryOfWinning") -> tuple:
    """Return a strategic deduplication key for a theory.

    Two theories are strategic duplicates when they share the same recommended
    option and choice pattern — different IDs or posture labels don't count
    as differentiation.
    """
    choice_pattern = tuple(sorted(
        (str(c.get("dimension", "")), str(c.get("selected_value", "")))
        for c in theory.strategic_choices
        if isinstance(c, dict)
    ))
    return (theory.recommended_option_id, choice_pattern)


def _check_theory_diversity(theories: list) -> None:
    """Raise ValueError when any two theories share the same strategic signature."""
    seen: dict[tuple, str] = {}
    for t in theories:
        sig = _theory_strategic_signature(t)
        if sig in seen:
            raise ValueError(
                f"[StrategyCoordinator] duplicate theory detected: "
                f"theory_id={t.theory_id!r} is strategically identical to "
                f"{seen[sig]!r}. "
                f"Set diversity_required=False or add more choices to configured dimensions."
            )
        seen[sig] = t.theory_id


class StrategyCoordinator:
    """Maps a completed AgentContext to a StrategicPosition.

    PH9.0: Accepts an optional StrategyConfig.
    PH9.1: Routes the config through ConfigurationResolver.
    PH9.3: Passes the resolved config through StrategyPlanner to produce
    a StrategyPlan.
    PH10.2: Invokes StrategicChoiceGenerator to produce three diverse
    StrategicChoiceSets (one per posture). Stored as _choice_sets.
    PH10.3: Invokes TheoryGenerator for each choice set to produce three
    TheoryOfWinning objects. Stored as _theories.
    PH10.5: Invokes TheoryEvaluator for each theory to produce three
    TheoryEvaluation objects. Stored as _evaluations.
    PH10.6: Invokes StrategySelector to pick the winning TheoryOfWinning.
    Stored as _selected_theory. StrategicPosition.theory_of_winning now
    reflects the selected theory rather than the legacy extraction path.
    """

    def __init__(
        self,
        config: StrategyConfig | None = None,
        raw_strategy_yaml: dict | None = None,
    ) -> None:
        raw = config if config is not None else StrategyConfig()
        self._config = ConfigurationResolver().resolve(raw)
        self._raw_strategy_yaml: dict = raw_strategy_yaml or {}
        self._plan = StrategyPlanner().build(self._config)
        self._choice_sets: list = []                   # set in build()
        self._theories: list = []                      # set in build()
        self._constraint_results: dict = {}            # set in build() — PH12.1
        self._evaluations: list = []                   # set in build()
        self._selected_theory: TheoryOfWinning | None = None   # set in build()
        self._selection: StrategySelection | None = None        # set in build()
        self._trace: StrategyTrace | None = None               # set in build()

    def build(self, ctx: "AgentContext") -> StrategicPosition:
        """Produce a StrategicPosition from a completed AgentContext.

        PH10.6 runtime:
          StrategyPlan → StrategicChoiceGenerator → list[StrategicChoiceSet]
          → TheoryGenerator (one per set) → list[TheoryOfWinning]
          → TheoryEvaluator (one per theory) → list[TheoryEvaluation]
          → StrategySelector → selected TheoryOfWinning
          → StrategicPosition (theory_of_winning from selected theory)

        Does not mutate ctx. Does not call an LLM. Does not generate prose.
        """
        # PH12.2a: resolve engagement strategy config upfront so all subsystems use it.
        _resolved_cfg = resolve_strategy_config(self._raw_strategy_yaml)

        # Wire evaluation_config criteria weights into the plan's evaluation model.
        # Weights are normalized to sum 1.0 (resolved_weight) for correct scoring.
        _eval_criteria = _resolved_cfg.canonical_snapshot.get("evaluation", {}).get("criteria", {})
        if _eval_criteria:
            _new_weights = {
                k: v["resolved_weight"]
                for k, v in _eval_criteria.items()
                if v.get("enabled", True) and v.get("resolved_weight", 0.0) > 0
            }
            if _new_weights:
                _new_em = self._plan.evaluation_model.model_copy(update={"weights": _new_weights})
                self._plan = self._plan.model_copy(update={"evaluation_model": _new_em})

        # PH10.2/PH12.0: generate StrategicChoiceSets (posture or configured)
        self._choice_sets = StrategicChoiceGenerator().build(self._plan, ctx)
        # PH10.3: generate one TheoryOfWinning per StrategicChoiceSet
        theory_gen = TheoryGenerator()
        self._theories = [theory_gen.build(cs, ctx) for cs in self._choice_sets]
        # PH12.0: validate theory diversity in configured mode only.
        # Legacy mode theories are expected to cluster around the preferred option.
        if self._plan.generation_policy.diversity_required and self._plan.dimension_configs:
            _check_theory_diversity(self._theories)
        # PH12.1: evaluate constraints per theory before scoring
        ce = ConstraintEvaluator()
        self._constraint_results = {
            t.theory_id: ce.evaluate(t, self._plan) for t in self._theories
        }
        # PH12.2b: evaluate theories WITHOUT content (content resolved after selection).
        # Restores PH12.1b ordering: evaluate → select → map → content_resolve.
        evaluator = TheoryEvaluator()
        self._evaluations = [
            evaluator.build(
                t, self._plan, ctx,
                self._constraint_results.get(t.theory_id),
                None,
            )
            for t in self._theories
        ]
        # PH12.1: detect score saturation before selection
        sat_detected, sat_msg = SaturationDetector().check(self._evaluations)
        # PH10.6: select the winning theory
        selector = StrategySelector()
        self._selected_theory = selector.select(
            self._theories, self._evaluations, self._plan
        )
        self._selection = selector._last_selection

        # PH12.2b: map ALL theories after selection (does not affect scores)
        mapper = OptionMapper(mapping_config=_resolved_cfg.resolved.mapping_config)
        all_mappings = {t.theory_id: mapper.map(t, ctx) for t in self._theories}
        winner_mapping = all_mappings[self._selection.winner_theory_id]

        _align_cfg = _resolved_cfg.resolved.alignment_config
        ap = AlignmentPolicy(
            minimum_challenge_margin=_align_cfg.minimum_challenge_margin,
            minimum_mapping_confidence=_align_cfg.minimum_challenge_confidence,
        )
        alignment = AlignmentEvaluator().evaluate(
            self._selected_theory, winner_mapping, self._selection, ctx, policy=ap
        )

        # Write-back: create new StrategySelection with alignment fields populated
        _tb = self._selection.tie_breaker_used
        _rationale = (
            f"Theory {self._selection.winner_theory_id} selected via "
            + (f"tie-breaker ({_tb})." if _tb else "highest score.")
        )
        _wm_extras = winner_mapping.model_extra or {}
        self._selection = StrategySelection(
            winner_theory_id=self._selection.winner_theory_id,
            winner_score=self._selection.winner_score,
            runner_up_theory_id=self._selection.runner_up_theory_id,
            runner_up_score=self._selection.runner_up_score,
            score_margin=self._selection.score_margin,
            tie_breaker_used=self._selection.tie_breaker_used,
            selection_status="selected",
            selection_rationale=_rationale,
            alignment_status=alignment.status,
            mapped_option_id=winner_mapping.mapped_option_id,
            saturation_detected=sat_detected,
            # PH12.2d: mapping metadata
            mapping_score=winner_mapping.mapping_score,
            mapping_confidence=winner_mapping.mapping_confidence,
            mapping_rationale=winner_mapping.mapping_rationale,
            mapping_status="mapped" if winner_mapping.mapped_option_id else "unmapped",
            mapping_margin=_wm_extras.get("mapping_margin"),
            runner_up_option_id=_wm_extras.get("runner_up_option_id"),
        )

        # PH12.2b: resolve theory content after selection
        content_cfg = _resolved_cfg.resolved.content
        _content_graph = ContentGraph().build(ctx)
        _content_resolver = ContentResolver(_content_graph, content_cfg)
        all_theory_contents: dict = {}
        for t in self._theories:
            t_mapping = all_mappings[t.theory_id]
            tc = _content_resolver.resolve(
                t, t_mapping.mapped_option_id, t_mapping.mapping_confidence
            )
            all_theory_contents[t.theory_id] = tc

        # PH12.2b: enrich content with per-item discrimination scores
        _tc_list_raw = [tc for tc in all_theory_contents.values() if tc is not None]
        _min_disc = content_cfg.minimum_discrimination_score
        try:
            _tc_list_raw = enrich_with_discrimination(
                _tc_list_raw, min_discrimination_score=_min_disc
            )
            all_theory_contents = {tc.theory_id: tc for tc in _tc_list_raw}
        except Exception as _disc_err:
            LOGGER.warning(
                "[StrategyCoordinator] discrimination enrichment skipped: %s", _disc_err
            )

        # PH12.2b: consistency guard — selection.mapped_option_id == winner theory_content
        winner_tc = all_theory_contents.get(self._selection.winner_theory_id)
        _consistency_guard: dict = {"passed": True, "rationale": "no winner theory_content"}
        if winner_tc is not None:
            tc_option = getattr(winner_tc, "mapped_option_id", None)
            sel_option = self._selection.mapped_option_id
            if tc_option and sel_option and tc_option != sel_option:
                _consistency_guard = {
                    "passed": False,
                    "winner_theory_id": self._selection.winner_theory_id,
                    "selection_mapped_option_id": sel_option,
                    "theory_content_mapped_option_id": tc_option,
                    "rationale": (
                        f"Mapping mismatch: selection says {sel_option!r} "
                        f"but winner theory_content says {tc_option!r}."
                    ),
                }
                LOGGER.error(
                    "[StrategyCoordinator] consistency guard FAILED: %s",
                    _consistency_guard["rationale"],
                )
            else:
                _consistency_guard = {
                    "passed": True,
                    "rationale": (
                        f"winner({self._selection.winner_theory_id}) "
                        f"maps to {sel_option!r} in both selection and theory_content."
                    ),
                }

        created_at = datetime.now(timezone.utc).isoformat()
        position_id = f"SP-{created_at[:10].replace('-', '')}-{(ctx.run_id or 'unknown')[:8]}"

        ec = ctx.executive_confidence or {}
        da = ctx.decision_analysis or {}

        # PH10.6: recommendation aligned with the selected theory
        recommendation = self._build_recommendation(
            self._selected_theory.recommended_option_id,
            self._selected_theory.recommended_option_title,
            ec,
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

        position = StrategicPosition(
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

            # Canonical spec structure — theory_of_winning from selector (PH10.6)
            theory_of_winning=self._selected_theory,
            recommendation=recommendation,
            justification=justification,
            execution=execution,
        )

        # PH11.0/PH11.2 — build StrategyTrace with full lineage chain
        _trace_id = f"STRAT-{self._plan.plan_id}"
        _ro = ctx.research_object or {}
        _research_id = _ro.get("id") or _ro.get("research_id") or ctx.run_id or None
        if not _research_id:
            raise ValueError(
                "StrategyCoordinator: no valid research or run identifier available "
                "for StrategyTrace lineage."
            )
        _lineage = build_strategy_lineage(
            research_id=_research_id,
            plan=self._plan,
            choice_sets=list(self._choice_sets),
            theories=list(self._theories),
            evaluations=list(self._evaluations),
            selection=self._selection,
            strategic_position=position,
            trace_id=_trace_id,
        )
        # PH12.1a — build structured audit blocks for StrategyTrace
        _theory_option_mappings = [
            {
                "theory_id": t.theory_id,
                "mapped_option_id": all_mappings[t.theory_id].mapped_option_id,
                "mapping_score": all_mappings[t.theory_id].mapping_score,
                "mapping_confidence": all_mappings[t.theory_id].mapping_confidence,
                "mapping_rationale": all_mappings[t.theory_id].mapping_rationale,
                "option_scores": all_mappings[t.theory_id].option_scores,
                "theory_postures": all_mappings[t.theory_id].theory_postures,
            }
            for t in self._theories
        ]
        _constraint_results_structured = {
            tid: [cr.model_dump() for cr in crs]
            for tid, crs in self._constraint_results.items()
        }
        _alignment_block = alignment.model_dump()
        _saturation_block = {"detected": sat_detected, "message": sat_msg}

        # PH12.2 — theory content trace blocks
        _theory_content_list = []
        _theory_content_lineage: dict = {}
        _theory_content_coverage: dict = {}
        _theory_content_confidence: dict = {}
        _all_content_fallbacks: list = []
        for t in self._theories:
            tc = all_theory_contents.get(t.theory_id)
            if tc is not None:
                _theory_content_list.append({
                    "theory_id": tc.theory_id,
                    "mapped_option_id": tc.mapped_option_id,
                    "recommendation_ids": tc.recommendation_ids,
                    "assumption_ids": tc.assumption_ids,
                    "risk_ids": tc.risk_ids,
                    "opportunity_ids": tc.opportunity_ids,
                    "evidence_ids": tc.evidence_ids,
                    "success_conditions": [sc.model_dump() for sc in tc.success_conditions],
                    "coverage": tc.coverage.model_dump(),
                    "confidence": tc.confidence.level,
                    # PH12.2b — distinctive/shared content split
                    "distinctive_assumption_ids": getattr(tc, "distinctive_assumption_ids", []),
                    "shared_assumption_ids": getattr(tc, "shared_assumption_ids", []),
                    "distinctive_risk_ids": getattr(tc, "distinctive_risk_ids", []),
                    "shared_risk_ids": getattr(tc, "shared_risk_ids", []),
                    "distinctive_opportunity_ids": getattr(tc, "distinctive_opportunity_ids", []),
                    "shared_opportunity_ids": getattr(tc, "shared_opportunity_ids", []),
                    "distinctive_recommendation_ids": getattr(tc, "distinctive_recommendation_ids", []),
                    "shared_recommendation_ids": getattr(tc, "shared_recommendation_ids", []),
                    "distinctive_evidence_ids": getattr(tc, "distinctive_evidence_ids", []),
                    "shared_evidence_ids": getattr(tc, "shared_evidence_ids", []),
                    "homogenization_state": getattr(tc, "homogenization_state", ""),
                })
                _theory_content_lineage[t.theory_id] = {
                    k: [e.model_dump() for e in v]
                    for k, v in tc.content_lineage.items()
                    if isinstance(v, list)
                }
                _theory_content_coverage[t.theory_id] = tc.coverage.model_dump()
                _theory_content_confidence[t.theory_id] = tc.confidence.model_dump()
                _all_content_fallbacks.extend(tc.content_fallbacks)

        # PH12.2a — build strategy configuration snapshot for trace
        # resolved uses the canonical short-name snapshot (same form used for fingerprinting)
        _strategy_config_snapshot = {
            "config_version": _resolved_cfg.config_version,
            "source": _resolved_cfg.source,
            "fingerprint": _resolved_cfg.fingerprint,
            "resolved": _resolved_cfg.canonical_snapshot,
            "defaults_applied": _resolved_cfg.defaults_applied,
            "deprecations": _resolved_cfg.deprecations,
            "warnings": _resolved_cfg.warnings,
        }

        _partial_threshold = content_cfg.partial_homogenization_threshold
        _full_threshold = content_cfg.full_homogenization_threshold
        _max_identical_dims = content_cfg.maximum_identical_dimensions
        _diff_result = compute_differentiation(
            [tc for tc in all_theory_contents.values() if tc is not None],
            partial_threshold=_partial_threshold,
            full_threshold=_full_threshold,
            maximum_identical_dimensions=_max_identical_dims,
        )
        _homogenization = _diff_result.get("homogenization_details", {})

        # PH12.2b: backfill homogenization_state into theory_content trace dicts
        _overall_hom_state = _diff_result.get("homogenization_state", "none") or "none"
        for _tc_dict in _theory_content_list:
            _tc_dict["homogenization_state"] = _overall_hom_state

        self._trace = StrategyTrace(
            trace_id=_trace_id,
            created_at=created_at,
            plan=self._plan,
            choice_sets=list(self._choice_sets),
            theories=list(self._theories),
            evaluations=list(self._evaluations),
            selection=self._selection,
            strategic_position=position,
            lineage=_lineage,
            theory_option_mappings=_theory_option_mappings,
            constraint_results=_constraint_results_structured,
            alignment=_alignment_block,
            saturation=_saturation_block,
            # PH12.2
            theory_content=_theory_content_list,
            theory_content_lineage=_theory_content_lineage,
            theory_content_coverage=_theory_content_coverage,
            theory_content_confidence=_theory_content_confidence,
            theory_differentiation=_diff_result.get("theory_differentiation", {}),
            content_homogenization=_homogenization,
            content_fallbacks=_all_content_fallbacks,
            # PH12.2b
            content_differentiation_state={
                "state": _overall_hom_state,
                "detected": _homogenization.get("detected", False),
                "identical_dimensions": _homogenization.get("identical_dimensions", []),
                "pairwise_similarity": _homogenization.get("pairwise_similarity", {}),
            },
            # PH12.2a — resolved strategy configuration snapshot
            strategy_configuration=_strategy_config_snapshot,
            strategy_config_fingerprint=_resolved_cfg.fingerprint,
            metadata={
                "framework": self._plan.framework,
                "plan_id": self._plan.plan_id,
                "choice_set_count": len(self._choice_sets),
                "theory_count": len(self._theories),
                "evaluation_count": len(self._evaluations),
                "selected_theory_id": self._selection.winner_theory_id,
                "score_margin": self._selection.score_margin,
                "tie_breaker_used": self._selection.tie_breaker_used,
                "research_id": _research_id,
                # PH12.1 diagnostics (kept in metadata for backward compatibility)
                "saturation_detected": sat_detected,
                "saturation_message": sat_msg,
                "alignment_status": alignment.status,
                "alignment_rationale": alignment.rationale,
                "mapped_option_id": alignment.mapped_option_id,
                # PH12.2b
                "consistency_guard": _consistency_guard,
            },
        )

        return position

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
            theory_id=f"TH-legacy-{recommended_id or 'unknown'}",
            source_choice_set_id="coordinator-legacy",
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
