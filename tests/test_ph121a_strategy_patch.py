"""PH12.1a — Strategy Completion Patch tests.

Covers:
  - AlignmentPolicy / ScoringPolicy: defaults and validation
  - ConfigurationResolver: alignment/scoring block validation (margins, penalties, confidence)
  - StrategyPlanner: policy threading into StrategyPlan
  - Engagement override: alignment/scoring YAML block parsing
  - OptionMapper: posture-category scoring (concentrated / diversified / staged)
  - AlignmentEvaluator: policy-aware confirmed / challenged / unresolved
  - StrategySelection write-back: alignment_status, mapped_option_id populated after build()
  - StrategyTrace: structured audit fields (theory_option_mappings, constraint_results, alignment, saturation)
  - MarkdownRenderer: table separator no longer produces ||---|
  - StrategyNarrative: failure mode uses statement key, not description
  - Backward compat: StrategyTrace without new fields still validates
"""
from __future__ import annotations

import types

import pytest

from functional_agents.editorial.markdown_renderer import MarkdownRenderer
from functional_agents.editorial.strategy_narrative import (
    build_strategy_narrative,
)
from functional_agents.strategy.alignment import AlignmentResult, OptionMapping
from functional_agents.strategy.alignment_evaluator import AlignmentEvaluator
from functional_agents.strategy.configuration_resolver import ConfigurationResolver
from functional_agents.strategy.option_mapper import OptionMapper
from functional_agents.strategy.strategic_position import TheoryOfWinning
from functional_agents.strategy.strategy_config import (
    AlignmentPolicy,
    ScoringPolicy,
    StrategyConfig,
    _SUPPORTED_MAPPING_CONFIDENCES,
)
from functional_agents.strategy.strategy_planner import StrategyPlanner
from functional_agents.strategy.strategy_selector import StrategySelection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_theory(
    theory_id: str = "TH-001",
    choices: list[dict] | None = None,
    recommended_option_id: str = "",
    failure_modes: list[dict] | None = None,
) -> TheoryOfWinning:
    return TheoryOfWinning(
        theory_id=theory_id,
        source_choice_set_id="CS-01",
        recommended_option_id=recommended_option_id,
        strategic_choices=choices or [],
        failure_modes=failure_modes or [],
    )


def _make_selection(
    winner_id: str = "TH-001",
    margin: float = 0.10,
    tie_breaker: str | None = None,
) -> StrategySelection:
    return StrategySelection(
        winner_theory_id=winner_id,
        winner_score=0.75,
        score_margin=margin,
        tie_breaker_used=tie_breaker,
    )


def _make_research(preferred_id: str = "", options: list[dict] | None = None) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        preferred_option={"option_id": preferred_id} if preferred_id else {},
        strategic_options=options or [],
    )


# ---------------------------------------------------------------------------
# AlignmentPolicy / ScoringPolicy defaults
# ---------------------------------------------------------------------------

class TestAlignmentPolicyScoringPolicyDefaults:
    def test_alignment_policy_defaults(self):
        ap = AlignmentPolicy()
        assert ap.preferred_option_authority is True
        assert ap.minimum_challenge_margin == pytest.approx(0.05)
        assert ap.unresolved_on_tie is True
        assert ap.minimum_mapping_confidence == "Medium"

    def test_scoring_policy_defaults(self):
        sp = ScoringPolicy()
        assert sp.constraint_violation_penalty == pytest.approx(0.25)
        assert sp.partial_constraint_penalty == pytest.approx(0.10)
        assert sp.wait_and_monitor_penalty == pytest.approx(0.15)
        assert sp.saturation_detection is True

    def test_strategy_config_carries_policies(self):
        cfg = StrategyConfig()
        assert isinstance(cfg.alignment_policy, AlignmentPolicy)
        assert isinstance(cfg.scoring_policy, ScoringPolicy)

    def test_supported_mapping_confidences(self):
        assert _SUPPORTED_MAPPING_CONFIDENCES == {"High", "Medium", "Low", "None"}


# ---------------------------------------------------------------------------
# ConfigurationResolver – validation of alignment/scoring blocks
# ---------------------------------------------------------------------------

class TestConfigurationResolverPolicyValidation:
    def _resolver(self):
        return ConfigurationResolver()

    def test_invalid_minimum_challenge_margin_raises(self):
        cfg = StrategyConfig()
        with pytest.raises(ValueError, match="minimum_challenge_margin"):
            self._resolver().resolve_from_engagement(
                cfg,
                {"alignment": {"minimum_challenge_margin": -0.01}},
            )

    def test_invalid_mapping_confidence_raises(self):
        cfg = StrategyConfig()
        with pytest.raises(ValueError, match="minimum_mapping_confidence"):
            self._resolver().resolve_from_engagement(
                cfg,
                {"alignment": {"minimum_mapping_confidence": "VeryHigh"}},
            )

    def test_invalid_constraint_violation_penalty_raises(self):
        cfg = StrategyConfig()
        with pytest.raises(ValueError, match="constraint_violation_penalty"):
            self._resolver().resolve_from_engagement(
                cfg,
                {"scoring": {"constraint_violation_penalty": 1.5}},
            )

    def test_invalid_partial_constraint_penalty_raises(self):
        cfg = StrategyConfig()
        with pytest.raises(ValueError, match="partial_constraint_penalty"):
            self._resolver().resolve_from_engagement(
                cfg,
                {"scoring": {"partial_constraint_penalty": -0.1}},
            )

    def test_valid_alignment_block_parsed(self):
        cfg = StrategyConfig()
        resolved = ConfigurationResolver().resolve_from_engagement(
            cfg,
            {"alignment": {
                "minimum_challenge_margin": 0.10,
                "minimum_mapping_confidence": "High",
                "unresolved_on_tie": False,
            }},
        )
        ap = resolved.alignment_policy
        assert ap.minimum_challenge_margin == pytest.approx(0.10)
        assert ap.minimum_mapping_confidence == "High"
        assert ap.unresolved_on_tie is False

    def test_valid_scoring_block_parsed(self):
        cfg = StrategyConfig()
        resolved = ConfigurationResolver().resolve_from_engagement(
            cfg,
            {"scoring": {
                "constraint_violation_penalty": 0.30,
                "wait_and_monitor_penalty": 0.20,
            }},
        )
        sp = resolved.scoring_policy
        assert sp.constraint_violation_penalty == pytest.approx(0.30)
        assert sp.wait_and_monitor_penalty == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# StrategyPlanner – policy threading
# ---------------------------------------------------------------------------

class TestStrategyPlannerPolicyThreading:
    def test_plan_carries_alignment_policy(self):
        custom_ap = AlignmentPolicy(minimum_challenge_margin=0.12)
        cfg = StrategyConfig(alignment_policy=custom_ap)
        plan = StrategyPlanner().build(cfg)
        assert plan.alignment_policy.minimum_challenge_margin == pytest.approx(0.12)

    def test_plan_carries_scoring_policy(self):
        custom_sp = ScoringPolicy(constraint_violation_penalty=0.35)
        cfg = StrategyConfig(scoring_policy=custom_sp)
        plan = StrategyPlanner().build(cfg)
        assert plan.scoring_policy.constraint_violation_penalty == pytest.approx(0.35)

    def test_plan_default_policies_when_config_is_default(self):
        plan = StrategyPlanner().build(StrategyConfig())
        assert isinstance(plan.alignment_policy, AlignmentPolicy)
        assert isinstance(plan.scoring_policy, ScoringPolicy)


# ---------------------------------------------------------------------------
# OptionMapper – posture-category scoring
# ---------------------------------------------------------------------------

class TestOptionMapperPostureScoring:
    def _map(self, choice_value: str, options: list[dict]) -> OptionMapping:
        theory = _make_theory(
            choices=[{
                "dimension": "geographic_scope",
                "selected_value": choice_value,
                "metadata": {"choice_title": choice_value},
            }]
        )
        research = _make_research(options=options)
        return OptionMapper().map(theory, research)

    def test_concentrated_maps_to_conc_option(self):
        options = [
            {"option_id": "OPT-CONC", "title": "Concentrated", "description": "focused single-state strategy"},
            {"option_id": "OPT-DIV", "title": "Diversified", "description": "multi-state portfolio diversified spread"},
        ]
        result = self._map("concentrated", options)
        assert result.mapped_option_id == "OPT-CONC"
        assert result.mapping_confidence in ("Medium", "High", "Low")

    def test_diversified_maps_to_div_option(self):
        options = [
            {"option_id": "OPT-CONC", "title": "Concentrated", "description": "focused single-state strategy"},
            {"option_id": "OPT-DIV", "title": "Diversified", "description": "multi-state portfolio diversified spread"},
        ]
        result = self._map("diversified", options)
        assert result.mapped_option_id == "OPT-DIV"

    def test_staged_maps_to_staged_option(self):
        options = [
            {"option_id": "OPT-CONC", "title": "Concentrated", "description": "focused single-state strategy"},
            {"option_id": "OPT-STGD", "title": "Staged", "description": "phased staged optionality contingency approach"},
        ]
        result = self._map("staged", options)
        assert result.mapped_option_id == "OPT-STGD"

    def test_no_options_returns_none_confidence(self):
        theory = _make_theory(choices=[{"selected_value": "concentrated"}])
        result = OptionMapper().map(theory, _make_research(options=[]))
        assert result.mapped_option_id is None
        assert result.mapping_confidence == "None"

    def test_no_posture_tokens_returns_none_confidence(self):
        theory = _make_theory(choices=[])
        options = [{"option_id": "OPT-A", "description": "something"}]
        result = OptionMapper().map(theory, _make_research(options=options))
        assert result.mapped_option_id is None


# ---------------------------------------------------------------------------
# AlignmentEvaluator – policy-aware status
# ---------------------------------------------------------------------------

class TestAlignmentEvaluatorPolicy:
    def _eval(self, policy=None, **kwargs):
        return AlignmentEvaluator().evaluate(**kwargs, policy=policy)

    def test_confirmed_when_same_option_sufficient_margin(self):
        theory = _make_theory()
        mapping = OptionMapping(mapped_option_id="OPT-A", mapping_confidence="High", mapping_score=0.8)
        selection = _make_selection(margin=0.15)
        research = _make_research(preferred_id="OPT-A")
        result = self._eval(
            selected_theory=theory, option_mapping=mapping,
            selection=selection, research=research,
        )
        assert result.status == "confirmed"

    def test_refined_when_same_option_narrow_margin(self):
        theory = _make_theory()
        mapping = OptionMapping(mapped_option_id="OPT-A", mapping_confidence="High", mapping_score=0.8)
        selection = _make_selection(margin=0.02)
        research = _make_research(preferred_id="OPT-A")
        result = self._eval(
            selected_theory=theory, option_mapping=mapping,
            selection=selection, research=research,
        )
        assert result.status == "refined"

    def test_challenged_when_different_option_sufficient_margin(self):
        theory = _make_theory()
        mapping = OptionMapping(mapped_option_id="OPT-B", mapping_confidence="High", mapping_score=0.8)
        selection = _make_selection(margin=0.15)
        research = _make_research(preferred_id="OPT-A")
        result = self._eval(
            selected_theory=theory, option_mapping=mapping,
            selection=selection, research=research,
        )
        assert result.status == "challenged"

    def test_unresolved_when_confidence_below_minimum(self):
        theory = _make_theory()
        mapping = OptionMapping(mapped_option_id="OPT-A", mapping_confidence="Low", mapping_score=0.1)
        selection = _make_selection(margin=0.15)
        research = _make_research(preferred_id="OPT-A")
        # default policy requires Medium confidence minimum
        result = self._eval(
            selected_theory=theory, option_mapping=mapping,
            selection=selection, research=research,
        )
        assert result.status == "unresolved"

    def test_unresolved_when_no_preferred_option(self):
        theory = _make_theory()
        mapping = OptionMapping(mapped_option_id="OPT-A", mapping_confidence="High", mapping_score=0.8)
        selection = _make_selection(margin=0.15)
        research = _make_research(preferred_id="")
        result = self._eval(
            selected_theory=theory, option_mapping=mapping,
            selection=selection, research=research,
        )
        assert result.status == "unresolved"

    def test_custom_policy_minimum_challenge_margin(self):
        theory = _make_theory()
        mapping = OptionMapping(mapped_option_id="OPT-B", mapping_confidence="High", mapping_score=0.8)
        selection = _make_selection(margin=0.08)
        research = _make_research(preferred_id="OPT-A")
        # margin 0.08 is above 0.05 default → challenged; but custom policy requires 0.20
        strict_policy = AlignmentPolicy(minimum_challenge_margin=0.20)
        result = self._eval(
            policy=strict_policy,
            selected_theory=theory, option_mapping=mapping,
            selection=selection, research=research,
        )
        assert result.status == "unresolved"

    def test_policy_conf_rank_none_maps_below_any_minimum(self):
        theory = _make_theory()
        mapping = OptionMapping(mapped_option_id="OPT-A", mapping_confidence="None", mapping_score=0.0)
        selection = _make_selection(margin=0.15)
        research = _make_research(preferred_id="OPT-A")
        result = self._eval(
            selected_theory=theory, option_mapping=mapping,
            selection=selection, research=research,
        )
        assert result.status == "unresolved"


# ---------------------------------------------------------------------------
# StrategySelection write-back fields
# ---------------------------------------------------------------------------

class TestStrategySelectionWriteBack:
    def test_selection_fields_exist(self):
        sel = StrategySelection(winner_theory_id="TH-01", winner_score=0.8)
        assert sel.alignment_status == ""
        assert sel.mapped_option_id is None
        assert sel.saturation_detected is False
        assert sel.selection_status == "selected"
        assert sel.selection_rationale == ""


# ---------------------------------------------------------------------------
# MarkdownRenderer – table separator fix
# ---------------------------------------------------------------------------

class TestMarkdownRendererTableSeparator:
    def test_separator_is_valid_markdown(self):
        renderer = MarkdownRenderer()
        table = {
            "headers": ["Option", "Score", "Status"],
            "rows": [["OPT-A", "0.75", "selected"]],
        }
        lines = renderer._render_table(table)
        sep_line = lines[1] if len(lines) > 1 else ""
        # Must not produce ||---| pattern
        assert "||" not in sep_line
        # Must produce valid GFM separator
        assert sep_line == "| --- | --- | --- |"

    def test_separator_scales_with_column_count(self):
        renderer = MarkdownRenderer()
        table = {"headers": ["A", "B"], "rows": []}
        lines = renderer._render_table(table)
        assert lines[1] == "| --- | --- |"


# ---------------------------------------------------------------------------
# StrategyNarrative – failure mode uses statement key
# ---------------------------------------------------------------------------

class TestStrategyNarrativeFailureMode:
    def _make_trace(self, failure_modes):
        """Minimal trace stub for build_strategy_narrative."""
        theory = _make_theory("TH-W", failure_modes=failure_modes)
        sel = StrategySelection(winner_theory_id="TH-W", winner_score=0.8)

        from functional_agents.strategy.theory_evaluation import TheoryEvaluation, CriterionScore
        eval_ = TheoryEvaluation(
            theory_id="TH-W",
            overall_score=0.8,
            confidence="High",
            criteria_scores={},
            strengths=[],
            weaknesses=[],
            residual_risks=[],
        )

        from functional_agents.strategy.strategy_plan import StrategyPlan
        from functional_agents.strategy.strategy_config import StrategyConfig
        plan = StrategyPlanner().build(StrategyConfig())

        from functional_agents.strategy.strategic_position import (
            StrategicPosition, StrategicRecommendation, StrategicJustification, StrategicExecution,
        )
        pos = StrategicPosition(
            position_id="SP-001",
            created_at="2026-01-01T00:00:00+00:00",
            theory_of_winning=theory,
            recommendation=StrategicRecommendation(),
            justification=StrategicJustification(),
            execution=StrategicExecution(),
        )

        from functional_agents.strategy.strategic_choice_set import StrategicChoiceSet
        cs = StrategicChoiceSet(id="CS-01", choices=[], rationale="test", overall_confidence="Medium")

        from functional_agents.strategy.strategy_trace import StrategyTrace
        from functional_agents.strategy.strategy_lineage import StrategyLineageLink
        lineage = [
            StrategyLineageLink(
                source_type="research_object", source_id="R-001",
                target_type="strategy_trace", target_id="STRAT-plan",
                relationship="derived_from",
            )
        ]
        trace = StrategyTrace(
            trace_id="STRAT-plan",
            created_at="2026-01-01T00:00:00+00:00",
            plan=plan,
            choice_sets=[cs],
            theories=[theory],
            evaluations=[eval_],
            selection=sel,
            strategic_position=pos,
            lineage=lineage,
            metadata={"research_id": "R-001"},
        )
        return trace

    def test_failure_mode_prefers_statement(self):
        fm = {
            "statement": "Critical grid interconnection risk",
            "description": "wrong description field",
            "severity": "High",
        }
        trace = self._make_trace([fm])
        narrative = build_strategy_narrative(trace)
        assert len(narrative.failure_modes) == 1
        assert "Critical grid interconnection risk" in narrative.failure_modes[0]
        assert "wrong description field" not in narrative.failure_modes[0]

    def test_failure_mode_falls_back_to_description(self):
        fm = {
            "description": "description fallback",
            "severity": "Medium",
        }
        trace = self._make_trace([fm])
        narrative = build_strategy_narrative(trace)
        assert "description fallback" in narrative.failure_modes[0]

    def test_failure_mode_includes_severity(self):
        fm = {
            "statement": "Regulatory risk",
            "severity": "High",
            "likelihood": "Medium",
        }
        trace = self._make_trace([fm])
        narrative = build_strategy_narrative(trace)
        assert "Severity" in narrative.failure_modes[0] or "High" in narrative.failure_modes[0]


# ---------------------------------------------------------------------------
# StrategyTrace – structured audit fields exist and are backward-compatible
# ---------------------------------------------------------------------------

class TestStrategyTraceStructuredFields:
    def _minimal_trace(self, **extra_kwargs):
        from functional_agents.strategy.strategy_config import StrategyConfig
        plan = StrategyPlanner().build(StrategyConfig())

        theory = _make_theory("TH-W")
        sel = StrategySelection(winner_theory_id="TH-W", winner_score=0.8)

        from functional_agents.strategy.theory_evaluation import TheoryEvaluation
        eval_ = TheoryEvaluation(
            theory_id="TH-W",
            overall_score=0.8,
            confidence="High",
            criteria_scores={},
            strengths=[],
            weaknesses=[],
            residual_risks=[],
        )

        from functional_agents.strategy.strategic_position import (
            StrategicPosition, StrategicRecommendation, StrategicJustification, StrategicExecution,
        )
        pos = StrategicPosition(
            position_id="SP-001",
            created_at="2026-01-01T00:00:00+00:00",
            theory_of_winning=theory,
            recommendation=StrategicRecommendation(),
            justification=StrategicJustification(),
            execution=StrategicExecution(),
        )

        from functional_agents.strategy.strategic_choice_set import StrategicChoiceSet
        cs = StrategicChoiceSet(id="CS-01", choices=[], rationale="test", overall_confidence="Medium")

        from functional_agents.strategy.strategy_trace import StrategyTrace
        from functional_agents.strategy.strategy_lineage import StrategyLineageLink
        lineage = [
            StrategyLineageLink(
                source_type="research_object", source_id="R-001",
                target_type="strategy_trace", target_id="STRAT-plan",
                relationship="derived_from",
            )
        ]
        return StrategyTrace(
            trace_id="STRAT-plan",
            created_at="2026-01-01T00:00:00+00:00",
            plan=plan,
            choice_sets=[cs],
            theories=[theory],
            evaluations=[eval_],
            selection=sel,
            strategic_position=pos,
            lineage=lineage,
            metadata={"research_id": "R-001"},
            **extra_kwargs,
        )

    def test_trace_without_new_fields_validates(self):
        """Backward compat: trace without new fields still validates (defaults to empty)."""
        trace = self._minimal_trace()
        assert trace.theory_option_mappings == []
        assert trace.constraint_results == {}
        assert trace.alignment == {}
        assert trace.saturation == {}

    def test_trace_with_theory_option_mappings(self):
        mappings = [
            {"theory_id": "TH-W", "mapped_option_id": "OPT-A", "mapping_confidence": "High"}
        ]
        trace = self._minimal_trace(theory_option_mappings=mappings)
        assert trace.theory_option_mappings[0]["mapped_option_id"] == "OPT-A"

    def test_trace_with_alignment_block(self):
        alignment = {"status": "confirmed", "preferred_option_id": "OPT-A", "rationale": "match"}
        trace = self._minimal_trace(alignment=alignment)
        assert trace.alignment["status"] == "confirmed"

    def test_trace_with_saturation_block(self):
        saturation = {"detected": True, "message": "scores within 0.01 of each other"}
        trace = self._minimal_trace(saturation=saturation)
        assert trace.saturation["detected"] is True

    def test_trace_with_constraint_results(self):
        cr = {"TH-W": [{"constraint": "required_condition:grid", "status": "satisfied", "score": 1.0}]}
        trace = self._minimal_trace(constraint_results=cr)
        assert trace.constraint_results["TH-W"][0]["status"] == "satisfied"
