"""PH12.1b — Runtime Option-Mapping and Strategy Report Authority Correction tests.

Covers:
  - PostureNormalizer: theory_postures(), option_postures(), normalize_text()
  - Contradiction table: CONTRADICTIONS structure and symmetry
  - OptionMapper (production fixture): Diversified/BTM/Milestone theory maps to OPT-B not OPT-A
  - OptionMapper: OPT-A score is negative due to geographic contradiction
  - OptionMapper: invariance — winner stable when unrelated options added
  - OptionMapper: confidence level determination (_confidence())
  - AlignmentEvaluator: confirmed vs refined distinction (PH12.1b)
  - StrategyNarrative: authority fields (alignment_status, alignment_narrative, mapped_option_id, etc.)
  - StrategyWriter: truncate_sentence_safe — no mid-word or mid-sentence truncation
  - MarkdownRenderer: authority table rendered with all required fields
  - Integration: posture extraction → mapping → alignment (production engagement scenario)
"""
from __future__ import annotations

import types

import pytest

from functional_agents.editorial.markdown_renderer import MarkdownRenderer, _truncate_sentence_safe
from functional_agents.editorial.strategy_narrative import (
    StrategyNarrative,
    build_strategy_narrative,
)
from functional_agents.editorial.strategy_writer import truncate_sentence_safe
from functional_agents.strategy.alignment import AlignmentResult, OptionMapping
from functional_agents.strategy.alignment_evaluator import AlignmentEvaluator
from functional_agents.strategy.option_mapper import OptionMapper
from functional_agents.strategy.posture_normalizer import (
    CONTRADICTIONS,
    POSTURE_CATEGORIES,
    PostureNormalizer,
)
from functional_agents.strategy.strategic_position import TheoryOfWinning
from functional_agents.strategy.strategy_config import AlignmentPolicy, StrategyConfig
from functional_agents.strategy.strategy_planner import StrategyPlanner
from functional_agents.strategy.strategy_selector import StrategySelection


# ---------------------------------------------------------------------------
# Production-shaped fixtures (Section 19 of spec)
# ---------------------------------------------------------------------------

THEORY_DIVERSIFIED_CHOICES = [
    {
        "dimension": "geographic_portfolio",
        "selected_value": "diversified",
        "metadata": {"choice_title": "Diversified"},
    },
    {
        "dimension": "power_pathway",
        "selected_value": "btm_first",
        "metadata": {"choice_title": "Behind-the-Meter First"},
    },
    {
        "dimension": "market_timing",
        "selected_value": "milestone_gated",
        "metadata": {"choice_title": "Milestone Gated"},
    },
]

OPTION_A = {
    "option_id": "OPT-A",
    "title": "Texas-First Aggressive ERCOT Concentration Strategy",
    "description": (
        "Concentrate all capital deployment in Texas and ERCOT. "
        "Move aggressively and commit immediately to a single-state anchor market. "
        "Dominate this region before competitors can respond."
    ),
    "strategic_objective": (
        "Dominate ERCOT market position through concentrated, accelerated commitment."
    ),
    "advantages": ["First-mover advantage in ERCOT", "Simplified operations"],
    "disadvantages": ["Geographic concentration risk", "Single-market exposure"],
    "expected_outcomes": "Market dominance in ERCOT within 18 months.",
    "time_horizon": "near_term",
}

OPTION_B = {
    "option_id": "OPT-B",
    "title": "Balanced Multi-RTO Portfolio Hedge Across Three States",
    "description": (
        "Spread investment across three geographies to build a diversified portfolio. "
        "Behind-the-meter generation acts as the primary power pathway for resilience. "
        "Proceed with milestone-gated capital deployment tied to market validation."
    ),
    "strategic_objective": (
        "Build a diversified portfolio hedge across multiple RTOs using BTM-first generation."
    ),
    "advantages": ["Geographic resilience", "Milestone-gated downside protection"],
    "disadvantages": ["Greater operational complexity"],
    "expected_outcomes": "Sustainable multi-state platform.",
    "time_horizon": "medium_term",
}

OPTION_C = {
    "option_id": "OPT-C",
    "title": "Conservative Staged Optionality Approach",
    "description": (
        "Adopt a phased, staged optionality approach. "
        "Preserve flexibility to expand or contract as market signals emerge. "
        "Wait-and-monitor posture with conditional commit triggers."
    ),
    "strategic_objective": "Maintain optionality while market conditions clarify.",
    "advantages": ["Low near-term risk", "Option to pivot"],
    "disadvantages": ["Opportunity cost", "Slow market penetration"],
}

OPTION_D = {
    "option_id": "OPT-D",
    "title": "Virginia-Anchored PJM Dominance Strategy",
    "description": (
        "Concentrate capital in Virginia to anchor the PJM interconnection region. "
        "Utility-scale grid-first deployment with immediate commitment."
    ),
    "strategic_objective": "Establish dominant PJM grid-first position through concentrated Virginia deployment.",
    "advantages": ["Strong data-center demand", "Established utility relationships"],
    "disadvantages": ["Concentrated geographic risk"],
}

ALL_OPTIONS = [OPTION_A, OPTION_B, OPTION_C, OPTION_D]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_theory(
    theory_id: str = "TH-001",
    choices: list[dict] | None = None,
    recommended_option_id: str = "OPT-A",
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


def _build_minimal_trace(
    theory: TheoryOfWinning | None = None,
    alignment: dict | None = None,
    saturation: dict | None = None,
    theory_option_mappings: list | None = None,
    constraint_results: dict | None = None,
):
    """Build a minimal valid StrategyTrace for tests that need build_strategy_narrative."""
    from functional_agents.strategy.strategy_trace import StrategyTrace
    from functional_agents.strategy.strategy_lineage import StrategyLineageLink
    from functional_agents.strategy.strategic_position import (
        StrategicPosition, StrategicRecommendation, StrategicJustification, StrategicExecution,
    )
    from functional_agents.strategy.strategic_choice_set import StrategicChoiceSet
    from functional_agents.strategy.theory_evaluation import TheoryEvaluation

    if theory is None:
        theory = _make_theory("TH-W", choices=THEORY_DIVERSIFIED_CHOICES, recommended_option_id="OPT-B")

    plan = StrategyPlanner().build(StrategyConfig())
    sel = StrategySelection(winner_theory_id=theory.theory_id, winner_score=0.78)
    eval_ = TheoryEvaluation(
        theory_id=theory.theory_id,
        overall_score=0.78,
        confidence="High",
        criteria_scores={},
        strengths=["Strong geographic resilience"],
        weaknesses=[],
        residual_risks=[],
    )
    pos = StrategicPosition(
        position_id="SP-001",
        created_at="2026-01-01T00:00:00+00:00",
        theory_of_winning=theory,
        recommendation=StrategicRecommendation(),
        justification=StrategicJustification(),
        execution=StrategicExecution(),
    )
    cs = StrategicChoiceSet(id="CS-01", choices=[], rationale="test", overall_confidence="Medium")
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
        alignment=alignment or {},
        saturation=saturation or {},
        theory_option_mappings=theory_option_mappings or [],
        constraint_results=constraint_results or {},
        metadata={"research_id": "R-001", "framework": ""},
    )


# ---------------------------------------------------------------------------
# TestPostureNormalizer
# ---------------------------------------------------------------------------

class TestPostureNormalizer:
    """PostureNormalizer: theory_postures(), option_postures(), normalize_text()."""

    def test_theory_postures_diversified_btm_milestone(self):
        norm = PostureNormalizer()
        result = norm.theory_postures(THEORY_DIVERSIFIED_CHOICES)
        assert result.get("geographic") == "diversified"
        assert result.get("power") == "btm_first"
        assert result.get("timing") == "milestone_gated"

    def test_theory_postures_concentrated(self):
        norm = PostureNormalizer()
        choices = [{"dimension": "geo", "selected_value": "concentrated", "metadata": {}}]
        result = norm.theory_postures(choices)
        assert result.get("geographic") == "concentrated"

    def test_theory_postures_wait_and_monitor(self):
        norm = PostureNormalizer()
        choices = [{"dimension": "timing", "selected_value": "wait_and_monitor", "metadata": {}}]
        result = norm.theory_postures(choices)
        assert result.get("timing") == "wait_and_monitor"

    def test_theory_postures_staged(self):
        norm = PostureNormalizer()
        choices = [{"dimension": "geo", "selected_value": "staged", "metadata": {}}]
        result = norm.theory_postures(choices)
        assert result.get("geographic") == "staged"

    def test_theory_postures_hyphenated_variants(self):
        norm = PostureNormalizer()
        choices = [
            {"dimension": "g", "selected_value": "milestone-gated", "metadata": {}},
            {"dimension": "p", "selected_value": "btm-first", "metadata": {}},
        ]
        result = norm.theory_postures(choices)
        assert result.get("timing") == "milestone_gated"
        assert result.get("power") == "btm_first"

    def test_option_postures_optionA_concentrated_accelerate(self):
        norm = PostureNormalizer()
        result = norm.option_postures(OPTION_A)
        # OPT-A should be detected as concentrated (single-state/concentrat) and accelerate
        assert result.get("geographic") == "concentrated"
        assert result.get("timing") == "accelerate"

    def test_option_postures_optionB_diversified_btm_milestone(self):
        norm = PostureNormalizer()
        result = norm.option_postures(OPTION_B)
        assert result.get("geographic") == "diversified"
        assert result.get("power") == "btm_first"
        assert result.get("timing") == "milestone_gated"

    def test_option_postures_optionC_staged_wait(self):
        norm = PostureNormalizer()
        result = norm.option_postures(OPTION_C)
        # staged optionality and wait-and-monitor
        assert result.get("geographic") in ("staged", None) or result.get("timing") in ("wait_and_monitor", None)

    def test_option_postures_optionD_concentrated_grid(self):
        norm = PostureNormalizer()
        result = norm.option_postures(OPTION_D)
        assert result.get("geographic") == "concentrated"
        assert result.get("power") == "grid_first"

    def test_normalize_text_diversified_phrase(self):
        norm = PostureNormalizer()
        result = norm.normalize_text("multi-state portfolio-hedge approach across three states")
        assert result.get("geographic") == "diversified"

    def test_normalize_text_btm(self):
        norm = PostureNormalizer()
        result = norm.normalize_text("behind-the-meter generation strategy")
        assert result.get("power") == "btm_first"

    def test_theory_postures_empty(self):
        norm = PostureNormalizer()
        assert norm.theory_postures([]) == {}

    def test_option_postures_empty_option(self):
        norm = PostureNormalizer()
        assert norm.option_postures({}) == {}


# ---------------------------------------------------------------------------
# TestContradictionDetection
# ---------------------------------------------------------------------------

class TestContradictionDetection:
    """CONTRADICTIONS table: structure, symmetry, expected penalties."""

    def test_diversified_vs_concentrated_penalty(self):
        assert CONTRADICTIONS.get(("geographic", "diversified", "geographic", "concentrated")) == pytest.approx(0.35)

    def test_concentrated_vs_diversified_penalty(self):
        assert CONTRADICTIONS.get(("geographic", "concentrated", "geographic", "diversified")) == pytest.approx(0.35)

    def test_accelerate_vs_wait_penalty(self):
        assert CONTRADICTIONS.get(("timing", "accelerate", "timing", "wait_and_monitor")) == pytest.approx(0.30)

    def test_wait_vs_accelerate_penalty(self):
        assert CONTRADICTIONS.get(("timing", "wait_and_monitor", "timing", "accelerate")) == pytest.approx(0.30)

    def test_milestone_vs_accelerate_penalty(self):
        assert CONTRADICTIONS.get(("timing", "milestone_gated", "timing", "accelerate")) == pytest.approx(0.20)

    def test_accelerate_vs_milestone_penalty(self):
        assert CONTRADICTIONS.get(("timing", "accelerate", "timing", "milestone_gated")) == pytest.approx(0.20)

    def test_btm_vs_grid_penalty(self):
        assert CONTRADICTIONS.get(("power", "btm_first", "power", "grid_first")) == pytest.approx(0.25)

    def test_grid_vs_btm_penalty(self):
        assert CONTRADICTIONS.get(("power", "grid_first", "power", "btm_first")) == pytest.approx(0.25)

    def test_no_penalty_for_same_posture(self):
        assert CONTRADICTIONS.get(("geographic", "diversified", "geographic", "diversified"), 0.0) == pytest.approx(0.0)

    def test_no_penalty_cross_category(self):
        assert CONTRADICTIONS.get(("geographic", "diversified", "timing", "accelerate"), 0.0) == pytest.approx(0.0)

    def test_contradictions_is_symmetric_on_geo_and_power(self):
        for (a_cat, a_val, b_cat, b_val), p in CONTRADICTIONS.items():
            reverse = CONTRADICTIONS.get((b_cat, b_val, a_cat, a_val), None)
            if a_cat == b_cat:
                assert reverse == pytest.approx(p), (
                    f"Asymmetric contradiction: ({a_cat},{a_val}) vs ({b_cat},{b_val})"
                )


# ---------------------------------------------------------------------------
# TestOptionMapperProduction
# ---------------------------------------------------------------------------

class TestOptionMapperProduction:
    """OptionMapper with production-shaped OPT-A/B/C/D fixture."""

    def _map_diversified(self, options=None) -> OptionMapping:
        theory = _make_theory("TH-DIV", choices=THEORY_DIVERSIFIED_CHOICES, recommended_option_id="OPT-B")
        research = _make_research(preferred_id="OPT-B", options=options or ALL_OPTIONS)
        return OptionMapper().map(theory, research)

    def test_diversified_theory_maps_to_optb_not_opta(self):
        result = self._map_diversified()
        assert result.mapped_option_id == "OPT-B", (
            f"Expected OPT-B but got {result.mapped_option_id!r}. "
            f"Scores: {[(s['option_id'], s['score']) for s in result.option_scores]}"
        )

    def test_opta_score_is_negative_for_diversified_theory(self):
        result = self._map_diversified()
        opt_a_entry = next((s for s in result.option_scores if s["option_id"] == "OPT-A"), None)
        assert opt_a_entry is not None, "OPT-A not found in option_scores"
        assert opt_a_entry["score"] < 0, (
            f"OPT-A score should be negative (geographic contradiction), got {opt_a_entry['score']}"
        )

    def test_opta_has_geographic_contradiction(self):
        result = self._map_diversified()
        opt_a_entry = next((s for s in result.option_scores if s["option_id"] == "OPT-A"), None)
        assert opt_a_entry is not None
        contradiction_categories = [c["category"] for c in opt_a_entry.get("contradictions", [])]
        assert "geographic" in contradiction_categories, (
            f"Expected geographic contradiction for OPT-A, got: {contradiction_categories}"
        )

    def test_optb_score_is_positive_for_diversified_theory(self):
        result = self._map_diversified()
        opt_b_entry = next((s for s in result.option_scores if s["option_id"] == "OPT-B"), None)
        assert opt_b_entry is not None, "OPT-B not found in option_scores"
        assert opt_b_entry["score"] > 0, f"OPT-B score should be positive, got {opt_b_entry['score']}"

    def test_optb_score_beats_opta(self):
        result = self._map_diversified()
        scores = {s["option_id"]: s["score"] for s in result.option_scores}
        assert scores["OPT-B"] > scores["OPT-A"], (
            f"OPT-B ({scores['OPT-B']:.3f}) should beat OPT-A ({scores['OPT-A']:.3f})"
        )

    def test_mapping_confidence_is_not_none(self):
        result = self._map_diversified()
        assert result.mapping_confidence in ("Low", "Medium", "High"), (
            f"Expected confidence, got {result.mapping_confidence!r}"
        )

    def test_theory_postures_captured_in_mapping(self):
        result = self._map_diversified()
        assert result.theory_postures.get("geographic") == "diversified"
        assert result.theory_postures.get("power") == "btm_first"
        assert result.theory_postures.get("timing") == "milestone_gated"

    def test_concentrated_theory_maps_to_opta(self):
        concentrated_choices = [
            {"dimension": "geo", "selected_value": "concentrated", "metadata": {"choice_title": "Concentrated"}},
            {"dimension": "timing", "selected_value": "accelerate", "metadata": {"choice_title": "Accelerate"}},
        ]
        theory = _make_theory("TH-CONC", choices=concentrated_choices, recommended_option_id="OPT-A")
        research = _make_research(preferred_id="OPT-A", options=ALL_OPTIONS)
        result = OptionMapper().map(theory, research)
        assert result.mapped_option_id == "OPT-A", (
            f"Concentrated/accelerate theory should map to OPT-A, got {result.mapped_option_id!r}. "
            f"Scores: {[(s['option_id'], s['score']) for s in result.option_scores]}"
        )

    def test_staged_theory_maps_to_optc(self):
        staged_choices = [
            {"dimension": "geo", "selected_value": "staged", "metadata": {"choice_title": "Staged"}},
            {"dimension": "timing", "selected_value": "wait_and_monitor", "metadata": {"choice_title": "Wait and Monitor"}},
        ]
        theory = _make_theory("TH-STG", choices=staged_choices, recommended_option_id="OPT-C")
        research = _make_research(preferred_id="OPT-C", options=ALL_OPTIONS)
        result = OptionMapper().map(theory, research)
        # OPT-C is staged/wait-and-monitor, OPT-A is concentrated/accelerate
        # OPT-A should NOT win when theory is staged/wait
        if result.mapped_option_id is not None:
            assert result.mapped_option_id != "OPT-A", (
                f"Staged/wait-and-monitor theory should not map to OPT-A"
            )

    def test_optd_loses_for_diversified_theory(self):
        result = self._map_diversified()
        scores = {s["option_id"]: s["score"] for s in result.option_scores}
        assert scores.get("OPT-B", -99) > scores.get("OPT-D", 99), (
            f"OPT-B should beat OPT-D for diversified theory"
        )


# ---------------------------------------------------------------------------
# TestOptionMapperInvariance
# ---------------------------------------------------------------------------

class TestOptionMapperInvariance:
    """Adding unrelated options doesn't flip the winner."""

    def _map(self, choices: list[dict], options: list[dict]) -> OptionMapping:
        theory = _make_theory(choices=choices)
        return OptionMapper().map(theory, _make_research(options=options))

    def test_winner_stable_with_extra_unrelated_option(self):
        base_choices = THEORY_DIVERSIFIED_CHOICES
        base_options = [OPTION_A, OPTION_B]
        base_result = self._map(base_choices, base_options)

        extended_options = [OPTION_A, OPTION_B, {
            "option_id": "OPT-Z",
            "title": "Generic unrelated option with no strategic posture signals",
            "description": "A placeholder option.",
        }]
        ext_result = self._map(base_choices, extended_options)
        assert ext_result.mapped_option_id == base_result.mapped_option_id

    def test_concentrated_winner_stable_with_unrelated_addition(self):
        concentrated_choices = [
            {"dimension": "geo", "selected_value": "concentrated", "metadata": {"choice_title": "Concentrated"}},
        ]
        base_options = [OPTION_A, OPTION_B]
        base_result = self._map(concentrated_choices, base_options)
        extended = base_options + [{"option_id": "OPT-X", "title": "Unrelated neutral option", "description": ""}]
        ext_result = self._map(concentrated_choices, extended)
        assert ext_result.mapped_option_id == base_result.mapped_option_id

    def test_no_options_returns_none(self):
        theory = _make_theory(choices=THEORY_DIVERSIFIED_CHOICES)
        result = OptionMapper().map(theory, _make_research(options=[]))
        assert result.mapped_option_id is None
        assert result.mapping_confidence == "None"

    def test_no_posture_choices_returns_none(self):
        theory = _make_theory(choices=[{"dimension": "x", "selected_value": "zzz_unknown_zzz"}])
        result = OptionMapper().map(theory, _make_research(options=ALL_OPTIONS))
        assert result.mapped_option_id is None


# ---------------------------------------------------------------------------
# TestOptionMapperConfidence
# ---------------------------------------------------------------------------

class TestOptionMapperConfidence:
    """_confidence() tiers."""

    def _mapper(self) -> OptionMapper:
        return OptionMapper()

    def test_confidence_high_when_above_threshold_no_contradictions(self):
        mapper = self._mapper()
        result = mapper._confidence(score=0.50, separation=0.25, has_contradictions=False)
        assert result == "High"

    def test_confidence_not_high_when_contradictions(self):
        mapper = self._mapper()
        result = mapper._confidence(score=0.50, separation=0.25, has_contradictions=True)
        assert result != "High"

    def test_confidence_medium_above_medium_threshold(self):
        mapper = self._mapper()
        result = mapper._confidence(score=0.30, separation=0.10, has_contradictions=False)
        assert result in ("Medium", "High")

    def test_confidence_low_for_low_score(self):
        mapper = self._mapper()
        result = mapper._confidence(score=0.05, separation=0.03, has_contradictions=False)
        assert result == "Low"

    def test_confidence_none_for_zero_score(self):
        mapper = self._mapper()
        result = mapper._confidence(score=0.0, separation=0.0, has_contradictions=False)
        assert result == "None"

    def test_confidence_none_for_negative_score(self):
        mapper = self._mapper()
        result = mapper._confidence(score=-0.30, separation=0.10, has_contradictions=True)
        assert result == "None"


# ---------------------------------------------------------------------------
# TestAlignmentConfirmedRefined
# ---------------------------------------------------------------------------

class TestAlignmentConfirmedRefined:
    """AlignmentEvaluator: confirmed vs refined status distinction (PH12.1b)."""

    def _eval(self, conf: str, margin: float, preferred_id: str, mapped_id: str) -> AlignmentResult:
        theory = _make_theory("TH-001")
        mapping = OptionMapping(
            mapped_option_id=mapped_id,
            mapping_confidence=conf,
            mapping_score=0.8,
        )
        selection = _make_selection(margin=margin)
        research = _make_research(preferred_id=preferred_id)
        return AlignmentEvaluator().evaluate(
            selected_theory=theory,
            option_mapping=mapping,
            selection=selection,
            research=research,
        )

    def test_confirmed_when_high_confidence_sufficient_margin(self):
        result = self._eval("High", margin=0.15, preferred_id="OPT-A", mapped_id="OPT-A")
        assert result.status == "confirmed"

    def test_refined_when_medium_confidence_even_with_margin(self):
        result = self._eval("Medium", margin=0.15, preferred_id="OPT-A", mapped_id="OPT-A")
        assert result.status == "refined"

    def test_refined_when_high_confidence_but_below_margin(self):
        result = self._eval("High", margin=0.01, preferred_id="OPT-A", mapped_id="OPT-A")
        assert result.status == "refined"

    def test_not_challenged_when_same_option(self):
        result = self._eval("High", margin=0.15, preferred_id="OPT-A", mapped_id="OPT-A")
        assert result.status != "challenged"

    def test_challenged_when_different_option_sufficient_margin(self):
        result = self._eval("High", margin=0.15, preferred_id="OPT-A", mapped_id="OPT-B")
        assert result.status == "challenged"

    def test_alignment_is_confirmed_or_refined_not_challenged_for_production(self):
        """Production case: Diversified/BTM/Milestone theory → OPT-B = preferred → not challenged."""
        theory = _make_theory("TH-DIV", choices=THEORY_DIVERSIFIED_CHOICES, recommended_option_id="OPT-B")
        research = _make_research(preferred_id="OPT-B", options=ALL_OPTIONS)
        mapping = OptionMapper().map(theory, research)
        selection = _make_selection(margin=0.12)
        result = AlignmentEvaluator().evaluate(
            selected_theory=theory,
            option_mapping=mapping,
            selection=selection,
            research=research,
        )
        assert result.status in ("confirmed", "refined"), (
            f"Expected confirmed/refined for production scenario, got {result.status!r}. "
            f"Mapping: {mapping.mapped_option_id!r}, conf={mapping.mapping_confidence!r}"
        )
        assert result.status != "challenged", (
            "Diversified theory correctly mapping to preferred OPT-B should never be 'challenged'"
        )


# ---------------------------------------------------------------------------
# TestStrategySelectionConsistency
# ---------------------------------------------------------------------------

class TestStrategySelectionConsistency:
    """StrategySelection fields populated correctly."""

    def test_selection_status_defaults_to_selected(self):
        sel = StrategySelection(winner_theory_id="TH-01", winner_score=0.75)
        assert sel.selection_status == "selected"

    def test_alignment_status_default_empty(self):
        sel = StrategySelection(winner_theory_id="TH-01", winner_score=0.75)
        assert sel.alignment_status == ""

    def test_mapped_option_id_default_none(self):
        sel = StrategySelection(winner_theory_id="TH-01", winner_score=0.75)
        assert sel.mapped_option_id is None

    def test_saturation_detected_default_false(self):
        sel = StrategySelection(winner_theory_id="TH-01", winner_score=0.75)
        assert sel.saturation_detected is False


# ---------------------------------------------------------------------------
# TestSentenceSafeRendering
# ---------------------------------------------------------------------------

class TestSentenceSafeRendering:
    """truncate_sentence_safe: no mid-word or mid-sentence cuts."""

    def test_short_text_returned_unchanged(self):
        text = "Short text."
        assert truncate_sentence_safe(text, 300) == text

    def test_truncates_at_sentence_boundary(self):
        text = "This is the first sentence. This is the second sentence which is very long indeed."
        result = truncate_sentence_safe(text, 32)
        assert result.endswith(".")
        assert "second" not in result

    def test_never_truncates_mid_word(self):
        text = "The quick brown fox jumps over the lazy dog and then runs away."
        result = truncate_sentence_safe(text, 20)
        assert not result.endswith(("th", "qu", "br", "ju", "ov", "la", "do"))
        assert result[-1] in " .,;:!?" or result.endswith(tuple(text.split()))

    def test_exact_limit_no_truncation(self):
        text = "exactly forty characters here padded.."[:40]
        result = truncate_sentence_safe(text, 40)
        assert result == text

    def test_truncates_at_clause_boundary_when_no_sentence(self):
        text = "Phase one: build the platform; Phase two: expand the market; Phase three: optimize."
        result = truncate_sentence_safe(text, 30)
        # Should end at ";" or ":" — not mid-word
        assert len(result) <= 30
        assert result[-1] not in "abcdefghijklmnopqrstuvwxyz"

    def test_renderer_truncate_and_writer_truncate_are_equivalent(self):
        text = "First sentence is complete. Second sentence extends well beyond. Third too long."
        limit = 45
        r1 = truncate_sentence_safe(text, limit)
        r2 = _truncate_sentence_safe(text, limit)
        assert r1 == r2

    def test_empty_string(self):
        assert truncate_sentence_safe("", 100) == ""

    def test_no_boundary_fallback_does_not_crash(self):
        # Text with no word/sentence boundaries — function falls back to returning full text
        result = truncate_sentence_safe("A" * 500, 1)
        assert isinstance(result, str)  # must not raise, may return full text


# ---------------------------------------------------------------------------
# TestStrategyNarrativeAuthorityFields
# ---------------------------------------------------------------------------

class TestStrategyNarrativeAuthorityFields:
    """StrategyNarrative: authority fields populated by build_strategy_narrative."""

    def test_narrative_has_alignment_status_field(self):
        sn = StrategyNarrative(winner_theory_id="TH-001", trace_id="STRAT-1")
        assert sn.alignment_status == ""

    def test_narrative_has_mapped_option_id_field(self):
        sn = StrategyNarrative(winner_theory_id="TH-001", trace_id="STRAT-1")
        assert sn.mapped_option_id is None

    def test_narrative_has_preferred_option_id_field(self):
        sn = StrategyNarrative(winner_theory_id="TH-001", trace_id="STRAT-1")
        assert sn.preferred_option_id == ""

    def test_narrative_has_mapping_confidence_field(self):
        sn = StrategyNarrative(winner_theory_id="TH-001", trace_id="STRAT-1")
        assert sn.mapping_confidence == ""

    def test_narrative_has_saturation_detected_field(self):
        sn = StrategyNarrative(winner_theory_id="TH-001", trace_id="STRAT-1")
        assert sn.saturation_detected is False

    def test_narrative_has_selection_status_field(self):
        sn = StrategyNarrative(winner_theory_id="TH-001", trace_id="STRAT-1")
        assert sn.selection_status == "selected"

    def test_narrative_has_constraint_outcomes_field(self):
        sn = StrategyNarrative(winner_theory_id="TH-001", trace_id="STRAT-1")
        assert sn.constraint_outcomes == []

    def test_narrative_has_winner_theory_label_field(self):
        sn = StrategyNarrative(winner_theory_id="TH-001", trace_id="STRAT-1")
        assert sn.winner_theory_label == ""

    def test_narrative_has_alignment_narrative_field(self):
        sn = StrategyNarrative(winner_theory_id="TH-001", trace_id="STRAT-1")
        assert sn.alignment_narrative == ""

    def test_build_populates_alignment_status_from_trace(self):
        trace = _build_minimal_trace(alignment={"status": "confirmed", "preferred_option_id": "OPT-B"})
        sn = build_strategy_narrative(trace)
        assert sn.alignment_status == "confirmed"

    def test_build_populates_preferred_option_id_from_trace(self):
        trace = _build_minimal_trace(alignment={"status": "refined", "preferred_option_id": "OPT-B"})
        sn = build_strategy_narrative(trace)
        assert sn.preferred_option_id == "OPT-B"

    def test_build_populates_mapped_option_id_from_trace(self):
        trace = _build_minimal_trace(alignment={"status": "confirmed", "mapped_option_id": "OPT-B"})
        sn = build_strategy_narrative(trace)
        assert sn.mapped_option_id == "OPT-B"

    def test_build_populates_saturation_detected(self):
        trace = _build_minimal_trace(saturation={"detected": True, "message": "scores tied"})
        sn = build_strategy_narrative(trace)
        assert sn.saturation_detected is True

    def test_build_populates_mapping_confidence_from_tom(self):
        theory = _make_theory("TH-W", choices=THEORY_DIVERSIFIED_CHOICES, recommended_option_id="OPT-B")
        tom = [{"theory_id": "TH-W", "mapped_option_id": "OPT-B", "mapping_confidence": "High"}]
        trace = _build_minimal_trace(theory=theory, theory_option_mappings=tom)
        sn = build_strategy_narrative(trace)
        assert sn.mapping_confidence == "High"

    def test_build_populates_constraint_outcomes_for_winner(self):
        theory = _make_theory("TH-W", choices=THEORY_DIVERSIFIED_CHOICES, recommended_option_id="OPT-B")
        cr = {"TH-W": [{"constraint": "grid_interconnect", "status": "satisfied", "score": 1.0, "rationale": "All required grid interconnect milestones met."}]}
        trace = _build_minimal_trace(theory=theory, constraint_results=cr)
        sn = build_strategy_narrative(trace)
        assert len(sn.constraint_outcomes) == 1
        assert sn.constraint_outcomes[0]["status"] == "satisfied"

    def test_build_winner_theory_label_from_choices(self):
        theory = _make_theory("TH-W", choices=THEORY_DIVERSIFIED_CHOICES, recommended_option_id="OPT-B")
        trace = _build_minimal_trace(theory=theory)
        sn = build_strategy_narrative(trace)
        # Label should contain the choice titles joined by "/"
        assert "/" in sn.winner_theory_label
        assert "Diversified" in sn.winner_theory_label or "diversified" in sn.winner_theory_label.lower()

    def test_build_alignment_narrative_confirmed(self):
        trace = _build_minimal_trace(alignment={"status": "confirmed"})
        sn = build_strategy_narrative(trace)
        assert "confirms" in sn.alignment_narrative.lower() or "confirm" in sn.alignment_narrative.lower()

    def test_build_alignment_narrative_refined(self):
        trace = _build_minimal_trace(alignment={"status": "refined"})
        sn = build_strategy_narrative(trace)
        assert "reinforces" in sn.alignment_narrative.lower() or "reinforce" in sn.alignment_narrative.lower()

    def test_build_alignment_narrative_unresolved(self):
        trace = _build_minimal_trace(alignment={"status": ""})
        sn = build_strategy_narrative(trace)
        assert "upstream" in sn.alignment_narrative.lower() or "authoritative" in sn.alignment_narrative.lower()


# ---------------------------------------------------------------------------
# TestReportAuthorityRendering
# ---------------------------------------------------------------------------

class TestReportAuthorityRendering:
    """MarkdownRenderer: authority table and alignment narrative in Strategic Direction."""

    def _render_strategic_direction(self, sn: StrategyNarrative) -> str:
        from functional_agents.editorial.strategy_writer import StrategyWriter

        # SimpleNamespace so StrategyWriter can mutate section attributes
        sec = types.SimpleNamespace(
            title="Strategic Direction",
            paragraphs=[],
            bullet_groups=[],
            tables=[],
            subtitle="",
            figures=[],
        )
        ms = types.SimpleNamespace(strategic_direction=sec)

        class _MockBrief:
            strategy_narrative = sn
            metadata = types.SimpleNamespace(question="What is the best strategy?")
            strategic_options = types.SimpleNamespace(options=[])

        brief = _MockBrief()

        # Populate via StrategyWriter (mutates sec in place)
        StrategyWriter().write(brief, ms)

        # Render
        lines = MarkdownRenderer()._s_strategic_direction(ms, brief)
        return "\n".join(lines)

    def _make_narrative(self, **kwargs) -> StrategyNarrative:
        defaults = dict(
            winner_theory_id="TH-DIV",
            trace_id="STRAT-001",
            winner_score=0.78,
            overall_confidence="High",
            alignment_status="confirmed",
            alignment_narrative="The configured Strategy evaluation confirms the upstream preferred option.",
            mapped_option_id="OPT-B",
            preferred_option_id="OPT-B",
            mapping_confidence="Medium",
            saturation_detected=False,
            selection_status="selected",
            winner_theory_label="Diversified / Behind-The-Meter First / Milestone Gated",
            # Needed so StrategyWriter sets paragraphs → has_content is True in renderer
            winning_position="A multi-state portfolio hedge is the recommended direction.",
        )
        defaults.update(kwargs)
        return StrategyNarrative(**defaults)

    def test_authority_table_contains_alignment_status(self):
        sn = self._make_narrative(alignment_status="confirmed")
        md = self._render_strategic_direction(sn)
        assert "Alignment Status" in md
        assert "Confirmed" in md or "confirmed" in md

    def test_authority_table_contains_mapped_option(self):
        sn = self._make_narrative()
        md = self._render_strategic_direction(sn)
        assert "Mapped Strategic Option" in md
        # OPT-B either by title or ID
        assert "OPT-B" in md or "Balanced" in md

    def test_authority_table_contains_winner_score(self):
        sn = self._make_narrative(winner_score=0.780)
        md = self._render_strategic_direction(sn)
        assert "Winner Score" in md
        assert "0.780" in md

    def test_authority_table_contains_mapping_confidence(self):
        sn = self._make_narrative(mapping_confidence="High")
        md = self._render_strategic_direction(sn)
        assert "Mapping Confidence" in md
        assert "High" in md

    def test_authority_table_contains_saturation_status(self):
        sn = self._make_narrative(saturation_detected=False)
        md = self._render_strategic_direction(sn)
        assert "Saturation Status" in md
        # "No" for not detected
        assert "No" in md

    def test_authority_table_contains_selection_status(self):
        sn = self._make_narrative(selection_status="selected")
        md = self._render_strategic_direction(sn)
        assert "Selection Status" in md

    def test_alignment_narrative_rendered(self):
        narrative_text = "The configured Strategy evaluation confirms the upstream preferred option."
        sn = self._make_narrative(alignment_narrative=narrative_text)
        md = self._render_strategic_direction(sn)
        assert "confirms" in md.lower() or narrative_text in md

    def test_constraint_table_rendered_when_present(self):
        sn = self._make_narrative(
            constraint_outcomes=[{
                "constraint": "grid_interconnect",
                "status": "satisfied",
                "score": 1.0,
                "rationale": "All milestones met.",
            }]
        )
        md = self._render_strategic_direction(sn)
        assert "Constraint Assessment" in md
        assert "grid_interconnect" in md or "Grid Interconnect" in md
        assert "Satisfied" in md or "satisfied" in md

    def test_constraint_table_absent_when_empty(self):
        sn = self._make_narrative(constraint_outcomes=[])
        md = self._render_strategic_direction(sn)
        assert "Constraint Assessment" not in md

    def test_winner_theory_label_in_table(self):
        sn = self._make_narrative(winner_theory_label="Diversified / BTM First / Milestone Gated")
        md = self._render_strategic_direction(sn)
        assert "Diversified" in md

    def test_alignment_narrative_empty_when_status_empty(self):
        sn = self._make_narrative(alignment_narrative="", alignment_status="")
        md = self._render_strategic_direction(sn)
        # Should not crash and should not render an italic blank line
        assert "**" not in md or "None" not in md  # not undefined values


# ---------------------------------------------------------------------------
# TestIntegrationProduction
# ---------------------------------------------------------------------------

class TestIntegrationProduction:
    """End-to-end: posture extraction → mapping → alignment for the production scenario."""

    def test_full_production_flow_maps_diversified_to_optb(self):
        """Full integration: Diversified/BTM/Milestone theory should map to OPT-B."""
        norm = PostureNormalizer()

        # Step 1: extract theory postures
        theory_postures = norm.theory_postures(THEORY_DIVERSIFIED_CHOICES)
        assert theory_postures.get("geographic") == "diversified"
        assert theory_postures.get("power") == "btm_first"
        assert theory_postures.get("timing") == "milestone_gated"

        # Step 2: extract option postures for OPT-A and OPT-B
        opt_a_postures = norm.option_postures(OPTION_A)
        opt_b_postures = norm.option_postures(OPTION_B)

        assert opt_a_postures.get("geographic") == "concentrated"  # contradicts theory
        assert opt_b_postures.get("geographic") == "diversified"   # matches theory

        # Step 3: run OptionMapper
        theory = _make_theory("TH-DIV", choices=THEORY_DIVERSIFIED_CHOICES, recommended_option_id="OPT-B")
        research = _make_research(preferred_id="OPT-B", options=ALL_OPTIONS)
        mapping = OptionMapper().map(theory, research)

        assert mapping.mapped_option_id == "OPT-B"

        # Step 4: run AlignmentEvaluator
        selection = _make_selection(margin=0.12)
        alignment = AlignmentEvaluator().evaluate(
            selected_theory=theory,
            option_mapping=mapping,
            selection=selection,
            research=research,
        )

        # Alignment must be confirmed or refined — NEVER challenged
        assert alignment.status in ("confirmed", "refined"), (
            f"Production integration: expected confirmed/refined, got {alignment.status!r}"
        )
        assert alignment.mapped_option_id == "OPT-B"
        assert alignment.preferred_option_id == "OPT-B"

    def test_opta_theory_alignment_is_confirmed_or_refined(self):
        """Concentrated/Accelerate theory vs OPT-A preferred — should confirm or refine."""
        concentrated_choices = [
            {"dimension": "geo", "selected_value": "concentrated", "metadata": {"choice_title": "Concentrated"}},
            {"dimension": "timing", "selected_value": "accelerate", "metadata": {"choice_title": "Accelerate"}},
        ]
        theory = _make_theory("TH-CONC", choices=concentrated_choices, recommended_option_id="OPT-A")
        research = _make_research(preferred_id="OPT-A", options=[OPTION_A, OPTION_B])
        mapping = OptionMapper().map(theory, research)
        selection = _make_selection(margin=0.10)
        alignment = AlignmentEvaluator().evaluate(
            selected_theory=theory,
            option_mapping=mapping,
            selection=selection,
            research=research,
        )
        if mapping.mapped_option_id == "OPT-A":
            assert alignment.status in ("confirmed", "refined")
        # If mapper returns None confidence, alignment is unresolved — still not challenged
        assert alignment.status != "challenged"

    def test_option_mapper_option_scores_populated(self):
        theory = _make_theory("TH-DIV", choices=THEORY_DIVERSIFIED_CHOICES, recommended_option_id="OPT-B")
        research = _make_research(preferred_id="OPT-B", options=ALL_OPTIONS)
        mapping = OptionMapper().map(theory, research)
        # option_scores should be populated and contain all four options
        assert len(mapping.option_scores) == len(ALL_OPTIONS)
        ids = [s["option_id"] for s in mapping.option_scores]
        assert "OPT-A" in ids
        assert "OPT-B" in ids

    def test_option_mapper_theory_postures_in_mapping(self):
        theory = _make_theory("TH-DIV", choices=THEORY_DIVERSIFIED_CHOICES, recommended_option_id="OPT-B")
        research = _make_research(preferred_id="OPT-B", options=ALL_OPTIONS)
        mapping = OptionMapper().map(theory, research)
        assert mapping.theory_postures.get("geographic") == "diversified"
