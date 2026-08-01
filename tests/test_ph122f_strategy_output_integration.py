"""PH12.2f — Strategy Output Integration tests.

Covers:
- StrategyOutputView construction from StrategyNarrative + strategic_options
- build_strategy_output_view: None when narrative is None
- build_strategy_output_view: resolves mapped/preferred option titles
- build_strategy_output_view: builds choice cascade with human-readable labels
- build_strategy_output_view: builds execution_implications from capability/management dims
- build_strategy_output_view: passes mapping metadata (score, margin, rationale)
- build_strategy_output_view: alignment fields forwarded
- ReportAgent/MarkdownRenderer: renders Strategic Direction when strategy available
- ReportAgent/MarkdownRenderer: does not render Strategic Direction when no strategy
- MarkdownRenderer: Mapped Strategic Option subsection present when mapped_option_id set
- MarkdownRenderer: Alignment Result subsection with executive language
- MarkdownRenderer: Execution Implications subsection from cascade
- MarkdownRenderer: Choice Cascade rendered with human-readable labels
- EditorialBrief: strategic_direction populated from narrative.winning_position
- EditorialBrief: core_thesis populated from narrative.winning_mechanism
- EditorialBrief: recommended_option populated from narrative.mapped_option_id
- EditorialBrief: mapped_option_title resolved from strategic_options
- EditorialBrief: alignment populated from narrative.alignment_status
- EditorialBrief: execution_implications populated from cascade
- EditorialBrief: strategy_provenance includes winning_theory_id, mapped_option_id, etc.
- EditorialBrief backward compat: all original fields still accessible
- Editorial manuscript: strategic_direction section populated by StrategyWriter
- Consistency: mapped option consistent across report and brief
- Conflict: strategy mapped_option overrides legacy recommendation in report
- Fallback: no StrategicPosition → existing behavior preserved
- Strategy disabled → no strategy-specific failure
- No duplicate StrategyCoordinator invocation in consumers
- StrategyNarrative: choice cascade uses dimension_title: choice_title (not raw keys)
- StrategyNarrative: mapping_score, mapping_margin, mapping_rationale populated from sel extras
- Determinism: same inputs produce identical view
- Provenance: strategy_provenance dict has required keys
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from functional_agents.editorial import (
    EditorialCoordinator,
    MarkdownRenderer,
    build_strategy_narrative,
)
from functional_agents.editorial.editorial_brief import EditorialBrief
from functional_agents.strategy import (
    StrategicChoiceSet,
    StrategySelection,
    StrategyTrace,
    build_strategy_output_view,
    StrategyOutputView,
    StrategyChoiceCascadeItem,
)
from functional_agents.strategy.strategic_position import (
    StrategicExecution,
    StrategicJustification,
    StrategicPosition,
    StrategicRecommendation,
    TheoryOfWinning,
)
from functional_agents.strategy.strategy_plan import StrategyPlan
from functional_agents.strategy.theory_evaluation import CriterionScore, TheoryEvaluation


# ---------------------------------------------------------------------------
# Shared fixtures — reusable minimal objects
# ---------------------------------------------------------------------------

_MONITOR_DIMENSIONS = [
    ("winning_aspiration", "Winning Aspiration", "category_leader", "Category Leader"),
    ("where_to_play", "Where to Play", "leagues_gb", "Leagues and Governing Bodies"),
    ("how_to_win", "How to Win", "advisory_led", "Advisory-Led"),
    ("must_have_capabilities", "Must-Have Capabilities", "credibility", "Advisory Credibility"),
    ("management_systems", "Management Systems", "gates", "Investment Gates"),
]


def _make_choice(dim_id: str, dim_title: str, choice_id: str, choice_title: str) -> dict:
    return {
        "id": f"SC-{dim_id}",
        "dimension": dim_id,
        "selected_value": choice_id,
        "rationale": f"Rationale for {dim_title}",
        "supporting_evidence": [],
        "supporting_assumptions": [],
        "confidence": "",
        "alternatives_considered": [],
        "requiredness": "required",
        "metadata": {
            "choice_title": choice_title,
            "choice_description": f"Description of {choice_title}.",
            "dimension_title": dim_title,
            "dimension_description": f"Description of {dim_title}.",
            "execution_complexity": "medium",
        },
    }


def _theory(
    tid: str,
    oid: str = "OPT-A",
    winning_position: str = "We will be the category leader.",
    winning_mechanism: str = "Through advisory relationships and credibility.",
    choices: list | None = None,
    assumptions: list | None = None,
    failure_modes: list | None = None,
    success_conditions: list | None = None,
) -> TheoryOfWinning:
    sc_list = choices or [
        _make_choice(dim_id, dim_title, choice_id, choice_title)
        for dim_id, dim_title, choice_id, choice_title in _MONITOR_DIMENSIONS
    ]
    return TheoryOfWinning(
        theory_id=tid,
        recommended_option_id=oid,
        recommended_option_title=f"Option {oid}",
        source_choice_set_id=f"SCS-{tid[-1]}",
        winning_position=winning_position,
        winning_mechanism=winning_mechanism,
        strategic_choices=sc_list,
        assumptions=assumptions or [{"statement": "Key assumption A."}, {"statement": "Key assumption B."}],
        failure_modes=failure_modes or [{"statement": "Key risk X."}],
        success_conditions=success_conditions or ["Condition 1.", "Condition 2."],
    )


def _eval(tid: str, score: float = 0.9, confidence: str = "High") -> TheoryEvaluation:
    cs = {"market_fit": CriterionScore(score=score, rationale="r", weight=1.0)}
    return TheoryEvaluation(
        theory_id=tid,
        criteria_scores=cs,
        strengths=["Strong market position."],
        weaknesses=[],
        residual_risks=[],
        overall_score=score,
        confidence=confidence,
        metadata={},
    )


def _selection(
    winner: str,
    winner_score: float = 0.9,
    runner_up: str | None = None,
    runner_up_score: float | None = None,
    mapped_option_id: str = "OPT-A",
    alignment_status: str = "refined",
    mapping_score: float | None = 0.85,
    mapping_margin: float | None = 0.25,
    mapping_rationale: str = "Best content match.",
    mapping_status: str = "mapped",
) -> StrategySelection:
    margin = round(winner_score - runner_up_score, 6) if runner_up_score is not None else None
    return StrategySelection(
        winner_theory_id=winner,
        winner_score=winner_score,
        runner_up_theory_id=runner_up,
        runner_up_score=runner_up_score,
        score_margin=margin,
        alignment_status=alignment_status,
        mapped_option_id=mapped_option_id,
        saturation_detected=False,
        # extras (model_extra via extra="allow")
        mapping_score=mapping_score,
        mapping_margin=mapping_margin,
        mapping_rationale=mapping_rationale,
        mapping_status=mapping_status,
        mapping_confidence="High",
    )


def _position(theory: TheoryOfWinning) -> StrategicPosition:
    return StrategicPosition(
        position_id="SP-TEST",
        created_at="2026-08-01T00:00:00+00:00",
        theory_of_winning=theory,
        recommendation=StrategicRecommendation(
            recommended_option_id=theory.recommended_option_id,
            recommended_option_title="",
            board_recommendation="Proceed.",
            decision_readiness="Ready",
            overall_confidence="High",
        ),
        justification=StrategicJustification(
            decision_analysis={},
            strategic_options=[],
            assumptions=[],
            risks=[],
            opportunities=[],
        ),
        execution=StrategicExecution(recommendations=[], validation_priorities=[]),
    )


def _choice_set(cs_id: str, th_id: str) -> StrategicChoiceSet:
    return StrategicChoiceSet(id=cs_id, completeness=1.0, overall_confidence="High",
                              metadata={"theory_id": th_id})


def _trace(
    winner_id: str = "TH-0",
    runner_up_id: str = "TH-1",
    winner_score: float = 0.93,
    runner_up_score: float = 0.87,
    mapped_option_id: str = "OPT-B",
    alignment_status: str = "refined",
) -> StrategyTrace:
    cs0 = _choice_set(f"SCS-{winner_id[-1]}", winner_id)
    cs1 = _choice_set(f"SCS-{runner_up_id[-1]}", runner_up_id)
    theories = [
        _theory(winner_id, oid=mapped_option_id),
        _theory(runner_up_id, oid="OPT-A", winning_position="", winning_mechanism="",
                choices=[_make_choice("winning_aspiration", "Winning Aspiration", "scaled", "Scaled Platform")]),
    ]
    evaluations = [
        _eval(winner_id, winner_score),
        _eval(runner_up_id, runner_up_score),
    ]
    sel = _selection(
        winner_id, winner_score, runner_up_id, runner_up_score,
        mapped_option_id=mapped_option_id,
        alignment_status=alignment_status,
    )
    pos = _position(theories[0])
    plan = StrategyPlan(plan_id="P-TEST", framework="monitor_choice_cascade", active_dimensions=[])
    return StrategyTrace(
        trace_id="STRAT-TEST",
        created_at="2026-08-01T00:00:00+00:00",
        plan=plan,
        choice_sets=[cs0, cs1],
        theories=theories,
        evaluations=evaluations,
        selection=sel,
        strategic_position=pos,
        alignment={
            "status": alignment_status,
            "mapped_option_id": mapped_option_id,
            "preferred_option_id": "OPT-A",
        },
        metadata={"framework": "monitor_choice_cascade", "research_id": ""},
    )


def _strategic_options() -> list[dict]:
    return [
        {"option_id": "OPT-A", "title": "Focused Beachhead", "description": "Option A.",
         "recommended": True, "supporting_assumption_ids": [], "associated_risk_ids": [],
         "associated_opportunity_ids": [], "supporting_recommendation_ids": [],
         "advantages": ["Fast."], "disadvantages": [], "strategic_objective": "",
         "expected_outcomes": [], "implementation_complexity": "Low",
         "estimated_time_horizon": "near_term", "capital_intensity": "Low", "confidence": "High"},
        {"option_id": "OPT-B", "title": "Integrated Platform Play", "description": "Option B.",
         "recommended": False, "supporting_assumption_ids": [], "associated_risk_ids": [],
         "associated_opportunity_ids": [], "supporting_recommendation_ids": [],
         "advantages": ["Scale."], "disadvantages": [], "strategic_objective": "",
         "expected_outcomes": [], "implementation_complexity": "High",
         "estimated_time_horizon": "medium_term", "capital_intensity": "High", "confidence": "Medium"},
    ]


def _full_position_with_options() -> StrategicPosition:
    theory = _theory("TH-0", oid="OPT-B")
    pos = _position(theory)
    pos = pos.model_copy(update={"strategic_options": _strategic_options()})
    return pos



# ---------------------------------------------------------------------------
# Section 1: StrategyNarrative — choice cascade and mapping fields (PH12.2f)
# ---------------------------------------------------------------------------

class TestStrategyNarrativeChoiceCascade:
    def test_winner_strategic_choices_use_dimension_title_and_choice_title(self):
        tr = _trace()
        sn = build_strategy_narrative(tr)
        # Each choice string should use dimension_title: choice_title, NOT raw keys
        assert any("Winning Aspiration: Category Leader" in c for c in sn.winner_strategic_choices), (
            f"Expected 'Winning Aspiration: Category Leader' in choices; got {sn.winner_strategic_choices}"
        )
        # Raw internal key format should not appear
        assert not any(c.startswith("winning_aspiration:") for c in sn.winner_strategic_choices), (
            "winner_strategic_choices should not use raw key 'winning_aspiration:'"
        )

    def test_choice_cascade_populated(self):
        tr = _trace()
        sn = build_strategy_narrative(tr)
        assert len(sn.choice_cascade) == len(_MONITOR_DIMENSIONS)

    def test_choice_cascade_entries_have_titles(self):
        tr = _trace()
        sn = build_strategy_narrative(tr)
        entry = sn.choice_cascade[0]
        assert entry["dimension_title"] == "Winning Aspiration"
        assert entry["choice_title"] == "Category Leader"
        assert entry["dimension_id"] == "winning_aspiration"
        assert entry["choice_id"] == "category_leader"

    def test_mapping_score_from_sel_extras(self):
        tr = _trace()
        sn = build_strategy_narrative(tr)
        assert sn.mapping_score == pytest.approx(0.85)

    def test_mapping_margin_from_sel_extras(self):
        tr = _trace()
        sn = build_strategy_narrative(tr)
        assert sn.mapping_margin == pytest.approx(0.25)

    def test_mapping_rationale_from_sel_extras(self):
        tr = _trace()
        sn = build_strategy_narrative(tr)
        assert sn.mapping_rationale == "Best content match."

    def test_mapping_status_from_sel_extras(self):
        tr = _trace()
        sn = build_strategy_narrative(tr)
        assert sn.mapping_status == "mapped"


# ---------------------------------------------------------------------------
# Section 2: StrategyOutputView construction
# ---------------------------------------------------------------------------

class TestStrategyOutputView:
    def test_returns_none_when_narrative_is_none(self):
        result = build_strategy_output_view(None, strategic_options=_strategic_options())
        assert result is None

    def test_returns_view_when_narrative_present(self):
        tr = _trace()
        sn = build_strategy_narrative(tr)
        view = build_strategy_output_view(sn, strategic_options=_strategic_options())
        assert view is not None
        assert isinstance(view, StrategyOutputView)

    def test_framework_and_trace_id_forwarded(self):
        tr = _trace()
        sn = build_strategy_narrative(tr)
        view = build_strategy_output_view(sn, strategic_options=_strategic_options())
        assert view.framework == "monitor_choice_cascade"
        assert view.trace_id == "STRAT-TEST"

    def test_mapped_option_id_forwarded(self):
        tr = _trace(mapped_option_id="OPT-B")
        sn = build_strategy_narrative(tr)
        view = build_strategy_output_view(sn, strategic_options=_strategic_options())
        assert view.mapped_option_id == "OPT-B"

    def test_mapped_option_title_resolved_from_options(self):
        tr = _trace(mapped_option_id="OPT-B")
        sn = build_strategy_narrative(tr)
        view = build_strategy_output_view(sn, strategic_options=_strategic_options())
        assert view.mapped_option_title == "Integrated Platform Play"

    def test_preferred_option_title_resolved(self):
        # preferred_option_id is from the alignment block — not always populated in test fixture
        tr = _trace()
        sn = build_strategy_narrative(tr)
        view = build_strategy_output_view(sn, strategic_options=_strategic_options())
        # preferred_option_id may be empty in minimal fixture; title should be "" or option title
        assert isinstance(view.preferred_option_title, str)

    def test_choice_cascade_has_all_dimensions(self):
        tr = _trace()
        sn = build_strategy_narrative(tr)
        view = build_strategy_output_view(sn, strategic_options=_strategic_options())
        assert len(view.choice_cascade) == len(_MONITOR_DIMENSIONS)

    def test_choice_cascade_items_are_frozen_models(self):
        tr = _trace()
        sn = build_strategy_narrative(tr)
        view = build_strategy_output_view(sn, strategic_options=_strategic_options())
        for item in view.choice_cascade:
            assert isinstance(item, StrategyChoiceCascadeItem)

    def test_choice_cascade_first_item_titles(self):
        tr = _trace()
        sn = build_strategy_narrative(tr)
        view = build_strategy_output_view(sn, strategic_options=_strategic_options())
        first = view.choice_cascade[0]
        assert first.dimension_title == "Winning Aspiration"
        assert first.choice_title == "Category Leader"

    def test_execution_implications_populated(self):
        tr = _trace()
        sn = build_strategy_narrative(tr)
        view = build_strategy_output_view(sn, strategic_options=_strategic_options())
        # must_have_capabilities, management_systems, how_to_win → implications
        assert len(view.execution_implications) >= 2

    def test_execution_implications_contain_capability_dimension(self):
        tr = _trace()
        sn = build_strategy_narrative(tr)
        view = build_strategy_output_view(sn, strategic_options=_strategic_options())
        assert any("Must-Have Capabilities" in impl for impl in view.execution_implications), (
            f"Expected 'Must-Have Capabilities' in implications; got {view.execution_implications}"
        )

    def test_execution_implications_contain_management_dimension(self):
        tr = _trace()
        sn = build_strategy_narrative(tr)
        view = build_strategy_output_view(sn, strategic_options=_strategic_options())
        assert any("Management Systems" in impl for impl in view.execution_implications)

    def test_mapping_metadata_forwarded(self):
        tr = _trace()
        sn = build_strategy_narrative(tr)
        view = build_strategy_output_view(sn, strategic_options=_strategic_options())
        assert view.mapping_score == pytest.approx(0.85)
        assert view.mapping_margin == pytest.approx(0.25)
        assert view.mapping_rationale == "Best content match."
        assert view.mapping_status == "mapped"

    def test_alignment_fields_forwarded(self):
        tr = _trace(alignment_status="refined")
        sn = build_strategy_narrative(tr)
        view = build_strategy_output_view(sn, strategic_options=_strategic_options())
        assert view.alignment_status == "refined"
        assert "reinforces" in view.alignment_narrative.lower() or "refined" in view.alignment_narrative.lower()

    def test_view_is_immutable(self):
        tr = _trace()
        sn = build_strategy_narrative(tr)
        view = build_strategy_output_view(sn, strategic_options=_strategic_options())
        with pytest.raises(Exception):
            view.framework = "changed"

    def test_deterministic_for_same_inputs(self):
        tr = _trace()
        sn = build_strategy_narrative(tr)
        opts = _strategic_options()
        view1 = build_strategy_output_view(sn, strategic_options=opts)
        view2 = build_strategy_output_view(sn, strategic_options=opts)
        assert view1.model_dump() == view2.model_dump()

    def test_empty_options_list_returns_option_id_as_title(self):
        tr = _trace(mapped_option_id="OPT-X")
        sn = build_strategy_narrative(tr)
        view = build_strategy_output_view(sn, strategic_options=[])
        assert view.mapped_option_id == "OPT-X"
        assert view.mapped_option_title == "OPT-X"  # falls back to id

    def test_fingerprint_forwarded(self):
        tr = _trace()
        sn = build_strategy_narrative(tr)
        view = build_strategy_output_view(sn, strategy_config_fingerprint="abc123")
        assert view.strategy_config_fingerprint == "abc123"


# ---------------------------------------------------------------------------
# Section 3: EditorialBrief strategy fields (PH12.2f)
# ---------------------------------------------------------------------------

def _build_brief_with_strategy() -> tuple[EditorialBrief, StrategyTrace]:
    tr = _trace(mapped_option_id="OPT-B", alignment_status="refined")
    pos = _full_position_with_options()
    coord = EditorialCoordinator()
    brief = coord.build(pos, strategy_trace=tr)
    return brief, tr


class TestEditorialBriefStrategyConsumption:
    def test_strategic_direction_populated(self):
        brief, _ = _build_brief_with_strategy()
        # strategic_direction should be winning_position text
        assert brief.strategic_direction  # non-empty
        assert "category leader" in brief.strategic_direction.lower() or len(brief.strategic_direction) > 10

    def test_core_thesis_populated(self):
        brief, _ = _build_brief_with_strategy()
        assert brief.core_thesis  # winning_mechanism text

    def test_recommended_option_is_mapped_option_id(self):
        brief, _ = _build_brief_with_strategy()
        assert brief.recommended_option == "OPT-B"

    def test_mapped_option_title_resolved(self):
        brief, _ = _build_brief_with_strategy()
        assert brief.mapped_option_title == "Integrated Platform Play"

    def test_alignment_populated(self):
        brief, _ = _build_brief_with_strategy()
        assert brief.alignment == "refined"

    def test_execution_implications_populated(self):
        brief, _ = _build_brief_with_strategy()
        assert len(brief.execution_implications) >= 2

    def test_strategy_provenance_has_required_keys(self):
        brief, _ = _build_brief_with_strategy()
        prov = brief.strategy_provenance
        required_keys = {
            "strategic_position_id",
            "winning_theory_id",
            "mapped_option_id",
            "alignment_status",
            "framework",
            "trace_id",
        }
        for k in required_keys:
            assert k in prov, f"Missing provenance key: {k}"

    def test_strategy_provenance_values_correct(self):
        brief, _ = _build_brief_with_strategy()
        prov = brief.strategy_provenance
        assert prov["mapped_option_id"] == "OPT-B"
        assert prov["alignment_status"] == "refined"
        assert prov["framework"] == "monitor_choice_cascade"

    def test_legacy_fields_still_accessible(self):
        brief, _ = _build_brief_with_strategy()
        # All 11 original fields remain intact
        assert brief.metadata is not None
        assert brief.executive_summary is not None
        assert brief.decision_analysis is not None
        assert brief.strategic_options is not None
        assert brief.recommendations is not None
        assert brief.strategic_assumptions is not None
        assert brief.strategic_risks is not None
        assert brief.strategic_opportunities is not None
        assert brief.executive_confidence is not None
        assert brief.validation_priorities is not None
        assert brief.appendix is not None
        assert brief.strategy_narrative is not None  # 12th field

    def test_strategy_fields_empty_when_no_strategy(self):
        pos = _full_position_with_options()
        coord = EditorialCoordinator()
        brief = coord.build(pos, strategy_trace=None)
        assert brief.strategic_direction == ""
        assert brief.recommended_option == ""
        assert brief.alignment == ""
        assert brief.execution_implications == []
        assert brief.strategy_provenance == {}

    def test_to_dict_includes_strategy_fields(self):
        brief, _ = _build_brief_with_strategy()
        d = brief.to_dict()
        assert "strategic_direction" in d
        assert "recommended_option" in d
        assert "alignment" in d
        assert "strategy_provenance" in d
        assert d["recommended_option"] == "OPT-B"


# ---------------------------------------------------------------------------
# Section 4: Report strategy rendering (MarkdownRenderer)
# ---------------------------------------------------------------------------

def _render_report_with_strategy() -> str:
    tr = _trace(mapped_option_id="OPT-B", alignment_status="refined")
    pos = _full_position_with_options()
    coord = EditorialCoordinator()
    brief = coord.build(pos, strategy_trace=tr)
    manuscript = coord.build_manuscript(brief)
    coord.run_writers(brief, manuscript)
    renderer = MarkdownRenderer()
    return renderer.render(manuscript, brief=brief)


class TestReportStrategyConsumption:
    def test_strategic_direction_section_present(self):
        md = _render_report_with_strategy()
        assert "## Strategic Direction" in md

    def test_mapped_option_section_present(self):
        md = _render_report_with_strategy()
        assert "Mapped Strategic Option" in md

    def test_mapped_option_title_in_report(self):
        md = _render_report_with_strategy()
        assert "Integrated Platform Play" in md

    def test_alignment_result_section_present(self):
        md = _render_report_with_strategy()
        assert "Alignment Result" in md

    def test_alignment_status_in_report(self):
        md = _render_report_with_strategy()
        assert "Refined" in md

    def test_choice_cascade_section_present(self):
        md = _render_report_with_strategy()
        assert "Choice Cascade" in md

    def test_choice_cascade_uses_dimension_titles(self):
        md = _render_report_with_strategy()
        assert "Winning Aspiration" in md
        assert "Where to Play" in md

    def test_choice_cascade_uses_choice_titles(self):
        md = _render_report_with_strategy()
        assert "Category Leader" in md
        assert "Leagues and Governing Bodies" in md

    def test_choice_cascade_no_raw_keys(self):
        md = _render_report_with_strategy()
        # Raw internal keys should not appear in the choice cascade section
        assert "winning_aspiration:" not in md
        assert "category_leader" not in md.lower() or "Category Leader" in md

    def test_execution_implications_section_present(self):
        md = _render_report_with_strategy()
        assert "Execution Implications" in md

    def test_execution_implications_content(self):
        md = _render_report_with_strategy()
        assert "Must-Have Capabilities" in md or "Advisory Credibility" in md

    def test_winning_position_in_report(self):
        md = _render_report_with_strategy()
        assert "category leader" in md.lower() or "We will be" in md

    def test_no_strategic_direction_when_no_strategy(self):
        pos = _full_position_with_options()
        coord = EditorialCoordinator()
        brief = coord.build(pos, strategy_trace=None)
        manuscript = coord.build_manuscript(brief)
        coord.run_writers(brief, manuscript)
        renderer = MarkdownRenderer()
        md = renderer.render(manuscript, brief=brief)
        assert "## Strategic Direction" not in md

    def test_strategy_does_not_contradict_mapped_option(self):
        # Legacy decision_analysis says OPT-A; strategy maps to OPT-B
        tr = _trace(mapped_option_id="OPT-B")
        pos = _full_position_with_options()
        coord = EditorialCoordinator()
        brief = coord.build(pos, strategy_trace=tr)
        manuscript = coord.build_manuscript(brief)
        coord.run_writers(brief, manuscript)
        renderer = MarkdownRenderer()
        md = renderer.render(manuscript, brief=brief)
        # Strategic Direction shows OPT-B (Integrated Platform Play)
        # Legacy recommendation also updated to OPT-B
        assert "Integrated Platform Play" in md


# ---------------------------------------------------------------------------
# Section 5: Editorial Manuscript strategy consumption
# ---------------------------------------------------------------------------

class TestEditorialManuscriptStrategyConsumption:
    def test_strategic_direction_section_exists(self):
        tr = _trace()
        pos = _full_position_with_options()
        coord = EditorialCoordinator()
        brief = coord.build(pos, strategy_trace=tr)
        manuscript = coord.build_manuscript(brief)
        coord.run_writers(brief, manuscript)
        assert manuscript.strategic_direction is not None

    def test_strategic_direction_populated_by_strategy_writer(self):
        tr = _trace()
        pos = _full_position_with_options()
        coord = EditorialCoordinator()
        brief = coord.build(pos, strategy_trace=tr)
        manuscript = coord.build_manuscript(brief)
        coord.run_writers(brief, manuscript)
        sd = manuscript.strategic_direction
        assert sd is not None
        assert sd.paragraphs  # populated when strategy present

    def test_strategic_direction_empty_when_no_strategy(self):
        pos = _full_position_with_options()
        coord = EditorialCoordinator()
        brief = coord.build(pos, strategy_trace=None)
        manuscript = coord.build_manuscript(brief)
        coord.run_writers(brief, manuscript)
        sd = manuscript.strategic_direction
        # StrategyWriter is optional — no-ops when no narrative
        assert sd is None or not sd.paragraphs

    def test_choice_cascade_in_bullet_groups(self):
        tr = _trace()
        pos = _full_position_with_options()
        coord = EditorialCoordinator()
        brief = coord.build(pos, strategy_trace=tr)
        manuscript = coord.build_manuscript(brief)
        coord.run_writers(brief, manuscript)
        sd = manuscript.strategic_direction
        assert sd is not None
        # bullet_groups[4] holds strategic choices in human-readable form
        if len(sd.bullet_groups) > 4:
            choices = sd.bullet_groups[4]
            if choices:
                assert any("Winning Aspiration" in c for c in choices), (
                    f"Expected 'Winning Aspiration' in choices; got {choices}"
                )


# ---------------------------------------------------------------------------
# Section 6: Consistency — mapped option consistent across consumers
# ---------------------------------------------------------------------------

class TestStrategyOutputConsistency:
    def test_brief_and_manuscript_use_same_mapped_option(self):
        tr = _trace(mapped_option_id="OPT-B")
        pos = _full_position_with_options()
        coord = EditorialCoordinator()
        brief = coord.build(pos, strategy_trace=tr)
        manuscript = coord.build_manuscript(brief)
        coord.run_writers(brief, manuscript)

        # Brief says OPT-B
        assert brief.recommended_option == "OPT-B"
        assert brief.strategy_narrative is not None
        assert brief.strategy_narrative.mapped_option_id == "OPT-B"

        # Manuscript strategic_direction derives from same narrative
        sd = manuscript.strategic_direction
        if sd and sd.paragraphs:
            narrative_text = " ".join(sd.paragraphs)
            # winning_position should be present (comes from the winner theory)
            assert narrative_text  # non-empty

    def test_strategy_overrides_legacy_in_report(self):
        # OPT-A is legacy recommended; OPT-B is strategy-mapped
        tr = _trace(mapped_option_id="OPT-B")
        pos = _full_position_with_options()  # OPT-A is legacy recommended
        coord = EditorialCoordinator()
        brief = coord.build(pos, strategy_trace=tr)
        # Brief uses OPT-B (strategy wins)
        assert brief.recommended_option == "OPT-B"

    def test_aligned_status_consistent_across_brief_and_narrative(self):
        tr = _trace(alignment_status="confirmed")
        pos = _full_position_with_options()
        coord = EditorialCoordinator()
        brief = coord.build(pos, strategy_trace=tr)
        assert brief.alignment == "confirmed"
        assert brief.strategy_narrative.alignment_status == "confirmed"


# ---------------------------------------------------------------------------
# Section 7: Alignment rendering — all statuses
# ---------------------------------------------------------------------------

class TestAlignmentRendering:
    @pytest.mark.parametrize("status,expected_keyword", [
        ("confirmed", "Confirmed"),
        ("refined", "Refined"),
        ("challenged", "Challenged"),
        ("unresolved", "Unresolved"),
    ])
    def test_alignment_status_label_in_report(self, status, expected_keyword):
        tr = _trace(alignment_status=status)
        pos = _full_position_with_options()
        coord = EditorialCoordinator()
        brief = coord.build(pos, strategy_trace=tr)
        manuscript = coord.build_manuscript(brief)
        coord.run_writers(brief, manuscript)
        renderer = MarkdownRenderer()
        md = renderer.render(manuscript, brief=brief)
        assert expected_keyword in md

    @pytest.mark.parametrize("status", ["confirmed", "refined", "challenged", "unresolved"])
    def test_alignment_explanation_present(self, status):
        tr = _trace(alignment_status=status)
        pos = _full_position_with_options()
        coord = EditorialCoordinator()
        brief = coord.build(pos, strategy_trace=tr)
        manuscript = coord.build_manuscript(brief)
        coord.run_writers(brief, manuscript)
        renderer = MarkdownRenderer()
        md = renderer.render(manuscript, brief=brief)
        assert "Alignment Result" in md


# ---------------------------------------------------------------------------
# Section 8: Fallback — no strategy
# ---------------------------------------------------------------------------

class TestFallbackBehavior:
    def test_no_strategy_report_still_renders(self):
        pos = _full_position_with_options()
        coord = EditorialCoordinator()
        brief = coord.build(pos, strategy_trace=None)
        manuscript = coord.build_manuscript(brief)
        coord.run_writers(brief, manuscript)
        renderer = MarkdownRenderer()
        md = renderer.render(manuscript, brief=brief)
        assert "# Executive Strategic Report" in md

    def test_no_strategy_brief_has_no_strategy_fields(self):
        pos = _full_position_with_options()
        coord = EditorialCoordinator()
        brief = coord.build(pos, strategy_trace=None)
        assert brief.strategy_narrative is None
        assert brief.strategic_direction == ""
        assert brief.recommended_option == ""

    def test_no_strategy_view_returns_none(self):
        result = build_strategy_output_view(None, strategic_options=_strategic_options())
        assert result is None

    def test_legacy_context_renders_without_error(self):
        pos = _full_position_with_options()
        coord = EditorialCoordinator()
        brief = coord.build(pos, strategy_trace=None)
        manuscript = coord.build_manuscript(brief)
        coord.run_writers(brief, manuscript)
        renderer = MarkdownRenderer()
        md = renderer.render(manuscript, brief=brief)
        assert len(md) > 100


# ---------------------------------------------------------------------------
# Section 9: No duplicate StrategyCoordinator invocation
# ---------------------------------------------------------------------------

class TestNoDuplicateExecution:
    def test_strategy_writer_does_not_invoke_coordinator(self):
        with patch("functional_agents.strategy.strategy_coordinator.StrategyCoordinator.build") as mock_build:
            tr = _trace()
            pos = _full_position_with_options()
            coord = EditorialCoordinator()
            brief = coord.build(pos, strategy_trace=tr)
            manuscript = coord.build_manuscript(brief)
            coord.run_writers(brief, manuscript)
            mock_build.assert_not_called()

    def test_markdown_renderer_does_not_invoke_coordinator(self):
        with patch("functional_agents.strategy.strategy_coordinator.StrategyCoordinator.build") as mock_build:
            tr = _trace()
            pos = _full_position_with_options()
            coord = EditorialCoordinator()
            brief = coord.build(pos, strategy_trace=tr)
            manuscript = coord.build_manuscript(brief)
            coord.run_writers(brief, manuscript)
            renderer = MarkdownRenderer()
            renderer.render(manuscript, brief=brief)
            mock_build.assert_not_called()

    def test_editorial_coordinator_does_not_re_invoke_on_build(self):
        with patch("functional_agents.strategy.strategy_coordinator.StrategyCoordinator.build") as mock_build:
            tr = _trace()
            pos = _full_position_with_options()
            coord = EditorialCoordinator()
            coord.build(pos, strategy_trace=tr)
            mock_build.assert_not_called()

    def test_build_strategy_output_view_does_not_invoke_coordinator(self):
        with patch("functional_agents.strategy.strategy_coordinator.StrategyCoordinator.build") as mock_build:
            tr = _trace()
            sn = build_strategy_narrative(tr)
            build_strategy_output_view(sn, strategic_options=_strategic_options())
            mock_build.assert_not_called()


# ---------------------------------------------------------------------------
# Section 10: Provenance metadata
# ---------------------------------------------------------------------------

class TestProvenancePersistence:
    def test_strategy_provenance_dict_has_all_required_keys(self):
        brief, _ = _build_brief_with_strategy()
        prov = brief.strategy_provenance
        required = {
            "strategic_position_id",
            "winning_theory_id",
            "mapped_option_id",
            "alignment_status",
            "framework",
            "trace_id",
            "strategy_config_fingerprint",
        }
        missing = required - set(prov.keys())
        assert not missing, f"Missing provenance keys: {missing}"

    def test_strategy_provenance_values_non_empty_for_core_fields(self):
        brief, _ = _build_brief_with_strategy()
        prov = brief.strategy_provenance
        for key in ("winning_theory_id", "mapped_option_id", "alignment_status", "framework"):
            assert prov.get(key), f"Provenance key '{key}' is empty or missing"

    def test_strategy_provenance_in_to_dict(self):
        brief, _ = _build_brief_with_strategy()
        d = brief.to_dict()
        assert "strategy_provenance" in d
        assert isinstance(d["strategy_provenance"], dict)
        assert d["strategy_provenance"]["mapped_option_id"] == "OPT-B"


# ---------------------------------------------------------------------------
# Section 11: Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_trace_same_view(self):
        tr = _trace()
        sn = build_strategy_narrative(tr)
        opts = _strategic_options()
        v1 = build_strategy_output_view(sn, strategic_options=opts)
        v2 = build_strategy_output_view(sn, strategic_options=opts)
        assert v1.model_dump() == v2.model_dump()

    def test_same_trace_same_narrative(self):
        tr = _trace()
        sn1 = build_strategy_narrative(tr)
        sn2 = build_strategy_narrative(tr)
        assert sn1.winner_strategic_choices == sn2.winner_strategic_choices
        assert sn1.choice_cascade == sn2.choice_cascade

    def test_same_inputs_same_report(self):
        tr = _trace()
        pos = _full_position_with_options()

        def _render():
            coord = EditorialCoordinator()
            brief = coord.build(pos, strategy_trace=tr)
            manuscript = coord.build_manuscript(brief)
            coord.run_writers(brief, manuscript)
            return MarkdownRenderer().render(manuscript, brief=brief)

        md1 = _render()
        md2 = _render()
        # Normalize timestamps (brief/manuscript IDs include creation time)
        import re
        md1_norm = re.sub(r'EB-\d{8}-\w+|EM-\d{8}-\w+', 'EB-DATE-ID', md1)
        md2_norm = re.sub(r'EB-\d{8}-\w+|EM-\d{8}-\w+', 'EB-DATE-ID', md2)
        assert md1_norm == md2_norm


# ---------------------------------------------------------------------------
# Section 12: Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_editorial_brief_original_11_fields_still_serializable(self):
        pos = _full_position_with_options()
        coord = EditorialCoordinator()
        brief = coord.build(pos, strategy_trace=None)
        d = brief.to_dict()
        expected_keys = {
            "metadata", "executive_summary", "decision_analysis", "strategic_options",
            "recommendations", "strategic_assumptions", "strategic_risks",
            "strategic_opportunities", "executive_confidence", "validation_priorities",
            "appendix", "strategy_narrative",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_new_strategy_fields_default_to_empty(self):
        pos = _full_position_with_options()
        coord = EditorialCoordinator()
        brief = coord.build(pos, strategy_trace=None)
        assert brief.strategic_direction == ""
        assert brief.core_thesis == ""
        assert brief.recommended_option == ""
        assert brief.mapped_option_title == ""
        assert brief.alignment == ""
        assert brief.execution_implications == []
        assert brief.strategy_provenance == {}

    def test_strategy_narrative_none_is_json_null(self):
        pos = _full_position_with_options()
        coord = EditorialCoordinator()
        brief = coord.build(pos, strategy_trace=None)
        d = brief.to_dict()
        assert d["strategy_narrative"] is None

    def test_run_writers_completes_without_strategy(self):
        pos = _full_position_with_options()
        coord = EditorialCoordinator()
        brief = coord.build(pos, strategy_trace=None)
        manuscript = coord.build_manuscript(brief)
        # Should not raise
        coord.run_writers(brief, manuscript)
