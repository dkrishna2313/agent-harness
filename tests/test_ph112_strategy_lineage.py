"""PH11.2 — StrategyTrace Lineage and Artifact Indexing tests.

Covers:
- StrategyLineageLink construction and field validation
- build_strategy_lineage() link count, categories, and specific targets
- TheoryOfWinning.source_choice_set_id field and TheoryGenerator populating it
- StrategyTrace rules 13-16 (lineage integrity when lineage is present)
- write_artifact_index() writes artifact.index.json with correct content
- End-to-end: coordinator run produces trace with lineage that round-trips
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from functional_agents.context import AgentContext
from functional_agents.strategy import (
    StrategyCoordinator,
    StrategyLineageLink,
    StrategyTrace,
)
from functional_agents.strategy.strategy_lineage import build_strategy_lineage
from functional_agents.strategy.strategy_trace import (
    write_artifact_index,
    write_strategy_trace,
)
from functional_agents.strategy.strategic_position import (
    StrategicExecution,
    StrategicJustification,
    StrategicPosition,
    StrategicRecommendation,
    TheoryOfWinning,
)
from functional_agents.strategy.strategic_choice import StrategicChoice
from functional_agents.strategy.strategic_choice_set import StrategicChoiceSet
from functional_agents.strategy.strategy_plan import StrategyPlan
from functional_agents.strategy.strategy_selector import StrategySelection
from functional_agents.strategy.theory_evaluation import CriterionScore, TheoryEvaluation
from functional_agents.strategy.theory_generator import TheoryGenerator


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _plan(plan_id: str = "P-TEST") -> StrategyPlan:
    return StrategyPlan(plan_id=plan_id, framework="executive", active_dimensions=[])


def _choice_set(sid: str) -> StrategicChoiceSet:
    ch = StrategicChoice(
        id=f"SC-{sid}", dimension="market", selected_value="OPT-A",
        rationale="r", confidence="High", supporting_assumptions=[], requiredness="optional",
    )
    return StrategicChoiceSet(
        id=sid, choices=[ch], overall_confidence="High",
        internal_conflicts=[], completeness=1.0, rationale="r",
    )


def _theory(tid: str, scid: str = "SCS-X") -> TheoryOfWinning:
    return TheoryOfWinning(theory_id=tid, source_choice_set_id=scid)


def _eval(tid: str, score: float = 0.8) -> TheoryEvaluation:
    return TheoryEvaluation(
        theory_id=tid,
        criteria_scores={"x": CriterionScore(score=score, rationale="r", weight=1.0)},
        strengths=[], weaknesses=[], residual_risks=[],
        overall_score=score, confidence="High", metadata={},
    )


def _selection(winner: str, runner: str | None = None) -> StrategySelection:
    return StrategySelection(winner_theory_id=winner, winner_score=0.8,
                             runner_up_theory_id=runner)


def _position(theory: TheoryOfWinning, pos_id: str = "SP-TEST") -> StrategicPosition:
    return StrategicPosition(
        position_id=pos_id, created_at="2026-07-26T00:00:00+00:00",
        theory_of_winning=theory,
        recommendation=StrategicRecommendation(
            recommended_option_id=theory.recommended_option_id,
            recommended_option_title="", board_recommendation="Go",
            decision_readiness="Ready", overall_confidence="High",
        ),
        justification=StrategicJustification(
            decision_analysis={}, strategic_options=[],
            assumptions=[], risks=[], opportunities=[],
        ),
        execution=StrategicExecution(recommendations=[], validation_priorities=[]),
    )


def _make_lineage_inputs(n: int = 3):
    """Return (plan, choice_sets, theories, evaluations, selection, position, trace_id)."""
    plan = _plan(f"P-N{n}")
    choice_sets = [_choice_set(f"SCS-{i}") for i in range(n)]
    theories = [_theory(f"TH-SCS-{i}", scid=f"SCS-{i}") for i in range(n)]
    evaluations = [_eval(f"TH-SCS-{i}", 0.9 - i * 0.1) for i in range(n)]
    winner = theories[0]
    runner = theories[1].theory_id if n > 1 else None
    sel = _selection(winner.theory_id, runner)
    pos = _position(winner)
    trace_id = f"STRAT-{plan.plan_id}"
    return plan, choice_sets, theories, evaluations, sel, pos, trace_id


def _make_trace_with_lineage(n: int = 3, research_id: str = "R-TEST") -> StrategyTrace:
    plan, choice_sets, theories, evaluations, sel, pos, trace_id = _make_lineage_inputs(n)
    lineage = build_strategy_lineage(
        research_id=research_id,
        plan=plan,
        choice_sets=choice_sets,
        theories=theories,
        evaluations=evaluations,
        selection=sel,
        strategic_position=pos,
        trace_id=trace_id,
    )
    return StrategyTrace(
        trace_id=trace_id,
        created_at="2026-07-26T00:00:00+00:00",
        plan=plan,
        choice_sets=choice_sets,
        theories=theories,
        evaluations=evaluations,
        selection=sel,
        strategic_position=pos,
        lineage=lineage,
        metadata={
            "framework": "executive", "plan_id": plan.plan_id,
            "choice_set_count": n, "theory_count": n, "evaluation_count": n,
            "selected_theory_id": sel.winner_theory_id,
            "score_margin": 0.0, "tie_breaker_used": None,
            "research_id": research_id,
        },
    )


def _full_ctx() -> AgentContext:
    return AgentContext(
        question="What should we do?", profiles=["test"], execution_profile="test",
        research_object={"id": "R-TEST"}, run_id="run001",
        strategic_options=[{
            "option_id": "OPT-A", "title": "Option A", "description": "First.",
            "strategic_objective": "Grow.", "expected_outcomes": ["O1"],
            "supporting_assumption_ids": [], "associated_risk_ids": [],
            "associated_opportunity_ids": [], "supporting_recommendation_ids": [],
            "advantages": ["Fast"], "disadvantages": ["Risky"],
            "implementation_complexity": "Low", "estimated_time_horizon": "Near-term",
            "capital_intensity": "Low", "confidence": "High", "recommended": True, "rationale": "Best.",
        }],
        assumptions=[], risks=[], opportunities=[], recommendations=[],
        decision_model={},
        decision_analysis={"recommended_option_id": "OPT-A", "rationale": "Best.",
                           "key_tradeoffs": [], "decision_matrix": []},
        executive_confidence={"overall_confidence": "High", "board_recommendation": "Proceed.",
                               "decision_readiness": "Ready", "confidence_drivers": [],
                               "confidence_limiters": [], "critical_unknowns": [],
                               "validation_priorities": []},
        preferred_option={"option_id": "OPT-A", "title": "Option A"},
        research_strategy={},
    )


# ---------------------------------------------------------------------------
# StrategyLineageLink construction and field validation
# ---------------------------------------------------------------------------

class TestStrategyLineageLinkConstruction:
    def _link(self, **kwargs) -> StrategyLineageLink:
        defaults = dict(
            source_type="research_object",
            source_id="R-001",
            target_type="strategy_plan",
            target_id="SPLAN-001",
            relationship="informs",
        )
        defaults.update(kwargs)
        return StrategyLineageLink(**defaults)

    def test_construction_with_required_fields(self):
        link = self._link()
        assert link.source_type == "research_object"
        assert link.source_id == "R-001"
        assert link.target_type == "strategy_plan"
        assert link.target_id == "SPLAN-001"
        assert link.relationship == "informs"

    def test_metadata_defaults_to_empty_dict(self):
        link = self._link()
        assert link.metadata == {}

    def test_metadata_can_be_provided(self):
        link = self._link(metadata={"note": "test"})
        assert link.metadata == {"note": "test"}

    def test_frozen_prevents_attribute_assignment(self):
        link = self._link()
        with pytest.raises(Exception):
            link.source_id = "NEW"  # type: ignore[misc]

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            StrategyLineageLink(
                source_type="a", source_id="b", target_type="c",
                target_id="d", relationship="e", extra_field="x",
            )

    def test_empty_source_id_rejected(self):
        with pytest.raises(ValueError, match="source_id"):
            self._link(source_id="")

    def test_whitespace_source_id_rejected(self):
        with pytest.raises(ValueError, match="source_id"):
            self._link(source_id="   ")

    def test_empty_target_id_rejected(self):
        with pytest.raises(ValueError, match="target_id"):
            self._link(target_id="")

    def test_whitespace_target_id_rejected(self):
        with pytest.raises(ValueError, match="target_id"):
            self._link(target_id="   ")

    def test_empty_relationship_rejected(self):
        with pytest.raises(ValueError, match="relationship"):
            self._link(relationship="")

    def test_whitespace_relationship_rejected(self):
        with pytest.raises(ValueError, match="relationship"):
            self._link(relationship="\t")

    def test_empty_source_type_rejected(self):
        with pytest.raises(ValueError, match="source_type"):
            self._link(source_type="")

    def test_empty_target_type_rejected(self):
        with pytest.raises(ValueError, match="target_type"):
            self._link(target_type="")

    def test_serialization_round_trip(self):
        link = self._link(metadata={"k": 1})
        data = link.model_dump(mode="json")
        restored = StrategyLineageLink.model_validate(data)
        assert restored == link

    def test_all_string_fields_preserved(self):
        link = self._link(
            source_type="theory_of_winning",
            source_id="TH-SCS-0",
            target_type="theory_evaluation",
            target_id="TH-SCS-0",
            relationship="evaluated_by",
        )
        assert link.source_type == "theory_of_winning"
        assert link.relationship == "evaluated_by"


# ---------------------------------------------------------------------------
# build_strategy_lineage
# ---------------------------------------------------------------------------

class TestBuildStrategyLineage:
    def _build(self, n: int = 3, research_id: str = "R-001"):
        plan, choice_sets, theories, evaluations, sel, pos, trace_id = _make_lineage_inputs(n)
        return build_strategy_lineage(
            research_id=research_id,
            plan=plan,
            choice_sets=choice_sets,
            theories=theories,
            evaluations=evaluations,
            selection=sel,
            strategic_position=pos,
            trace_id=trace_id,
        ), plan, choice_sets, theories, evaluations, sel, pos, trace_id

    def test_returns_list_of_lineage_links(self):
        links, *_ = self._build()
        assert isinstance(links, list)
        assert all(isinstance(lk, StrategyLineageLink) for lk in links)

    def test_link_count_formula_n3(self):
        links, *_ = self._build(n=3)
        assert len(links) == 4 * 3 + 4  # 16

    def test_link_count_formula_n1(self):
        links, *_ = self._build(n=1)
        assert len(links) == 4 * 1 + 4  # 8

    def test_link_count_formula_n5(self):
        links, *_ = self._build(n=5)
        assert len(links) == 4 * 5 + 4  # 24

    def test_first_link_is_research_to_plan(self):
        links, plan, *_ = self._build(research_id="R-FIRST")
        assert links[0].source_type == "research_object"
        assert links[0].source_id == "R-FIRST"
        assert links[0].target_type == "strategy_plan"
        assert links[0].target_id == plan.plan_id
        assert links[0].relationship == "informs"

    def test_plan_to_choice_set_links_present(self):
        links, plan, choice_sets, *_ = self._build(n=3)
        gen_links = [lk for lk in links
                     if lk.source_type == "strategy_plan" and lk.relationship == "generates"]
        assert len(gen_links) == 3
        target_ids = {lk.target_id for lk in gen_links}
        assert target_ids == {cs.id for cs in choice_sets}

    def test_choice_set_to_theory_links_present(self):
        links, _, choice_sets, theories, *_ = self._build(n=3)
        prod_links = [lk for lk in links
                      if lk.source_type == "strategic_choice_set" and lk.relationship == "produces"]
        assert len(prod_links) == 3
        # source_ids should match source_choice_set_id on each theory
        source_ids = {lk.source_id for lk in prod_links}
        assert source_ids == {t.source_choice_set_id for t in theories}

    def test_theory_to_eval_links_present(self):
        links, _, _, theories, *_ = self._build(n=3)
        ev_links = [lk for lk in links
                    if lk.source_type == "theory_of_winning" and lk.relationship == "evaluated_by"]
        assert len(ev_links) == 3
        source_ids = {lk.source_id for lk in ev_links}
        assert source_ids == {t.theory_id for t in theories}

    def test_eval_to_selection_links_present(self):
        links, *_, sel, pos, trace_id = self._build(n=3)
        sel_id = f"SEL-STRAT-P-N3"
        contrib_links = [lk for lk in links
                         if lk.source_type == "theory_evaluation"
                         and lk.relationship == "contributes_to"]
        assert len(contrib_links) == 3
        assert all(lk.target_id == sel_id for lk in contrib_links)

    def test_selection_selects_winner_link(self):
        links, _, _, _, _, sel, _, trace_id = self._build(n=3)
        sel_links = [lk for lk in links
                     if lk.source_type == "strategy_selection" and lk.relationship == "selects"]
        assert len(sel_links) == 1
        assert sel_links[0].target_type == "theory_of_winning"
        assert sel_links[0].target_id == sel.winner_theory_id

    def test_selection_link_source_id_format(self):
        links, _, _, _, _, sel, pos, trace_id = self._build(n=3)
        sel_link = next(lk for lk in links
                        if lk.source_type == "strategy_selection" and lk.relationship == "selects")
        assert sel_link.source_id == f"SEL-{trace_id}"

    def test_winner_grounds_position(self):
        links, _, _, _, _, sel, pos, _ = self._build(n=3)
        grounds = [lk for lk in links
                   if lk.relationship == "grounds"]
        assert len(grounds) == 1
        assert grounds[0].source_id == sel.winner_theory_id
        assert grounds[0].target_type == "strategic_position"
        assert grounds[0].target_id == pos.position_id

    def test_position_captured_in_trace(self):
        links, _, _, _, _, sel, pos, trace_id = self._build(n=3)
        cap = [lk for lk in links if lk.relationship == "captured_in"]
        assert len(cap) == 1
        assert cap[0].source_id == pos.position_id
        assert cap[0].target_type == "strategy_trace"
        assert cap[0].target_id == trace_id

    def test_all_links_are_immutable(self):
        links, *_ = self._build()
        for lk in links:
            with pytest.raises(Exception):
                lk.source_id = "x"  # type: ignore[misc]

    def test_no_duplicate_composite_keys(self):
        links, *_ = self._build(n=3)
        keys = [(lk.source_type, lk.source_id, lk.target_type, lk.target_id, lk.relationship)
                for lk in links]
        assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# TheoryOfWinning.source_choice_set_id field
# ---------------------------------------------------------------------------

class TestSourceChoiceSetId:
    def test_field_exists_on_theory_of_winning(self):
        t = TheoryOfWinning(theory_id="TH-X", source_choice_set_id="SCS-X")
        assert hasattr(t, "source_choice_set_id")

    def test_construction_without_source_choice_set_id_raises(self):
        with pytest.raises(ValidationError):
            TheoryOfWinning(theory_id="TH-X")

    def test_field_can_be_set_explicitly(self):
        t = TheoryOfWinning(theory_id="TH-X", source_choice_set_id="SCS-0")
        assert t.source_choice_set_id == "SCS-0"

    def test_theory_generator_sets_source_choice_set_id(self):
        import types
        cs = _choice_set("SCS-GEN")
        research = types.SimpleNamespace(
            executive_confidence=None, decision_analysis=None, preferred_option=None,
            strategic_options=[], assumptions=[], risks=[], research_object={},
        )
        theory = TheoryGenerator().build(cs, research)
        assert theory.source_choice_set_id == cs.id

    def test_theory_generator_theory_id_matches_convention(self):
        import types
        cs = _choice_set("SCS-ID")
        research = types.SimpleNamespace(
            executive_confidence=None, decision_analysis=None, preferred_option=None,
            strategic_options=[], assumptions=[], risks=[], research_object={},
        )
        theory = TheoryGenerator().build(cs, research)
        assert theory.theory_id == f"TH-{cs.id}"
        assert theory.source_choice_set_id == cs.id

    def test_coordinator_theories_all_have_source_choice_set_id(self):
        ctx = _full_ctx()
        coord = StrategyCoordinator()
        coord.build(ctx)
        for theory in coord._theories:
            assert theory.source_choice_set_id, (
                f"theory_id={theory.theory_id!r} has empty source_choice_set_id"
            )

    def test_coordinator_source_ids_match_choice_sets(self):
        ctx = _full_ctx()
        coord = StrategyCoordinator()
        coord.build(ctx)
        cs_ids = {cs.id for cs in coord._choice_sets}
        for theory in coord._theories:
            assert theory.source_choice_set_id in cs_ids


# ---------------------------------------------------------------------------
# StrategyTrace rules 13-16 (lineage integrity)
# ---------------------------------------------------------------------------

class TestStrategyTraceLineageRules:
    def _base_trace_kwargs(self, n: int = 2):
        plan, choice_sets, theories, evaluations, sel, pos, trace_id = _make_lineage_inputs(n)
        return dict(
            trace_id=trace_id, created_at="2026-07-26T00:00:00+00:00",
            plan=plan, choice_sets=choice_sets, theories=theories,
            evaluations=evaluations, selection=sel, strategic_position=pos,
            metadata={},
        )

    def test_empty_lineage_skips_rules_15_to_18(self):
        # StrategyTrace without lineage — rules 15-18 not triggered
        plan = _plan()
        cs = _choice_set("SCS-0")
        t = _theory("TH-SCS-0", "SCS-0")  # scid matches choice_set (required by rule 14)
        ev = _eval("TH-SCS-0")
        sel = _selection("TH-SCS-0")
        pos = _position(t)
        # Rules 15-18 are gated on self.lineage being non-empty — must not raise
        trace = StrategyTrace(
            trace_id="STRAT-P-TEST", created_at="2026-07-26T00:00:00+00:00",
            plan=plan, choice_sets=[cs], theories=[t], evaluations=[ev],
            selection=sel, strategic_position=pos, lineage=[], metadata={},
        )
        assert trace.lineage == []

    def test_rule_13_empty_source_choice_set_id_rejected(self):
        # After PH11.2a: empty scid rejected at TheoryOfWinning construction, not StrategyTrace
        with pytest.raises(ValidationError, match="source_choice_set_id"):
            TheoryOfWinning(theory_id="TH-X", source_choice_set_id="")

    def test_rule_14_unknown_source_choice_set_id_rejected(self):
        kwargs = self._base_trace_kwargs(n=2)
        theories = list(kwargs["theories"])
        # Assign a source_choice_set_id that does not match any choice_set
        theories[0] = TheoryOfWinning(
            theory_id=theories[0].theory_id,
            source_choice_set_id="SCS-NONEXISTENT",  # violates rule 14
        )
        kwargs["theories"] = theories
        plan, cs_list, orig_theories, evals, sel, pos, trace_id = _make_lineage_inputs(2)
        lineage = build_strategy_lineage(
            research_id="R-TEST", plan=plan, choice_sets=cs_list,
            theories=orig_theories, evaluations=evals, selection=sel,
            strategic_position=pos, trace_id=trace_id,
        )
        with pytest.raises(ValueError, match="not found in choice_sets"):
            StrategyTrace(**{**kwargs, "lineage": lineage})

    def test_rule_15_duplicate_lineage_link_rejected(self):
        plan, choice_sets, theories, evaluations, sel, pos, trace_id = _make_lineage_inputs(2)
        lineage = build_strategy_lineage(
            research_id="R-TEST", plan=plan, choice_sets=choice_sets,
            theories=theories, evaluations=evaluations, selection=sel,
            strategic_position=pos, trace_id=trace_id,
        )
        dup_link = lineage[0]
        with pytest.raises(ValueError, match="duplicate lineage link"):
            StrategyTrace(
                trace_id=trace_id, created_at="2026-07-26T00:00:00+00:00",
                plan=plan, choice_sets=choice_sets, theories=theories,
                evaluations=evaluations, selection=sel, strategic_position=pos,
                lineage=list(lineage) + [dup_link], metadata={},
            )

    def test_rule_16_unknown_theory_target_rejected(self):
        plan, choice_sets, theories, evaluations, sel, pos, trace_id = _make_lineage_inputs(2)
        lineage = build_strategy_lineage(
            research_id="R-TEST", plan=plan, choice_sets=choice_sets,
            theories=theories, evaluations=evaluations, selection=sel,
            strategic_position=pos, trace_id=trace_id,
        )
        bad_link = StrategyLineageLink(
            source_type="strategy_selection",
            source_id=f"SEL-{trace_id}",
            target_type="theory_of_winning",
            target_id="TH-NONEXISTENT",
            relationship="selects",
        )
        with pytest.raises(ValueError, match="theory_of_winning"):
            StrategyTrace(
                trace_id=trace_id, created_at="2026-07-26T00:00:00+00:00",
                plan=plan, choice_sets=choice_sets, theories=theories,
                evaluations=evaluations, selection=sel, strategic_position=pos,
                lineage=list(lineage) + [bad_link], metadata={},
            )

    def test_rule_16_unknown_choice_set_target_rejected(self):
        plan, choice_sets, theories, evaluations, sel, pos, trace_id = _make_lineage_inputs(2)
        lineage = build_strategy_lineage(
            research_id="R-TEST", plan=plan, choice_sets=choice_sets,
            theories=theories, evaluations=evaluations, selection=sel,
            strategic_position=pos, trace_id=trace_id,
        )
        bad_link = StrategyLineageLink(
            source_type="strategy_plan",
            source_id=plan.plan_id,
            target_type="strategic_choice_set",
            target_id="SCS-NONEXISTENT",
            relationship="generates",
        )
        with pytest.raises(ValueError, match="strategic_choice_set"):
            StrategyTrace(
                trace_id=trace_id, created_at="2026-07-26T00:00:00+00:00",
                plan=plan, choice_sets=choice_sets, theories=theories,
                evaluations=evaluations, selection=sel, strategic_position=pos,
                lineage=list(lineage) + [bad_link], metadata={},
            )

    def test_rule_16_unknown_evaluation_target_rejected(self):
        plan, choice_sets, theories, evaluations, sel, pos, trace_id = _make_lineage_inputs(2)
        lineage = build_strategy_lineage(
            research_id="R-TEST", plan=plan, choice_sets=choice_sets,
            theories=theories, evaluations=evaluations, selection=sel,
            strategic_position=pos, trace_id=trace_id,
        )
        bad_link = StrategyLineageLink(
            source_type="theory_of_winning",
            source_id=theories[0].theory_id,
            target_type="theory_evaluation",
            target_id="TH-NONEXISTENT",
            relationship="evaluated_by",
        )
        with pytest.raises(ValueError, match="theory_evaluation"):
            StrategyTrace(
                trace_id=trace_id, created_at="2026-07-26T00:00:00+00:00",
                plan=plan, choice_sets=choice_sets, theories=theories,
                evaluations=evaluations, selection=sel, strategic_position=pos,
                lineage=list(lineage) + [bad_link], metadata={},
            )

    def test_valid_lineage_passes_validation(self):
        trace = _make_trace_with_lineage(n=3)
        assert len(trace.lineage) == 16
        assert isinstance(trace, StrategyTrace)


# ---------------------------------------------------------------------------
# write_artifact_index
# ---------------------------------------------------------------------------

class TestArtifactIndex:
    def test_writes_artifact_index_json(self, tmp_path):
        trace = _make_trace_with_lineage()
        st_path = write_strategy_trace(trace, tmp_path)
        write_artifact_index(trace, st_path, tmp_path)
        assert (tmp_path / "artifact.index.json").exists()

    def test_index_is_valid_utf8_json(self, tmp_path):
        trace = _make_trace_with_lineage()
        st_path = write_strategy_trace(trace, tmp_path)
        write_artifact_index(trace, st_path, tmp_path)
        raw = (tmp_path / "artifact.index.json").read_text(encoding="utf-8")
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_index_has_schema_version(self, tmp_path):
        trace = _make_trace_with_lineage()
        st_path = write_strategy_trace(trace, tmp_path)
        write_artifact_index(trace, st_path, tmp_path)
        data = json.loads((tmp_path / "artifact.index.json").read_text(encoding="utf-8"))
        assert "schema_version" in data
        assert "ph11.2" in data["schema_version"]

    def test_index_has_entries_list(self, tmp_path):
        trace = _make_trace_with_lineage()
        st_path = write_strategy_trace(trace, tmp_path)
        write_artifact_index(trace, st_path, tmp_path)
        data = json.loads((tmp_path / "artifact.index.json").read_text(encoding="utf-8"))
        assert isinstance(data["entries"], list)
        assert len(data["entries"]) == 1

    def test_index_entry_artifact_type(self, tmp_path):
        trace = _make_trace_with_lineage()
        st_path = write_strategy_trace(trace, tmp_path)
        write_artifact_index(trace, st_path, tmp_path)
        data = json.loads((tmp_path / "artifact.index.json").read_text(encoding="utf-8"))
        assert data["entries"][0]["artifact_type"] == "strategy_trace"

    def test_index_entry_trace_id(self, tmp_path):
        trace = _make_trace_with_lineage()
        st_path = write_strategy_trace(trace, tmp_path)
        write_artifact_index(trace, st_path, tmp_path)
        data = json.loads((tmp_path / "artifact.index.json").read_text(encoding="utf-8"))
        assert data["entries"][0]["trace_id"] == trace.trace_id

    def test_index_entry_path(self, tmp_path):
        trace = _make_trace_with_lineage()
        st_path = write_strategy_trace(trace, tmp_path)
        write_artifact_index(trace, st_path, tmp_path)
        data = json.loads((tmp_path / "artifact.index.json").read_text(encoding="utf-8"))
        assert data["entries"][0]["path"] == str(st_path)

    def test_index_entry_lineage_count(self, tmp_path):
        trace = _make_trace_with_lineage(n=3)
        st_path = write_strategy_trace(trace, tmp_path)
        write_artifact_index(trace, st_path, tmp_path)
        data = json.loads((tmp_path / "artifact.index.json").read_text(encoding="utf-8"))
        assert data["entries"][0]["lineage_count"] == len(trace.lineage)

    def test_index_entry_research_id(self, tmp_path):
        trace = _make_trace_with_lineage(research_id="R-CUSTOM")
        st_path = write_strategy_trace(trace, tmp_path)
        write_artifact_index(trace, st_path, tmp_path)
        data = json.loads((tmp_path / "artifact.index.json").read_text(encoding="utf-8"))
        assert data["entries"][0]["research_id"] == "R-CUSTOM"

    def test_second_write_replaces_strategy_trace_entry(self, tmp_path):
        trace1 = _make_trace_with_lineage(n=1, research_id="R-FIRST")
        st_path1 = write_strategy_trace(trace1, tmp_path)
        write_artifact_index(trace1, st_path1, tmp_path)

        trace2 = _make_trace_with_lineage(n=3, research_id="R-SECOND")
        st_path2 = write_strategy_trace(trace2, tmp_path)
        write_artifact_index(trace2, st_path2, tmp_path)

        data = json.loads((tmp_path / "artifact.index.json").read_text(encoding="utf-8"))
        st_entries = [e for e in data["entries"] if e["artifact_type"] == "strategy_trace"]
        assert len(st_entries) == 1
        assert st_entries[0]["research_id"] == "R-SECOND"

    def test_returns_written_path(self, tmp_path):
        trace = _make_trace_with_lineage()
        st_path = write_strategy_trace(trace, tmp_path)
        idx_path = write_artifact_index(trace, st_path, tmp_path)
        assert idx_path == tmp_path / "artifact.index.json"

    def test_creates_output_dir_when_missing(self, tmp_path):
        nested = tmp_path / "a" / "b"
        trace = _make_trace_with_lineage()
        st_path = nested / "strategy.trace.json"
        write_artifact_index(trace, st_path, nested)
        assert (nested / "artifact.index.json").exists()


# ---------------------------------------------------------------------------
# End-to-end: coordinator run → lineage in trace → round-trip
# ---------------------------------------------------------------------------

class TestEndToEndLineage:
    def test_coordinator_trace_has_lineage(self):
        ctx = _full_ctx()
        coord = StrategyCoordinator()
        coord.build(ctx)
        assert coord._trace is not None
        assert len(coord._trace.lineage) > 0

    def test_lineage_count_formula(self):
        ctx = _full_ctx()
        coord = StrategyCoordinator()
        coord.build(ctx)
        n = len(coord._choice_sets)
        expected = 4 * n + 4
        assert len(coord._trace.lineage) == expected

    def test_lineage_round_trips_through_json(self, tmp_path):
        ctx = _full_ctx()
        coord = StrategyCoordinator()
        coord.build(ctx)
        path = write_strategy_trace(coord._trace, tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        restored = StrategyTrace.from_dict(data)
        assert isinstance(restored, StrategyTrace)
        assert len(restored.lineage) == len(coord._trace.lineage)

    def test_lineage_winner_link_matches_selection(self):
        ctx = _full_ctx()
        coord = StrategyCoordinator()
        coord.build(ctx)
        winner_id = coord._selection.winner_theory_id
        selects_links = [lk for lk in coord._trace.lineage
                         if lk.relationship == "selects"]
        assert len(selects_links) == 1
        assert selects_links[0].target_id == winner_id

    def test_lineage_position_link_matches_position(self):
        ctx = _full_ctx()
        coord = StrategyCoordinator()
        sp = coord.build(ctx)
        grounds_links = [lk for lk in coord._trace.lineage
                         if lk.relationship == "grounds"]
        assert len(grounds_links) == 1
        assert grounds_links[0].target_id == sp.position_id

    def test_metadata_research_id_populated(self):
        ctx = _full_ctx()
        coord = StrategyCoordinator()
        coord.build(ctx)
        assert coord._trace.metadata.get("research_id") == "R-TEST"

    def test_artifact_index_written_by_coordinator_via_helpers(self, tmp_path):
        ctx = _full_ctx()
        coord = StrategyCoordinator()
        coord.build(ctx)
        st_path = write_strategy_trace(coord._trace, tmp_path)
        idx_path = write_artifact_index(coord._trace, st_path, tmp_path)
        data = json.loads(idx_path.read_text(encoding="utf-8"))
        entries = [e for e in data["entries"] if e["artifact_type"] == "strategy_trace"]
        assert len(entries) == 1
        assert entries[0]["trace_id"] == coord._trace.trace_id
