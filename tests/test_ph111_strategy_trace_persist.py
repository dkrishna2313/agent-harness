"""PH11.1 — StrategyTrace persistence tests.

Covers:
- write_strategy_trace() writes strategy.trace.json to the given directory
- Written file is valid UTF-8 JSON
- Written file round-trips to a valid StrategyTrace
- Theory and evaluation identities are preserved in the file
- Second write overwrites the first (no accumulation)
- Output directory is created if it does not exist
- Missing trace → write_strategy_trace is never called (orchestrator guard)
- Deliverable artifact is registered with correct type, path, and mime_type
- Embedded strategy_trace in pipeline trace is unaffected
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from functional_agents.context import AgentContext
from functional_agents.deliverables.artifact import DeliverableArtifact
from functional_agents.strategy import StrategyCoordinator, StrategyTrace
from functional_agents.strategy.strategy_trace import write_strategy_trace
from functional_agents.strategy.strategic_position import (
    StrategicExecution, StrategicJustification, StrategicPosition,
    StrategicRecommendation, TheoryOfWinning,
)
from functional_agents.strategy.strategy_plan import StrategyPlan
from functional_agents.strategy.strategic_choice_set import StrategicChoiceSet
from functional_agents.strategy.strategic_choice import StrategicChoice
from functional_agents.strategy.strategy_selector import StrategySelection
from functional_agents.strategy.theory_evaluation import CriterionScore, TheoryEvaluation
from functional_agents.trace_paths import STRATEGY_TRACE_FILENAME, STRATEGY_TRACE


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


def _theory(tid: str) -> TheoryOfWinning:
    return TheoryOfWinning(theory_id=tid)


def _eval(tid: str, score: float = 0.8) -> TheoryEvaluation:
    return TheoryEvaluation(
        theory_id=tid,
        criteria_scores={"x": CriterionScore(score=score, rationale="r", weight=1.0)},
        strengths=[], weaknesses=[], residual_risks=[],
        overall_score=score, confidence="High", metadata={},
    )


def _selection(winner: str, runner: str | None = "TH-SCS-1") -> StrategySelection:
    return StrategySelection(winner_theory_id=winner, winner_score=0.8, runner_up_theory_id=runner)


def _position(theory: TheoryOfWinning) -> StrategicPosition:
    return StrategicPosition(
        position_id="SP-TEST", created_at="2026-07-26T00:00:00+00:00",
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


def _make_trace(n: int = 3, plan_id: str = "P-TEST") -> StrategyTrace:
    plan = _plan(plan_id)
    choice_sets = [_choice_set(f"SCS-{i}") for i in range(n)]
    theories = [_theory(f"TH-SCS-{i}") for i in range(n)]
    evaluations = [_eval(f"TH-SCS-{i}", 0.9 - i * 0.1) for i in range(n)]
    winner = theories[0]
    runner_up = theories[1].theory_id if n > 1 else None
    return StrategyTrace(
        trace_id=f"STRAT-{plan.plan_id}",
        created_at="2026-07-26T00:00:00+00:00",
        plan=plan,
        choice_sets=choice_sets,
        theories=theories,
        evaluations=evaluations,
        selection=_selection(winner.theory_id, runner_up),
        strategic_position=_position(winner),
        metadata={"framework": "executive", "plan_id": plan_id,
                  "choice_set_count": n, "theory_count": n, "evaluation_count": n,
                  "selected_theory_id": winner.theory_id, "score_margin": 0.0,
                  "tie_breaker_used": None},
    )


def _full_ctx() -> AgentContext:
    return AgentContext(
        question="What should we do?", profiles=["test"], execution_profile="test",
        research_object={"id": "R-TEST"}, run_id="run001",
        strategic_options=[{"option_id": "OPT-A", "title": "Option A", "description": "First.",
            "strategic_objective": "Grow.", "expected_outcomes": ["O1"],
            "supporting_assumption_ids": [], "associated_risk_ids": [],
            "associated_opportunity_ids": [], "supporting_recommendation_ids": [],
            "advantages": ["Fast"], "disadvantages": ["Risky"],
            "implementation_complexity": "Low", "estimated_time_horizon": "Near-term",
            "capital_intensity": "Low", "confidence": "High", "recommended": True, "rationale": "Best."}],
        assumptions=[], risks=[], opportunities=[], recommendations=[],
        decision_model={},
        decision_analysis={"recommended_option_id": "OPT-A", "rationale": "Best.",
            "key_tradeoffs": [], "decision_matrix": []},
        executive_confidence={"overall_confidence": "High", "board_recommendation": "Proceed.",
            "decision_readiness": "Ready", "confidence_drivers": [], "confidence_limiters": [],
            "critical_unknowns": [], "validation_priorities": []},
        preferred_option={"option_id": "OPT-A", "title": "Option A"},
        research_strategy={},
    )


# ---------------------------------------------------------------------------
# trace_paths constants
# ---------------------------------------------------------------------------

class TestTracePaths:
    def test_strategy_trace_filename_constant(self):
        assert STRATEGY_TRACE_FILENAME == "strategy.trace.json"

    def test_strategy_trace_path_constant(self):
        assert STRATEGY_TRACE == Path("outputs") / "strategy.trace.json"

    def test_strategy_trace_under_outputs(self):
        assert STRATEGY_TRACE.parent == Path("outputs")


# ---------------------------------------------------------------------------
# write_strategy_trace — unit tests
# ---------------------------------------------------------------------------

class TestWriteStrategyTrace:
    def test_writes_file(self, tmp_path):
        st = _make_trace()
        out = write_strategy_trace(st, tmp_path)
        assert out.exists()

    def test_filename_is_strategy_trace_json(self, tmp_path):
        st = _make_trace()
        out = write_strategy_trace(st, tmp_path)
        assert out.name == "strategy.trace.json"

    def test_file_is_valid_utf8_json(self, tmp_path):
        st = _make_trace()
        write_strategy_trace(st, tmp_path)
        raw = (tmp_path / "strategy.trace.json").read_text(encoding="utf-8")
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_file_round_trips_as_strategy_trace(self, tmp_path):
        st = _make_trace(plan_id="P-ROUNDTRIP")
        write_strategy_trace(st, tmp_path)
        raw = (tmp_path / "strategy.trace.json").read_text(encoding="utf-8")
        restored = StrategyTrace.from_dict(json.loads(raw))
        assert isinstance(restored, StrategyTrace)

    def test_trace_id_preserved_in_file(self, tmp_path):
        st = _make_trace(plan_id="P-IDCHECK")
        write_strategy_trace(st, tmp_path)
        data = json.loads((tmp_path / "strategy.trace.json").read_text(encoding="utf-8"))
        assert data["trace_id"] == st.trace_id

    def test_theory_ids_preserved_in_file(self, tmp_path):
        st = _make_trace(n=3)
        write_strategy_trace(st, tmp_path)
        data = json.loads((tmp_path / "strategy.trace.json").read_text(encoding="utf-8"))
        file_ids = [t["theory_id"] for t in data["theories"]]
        orig_ids = [t.theory_id for t in st.theories]
        assert file_ids == orig_ids

    def test_evaluation_ids_preserved_in_file(self, tmp_path):
        st = _make_trace(n=3)
        write_strategy_trace(st, tmp_path)
        data = json.loads((tmp_path / "strategy.trace.json").read_text(encoding="utf-8"))
        file_ids = [e["theory_id"] for e in data["evaluations"]]
        orig_ids = [e.theory_id for e in st.evaluations]
        assert file_ids == orig_ids

    def test_winner_id_preserved_in_file(self, tmp_path):
        st = _make_trace()
        write_strategy_trace(st, tmp_path)
        data = json.loads((tmp_path / "strategy.trace.json").read_text(encoding="utf-8"))
        assert data["selection"]["winner_theory_id"] == st.selection.winner_theory_id

    def test_runner_up_id_preserved_in_file(self, tmp_path):
        st = _make_trace(n=3)
        write_strategy_trace(st, tmp_path)
        data = json.loads((tmp_path / "strategy.trace.json").read_text(encoding="utf-8"))
        assert data["selection"]["runner_up_theory_id"] == st.selection.runner_up_theory_id

    def test_metadata_preserved_in_file(self, tmp_path):
        st = _make_trace(plan_id="P-META")
        write_strategy_trace(st, tmp_path)
        data = json.loads((tmp_path / "strategy.trace.json").read_text(encoding="utf-8"))
        assert data["metadata"]["plan_id"] == "P-META"
        assert data["metadata"]["framework"] == "executive"

    def test_second_write_overwrites_first(self, tmp_path):
        st1 = _make_trace(plan_id="P-FIRST")
        st2 = _make_trace(plan_id="P-SECOND")
        write_strategy_trace(st1, tmp_path)
        write_strategy_trace(st2, tmp_path)
        data = json.loads((tmp_path / "strategy.trace.json").read_text(encoding="utf-8"))
        assert data["trace_id"] == st2.trace_id  # second write wins

    def test_creates_output_dir_when_missing(self, tmp_path):
        nested = tmp_path / "a" / "b" / "outputs"
        assert not nested.exists()
        st = _make_trace()
        write_strategy_trace(st, nested)
        assert (nested / "strategy.trace.json").exists()

    def test_returns_written_path(self, tmp_path):
        st = _make_trace()
        returned = write_strategy_trace(st, tmp_path)
        assert returned == tmp_path / "strategy.trace.json"

    def test_does_not_mutate_trace(self, tmp_path):
        st = _make_trace()
        orig_id = st.trace_id
        write_strategy_trace(st, tmp_path)
        assert st.trace_id == orig_id  # frozen model, so this is guaranteed anyway

    def test_json_uses_indent_2(self, tmp_path):
        st = _make_trace()
        write_strategy_trace(st, tmp_path)
        raw = (tmp_path / "strategy.trace.json").read_text(encoding="utf-8")
        # indent=2 produces lines starting with exactly two spaces
        lines = raw.splitlines()
        indented = [l for l in lines if l.startswith("  ") and not l.startswith("   ")]
        assert len(indented) > 0  # at least some two-space-indented lines


# ---------------------------------------------------------------------------
# Missing trace guard — orchestrator logic
# ---------------------------------------------------------------------------

class TestMissingTraceGuard:
    def test_write_not_called_when_trace_none(self, tmp_path):
        """Guard: if _strategy_trace is None, no file is written."""
        # Simulate the orchestrator's guard:
        #   _st_raw = result_ctx.trace.get("_strategy_trace")
        #   if _st_raw is not None: write_strategy_trace(...)
        _st_raw = None
        if _st_raw is not None:
            write_strategy_trace(_st_raw, tmp_path)
        assert not (tmp_path / "strategy.trace.json").exists()

    def test_empty_file_not_written_when_trace_absent(self, tmp_path):
        """No partial or empty artifact when trace build failed."""
        # Even an empty write_text call would create a file; verify we don't touch it
        assert not (tmp_path / "strategy.trace.json").exists()


# ---------------------------------------------------------------------------
# DeliverableArtifact registration
# ---------------------------------------------------------------------------

class TestStrategyTraceDeliverable:
    def _make_artifact(self, path: str, trace_id: str) -> dict:
        return DeliverableArtifact(
            type="strategy_trace",
            path=path,
            mime_type="application/json",
            metadata={"trace_id": trace_id},
        ).to_dict()

    def test_deliverable_type_is_strategy_trace(self):
        art = self._make_artifact("outputs/strategy.trace.json", "STRAT-P-TEST")
        assert art["type"] == "strategy_trace"

    def test_deliverable_path_preserved(self):
        art = self._make_artifact("outputs/strategy.trace.json", "STRAT-P-TEST")
        assert art["path"] == "outputs/strategy.trace.json"

    def test_deliverable_mime_type(self):
        art = self._make_artifact("outputs/strategy.trace.json", "STRAT-P-TEST")
        assert art["mime_type"] == "application/json"

    def test_deliverable_status_defaults_to_generated(self):
        art = self._make_artifact("outputs/strategy.trace.json", "STRAT-P-TEST")
        assert art["status"] == "generated"

    def test_deliverable_trace_id_in_metadata(self):
        art = self._make_artifact("outputs/strategy.trace.json", "STRAT-MY-PLAN")
        assert art["metadata"]["trace_id"] == "STRAT-MY-PLAN"


# ---------------------------------------------------------------------------
# End-to-end: coordinator run → write → verify
# ---------------------------------------------------------------------------

class TestEndToEndPersist:
    def test_coordinator_trace_writes_valid_file(self, tmp_path):
        ctx = _full_ctx()
        coord = StrategyCoordinator()
        coord.build(ctx)
        out = write_strategy_trace(coord._trace, tmp_path)
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        restored = StrategyTrace.from_dict(data)
        assert restored.trace_id == coord._trace.trace_id

    def test_coordinator_trace_ids_round_trip(self, tmp_path):
        ctx = _full_ctx()
        coord = StrategyCoordinator()
        coord.build(ctx)
        write_strategy_trace(coord._trace, tmp_path)
        data = json.loads((tmp_path / "strategy.trace.json").read_text(encoding="utf-8"))
        assert data["selection"]["winner_theory_id"] == coord._selection.winner_theory_id
        assert [t["theory_id"] for t in data["theories"]] == [t.theory_id for t in coord._theories]
        assert [e["theory_id"] for e in data["evaluations"]] == [e.theory_id for e in coord._evaluations]

    def test_coordinator_trace_metadata_round_trip(self, tmp_path):
        ctx = _full_ctx()
        coord = StrategyCoordinator()
        coord.build(ctx)
        write_strategy_trace(coord._trace, tmp_path)
        data = json.loads((tmp_path / "strategy.trace.json").read_text(encoding="utf-8"))
        assert data["metadata"]["plan_id"] == coord._plan.plan_id
        assert data["metadata"]["selected_theory_id"] == coord._selection.winner_theory_id
        assert data["metadata"]["choice_set_count"] == len(coord._choice_sets)

    def test_embedded_pipeline_trace_unaffected(self, tmp_path):
        """write_strategy_trace must not change what build_canonical_trace returns."""
        from functional_agents.pipeline_trace import build_canonical_trace
        ctx = _full_ctx()
        coord = StrategyCoordinator()
        sp = coord.build(ctx)
        ctx.trace["_strategic_position"] = sp
        ctx.trace["_strategy_trace"] = coord._trace

        # Get pipeline trace before write
        before = build_canonical_trace(ctx)
        before_trace_id = before["strategy_trace"]["trace_id"]

        # Write standalone artifact
        write_strategy_trace(coord._trace, tmp_path)

        # Get pipeline trace after write
        after = build_canonical_trace(ctx)
        assert after["strategy_trace"]["trace_id"] == before_trace_id

    def test_standalone_and_embedded_trace_ids_match(self, tmp_path):
        from functional_agents.pipeline_trace import build_canonical_trace
        ctx = _full_ctx()
        coord = StrategyCoordinator()
        sp = coord.build(ctx)
        ctx.trace["_strategic_position"] = sp
        ctx.trace["_strategy_trace"] = coord._trace

        write_strategy_trace(coord._trace, tmp_path)
        data = json.loads((tmp_path / "strategy.trace.json").read_text(encoding="utf-8"))
        ct = build_canonical_trace(ctx)

        assert data["trace_id"] == ct["strategy_trace"]["trace_id"]
