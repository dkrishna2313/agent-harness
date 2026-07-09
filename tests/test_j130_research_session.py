"""Tests for ResearchSession architecture (J13.0).

Covers:
  1.  ResearchState: creation, from_context, to_dict, from_dict roundtrip
  2.  IterationRecord: creation, to_dict, from_dict roundtrip
  3.  Snapshot: from_state, to_dict, from_dict, to_research_state roundtrip
  4.  ResearchSession: create, add_iteration, take_snapshot, complete, archive
  5.  ResearchSession: to_dict, from_dict serialization roundtrip
  6.  SessionStore: create writes file to disk
  7.  SessionStore: save overwrites file
  8.  SessionStore: load returns equivalent session
  9.  SessionStore: archive transitions status to ARCHIVED
  10. SessionStore: continue_session reactivates to ACTIVE
  11. SessionStore: list_sessions returns sorted IDs
  12. SessionStore: SessionNotFoundError on missing session
  13. ResearchState: from_context reads all six artifact slots
  14. Snapshot.to_research_state preserves all artifact fields
  15. Iteration history preserved across save/load roundtrip
  16. Snapshots preserved across save/load roundtrip
  17. Session metadata preserved across save/load roundtrip
  18. Session status transitions: ACTIVE → COMPLETED → ARCHIVED
  19. Package-level imports resolve (__init__.py exports)
  20. No LLM calls in session package
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(**overrides: Any):
    """Build a minimal AgentContext-like object for testing ResearchState.from_context."""
    from functional_agents.context import AgentContext
    ctx = AgentContext(
        question="What is the power demand of AI data centers?",
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        run_id="testrun001",
    )
    ctx.research_object = {"research_id": "R-TEST-001", "question": ctx.question}
    ctx.engagement = {"client": "test", "engagement_id": "ENG-001"}
    ctx.decision_model = {"decision_model_id": "DM-001"}
    ctx.research_gap_analysis = {"gap_count": 3}
    ctx.executive_confidence = {"overall_confidence": "Medium"}
    ctx.iteration_plan = {"iteration_needed": True, "priority_research_tasks": []}
    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


def _make_research_state() -> Any:
    from functional_agents.session import ResearchState
    return ResearchState(
        engagement={"client": "test"},
        research_object={"research_id": "R-001"},
        decision_model={"id": "DM-001"},
        research_gap_analysis={"gaps": []},
        executive_confidence={"confidence": "High"},
        iteration_plan={"iteration_needed": False},
        updated_at="2026-07-09T00:00:00+00:00",
    )


def _make_session() -> Any:
    from functional_agents.session import ResearchSession
    state = _make_research_state()
    return ResearchSession.create(
        metadata={"run_id": "r001", "profiles": ["ai_data_centers"]},
        research_state=state,
    )


# ---------------------------------------------------------------------------
# Section 1: Package-level imports
# ---------------------------------------------------------------------------

class TestPackageImports:
    def test_all_public_names_importable(self):
        from functional_agents.session import (
            ResearchSession,
            SessionStatus,
            ResearchState,
            IterationRecord,
            Snapshot,
            SessionStore,
            SessionNotFoundError,
        )
        assert ResearchSession is not None
        assert SessionStatus is not None
        assert ResearchState is not None
        assert IterationRecord is not None
        assert Snapshot is not None
        assert SessionStore is not None
        assert SessionNotFoundError is not None

    def test_session_status_constants(self):
        from functional_agents.session import SessionStatus
        assert SessionStatus.ACTIVE == "active"
        assert SessionStatus.COMPLETED == "completed"
        assert SessionStatus.ARCHIVED == "archived"

    def test_no_llm_imports_in_session_package(self):
        import functional_agents.session.research_state as rs
        import functional_agents.session.iteration_record as ir
        import functional_agents.session.snapshot as sn
        import functional_agents.session.research_session as sess
        import functional_agents.session.session_store as store
        import inspect
        for mod in [rs, ir, sn, sess, store]:
            src = inspect.getsource(mod)
            assert "claude_client" not in src
            assert "ClaudeClient" not in src
            assert "import anthropic" not in src


# ---------------------------------------------------------------------------
# Section 2: ResearchState
# ---------------------------------------------------------------------------

class TestResearchState:
    def test_creation_with_defaults(self):
        from functional_agents.session import ResearchState
        state = ResearchState()
        assert state.engagement == {}
        assert state.research_object == {}
        assert state.decision_model == {}
        assert state.research_gap_analysis == {}
        assert state.executive_confidence == {}
        assert state.iteration_plan == {}
        assert state.updated_at == ""

    def test_from_context_reads_all_six_slots(self):
        from functional_agents.session import ResearchState
        ctx = _make_context()
        state = ResearchState.from_context(ctx)
        assert state.engagement == ctx.engagement
        assert state.research_object == ctx.research_object
        assert state.decision_model == ctx.decision_model
        assert state.research_gap_analysis == ctx.research_gap_analysis
        assert state.executive_confidence == ctx.executive_confidence
        assert state.iteration_plan == ctx.iteration_plan
        assert state.updated_at != ""

    def test_from_context_makes_copies(self):
        from functional_agents.session import ResearchState
        ctx = _make_context()
        state = ResearchState.from_context(ctx)
        ctx.research_object["mutated"] = True
        assert "mutated" not in state.research_object

    def test_from_context_handles_empty_slots(self):
        from functional_agents.session import ResearchState
        ctx = _make_context()
        ctx.engagement = {}
        ctx.decision_model = {}
        ctx.research_gap_analysis = {}
        ctx.executive_confidence = {}
        ctx.iteration_plan = {}
        state = ResearchState.from_context(ctx)
        assert state.engagement == {}
        assert state.decision_model == {}

    def test_to_dict_has_all_keys(self):
        state = _make_research_state()
        d = state.to_dict()
        assert set(d.keys()) == {
            "engagement", "research_object", "decision_model",
            "research_gap_analysis", "executive_confidence",
            "iteration_plan", "updated_at",
        }

    def test_to_dict_from_dict_roundtrip(self):
        from functional_agents.session import ResearchState
        original = _make_research_state()
        d = original.to_dict()
        restored = ResearchState.from_dict(d)
        assert restored.engagement == original.engagement
        assert restored.research_object == original.research_object
        assert restored.decision_model == original.decision_model
        assert restored.research_gap_analysis == original.research_gap_analysis
        assert restored.executive_confidence == original.executive_confidence
        assert restored.iteration_plan == original.iteration_plan
        assert restored.updated_at == original.updated_at

    def test_from_dict_handles_missing_keys(self):
        from functional_agents.session import ResearchState
        state = ResearchState.from_dict({})
        assert state.engagement == {}
        assert state.research_object == {}
        assert state.updated_at == ""


# ---------------------------------------------------------------------------
# Section 3: IterationRecord
# ---------------------------------------------------------------------------

class TestIterationRecord:
    def test_creation(self):
        from functional_agents.session import IterationRecord
        record = IterationRecord(
            iteration_number=0,
            timestamp="2026-07-09T00:00:00+00:00",
            trigger="initial",
            summary="Pipeline started",
            completed_tasks=[],
            notes="",
        )
        assert record.iteration_number == 0
        assert record.trigger == "initial"
        assert record.completed_tasks == []

    def test_to_dict_keys(self):
        from functional_agents.session import IterationRecord
        record = IterationRecord(
            iteration_number=1,
            timestamp="ts",
            trigger="continuation",
            summary="Second run",
            completed_tasks=["IRT-001", "IRT-002"],
            notes="note",
        )
        d = record.to_dict()
        assert set(d.keys()) == {
            "iteration_number", "timestamp", "trigger", "summary",
            "completed_tasks", "notes",
        }
        assert d["completed_tasks"] == ["IRT-001", "IRT-002"]

    def test_to_dict_from_dict_roundtrip(self):
        from functional_agents.session import IterationRecord
        original = IterationRecord(
            iteration_number=2,
            timestamp="2026-07-09T12:00:00+00:00",
            trigger="replan",
            summary="Re-planning after QA request",
            completed_tasks=["IRT-003"],
            notes="QA flagged gap",
        )
        restored = IterationRecord.from_dict(original.to_dict())
        assert restored.iteration_number == original.iteration_number
        assert restored.timestamp == original.timestamp
        assert restored.trigger == original.trigger
        assert restored.summary == original.summary
        assert restored.completed_tasks == original.completed_tasks
        assert restored.notes == original.notes

    def test_from_dict_empty_dict(self):
        from functional_agents.session import IterationRecord
        record = IterationRecord.from_dict({})
        assert record.iteration_number == 0
        assert record.trigger == "initial"
        assert record.completed_tasks == []

    def test_to_dict_makes_list_copy(self):
        from functional_agents.session import IterationRecord
        original = IterationRecord(0, "ts", "initial", "s", ["IRT-001"], "")
        d = original.to_dict()
        d["completed_tasks"].append("IRT-999")
        assert "IRT-999" not in original.completed_tasks


# ---------------------------------------------------------------------------
# Section 4: Snapshot
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_from_state_generates_snapshot_id(self):
        from functional_agents.session import Snapshot
        state = _make_research_state()
        snap = Snapshot.from_state(state, iteration_number=0)
        assert snap.snapshot_id.startswith("SNAP-000-")
        assert len(snap.snapshot_id) > 9
        assert snap.iteration_number == 0
        assert snap.created_at != ""

    def test_from_state_snapshot_id_uses_iteration_number(self):
        from functional_agents.session import Snapshot
        state = _make_research_state()
        snap = Snapshot.from_state(state, iteration_number=3)
        assert snap.snapshot_id.startswith("SNAP-003-")

    def test_from_state_serializes_research_state(self):
        from functional_agents.session import Snapshot
        state = _make_research_state()
        snap = Snapshot.from_state(state, iteration_number=0)
        assert snap.state["research_object"] == {"research_id": "R-001"}
        assert snap.state["executive_confidence"] == {"confidence": "High"}

    def test_to_research_state_roundtrip(self):
        from functional_agents.session import Snapshot
        state = _make_research_state()
        snap = Snapshot.from_state(state, iteration_number=0)
        restored = snap.to_research_state()
        assert restored.research_object == state.research_object
        assert restored.engagement == state.engagement
        assert restored.iteration_plan == state.iteration_plan

    def test_to_dict_from_dict_roundtrip(self):
        from functional_agents.session import Snapshot
        state = _make_research_state()
        original = Snapshot.from_state(state, iteration_number=2)
        d = original.to_dict()
        restored = Snapshot.from_dict(d)
        assert restored.snapshot_id == original.snapshot_id
        assert restored.created_at == original.created_at
        assert restored.iteration_number == original.iteration_number
        assert restored.state == original.state

    def test_from_dict_empty(self):
        from functional_agents.session import Snapshot
        snap = Snapshot.from_dict({})
        assert snap.snapshot_id == ""
        assert snap.iteration_number == 0
        assert snap.state == {}

    def test_to_dict_keys(self):
        from functional_agents.session import Snapshot
        snap = Snapshot.from_state(_make_research_state(), iteration_number=0)
        assert set(snap.to_dict().keys()) == {
            "snapshot_id", "created_at", "iteration_number", "state"
        }


# ---------------------------------------------------------------------------
# Section 5: ResearchSession
# ---------------------------------------------------------------------------

class TestResearchSessionCreation:
    def test_create_returns_active_session(self):
        from functional_agents.session import ResearchSession, SessionStatus
        session = _make_session()
        assert session.status == SessionStatus.ACTIVE
        assert session.session_id.startswith("SS-")
        assert session.created_at != ""
        assert session.updated_at != ""

    def test_create_session_id_format(self):
        from functional_agents.session import ResearchSession
        import re
        session = _make_session()
        # SS-YYYYMMDD-HHMMSS-hex6
        assert re.match(r"SS-\d{8}-\d{6}-[0-9a-f]{6}", session.session_id)

    def test_create_metadata_preserved(self):
        from functional_agents.session import ResearchSession
        state = _make_research_state()
        metadata = {"run_id": "r001", "profiles": ["ai_dc"], "run_mode": "research"}
        session = ResearchSession.create(metadata=metadata, research_state=state)
        assert session.metadata["run_id"] == "r001"
        assert session.metadata["profiles"] == ["ai_dc"]
        assert session.metadata["run_mode"] == "research"

    def test_create_iteration_history_empty(self):
        session = _make_session()
        assert session.iteration_history == []

    def test_create_snapshots_empty(self):
        session = _make_session()
        assert session.snapshots == []

    def test_add_iteration_appends(self):
        from functional_agents.session import IterationRecord
        session = _make_session()
        record = IterationRecord(0, "ts", "initial", "started")
        session.add_iteration(record)
        assert len(session.iteration_history) == 1
        assert session.iteration_history[0].trigger == "initial"

    def test_add_iteration_updates_updated_at(self):
        from functional_agents.session import IterationRecord
        session = _make_session()
        before = session.updated_at
        record = IterationRecord(0, "ts", "initial", "s")
        session.add_iteration(record)
        assert session.updated_at >= before

    def test_take_snapshot_appends(self):
        session = _make_session()
        snap = session.take_snapshot()
        assert len(session.snapshots) == 1
        assert snap.snapshot_id.startswith("SNAP-")

    def test_take_snapshot_uses_iteration_count(self):
        from functional_agents.session import IterationRecord
        session = _make_session()
        session.add_iteration(IterationRecord(0, "ts", "initial", "s"))
        session.add_iteration(IterationRecord(1, "ts", "continuation", "s"))
        snap = session.take_snapshot()
        assert snap.iteration_number == 2

    def test_complete_transitions_status(self):
        from functional_agents.session import SessionStatus
        session = _make_session()
        session.complete()
        assert session.status == SessionStatus.COMPLETED

    def test_archive_transitions_status(self):
        from functional_agents.session import SessionStatus
        session = _make_session()
        session.archive()
        assert session.status == SessionStatus.ARCHIVED

    def test_complete_then_archive(self):
        from functional_agents.session import SessionStatus
        session = _make_session()
        session.complete()
        session.archive()
        assert session.status == SessionStatus.ARCHIVED


# ---------------------------------------------------------------------------
# Section 6: ResearchSession serialization
# ---------------------------------------------------------------------------

class TestResearchSessionSerialization:
    def test_to_dict_keys(self):
        session = _make_session()
        d = session.to_dict()
        assert set(d.keys()) == {
            "session_id", "created_at", "updated_at", "status",
            "metadata", "research_state", "iteration_history", "snapshots",
        }

    def test_to_dict_from_dict_empty_session(self):
        from functional_agents.session import ResearchSession, SessionStatus
        original = _make_session()
        restored = ResearchSession.from_dict(original.to_dict())
        assert restored.session_id == original.session_id
        assert restored.created_at == original.created_at
        assert restored.status == original.status
        assert restored.metadata == original.metadata
        assert restored.iteration_history == []
        assert restored.snapshots == []

    def test_to_dict_from_dict_with_history_and_snapshots(self):
        from functional_agents.session import ResearchSession, IterationRecord
        original = _make_session()
        original.add_iteration(IterationRecord(0, "ts1", "initial", "started", [], ""))
        original.add_iteration(IterationRecord(1, "ts2", "continuation", "continued", ["IRT-001"], "note"))
        original.take_snapshot()
        original.complete()
        restored = ResearchSession.from_dict(original.to_dict())
        assert len(restored.iteration_history) == 2
        assert restored.iteration_history[1].completed_tasks == ["IRT-001"]
        assert len(restored.snapshots) == 1
        assert restored.status == "completed"

    def test_from_dict_empty_dict(self):
        from functional_agents.session import ResearchSession, SessionStatus
        session = ResearchSession.from_dict({})
        assert session.session_id == ""
        assert session.status == SessionStatus.ACTIVE
        assert session.iteration_history == []
        assert session.snapshots == []

    def test_research_state_nested_in_to_dict(self):
        session = _make_session()
        d = session.to_dict()
        assert isinstance(d["research_state"], dict)
        assert "research_object" in d["research_state"]

    def test_json_roundtrip(self):
        from functional_agents.session import ResearchSession, IterationRecord
        original = _make_session()
        original.add_iteration(IterationRecord(0, "ts", "initial", "started"))
        original.take_snapshot()
        serialized = json.dumps(original.to_dict(), indent=2)
        restored = ResearchSession.from_dict(json.loads(serialized))
        assert restored.session_id == original.session_id
        assert len(restored.iteration_history) == 1
        assert len(restored.snapshots) == 1


# ---------------------------------------------------------------------------
# Section 7: SessionStore — create / save / load
# ---------------------------------------------------------------------------

class TestSessionStoreSaveLoad:
    def test_create_writes_file(self, tmp_path):
        from functional_agents.session import SessionStore
        store = SessionStore(base_dir=tmp_path)
        session = _make_session()
        path = store.create(session)
        assert path.exists()
        assert path.name == f"{session.session_id}.json"

    def test_create_file_is_valid_json(self, tmp_path):
        from functional_agents.session import SessionStore
        store = SessionStore(base_dir=tmp_path)
        session = _make_session()
        path = store.create(session)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["session_id"] == session.session_id

    def test_load_returns_equivalent_session(self, tmp_path):
        from functional_agents.session import SessionStore
        store = SessionStore(base_dir=tmp_path)
        original = _make_session()
        store.create(original)
        loaded = store.load(original.session_id)
        assert loaded.session_id == original.session_id
        assert loaded.status == original.status
        assert loaded.metadata == original.metadata

    def test_load_restores_research_state(self, tmp_path):
        from functional_agents.session import SessionStore
        store = SessionStore(base_dir=tmp_path)
        original = _make_session()
        store.create(original)
        loaded = store.load(original.session_id)
        assert loaded.research_state.research_object == {"research_id": "R-001"}
        assert loaded.research_state.executive_confidence == {"confidence": "High"}

    def test_save_overwrites_existing_file(self, tmp_path):
        from functional_agents.session import SessionStore, SessionStatus
        store = SessionStore(base_dir=tmp_path)
        session = _make_session()
        store.create(session)
        session.complete()
        store.save(session)
        loaded = store.load(session.session_id)
        assert loaded.status == SessionStatus.COMPLETED

    def test_load_raises_session_not_found(self, tmp_path):
        from functional_agents.session import SessionStore, SessionNotFoundError
        store = SessionStore(base_dir=tmp_path)
        with pytest.raises(SessionNotFoundError):
            store.load("SS-00000000-000000-000000")

    def test_exists_true_after_create(self, tmp_path):
        from functional_agents.session import SessionStore
        store = SessionStore(base_dir=tmp_path)
        session = _make_session()
        store.create(session)
        assert store.exists(session.session_id) is True

    def test_exists_false_before_create(self, tmp_path):
        from functional_agents.session import SessionStore
        store = SessionStore(base_dir=tmp_path)
        assert store.exists("SS-00000000-000000-000000") is False

    def test_creates_dir_automatically(self, tmp_path):
        from functional_agents.session import SessionStore
        nested = tmp_path / "deep" / "sessions"
        store = SessionStore(base_dir=nested)
        session = _make_session()
        store.create(session)
        assert nested.exists()


# ---------------------------------------------------------------------------
# Section 8: SessionStore — archive
# ---------------------------------------------------------------------------

class TestSessionStoreArchive:
    def test_archive_transitions_status(self, tmp_path):
        from functional_agents.session import SessionStore, SessionStatus
        store = SessionStore(base_dir=tmp_path)
        session = _make_session()
        store.create(session)
        store.archive(session.session_id)
        loaded = store.load(session.session_id)
        assert loaded.status == SessionStatus.ARCHIVED

    def test_archive_persists_to_disk(self, tmp_path):
        from functional_agents.session import SessionStore
        store = SessionStore(base_dir=tmp_path)
        session = _make_session()
        store.create(session)
        store.archive(session.session_id)
        data = json.loads((tmp_path / f"{session.session_id}.json").read_text())
        assert data["status"] == "archived"

    def test_archive_missing_session_raises(self, tmp_path):
        from functional_agents.session import SessionStore, SessionNotFoundError
        store = SessionStore(base_dir=tmp_path)
        with pytest.raises(SessionNotFoundError):
            store.archive("SS-00000000-000000-000000")


# ---------------------------------------------------------------------------
# Section 9: SessionStore — continue_session
# ---------------------------------------------------------------------------

class TestSessionStoreContinue:
    def test_continue_reactivates_completed_session(self, tmp_path):
        from functional_agents.session import SessionStore, SessionStatus
        store = SessionStore(base_dir=tmp_path)
        session = _make_session()
        session.complete()
        store.create(session)
        continued = store.continue_session(session.session_id)
        assert continued.status == SessionStatus.ACTIVE

    def test_continue_reactivates_archived_session(self, tmp_path):
        from functional_agents.session import SessionStore, SessionStatus
        store = SessionStore(base_dir=tmp_path)
        session = _make_session()
        session.archive()
        store.create(session)
        continued = store.continue_session(session.session_id)
        assert continued.status == SessionStatus.ACTIVE

    def test_continue_does_not_auto_save(self, tmp_path):
        from functional_agents.session import SessionStore, SessionStatus
        store = SessionStore(base_dir=tmp_path)
        session = _make_session()
        session.complete()
        store.create(session)
        store.continue_session(session.session_id)
        # Re-load: should still be completed (continue_session did not auto-save)
        reloaded = store.load(session.session_id)
        assert reloaded.status == SessionStatus.COMPLETED

    def test_continue_missing_session_raises(self, tmp_path):
        from functional_agents.session import SessionStore, SessionNotFoundError
        store = SessionStore(base_dir=tmp_path)
        with pytest.raises(SessionNotFoundError):
            store.continue_session("SS-00000000-000000-000000")


# ---------------------------------------------------------------------------
# Section 10: SessionStore — list_sessions
# ---------------------------------------------------------------------------

class TestSessionStoreList:
    def test_list_empty_when_no_sessions(self, tmp_path):
        from functional_agents.session import SessionStore
        store = SessionStore(base_dir=tmp_path)
        assert store.list_sessions() == []

    def test_list_empty_when_dir_absent(self, tmp_path):
        from functional_agents.session import SessionStore
        store = SessionStore(base_dir=tmp_path / "nonexistent")
        assert store.list_sessions() == []

    def test_list_returns_session_ids(self, tmp_path):
        from functional_agents.session import SessionStore
        store = SessionStore(base_dir=tmp_path)
        s1 = _make_session()
        s2 = _make_session()
        store.create(s1)
        store.create(s2)
        ids = store.list_sessions()
        assert s1.session_id in ids
        assert s2.session_id in ids

    def test_list_returns_sorted_ids(self, tmp_path):
        from functional_agents.session import SessionStore
        store = SessionStore(base_dir=tmp_path)
        sessions = [_make_session() for _ in range(3)]
        for s in sessions:
            store.create(s)
        ids = store.list_sessions()
        assert ids == sorted(ids)

    def test_list_ignores_non_session_files(self, tmp_path):
        from functional_agents.session import SessionStore
        store = SessionStore(base_dir=tmp_path)
        (tmp_path / "README.txt").write_text("ignore me", encoding="utf-8")
        (tmp_path / "other.json").write_text("{}", encoding="utf-8")
        s = _make_session()
        store.create(s)
        ids = store.list_sessions()
        assert ids == [s.session_id]


# ---------------------------------------------------------------------------
# Section 11: Iteration history and snapshot persistence (full roundtrip)
# ---------------------------------------------------------------------------

class TestFullRoundtrip:
    def test_iteration_history_roundtrip_through_store(self, tmp_path):
        from functional_agents.session import SessionStore, IterationRecord
        store = SessionStore(base_dir=tmp_path)
        session = _make_session()
        session.add_iteration(IterationRecord(
            iteration_number=0,
            timestamp="2026-07-09T00:00:00+00:00",
            trigger="initial",
            summary="First run",
            completed_tasks=[],
            notes="",
        ))
        session.add_iteration(IterationRecord(
            iteration_number=1,
            timestamp="2026-07-09T01:00:00+00:00",
            trigger="continuation",
            summary="Second run",
            completed_tasks=["IRT-001", "IRT-002"],
            notes="completed tasks",
        ))
        store.save(session)
        loaded = store.load(session.session_id)
        assert len(loaded.iteration_history) == 2
        assert loaded.iteration_history[0].trigger == "initial"
        assert loaded.iteration_history[1].completed_tasks == ["IRT-001", "IRT-002"]
        assert loaded.iteration_history[1].notes == "completed tasks"

    def test_snapshots_roundtrip_through_store(self, tmp_path):
        from functional_agents.session import SessionStore, IterationRecord
        store = SessionStore(base_dir=tmp_path)
        session = _make_session()
        session.add_iteration(IterationRecord(0, "ts", "initial", "s"))
        snap = session.take_snapshot()
        store.save(session)
        loaded = store.load(session.session_id)
        assert len(loaded.snapshots) == 1
        assert loaded.snapshots[0].snapshot_id == snap.snapshot_id
        assert loaded.snapshots[0].state["research_object"] == {"research_id": "R-001"}

    def test_snapshot_state_includes_all_artifacts(self, tmp_path):
        from functional_agents.session import SessionStore
        store = SessionStore(base_dir=tmp_path)
        session = _make_session()
        session.take_snapshot()
        store.save(session)
        loaded = store.load(session.session_id)
        snap_state = loaded.snapshots[0].state
        assert "engagement" in snap_state
        assert "research_object" in snap_state
        assert "decision_model" in snap_state
        assert "research_gap_analysis" in snap_state
        assert "executive_confidence" in snap_state
        assert "iteration_plan" in snap_state

    def test_snapshot_to_research_state_after_store_roundtrip(self, tmp_path):
        from functional_agents.session import SessionStore
        store = SessionStore(base_dir=tmp_path)
        session = _make_session()
        session.take_snapshot()
        store.save(session)
        loaded = store.load(session.session_id)
        restored_state = loaded.snapshots[0].to_research_state()
        assert restored_state.executive_confidence == {"confidence": "High"}
        assert restored_state.research_object == {"research_id": "R-001"}
