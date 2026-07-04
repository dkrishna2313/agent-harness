"""Tests for the J11.2 DeliverableBundle and DeliverableBundleGenerator.

Covers: DeliverableBundle construction and serialisation, bundle generation
from a single AgentContext, multi-type generation, registry integration,
artifact id assignment, no-duplicate-reasoning invariant, and trace capture.
"""

from __future__ import annotations

import json

import pytest

from functional_agents.context import AgentContext
from functional_agents.deliverables import (
    DeliverableBundle,
    DeliverableBundleGenerator,
    DeliverableRegistry,
    ExecutiveBriefGenerator,
    MarkdownReportGenerator,
    default_registry,
)
from functional_agents.deliverables.artifact import DeliverableArtifact
from functional_agents.pipeline_trace import build_canonical_trace
from research_agent.schemas import ResearchMemo


def _ctx() -> AgentContext:
    ctx = AgentContext(
        question="Q?",
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={"research_id": "R-TEST-001"},
    )
    ctx.trace["_report_memo"] = ResearchMemo(
        title="Report: Q?", question="Q?", executive_summary="Summary."
    )
    return ctx


# ---------------------------------------------------------------------------
# DeliverableBundle — construction and serialisation
# ---------------------------------------------------------------------------

def test_deliverable_bundle_defaults():
    bundle = DeliverableBundle()
    assert bundle.bundle_id == ""
    assert bundle.engagement_id == ""
    assert bundle.reasoning_graph_id == ""
    assert bundle.created_at == ""
    assert bundle.deliverables == []
    assert bundle.metadata == {}


def test_deliverable_bundle_to_dict_includes_generated_key_always():
    bundle = DeliverableBundle(bundle_id="b1")
    d = bundle.to_dict()
    assert "generated" in d
    assert d["generated"] == []


def test_deliverable_bundle_to_dict_omits_empty_optional_fields():
    bundle = DeliverableBundle(bundle_id="b1")
    d = bundle.to_dict()
    assert "engagement_id" not in d
    assert "reasoning_graph_id" not in d
    assert "created_at" not in d
    assert "metadata" not in d


def test_deliverable_bundle_to_dict_includes_populated_optional_fields():
    bundle = DeliverableBundle(
        bundle_id="b1",
        engagement_id="eng-1",
        reasoning_graph_id="R-001",
        created_at="2026-07-03T00:00:00Z",
        metadata={"source": "test"},
    )
    d = bundle.to_dict()
    assert d["bundle_id"] == "b1"
    assert d["engagement_id"] == "eng-1"
    assert d["reasoning_graph_id"] == "R-001"
    assert d["created_at"] == "2026-07-03T00:00:00Z"
    assert d["metadata"] == {"source": "test"}


def test_deliverable_bundle_to_dict_serialises_artifacts():
    artifact = DeliverableArtifact(type="markdown", path="out/markdown.md",
                                   mime_type="text/markdown",
                                   metadata={"status": "generated"})
    bundle = DeliverableBundle(bundle_id="b1", deliverables=[artifact])
    d = bundle.to_dict()
    assert len(d["generated"]) == 1
    assert d["generated"][0]["type"] == "markdown"
    assert d["generated"][0]["status"] == "generated"


def test_deliverable_bundle_to_dict_is_json_serialisable():
    artifact = DeliverableArtifact(type="markdown", path="out/x.md",
                                   mime_type="text/markdown",
                                   metadata={"status": "generated"})
    bundle = DeliverableBundle(bundle_id="b1", deliverables=[artifact])
    json.dumps(bundle.to_dict())  # must not raise


# ---------------------------------------------------------------------------
# DeliverableBundleGenerator — single type
# ---------------------------------------------------------------------------

def test_bundle_generator_creates_bundle_for_single_type(tmp_path):
    ctx = _ctx()
    gen = DeliverableBundleGenerator()
    bundle = gen.generate(ctx, ["markdown"], tmp_path)

    assert isinstance(bundle, DeliverableBundle)
    assert len(bundle.deliverables) == 1
    assert bundle.deliverables[0].type == "markdown"
    assert (tmp_path / "markdown.md").exists()


def test_bundle_generator_creates_bundle_for_two_types(tmp_path):
    ctx = _ctx()
    gen = DeliverableBundleGenerator()
    bundle = gen.generate(ctx, ["markdown", "executive_brief"], tmp_path)

    assert len(bundle.deliverables) == 2
    types = [a.type for a in bundle.deliverables]
    assert "markdown" in types
    assert "executive_brief" in types
    assert (tmp_path / "markdown.md").exists()
    assert (tmp_path / "executive-brief.md").exists()


def test_bundle_generator_assigns_artifact_ids(tmp_path):
    ctx = _ctx()
    gen = DeliverableBundleGenerator()
    bundle = gen.generate(ctx, ["markdown", "executive_brief"], tmp_path,
                          bundle_id="abcdef12-test")

    for artifact in bundle.deliverables:
        assert artifact.id, "artifact.id must be non-empty"
        assert artifact.id.startswith("abcdef12")


def test_bundle_generator_artifact_paths_set(tmp_path):
    ctx = _ctx()
    gen = DeliverableBundleGenerator()
    bundle = gen.generate(ctx, ["markdown", "executive_brief"], tmp_path)

    for artifact in bundle.deliverables:
        assert artifact.path is not None
        assert artifact.mime_type == "text/markdown"


def test_bundle_generator_assigns_bundle_id_when_not_provided(tmp_path):
    ctx = _ctx()
    bundle = DeliverableBundleGenerator().generate(ctx, ["markdown"], tmp_path)
    assert bundle.bundle_id  # non-empty UUID was assigned


def test_bundle_generator_uses_provided_bundle_id(tmp_path):
    ctx = _ctx()
    bundle = DeliverableBundleGenerator().generate(
        ctx, ["markdown"], tmp_path, bundle_id="explicit-id-123"
    )
    assert bundle.bundle_id == "explicit-id-123"


def test_bundle_generator_sets_reasoning_graph_id_from_context(tmp_path):
    ctx = _ctx()
    bundle = DeliverableBundleGenerator().generate(ctx, ["markdown"], tmp_path)
    assert bundle.reasoning_graph_id == "R-TEST-001"


def test_bundle_generator_accepts_engagement_id(tmp_path):
    ctx = _ctx()
    bundle = DeliverableBundleGenerator().generate(
        ctx, ["markdown"], tmp_path, engagement_id="eng-hyperscaler"
    )
    assert bundle.engagement_id == "eng-hyperscaler"


def test_bundle_generator_accepts_created_at(tmp_path):
    ctx = _ctx()
    bundle = DeliverableBundleGenerator().generate(
        ctx, ["markdown"], tmp_path, created_at="2026-07-03T00:00:00Z"
    )
    assert bundle.created_at == "2026-07-03T00:00:00Z"


# ---------------------------------------------------------------------------
# No duplicate reasoning — core invariant
# ---------------------------------------------------------------------------

def test_bundle_generator_does_not_invoke_any_functional_agent(tmp_path):
    ctx = _ctx()
    ctx.agent_history.append({"agent": "FakeAgent", "status": "success"})
    history_before = len(ctx.agent_history)

    DeliverableBundleGenerator().generate(ctx, ["markdown", "executive_brief"], tmp_path)

    assert len(ctx.agent_history) == history_before, (
        "DeliverableBundleGenerator must not add to agent_history"
    )


def test_bundle_generator_does_not_mutate_reasoning_fields(tmp_path):
    ctx = _ctx()
    ctx.risks = [{"risk_id": "R1", "statement": "Risk one", "severity": "high"}]
    ctx.assumptions = [{"assumption_id": "A1", "statement": "Assumption one", "importance": "critical"}]

    risks_before = list(ctx.risks)
    assumptions_before = list(ctx.assumptions)

    DeliverableBundleGenerator().generate(ctx, ["markdown", "executive_brief"], tmp_path)

    assert ctx.risks == risks_before
    assert ctx.assumptions == assumptions_before


# ---------------------------------------------------------------------------
# Context mutation — presentation state only
# ---------------------------------------------------------------------------

def test_bundle_generator_sets_context_deliverable_bundle(tmp_path):
    ctx = _ctx()
    bundle = DeliverableBundleGenerator().generate(
        ctx, ["markdown", "executive_brief"], tmp_path
    )
    assert ctx.deliverable_bundle
    assert ctx.deliverable_bundle["bundle_id"] == bundle.bundle_id
    assert len(ctx.deliverable_bundle["generated"]) == 2


def test_bundle_generator_context_remains_json_serialisable(tmp_path):
    ctx = _ctx()
    DeliverableBundleGenerator().generate(ctx, ["markdown", "executive_brief"], tmp_path)
    import dataclasses
    json.dumps(dataclasses.asdict(ctx), default=str)


# ---------------------------------------------------------------------------
# Trace capture
# ---------------------------------------------------------------------------

def test_canonical_trace_includes_deliverable_bundle_key_after_generation(tmp_path):
    ctx = _ctx()
    DeliverableBundleGenerator().generate(ctx, ["markdown", "executive_brief"], tmp_path)

    trace = build_canonical_trace(ctx)
    assert "deliverable_bundle" in trace
    assert trace["deliverable_bundle"] is not None
    assert "bundle_id" in trace["deliverable_bundle"]
    assert len(trace["deliverable_bundle"]["generated"]) == 2


def test_canonical_trace_deliverable_bundle_is_none_when_no_bundle_generated():
    ctx = _ctx()
    trace = build_canonical_trace(ctx)
    assert "deliverable_bundle" in trace
    assert trace["deliverable_bundle"] is None


def test_canonical_trace_deliverables_list_unchanged_by_bundle(tmp_path):
    ctx = _ctx()
    # Simulate ReportAgent having already produced the markdown deliverable.
    ctx.deliverables.append({"type": "markdown", "status": "generated"})

    DeliverableBundleGenerator().generate(ctx, ["executive_brief"], tmp_path)

    trace = build_canonical_trace(ctx)
    assert trace["deliverables"] == [{"type": "markdown", "status": "generated"}]


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_bundle_generator_uses_default_registry_by_default(tmp_path):
    ctx = _ctx()
    gen = DeliverableBundleGenerator()
    bundle = gen.generate(ctx, ["markdown", "executive_brief"], tmp_path)
    assert len(bundle.deliverables) == 2


def test_bundle_generator_accepts_custom_registry(tmp_path):
    ctx = _ctx()
    reg = DeliverableRegistry()
    reg.register(MarkdownReportGenerator())
    gen = DeliverableBundleGenerator(registry=reg)
    bundle = gen.generate(ctx, ["markdown"], tmp_path)
    assert len(bundle.deliverables) == 1
    assert bundle.deliverables[0].type == "markdown"


def test_bundle_generator_raises_for_unknown_type(tmp_path):
    ctx = _ctx()
    with pytest.raises(KeyError, match="No deliverable generator registered"):
        DeliverableBundleGenerator().generate(ctx, ["powerpoint"], tmp_path)


def test_default_registry_supports_bundle_generation_for_both_registered_types(tmp_path):
    ctx = _ctx()
    bundle = DeliverableBundleGenerator(registry=default_registry).generate(
        ctx, ["markdown", "executive_brief"], tmp_path
    )
    assert len(bundle.deliverables) == 2
    types = {a.type for a in bundle.deliverables}
    assert types == {"markdown", "executive_brief"}


# ---------------------------------------------------------------------------
# to_dict round-trip in trace
# ---------------------------------------------------------------------------

def test_bundle_to_dict_in_trace_matches_bundle_object(tmp_path):
    ctx = _ctx()
    bundle = DeliverableBundleGenerator().generate(
        ctx, ["markdown", "executive_brief"], tmp_path,
        bundle_id="fixed-id-000"
    )
    trace = build_canonical_trace(ctx)
    assert trace["deliverable_bundle"]["bundle_id"] == "fixed-id-000"
    gen_types = {g["type"] for g in trace["deliverable_bundle"]["generated"]}
    assert gen_types == {"markdown", "executive_brief"}
