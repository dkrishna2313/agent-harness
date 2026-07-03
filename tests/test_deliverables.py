"""Tests for the J11.0 Strategic Deliverables Framework.

Covers: DeliverableRequest/DeliverableArtifact round-tripping, the
DeliverableGenerator contract, DeliverableRegistry dispatch, and the
MarkdownReportGenerator wrapping ReportAgent's existing markdown assembly —
including an end-to-end ReportAgent run confirming AgentContext.deliverable_request
/ deliverables are populated and existing report/artifact behaviour is unchanged.
"""

from __future__ import annotations

import json

import pytest

from functional_agents.context import AgentContext
from functional_agents.deliverables import (
    DeliverableArtifact,
    DeliverableGenerator,
    DeliverableRegistry,
    DeliverableRequest,
    MarkdownReportGenerator,
    default_registry,
)
from functional_agents.report_agent import build_markdown_report_content
from functional_agents.run_agent import run_agent
from research_agent.schemas import ResearchMemo

_FIXTURES = "fixtures"


def _memo() -> ResearchMemo:
    return ResearchMemo(title="Report: Q?", question="Q?", executive_summary="Summary.")


# ---------------------------------------------------------------------------
# DeliverableRequest / DeliverableArtifact — canonical object round-tripping
# ---------------------------------------------------------------------------

def test_deliverable_request_default_type_is_markdown():
    req = DeliverableRequest()
    assert req.type == "markdown"
    assert req.id == ""
    assert req.audience == ""
    assert req.title == ""
    assert req.format == "markdown"
    assert req.options == {}


def test_deliverable_request_round_trips_through_dict():
    req = DeliverableRequest(
        id="d1", type="markdown", audience="board", title="Q3 Update",
        options={"foo": "bar"},
    )
    restored = DeliverableRequest.from_dict(req.to_dict())
    assert restored == req


def test_deliverable_request_from_dict_handles_none_and_empty():
    assert DeliverableRequest.from_dict(None) == DeliverableRequest()
    assert DeliverableRequest.from_dict({}) == DeliverableRequest()


def test_deliverable_artifact_to_dict_matches_trace_schema():
    artifact = DeliverableArtifact(type="markdown", metadata={"status": "generated"})
    assert artifact.to_dict() == {"type": "markdown", "status": "generated"}


def test_deliverable_artifact_to_dict_defaults_status_when_metadata_omitted():
    artifact = DeliverableArtifact(type="markdown")
    assert artifact.to_dict() == {"type": "markdown", "status": "generated"}


def test_deliverable_artifact_to_dict_includes_optional_fields_when_present():
    artifact = DeliverableArtifact(
        type="markdown", id="a1", path="outputs/x.md",
        mime_type="text/markdown", metadata={"status": "generated", "bytes": 42},
    )
    d = artifact.to_dict()
    assert d["id"] == "a1"
    assert d["path"] == "outputs/x.md"
    assert d["mime_type"] == "text/markdown"
    assert d["status"] == "generated"
    # "status" is echoed as a top-level key; it is not duplicated inside metadata.
    assert d["metadata"] == {"bytes": 42}


# ---------------------------------------------------------------------------
# DeliverableGenerator contract
# ---------------------------------------------------------------------------

def test_deliverable_generator_is_abstract():
    with pytest.raises(TypeError):
        DeliverableGenerator()  # type: ignore[abstract]


def test_markdown_report_generator_declares_its_type():
    assert MarkdownReportGenerator.deliverable_type == "markdown"
    assert isinstance(MarkdownReportGenerator(), DeliverableGenerator)


def test_deliverable_generator_signature_is_context_and_output_path():
    import inspect
    sig = inspect.signature(DeliverableGenerator.generate)
    assert list(sig.parameters) == ["self", "context", "output_path"]


# ---------------------------------------------------------------------------
# DeliverableRegistry
# ---------------------------------------------------------------------------

def test_registry_register_and_get():
    reg = DeliverableRegistry()
    gen = MarkdownReportGenerator()
    reg.register(gen)
    assert reg.get("markdown") is gen


def test_registry_get_unknown_type_raises_keyerror():
    reg = DeliverableRegistry()
    with pytest.raises(KeyError, match="No deliverable generator registered"):
        reg.get("powerpoint")


def test_registry_register_requires_deliverable_type():
    class _Bad(DeliverableGenerator):
        deliverable_type = ""

        def generate(self, context, output_path):
            raise NotImplementedError

    with pytest.raises(ValueError):
        DeliverableRegistry().register(_Bad())


def test_default_registry_has_markdown_generator_registered():
    gen = default_registry.get("markdown")
    assert isinstance(gen, MarkdownReportGenerator)


def test_registry_generate_dispatches_on_request_type(tmp_path):
    ctx = AgentContext(
        question="Q?", profiles=["ai_data_centers"], execution_profile="ai_data_centers",
        research_object={"research_id": "R"},
    )
    ctx.trace["_report_memo"] = _memo()
    request = DeliverableRequest(type="markdown")

    artifact = default_registry.generate(ctx, request, tmp_path / "report.md")

    assert artifact.type == "markdown"
    assert (tmp_path / "report.md").exists()


# ---------------------------------------------------------------------------
# MarkdownReportGenerator — wraps build_markdown_report_content + write_markdown
# ---------------------------------------------------------------------------

def test_markdown_report_generator_produces_generated_artifact(tmp_path):
    ctx = AgentContext(
        question="Q?", profiles=["ai_data_centers"], execution_profile="ai_data_centers",
        research_object={"research_id": "R"},
    )
    ctx.trace["_report_memo"] = _memo()

    artifact = MarkdownReportGenerator().generate(ctx, tmp_path / "report.md")

    assert artifact.type == "markdown"
    assert artifact.mime_type == "text/markdown"
    assert artifact.to_dict()["status"] == "generated"
    assert artifact.path == str(tmp_path / "report.md")
    assert (tmp_path / "report.md").exists()


def test_markdown_report_generator_content_matches_build_markdown_report_content(tmp_path):
    ctx = AgentContext(
        question="Q?", profiles=["ai_data_centers"], execution_profile="ai_data_centers",
        research_object={"research_id": "R"}, hypotheses=[{"id": "H1", "title": "T"}],
    )
    memo = _memo()
    ctx.trace["_report_memo"] = memo
    expected = build_markdown_report_content(ctx, memo)

    artifact = MarkdownReportGenerator().generate(ctx, tmp_path / "report.md")

    assert (tmp_path / "report.md").read_text(encoding="utf-8") == expected


def test_markdown_report_generator_never_touches_functional_agents():
    """Generators must never invoke Functional Agents (J11.0 constraint).

    MarkdownReportGenerator imports report_agent.build_markdown_report_content
    (a pure function) but must never import the ReportAgent class itself, nor
    the FunctionalAgent base class any Functional Agent derives from.
    """
    import ast
    import inspect

    from functional_agents.deliverables import markdown_report

    tree = ast.parse(inspect.getsource(markdown_report))
    imported_symbols = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "ReportAgent" not in imported_symbols
    assert "FunctionalAgent" not in imported_symbols


# ---------------------------------------------------------------------------
# End-to-end: ReportAgent delegates to the registry (byte-identical behaviour)
# ---------------------------------------------------------------------------

def test_report_agent_populates_deliverable_context_fields(tmp_path):
    out = tmp_path / "report.md"
    res = run_agent("report", f"{_FIXTURES}/report_start.json", no_llm=True, out_path=str(out))

    ctx = res["context"]
    assert len(ctx["deliverables"]) == 1
    assert ctx["deliverables"][0]["type"] == "markdown"
    assert ctx["deliverables"][0]["status"] == "generated"
    assert ctx["deliverable_request"]["type"] == "markdown"

    # Existing artifact contract is unchanged.
    assert ctx["artifacts"]["report_path"] == str(out)
    assert ctx["artifacts"]["trace_path"] == str(out.with_suffix(".trace.json"))
    assert out.exists()


def test_report_agent_context_is_still_json_serializable(tmp_path):
    out = tmp_path / "report.md"
    res = run_agent("report", f"{_FIXTURES}/report_start.json", no_llm=True, out_path=str(out))
    # deliverable_request/deliverables must round-trip through JSON like every
    # other AgentContext field (run_agent.py's context_to_jsonable contract).
    json.dumps(res["context"])
