import json

from typer.testing import CliRunner

from dc_power_agent.claude_client import ClaudeClient, MockClaudeClient
from dc_power_agent.cli import _build_client, app


def test_cli_writes_markdown(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "rubin.md").write_text(
        "NVIDIA Rubin sources mention rack-scale power and cooling.",
        encoding="utf-8",
    )
    out = tmp_path / "outputs" / "rubin.md"

    result = CliRunner().invoke(
        app,
        [
            "Explain NVIDIA Rubin rack architecture",
            "--sources",
            str(sources),
            "--out",
            str(out),
            "--mock",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Loaded 1 source file(s):" not in result.output
    assert out.exists()
    assert out.with_suffix(".trace.json").exists()
    content = out.read_text(encoding="utf-8")
    assert "## Executive Summary" in content
    assert "## Evaluation Warnings" in content


def test_cli_can_show_loaded_sources(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "rubin.md").write_text(
        "NVIDIA Rubin sources mention rack-scale power and cooling.",
        encoding="utf-8",
    )
    out = tmp_path / "outputs" / "rubin.md"

    result = CliRunner().invoke(
        app,
        [
            "Explain NVIDIA Rubin rack architecture",
            "--sources",
            str(sources),
            "--out",
            str(out),
            "--show-sources",
            "--mock",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Loaded 1 source file(s):" in result.output
    assert "- rubin.md (58 characters)" in result.output


def test_cli_writes_trace_json(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "rubin.md").write_text(
        "NVIDIA Rubin architecture depends on rack power, cooling, and networking.",
        encoding="utf-8",
    )
    out = tmp_path / "outputs" / "rubin.md"

    result = CliRunner().invoke(
        app,
        [
            "Explain NVIDIA Rubin rack architecture",
            "--sources",
            str(sources),
            "--out",
            str(out),
            "--mock",
        ],
    )

    assert result.exit_code == 0, result.output
    trace_path = out.with_suffix(".trace.json")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["question"] == "Explain NVIDIA Rubin rack architecture"
    assert "rack architecture" in trace["question_topics_detected"]
    assert trace["documents_loaded"] == 1
    assert trace["documents"][0]["filename"] == "rubin.md"
    assert trace["documents"][0]["evidence_item_count"] >= 3
    assert trace["evidence_items"]
    assert trace["evidence_items"][0]["evidence_id"] == "E001"
    assert "overall_score" in trace["evidence_items"][0]
    assert trace["evidence_ranking"]
    assert trace["evidence_ranking"][0]["evidence_id"]
    assert "overall_score" in trace["evidence_ranking"][0]
    assert trace["evaluation_warnings"]
    assert trace["mock_mode"] is True


def test_cli_debug_prints_run_summary(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "rubin.md").write_text(
        "NVIDIA Rubin architecture depends on rack power, cooling, and networking.",
        encoding="utf-8",
    )
    out = tmp_path / "outputs" / "rubin.md"

    result = CliRunner().invoke(
        app,
        [
            "Explain NVIDIA Rubin architecture",
            "--sources",
            str(sources),
            "--out",
            str(out),
            "--debug",
            "--mock",
            "--top-evidence",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    for expected in [
        "Debug summary:",
        "Question: Explain NVIDIA Rubin architecture",
        "Question topics detected:",
        "Source directory:",
        "Output path:",
        "Documents loaded: 1",
        "Evidence items per document:",
        "Total evidence items:",
        "Evidence items used for synthesis: 2",
        "Top evidence items:",
        "overall",
        "Memo sections generated:",
        "Evaluation warning count:",
        "Trace file path:",
    ]:
        assert expected in result.output


def test_api_key_present_without_mock_uses_real_claude_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    client, warnings = _build_client(mock=False, live_llm=False, model=None)

    assert isinstance(client, ClaudeClient)
    assert client.is_mock is False
    assert warnings == []


def test_api_key_present_with_mock_uses_mock_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    client, warnings = _build_client(mock=True, live_llm=False, model=None)

    assert isinstance(client, MockClaudeClient)
    assert client.is_mock is True
    assert warnings == []


def test_missing_api_key_without_mock_falls_back_with_warning(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    client, warnings = _build_client(mock=False, live_llm=False, model=None)

    assert isinstance(client, MockClaudeClient)
    assert client.is_mock is True
    assert warnings == [
        "Claude warning: ANTHROPIC_API_KEY is missing; using deterministic mock client."
    ]
