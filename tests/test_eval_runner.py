from typer.testing import CliRunner

from dc_power_agent.eval_runner import app, load_eval_questions


def test_load_eval_questions_reads_simple_yaml(tmp_path):
    evals = tmp_path / "questions.yaml"
    evals.write_text(
        'questions:\n  - "How does power affect Rubin racks?"\n  - "What cooling changes matter?"\n',
        encoding="utf-8",
    )

    questions = load_eval_questions(evals)

    assert questions == [
        "How does power affect Rubin racks?",
        "What cooling changes matter?",
    ]


def test_eval_runner_writes_markdown_report(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "rubin.md").write_text(
        "Rubin rack architecture affects power, cooling, and networking.",
        encoding="utf-8",
    )
    evals = tmp_path / "questions.yaml"
    evals.write_text(
        'questions:\n  - "How does rack power affect deployment?"\n',
        encoding="utf-8",
    )
    out = tmp_path / "outputs" / "eval_report.md"

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--sources",
            str(sources),
            "--evals",
            str(evals),
            "--out",
            str(out),
            "--mock",
        ],
    )

    assert result.exit_code == 0, result.output
    memo_path = tmp_path / "outputs" / "evals" / "eval_001.md"
    trace_path = tmp_path / "outputs" / "evals" / "eval_001.trace.json"
    assert memo_path.exists()
    assert trace_path.exists()
    report = out.read_text(encoding="utf-8")
    assert "# Evaluation Report" in report
    assert "How does rack power affect deployment?" in report
    assert "Detected topics:" in report
    assert "Evidence count:" in report
    assert "Citation count:" in report
    assert f"Memo: {memo_path}" in report
    assert f"Trace: {trace_path}" in report
    assert "Warning count:" in report
