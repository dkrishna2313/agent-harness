from research_agent.loaders import load_sources


def test_load_sources_reads_supported_text_files(tmp_path):
    (tmp_path / "a.md").write_text("# Rubin\nPower and cooling notes.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Vera Rubin rack architecture.", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("ignore me", encoding="utf-8")
    (tmp_path / ".hidden.txt").write_text("ignore hidden", encoding="utf-8")

    collection = load_sources(tmp_path)

    assert collection.errors == []
    assert [document.title for document in collection.documents] == ["a", "b"]
    assert {document.extension for document in collection.documents} == {".md", ".txt"}


def test_load_sources_reports_missing_directory(tmp_path):
    missing = tmp_path / "missing"

    collection = load_sources(missing)

    assert collection.documents == []
    assert collection.errors
    assert collection.errors[0].exception_type == "FileNotFoundError"
