"""Tests for _load_build_config and the --config CLI flag on `knowledge build`."""
from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from knowledge.cli import _load_build_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def _make_source_dirs(tmp_path: Path, *names: str) -> list[Path]:
    dirs = []
    for name in names:
        d = tmp_path / name
        d.mkdir()
        dirs.append(d)
    return dirs


# ---------------------------------------------------------------------------
# Happy-path: loading valid configs
# ---------------------------------------------------------------------------

def test_valid_config_returns_sources_and_profiles(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_source_dirs(tmp_path, "sports_sources")
    cfg = _write_yaml(tmp_path, """\
        name: sports
        profiles:
          - sports
        sources:
          - sports_sources
    """)
    sources, profiles = _load_build_config(cfg)
    assert profiles == ["sports"]
    assert len(sources) == 1
    assert sources[0] == (tmp_path / "sports_sources").resolve()


def test_multiple_sources_loaded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_source_dirs(tmp_path, "dir_a", "dir_b")
    cfg = _write_yaml(tmp_path, """\
        name: multi
        profiles:
          - sports
        sources:
          - dir_a
          - dir_b
    """)
    sources, _ = _load_build_config(cfg)
    assert len(sources) == 2


def test_multiple_profiles_loaded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_source_dirs(tmp_path, "sports_sources")
    cfg = _write_yaml(tmp_path, """\
        name: multi_profile
        profiles:
          - sports
          - general
        sources:
          - sports_sources
    """)
    _, profiles = _load_build_config(cfg)
    assert profiles == ["sports", "general"]


def test_profiles_are_strings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_source_dirs(tmp_path, "sports_sources")
    cfg = _write_yaml(tmp_path, """\
        name: sports
        profiles:
          - sports
        sources:
          - sports_sources
    """)
    _, profiles = _load_build_config(cfg)
    assert all(isinstance(p, str) for p in profiles)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def test_relative_path_resolved_from_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_source_dirs(tmp_path, "sports_sources")
    cfg = _write_yaml(tmp_path, """\
        name: sports
        profiles: [sports]
        sources:
          - sports_sources
    """)
    sources, _ = _load_build_config(cfg)
    assert sources[0] == (tmp_path / "sports_sources").resolve()


def test_absolute_path_used_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    abs_dir = tmp_path / "abs_source"
    abs_dir.mkdir()
    cfg = _write_yaml(tmp_path, f"""\
        name: sports
        profiles: [sports]
        sources:
          - {abs_dir}
    """)
    sources, _ = _load_build_config(cfg)
    assert sources[0] == abs_dir.resolve()


def test_tilde_expansion(tmp_path, monkeypatch):
    home = tmp_path / "fake_home"
    home.mkdir()
    source_dir = home / "my_sources"
    source_dir.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    cfg = _write_yaml(tmp_path, """\
        name: tilde
        profiles: [sports]
        sources:
          - ~/my_sources
    """)
    sources, _ = _load_build_config(cfg)
    assert sources[0].exists()


def test_env_var_expansion(tmp_path, monkeypatch):
    source_dir = tmp_path / "env_sources"
    source_dir.mkdir()
    monkeypatch.setenv("MY_SOURCES_DIR", str(source_dir))
    monkeypatch.chdir(tmp_path)
    cfg = _write_yaml(tmp_path, """\
        name: env
        profiles: [sports]
        sources:
          - ${MY_SOURCES_DIR}
    """)
    sources, _ = _load_build_config(cfg)
    assert sources[0] == source_dir.resolve()


# ---------------------------------------------------------------------------
# Error: missing config file
# ---------------------------------------------------------------------------

def test_missing_config_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        _load_build_config(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# Error: invalid YAML
# ---------------------------------------------------------------------------

def test_invalid_yaml_raises_value_error(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("key: [\nunclosed bracket", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML"):
        _load_build_config(cfg)


# ---------------------------------------------------------------------------
# Error: non-mapping YAML root
# ---------------------------------------------------------------------------

def test_yaml_list_root_raises_value_error(tmp_path):
    cfg = _write_yaml(tmp_path, "- item1\n- item2\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        _load_build_config(cfg)


# ---------------------------------------------------------------------------
# Error: missing required fields
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing_field", ["name", "profiles", "sources"])
def test_missing_required_field_raises(tmp_path, monkeypatch, missing_field):
    monkeypatch.chdir(tmp_path)
    _make_source_dirs(tmp_path, "sports_sources")
    full = {"name": "sports", "profiles": ["sports"], "sources": ["sports_sources"]}
    del full[missing_field]
    lines = "\n".join(
        f"{k}: {v}" if isinstance(v, str) else f"{k}:\n" + "\n".join(f"  - {i}" for i in v)
        for k, v in full.items()
    )
    cfg = tmp_path / "config.yaml"
    cfg.write_text(lines, encoding="utf-8")
    with pytest.raises(ValueError, match=f"'{missing_field}'"):
        _load_build_config(cfg)


# ---------------------------------------------------------------------------
# Error: empty lists
# ---------------------------------------------------------------------------

def test_empty_profiles_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_source_dirs(tmp_path, "sports_sources")
    cfg = _write_yaml(tmp_path, """\
        name: sports
        profiles: []
        sources:
          - sports_sources
    """)
    with pytest.raises(ValueError, match="profiles"):
        _load_build_config(cfg)


def test_empty_sources_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _write_yaml(tmp_path, """\
        name: sports
        profiles:
          - sports
        sources: []
    """)
    with pytest.raises(ValueError, match="sources"):
        _load_build_config(cfg)


# ---------------------------------------------------------------------------
# Error: non-existent source directory
# ---------------------------------------------------------------------------

def test_nonexistent_source_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _write_yaml(tmp_path, """\
        name: sports
        profiles: [sports]
        sources:
          - this_directory_does_not_exist
    """)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _load_build_config(cfg)


# ---------------------------------------------------------------------------
# CLI mutual-exclusion checks (via typer CliRunner)
# ---------------------------------------------------------------------------

from typer.testing import CliRunner
from knowledge.cli import app as knowledge_app

runner = CliRunner()


def test_cli_config_and_sources_mutually_exclusive(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("name: x\nprofiles: [p]\nsources: [s]\n")
    result = runner.invoke(
        knowledge_app,
        ["build", "--config", str(cfg), "--sources", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "mutually exclusive" in (result.stderr or result.output)


def test_cli_config_and_profiles_mutually_exclusive(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_source_dirs(tmp_path, "sports_sources")
    cfg = _write_yaml(tmp_path, """\
        name: sports
        profiles: [sports]
        sources:
          - sports_sources
    """)
    result = runner.invoke(
        knowledge_app,
        ["build", "--config", str(cfg), "--profiles", "sports", "--skip-extraction"],
    )
    assert result.exit_code == 1
    assert "mutually exclusive" in (result.stderr or result.output)


def test_cli_missing_config_file(tmp_path):
    result = runner.invoke(
        knowledge_app,
        ["build", "--config", str(tmp_path / "does_not_exist.yaml")],
    )
    assert result.exit_code == 1
    assert "not found" in (result.stderr or result.output)


def test_cli_invalid_yaml_config(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("key: [\nbad yaml", encoding="utf-8")
    result = runner.invoke(
        knowledge_app,
        ["build", "--config", str(cfg)],
    )
    assert result.exit_code == 1
    assert "Invalid YAML" in (result.stderr or result.output)


def test_cli_sources_flag_still_works(tmp_path, monkeypatch):
    """Existing --sources flag must not be broken."""
    monkeypatch.chdir(tmp_path)
    _make_source_dirs(tmp_path, "sports_sources")
    result = runner.invoke(
        knowledge_app,
        [
            "build",
            "--sources", str(tmp_path / "sports_sources"),
            "--profiles", "sports",
            "--skip-extraction",
            "--log-level", "ERROR",
        ],
    )
    # Exit code 0 expected (no sources to process produces a clean report with 0 failures)
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Bundled config files exist and are valid YAML
# ---------------------------------------------------------------------------

_CONFIGS_DIR = Path("knowledge/configs")


@pytest.mark.skipif(
    not _CONFIGS_DIR.exists(),
    reason="knowledge/configs/ directory not present",
)
@pytest.mark.parametrize("config_name", ["sports.yaml", "smr.yaml", "ai_data_centers.yaml"])
def test_bundled_configs_are_valid_yaml(config_name):
    import yaml
    cfg = _CONFIGS_DIR / config_name
    raw = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    assert "name" in raw
    assert "profiles" in raw
    assert "sources" in raw
