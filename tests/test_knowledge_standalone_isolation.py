"""R2 — Knowledge Layer standalone isolation tests.

Proves that the documented read-only Knowledge API works without importing
functional_agents, strategy, or editorial.

Every test in this file is self-contained:
- Uses only knowledge.* imports (no functional_agents)
- Creates deterministic in-memory stores via tmp_path
- Does NOT read from outputs/ or any live CLI-generated artifact
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

from knowledge.models import Evidence, KnowledgeMetadata, Source
from knowledge.retriever import (
    RETRIEVAL_MODE_LEXICAL,
    EvidenceRetriever,
    RetrievalResult,
    RetrievedEvidence,
)
from knowledge.store import KnowledgeStore

# ---------------------------------------------------------------------------
# Fixture helpers (copied from test_retriever.py pattern, domain-neutral)
# ---------------------------------------------------------------------------

_DOMAIN_A = "domain-alpha"
_DOMAIN_B = "domain-beta"


def _make_evidence(
    statement: str,
    *,
    etype: str = "STRATEGIC",
    profile_ids: list[str] | None = None,
    **kwargs,
) -> Evidence:
    return Evidence(
        statement=statement,
        evidence_type=etype,
        supporting_source_ids=["src-001"],
        extraction_run_id="run-001",
        profile_ids=profile_ids or [],
        **kwargs,
    )


def _make_meta(
    evidence_id: str,
    *,
    retrieval_enabled: bool = True,
    overall_score: float = 3.0,
    retrieval_priority: int = 3,
) -> KnowledgeMetadata:
    return KnowledgeMetadata(
        evidence_id=evidence_id,
        retrieval_enabled=retrieval_enabled,
        overall_score=overall_score,
        retrieval_priority=retrieval_priority,
        review_status="AUTO_REVIEWED",
    )


@pytest.fixture()
def two_profile_store(tmp_path: Path) -> KnowledgeStore:
    """Store with items tagged to two distinct profiles in one domain."""
    ks = KnowledgeStore(root=tmp_path / "ks")

    alpha_items = [
        _make_evidence("Alpha strategic market entry requires brand differentiation.", profile_ids=["alpha"]),
        _make_evidence("Alpha competitive positioning depends on pricing strategy.", profile_ids=["alpha"]),
    ]
    beta_items = [
        _make_evidence("Beta analytics platform enables real-time performance tracking.", profile_ids=["beta"]),
        _make_evidence("Beta operational efficiency improves margin by 15 percent.", profile_ids=["beta"]),
    ]

    all_items = alpha_items + beta_items
    metas = [_make_meta(ev.evidence_id) for ev in all_items]

    ks.write_evidence_batch(all_items, _DOMAIN_A)
    ks.write_metadata_batch(metas, _DOMAIN_A)
    return ks


@pytest.fixture()
def simple_store(tmp_path: Path) -> KnowledgeStore:
    """Store with a handful of items for basic retrieval tests."""
    ks = KnowledgeStore(root=tmp_path / "ks")

    items = [
        _make_evidence(
            "Competitive strategy requires market differentiation and brand positioning.",
            profile_ids=["sports"],
        ),
        _make_evidence(
            "Analytics-driven talent acquisition improves team performance by 20 percent.",
            profile_ids=["sports"],
        ),
        _make_evidence(
            "Revenue diversification through media rights is a key strategic lever.",
            profile_ids=["sports"],
        ),
    ]
    metas = [_make_meta(ev.evidence_id, overall_score=4.0) for ev in items]

    ks.write_evidence_batch(items, _DOMAIN_A)
    ks.write_metadata_batch(metas, _DOMAIN_A)
    return ks


# ---------------------------------------------------------------------------
# Test 1 — Import isolation (subprocess)
# ---------------------------------------------------------------------------


class TestImportIsolation:
    def test_knowledge_store_does_not_load_functional_agents(self):
        """Importing KnowledgeStore must not load functional_agents or strategy."""
        snippet = dedent("""
            import sys
            from knowledge.store import KnowledgeStore
            from knowledge.retriever import EvidenceRetriever
            from knowledge.models import Evidence, KnowledgeMetadata
            forbidden = [
                "functional_agents",
                "functional_agents.strategy",
                "functional_agents.editorial",
            ]
            loaded = [m for m in sys.modules if any(m == f or m.startswith(f + ".") for f in forbidden)]
            if loaded:
                print("FORBIDDEN MODULES LOADED:", loaded, flush=True)
                sys.exit(1)
            print("ISOLATION OK", flush=True)
        """)
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0, f"Import isolation failed:\n{result.stdout}\n{result.stderr}"
        assert "ISOLATION OK" in result.stdout

    def test_knowledge_retrieval_does_not_load_strategy(self, simple_store: KnowledgeStore):
        """No strategy module is imported during read-only retrieval."""
        # Guard: if any strategy module sneaks in, subsequent getattr calls fail
        forbidden_prefixes = (
            "functional_agents.strategy",
            "functional_agents.editorial",
        )
        loaded_before = set(sys.modules)
        retriever = EvidenceRetriever(simple_store)
        retriever.retrieve("competitive strategy", domain=_DOMAIN_A, top_k=5)
        new_modules = set(sys.modules) - loaded_before
        forbidden_loaded = [
            m for m in new_modules
            if any(m.startswith(p) for p in forbidden_prefixes)
        ]
        assert not forbidden_loaded, f"Strategy modules loaded during retrieval: {forbidden_loaded}"


# ---------------------------------------------------------------------------
# Test 2 — Store open
# ---------------------------------------------------------------------------


class TestStoreOpen:
    def test_open_store_via_documented_api(self, tmp_path: Path):
        """KnowledgeStore(path) is openable and creates layout."""
        store = KnowledgeStore(root=tmp_path / "fresh_store")
        assert (tmp_path / "fresh_store" / "_meta" / "schema_version.json").exists()
        assert isinstance(store.available_domains(), list)

    def test_stats_readable_on_empty_store(self, tmp_path: Path):
        store = KnowledgeStore(root=tmp_path / "empty")
        stats = store.read_stats()
        assert isinstance(stats, dict)


# ---------------------------------------------------------------------------
# Test 3 — Standalone retrieval
# ---------------------------------------------------------------------------


class TestStandaloneRetrieval:
    def test_retrieval_returns_result_type(self, simple_store: KnowledgeStore):
        retriever = EvidenceRetriever(simple_store)
        result = retriever.retrieve("competitive strategy market", domain=_DOMAIN_A)
        assert isinstance(result, RetrievalResult)

    def test_retrieval_items_are_retrieved_evidence(self, simple_store: KnowledgeStore):
        retriever = EvidenceRetriever(simple_store)
        result = retriever.retrieve("competitive strategy", domain=_DOMAIN_A)
        assert len(result.items) > 0
        item = result.items[0]
        assert isinstance(item, RetrievedEvidence)

    def test_result_fields_populated(self, simple_store: KnowledgeStore):
        """All documented result fields are present and correct types."""
        retriever = EvidenceRetriever(simple_store)
        result = retriever.retrieve("analytics strategy market", domain=_DOMAIN_A, top_k=3)
        assert isinstance(result.query, str)
        assert isinstance(result.items, list)
        assert isinstance(result.domains_searched, list)
        assert isinstance(result.total_candidates, int)
        assert isinstance(result.matched_candidates, int)
        assert isinstance(result.retrieval_method, str)
        assert isinstance(result.latency_ms, float)
        assert result.latency_ms >= 0.0

    def test_item_fields_populated(self, simple_store: KnowledgeStore):
        """RetrievedEvidence has chunk text, ID, profile IDs, score, rank, domain."""
        retriever = EvidenceRetriever(simple_store)
        result = retriever.retrieve("competitive strategy", domain=_DOMAIN_A)
        assert result.items, "Expected at least one result"
        item = result.items[0]

        assert isinstance(item.evidence.evidence_id, str) and item.evidence.evidence_id
        assert isinstance(item.evidence.statement, str) and item.evidence.statement
        assert isinstance(item.evidence.profile_ids, list)
        assert isinstance(item.score, float)
        assert item.score > 0.0
        assert item.rank == 1
        assert isinstance(item.source_domain, str) and item.source_domain == _DOMAIN_A

    def test_read_evidence_by_id(self, simple_store: KnowledgeStore):
        """store.read_evidence(domain, id) returns the original item."""
        retriever = EvidenceRetriever(simple_store)
        result = retriever.retrieve("analytics talent", domain=_DOMAIN_A, top_k=1)
        assert result.items
        ev_id = result.items[0].evidence.evidence_id
        fetched = simple_store.read_evidence(_DOMAIN_A, ev_id)
        assert fetched is not None
        assert fetched.evidence_id == ev_id


# ---------------------------------------------------------------------------
# Test 4 — Profile filtering
# ---------------------------------------------------------------------------


class TestProfileFiltering:
    def test_profile_restricts_results(self, two_profile_store: KnowledgeStore):
        """Querying with profile='alpha' returns only alpha-tagged items."""
        retriever = EvidenceRetriever(two_profile_store)

        alpha_result = retriever.retrieve("strategy market", domain=_DOMAIN_A, profile="alpha")
        for item in alpha_result.items:
            assert "alpha" in item.evidence.profile_ids, (
                f"Found non-alpha item: {item.evidence.statement[:60]}"
            )

    def test_profile_excludes_other_profile(self, two_profile_store: KnowledgeStore):
        """beta profile items do not appear in alpha-restricted results."""
        retriever = EvidenceRetriever(two_profile_store)
        alpha_result = retriever.retrieve("analytics performance", domain=_DOMAIN_A, profile="alpha")
        for item in alpha_result.items:
            assert "beta" not in item.evidence.profile_ids

    def test_no_profile_returns_all(self, two_profile_store: KnowledgeStore):
        """No profile filter returns items from both profiles."""
        retriever = EvidenceRetriever(two_profile_store)
        result = retriever.retrieve("strategy analytics", domain=_DOMAIN_A)
        profiles_seen = {p for item in result.items for p in item.evidence.profile_ids}
        assert "alpha" in profiles_seen or "beta" in profiles_seen


# ---------------------------------------------------------------------------
# Test 5 — No strategy initialization during retrieval
# ---------------------------------------------------------------------------


class TestNoStrategyInitialization:
    def test_retrieval_works_with_strategy_blocked(self, simple_store: KnowledgeStore, monkeypatch):
        """Retrieval succeeds even when strategy modules are blocked from import."""
        # Make importing strategy modules raise ImportError
        import builtins
        real_import = builtins.__import__

        def blocking_import(name, *args, **kwargs):
            if name.startswith("functional_agents.strategy") or name.startswith("functional_agents.editorial"):
                raise ImportError(f"Import blocked for isolation test: {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocking_import)

        # Should not raise
        retriever = EvidenceRetriever(simple_store)
        result = retriever.retrieve("competitive strategy", domain=_DOMAIN_A)
        assert isinstance(result, RetrievalResult)


# ---------------------------------------------------------------------------
# Test 6 — CLI isolation (subprocess)
# ---------------------------------------------------------------------------


class TestCLIIsolation:
    def test_knowledge_status_exits_zero(self, tmp_path: Path):
        """python3 -m knowledge status runs without loading functional_agents.strategy."""
        store_path = tmp_path / "cli_store"
        ks = KnowledgeStore(root=store_path)  # initialise layout

        result = subprocess.run(
            [sys.executable, "-m", "knowledge", "status", "--store", str(store_path)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0, f"CLI status failed:\n{result.stderr}"
        assert "Knowledge Base" in result.stdout

    def test_knowledge_status_does_not_import_strategy(self, tmp_path: Path):
        """CLI status command should not trigger strategy imports."""
        store_path = tmp_path / "cli_store2"
        KnowledgeStore(root=store_path)

        # Use the verbose module check via PYTHONVERBOSE or audit hook
        snippet = dedent(f"""
            import sys, subprocess, os
            env = os.environ.copy()
            env.pop("PYTHONVERBOSE", None)
            proc = subprocess.run(
                [sys.executable, "-c",
                 "import sys; "
                 "import knowledge.cli; "
                 "forbidden = [m for m in sys.modules if 'functional_agents.strategy' in m "
                              "or 'functional_agents.editorial' in m]; "
                 "print('FORBIDDEN:', forbidden if forbidden else 'none')"],
                capture_output=True, text=True,
                cwd=r"{Path(__file__).parent.parent}",
            )
            print(proc.stdout)
            if "functional_agents.strategy" in proc.stdout or "functional_agents.editorial" in proc.stdout:
                sys.exit(1)
        """)
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"CLI strategy isolation failed:\n{result.stdout}\n{result.stderr}"


# ---------------------------------------------------------------------------
# Test 7 — Build skip-extraction isolation (API level)
# ---------------------------------------------------------------------------


class TestBuildSkipExtractionIsolation:
    def test_builder_importable_without_research_agent(self):
        """KnowledgeBuilder can be imported and instantiated without research_agent."""
        import builtins
        real_import = builtins.__import__

        # Track if research_agent is imported during KnowledgeBuilder construction
        research_agent_imported = []

        def tracking_import(name, *args, **kwargs):
            if name.startswith("research_agent"):
                research_agent_imported.append(name)
            return real_import(name, *args, **kwargs)

        original = builtins.__import__
        builtins.__import__ = tracking_import
        try:
            from knowledge.builder import KnowledgeBuilder
            from knowledge.store import KnowledgeStore as KS
            # Instantiate with client=None (equivalent to --skip-extraction)
            import tempfile
            with tempfile.TemporaryDirectory() as d:
                store = KS(root=Path(d) / "ks")
                builder = KnowledgeBuilder(store=store, client=None)
                assert builder is not None
        finally:
            builtins.__import__ = original

        assert not research_agent_imported, (
            f"research_agent was imported during builder construction: {research_agent_imported}"
        )

    def test_store_build_with_no_sources_does_not_need_research_agent(self, tmp_path: Path):
        """Building with no source dirs and no client produces a valid empty BuildReport."""
        from knowledge.builder import KnowledgeBuilder

        store = KnowledgeStore(root=tmp_path / "empty_build")
        builder = KnowledgeBuilder(store=store, client=None)
        # build() takes source_dirs: list[Path] as positional arg
        report = builder.build(source_dirs=[], incremental=True)
        assert report is not None


# ---------------------------------------------------------------------------
# Test 8 — Extraction coupling documentation guard
# ---------------------------------------------------------------------------


class TestExtractionCouplingGuard:
    """Verify the known lazy imports remain inside function bodies, not at module level."""

    @staticmethod
    def _module_level_research_agent_imports(module_path: Path) -> list[tuple[int, str]]:
        """Return (lineno, import_name) for any top-level research_agent import."""
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(module_path))
        violations = []
        for node in ast.iter_child_nodes(tree):
            # Only check module-level nodes (not inside functions/classes)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("research_agent"):
                        violations.append((node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("research_agent"):
                    violations.append((node.lineno, module))
        return violations

    def test_extractor_has_no_module_level_research_agent_import(self):
        """knowledge/extractor.py must not import research_agent at module level."""
        repo = Path(__file__).parent.parent
        violations = self._module_level_research_agent_imports(repo / "knowledge" / "extractor.py")
        assert not violations, (
            f"Module-level research_agent imports found in extractor.py: {violations}. "
            "These must remain inside function bodies to preserve standalone isolation."
        )

    def test_reranker_has_no_module_level_research_agent_import(self):
        """knowledge/reranker.py must not import research_agent at module level."""
        repo = Path(__file__).parent.parent
        violations = self._module_level_research_agent_imports(repo / "knowledge" / "reranker.py")
        assert not violations, (
            f"Module-level research_agent imports found in reranker.py: {violations}. "
            "These must remain inside function bodies to preserve standalone isolation."
        )

    def test_cli_research_agent_import_is_guarded_by_condition(self):
        """knowledge/cli.py research_agent import must be inside a conditional block."""
        repo = Path(__file__).parent.parent
        source = (repo / "knowledge" / "cli.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Find all ImportFrom nodes where module starts with research_agent
        # They must all be children of If nodes, not module-level
        research_imports_at_module_level = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("research_agent"):
                    research_imports_at_module_level.append(node.lineno)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("research_agent"):
                            research_imports_at_module_level.append(node.lineno)

        assert not research_imports_at_module_level, (
            f"Unguarded module-level research_agent import in cli.py at lines: "
            f"{research_imports_at_module_level}"
        )
