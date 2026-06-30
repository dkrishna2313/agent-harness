"""KnowledgeBuilder — offline knowledge construction pipeline.

Responsibilities:
  - Ingest Sources from source directories
  - Fingerprint Sources and detect unchanged ones (incremental builds)
  - Extract Evidence from new/changed Sources
  - Generate Evidence embeddings
  - Persist everything into the KnowledgeStore
  - Record one ExtractionRun per builder execution

This module does NOT perform strategic reasoning.
It does NOT modify the J7 research pipeline.
"""

from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable

from .embedder import EmbeddingBackend, NullEmbedder, get_default_embedder
from .extractor import _PROMPT_VERSION, extract_evidence_from_source
from .fingerprint import compute_text_fingerprint
from .models import Evidence, ExtractionRun, Source, SourceManifestEntry
from .source_normalizer import normalize_source
from .store import KnowledgeStore

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain auto-detection from directory name
# ---------------------------------------------------------------------------

_DOMAIN_MAP: dict[str, str] = {
    "smr_sources": "smr",
    "smr": "smr",
    "nvidia": "ai_data_centers",
    "ai_data_centers": "ai_data_centers",
    "infrastructure": "infrastructure",
    "market": "economics",
    "economics": "economics",
    "networking": "networking",
    "transmission": "transmission",
    "nuclear_policy": "nuclear_policy",
    "sources": "general",
    "empty_sources": "empty",
}

_SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def infer_domain(directory: Path) -> str:
    name = directory.name.lower().rstrip("/")
    return _DOMAIN_MAP.get(name, name)


# ---------------------------------------------------------------------------
# BuildReport
# ---------------------------------------------------------------------------


@dataclass
class DomainBuildSummary:
    """Per-domain breakdown within a build run."""

    domain: str
    sources_rebuilt: int = 0
    sources_skipped: int = 0
    sources_failed: int = 0
    evidence_objects: int = 0
    embeddings_generated: int = 0
    runtime_ready: bool = False


@dataclass
class BuildReport:
    """Summary of a KnowledgeBuilder run."""

    sources_scanned: int = 0
    sources_skipped: int = 0
    sources_rebuilt: int = 0
    sources_failed: int = 0
    evidence_objects: int = 0
    duplicates_merged: int = 0
    embeddings_generated: int = 0
    embeddings_skipped: int = 0
    total_runtime_seconds: float = 0.0
    extraction_run_id: str = ""
    errors: list[str] = field(default_factory=list)
    # J8.7 — per-domain breakdown and runtime readiness
    per_domain: dict[str, DomainBuildSummary] = field(default_factory=dict)
    runtime_ready: bool = False

    @property
    def cache_hit_ratio(self) -> float:
        total = self.sources_scanned
        return round(self.sources_skipped / total, 3) if total else 0.0

    def summary_lines(self) -> list[str]:
        lines = [
            f"Sources scanned:       {self.sources_scanned}",
            f"Sources skipped:       {self.sources_skipped}  (cache_hit={self.cache_hit_ratio:.1%})",
            f"Sources rebuilt:       {self.sources_rebuilt}",
            f"Sources failed:        {self.sources_failed}",
            f"Evidence objects:      {self.evidence_objects}",
            f"Duplicates merged:     {self.duplicates_merged}",
            f"Embeddings generated:  {self.embeddings_generated}",
            f"Embeddings skipped:    {self.embeddings_skipped}",
            f"Total runtime:         {self.total_runtime_seconds:.1f}s",
            f"ExtractionRun ID:      {self.extraction_run_id}",
            f"Runtime ready:         {'YES' if self.runtime_ready else 'NO — run embed to enable semantic retrieval'}",
        ]
        if self.per_domain:
            lines.append("\nPer-domain breakdown:")
            for domain, ds in sorted(self.per_domain.items()):
                ready = "ready" if ds.runtime_ready else "NOT READY"
                lines.append(
                    f"  {domain:24} rebuilt={ds.sources_rebuilt:3}  "
                    f"evidence={ds.evidence_objects:4}  embeddings={ds.embeddings_generated:4}  [{ready}]"
                )
        return lines

    def print(self) -> None:
        print("\n=== Knowledge Builder Report ===")
        for line in self.summary_lines():
            print(line)
        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for err in self.errors[:10]:
                print(f"  {err}")


# ---------------------------------------------------------------------------
# KnowledgeBuilder
# ---------------------------------------------------------------------------


class KnowledgeBuilder:
    """Constructs and maintains the persistent Knowledge Base.

    Parameters
    ----------
    store:
        KnowledgeStore instance. Defaults to knowledge_store/ in the CWD.
    client:
        ClaudeClient or MockClaudeClient for evidence extraction.
        If None, evidence extraction is skipped (source ingestion only).
    embedder:
        EmbeddingBackend. Defaults to best available (LocalEmbedder if
        sentence-transformers is installed, else NullEmbedder).
    model_version:
        Model identifier recorded in ExtractionRun provenance.
    workers:
        Number of concurrent source-processing threads.
    """

    def __init__(
        self,
        store: KnowledgeStore | None = None,
        client: object = None,
        embedder: EmbeddingBackend | None = None,
        model_version: str = "claude-sonnet-4-6",
        workers: int = 1,
    ) -> None:
        self.store = store or KnowledgeStore()
        self.client = client
        self.embedder: EmbeddingBackend = embedder if embedder is not None else get_default_embedder()
        self.model_version = model_version
        self.workers = max(1, workers)

        if isinstance(self.embedder, NullEmbedder):
            LOGGER.warning(
                "builder: no embedding backend available — embeddings will be skipped. "
                "Install sentence-transformers for real embeddings."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        source_dirs: list[Path],
        *,
        domain_overrides: dict[str, str] | None = None,
        incremental: bool = True,
        force: bool = False,
        profile_ids: list[str] | None = None,
    ) -> BuildReport:
        """Run the Knowledge Builder over one or more source directories.

        Parameters
        ----------
        source_dirs:
            Directories to ingest. Each may contain PDFs, TXTs, or DOCXs.
        domain_overrides:
            Map directory name → domain, overriding auto-detection.
        incremental:
            If True, skip sources whose fingerprint has not changed.
        force:
            If True, rebuild all sources regardless of fingerprint.
        profile_ids:
            Profile IDs to tag all produced evidence with.
        """
        started_at = datetime.utcnow()
        t0 = time.monotonic()

        run = ExtractionRun(
            model_version=self.model_version,
            prompt_version=_PROMPT_VERSION,
            started_at=started_at,
        )
        self.store.write_extraction_run(run)

        manifest = self.store.load_manifest()
        report = BuildReport(extraction_run_id=run.run_id)

        for source_dir in source_dirs:
            source_dir = Path(source_dir)
            if not source_dir.exists():
                LOGGER.warning("builder: source directory does not exist: %s", source_dir)
                continue

            domain = (domain_overrides or {}).get(source_dir.name, infer_domain(source_dir))
            if domain == "empty":
                LOGGER.debug("builder: skipping empty_sources directory")
                continue

            LOGGER.info("builder: scanning %s  domain=%s", source_dir, domain)
            self._process_directory(
                source_dir,
                domain=domain,
                manifest=manifest,
                run=run,
                report=report,
                incremental=incremental and not force,
                profile_ids=profile_ids or [],
            )

        # Finalise ExtractionRun
        elapsed = time.monotonic() - t0
        report.total_runtime_seconds = round(elapsed, 2)
        run = ExtractionRun(
            run_id=run.run_id,
            model_version=run.model_version,
            prompt_version=run.prompt_version,
            started_at=run.started_at,
            completed_at=datetime.utcnow(),
            status="COMPLETED",
            sources_scanned=report.sources_scanned,
            sources_skipped=report.sources_skipped,
            sources_rebuilt=report.sources_rebuilt,
            evidence_ids_produced=run.evidence_ids_produced,
            duplicates_merged=report.duplicates_merged,
            embeddings_generated=report.embeddings_generated,
            duration_seconds=elapsed,
        )
        self.store.write_extraction_run(run)
        self.store.save_manifest(manifest)
        self._write_stats(report)

        # J8.7 — post-build health check to set runtime_ready
        try:
            from .health import check_store_health
            health = check_store_health(self.store)
            for dh in health.domains:
                if dh.domain in report.per_domain:
                    report.per_domain[dh.domain].runtime_ready = dh.runtime_ready
            report.runtime_ready = health.runtime_ready
        except Exception as exc:
            LOGGER.warning("builder: post-build health check failed — %s", exc)

        return report

    def ingest_directory(
        self,
        source_dir: str | Path,
        domain: str | None = None,
        *,
        incremental: bool = True,
        force: bool = False,
        profile_ids: list[str] | None = None,
    ) -> BuildReport:
        """Convenience method: build from a single directory."""
        source_dir = Path(source_dir)
        domain = domain or infer_domain(source_dir)
        return self.build(
            [source_dir],
            domain_overrides={source_dir.name: domain},
            incremental=incremental,
            force=force,
            profile_ids=profile_ids,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_directory(
        self,
        directory: Path,
        domain: str,
        manifest: dict[str, SourceManifestEntry],
        run: ExtractionRun,
        report: BuildReport,
        incremental: bool,
        profile_ids: list[str],
    ) -> None:
        source_files = self._discover_files(directory)
        LOGGER.info("builder: found %d files in %s", len(source_files), directory)

        existing_fingerprints = self.store.get_statement_fingerprints(domain)

        if self.workers > 1:
            self._process_files_concurrent(
                source_files, domain, manifest, run, report, incremental, profile_ids, existing_fingerprints
            )
        else:
            for path in source_files:
                self._process_file(
                    path, domain, manifest, run, report, incremental, profile_ids, existing_fingerprints
                )

    def _process_files_concurrent(
        self,
        files: list[Path],
        domain: str,
        manifest: dict[str, SourceManifestEntry],
        run: ExtractionRun,
        report: BuildReport,
        incremental: bool,
        profile_ids: list[str],
        existing_fingerprints: set[str],
    ) -> None:
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(
                    self._process_file,
                    path, domain, manifest, run, report, incremental, profile_ids, existing_fingerprints
                ): path
                for path in files
            }
            for future in as_completed(futures):
                path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    LOGGER.error("builder: unhandled error processing %s — %s", path, exc)
                    report.errors.append(f"{path}: {exc}")
                    report.sources_failed += 1

    def _process_file(
        self,
        path: Path,
        domain: str,
        manifest: dict[str, SourceManifestEntry],
        run: ExtractionRun,
        report: BuildReport,
        incremental: bool,
        profile_ids: list[str],
        existing_fingerprints: set[str],
    ) -> None:
        report.sources_scanned += 1
        # Ensure per-domain summary entry exists
        if domain not in report.per_domain:
            report.per_domain[domain] = DomainBuildSummary(domain=domain)

        # Extract canonical text
        try:
            canonical_text = self._extract_text(path)
        except Exception as exc:
            LOGGER.warning("builder: failed to extract text from %s — %s", path, exc)
            report.sources_failed += 1
            report.per_domain[domain].sources_failed += 1
            report.errors.append(f"{path.name}: text extraction failed — {exc}")
            return

        fingerprint = compute_text_fingerprint(canonical_text)
        source_id = Source.compute_source_id(fingerprint)

        # Incremental check
        if incremental and source_id in manifest:
            entry = manifest[source_id]
            if entry.fingerprint == fingerprint:
                LOGGER.debug("builder: skip unchanged source %s", path.name)
                report.sources_skipped += 1
                report.per_domain[domain].sources_skipped += 1
                return

        LOGGER.info("builder: processing %s  source_id=%s", path.name, source_id)

        # Normalise source provenance (P1 PDF metadata → P2 text regex → P3 LLM)
        norm = normalize_source(path, canonical_text, use_llm=(self.client is not None))

        # Build Source record
        source = Source(
            source_id=source_id,
            uri=str(path.resolve()),
            title=norm.title or path.stem.replace("_", " ").replace("-", " "),
            subtitle=norm.subtitle,
            author=norm.author,
            organization=norm.organization,
            publisher=norm.publisher,
            publication_date=norm.publication_date,
            retrieved_date=date.today(),
            fingerprint=fingerprint,
            document_type=self._detect_type(path),
            domain=domain,
            copyright=norm.copyright,
            canonical_text=canonical_text,
            page_count=norm.page_count,
            document_version=norm.document_version,
            document_number=norm.document_number,
        )

        # Persist Source
        self.store.write_source(source)

        # Evidence extraction
        evidence_list: list[Evidence] = []
        duplicates = 0
        if self.client is not None:
            try:
                ev_list, meta_list, duplicates = extract_evidence_from_source(
                    source,
                    extraction_run_id=run.run_id,
                    client=self.client,
                    existing_fingerprints=existing_fingerprints,
                    profile_ids=profile_ids,
                )
                if ev_list:
                    self.store.write_evidence_batch(ev_list, domain)
                    self.store.write_metadata_batch(meta_list, domain)
                evidence_list = ev_list
                report.evidence_objects += len(ev_list)
                report.duplicates_merged += duplicates
                report.per_domain[domain].evidence_objects += len(ev_list)
                # Track produced IDs in run (thread-safe append to list is GIL-protected in CPython)
                run.evidence_ids_produced.extend(ev.evidence_id for ev in ev_list)
            except Exception as exc:
                LOGGER.error("builder: evidence extraction failed for %s — %s", path.name, exc)
                report.errors.append(f"{path.name}: evidence extraction failed — {exc}")
        else:
            LOGGER.debug("builder: no client configured — skipping evidence extraction for %s", path.name)

        # Embedding generation
        embeddings_generated = self._generate_embeddings(evidence_list)
        report.embeddings_generated += embeddings_generated
        report.embeddings_skipped += len(evidence_list) - embeddings_generated
        report.per_domain[domain].embeddings_generated += embeddings_generated

        # Evict any stale manifest entry that shares this URI (content changed → new source_id)
        uri_str = str(path.resolve())
        stale_ids = [sid for sid, entry in manifest.items() if entry.uri == uri_str and sid != source_id]
        for stale_id in stale_ids:
            LOGGER.debug("builder: evicting stale manifest entry for %s (old source_id=%s)", path.name, stale_id)
            del manifest[stale_id]

        # Update manifest
        manifest[source_id] = SourceManifestEntry(
            source_id=source_id,
            fingerprint=fingerprint,
            domain=domain,
            uri=str(path.resolve()),
            evidence_ids=[ev.evidence_id for ev in evidence_list],
            last_built=datetime.utcnow(),
            extraction_run_id=run.run_id,
        )
        report.sources_rebuilt += 1
        report.per_domain[domain].sources_rebuilt += 1

    def _generate_embeddings(self, evidence_list: list[Evidence]) -> int:
        if not evidence_list:
            return 0
        texts = [ev.statement for ev in evidence_list]
        vectors = self.embedder.embed_batch(texts)
        generated = 0
        for ev, vec in zip(evidence_list, vectors):
            if vec is not None:
                self.store.write_embedding(ev.evidence_id, vec)
                generated += 1
        return generated

    @staticmethod
    def _discover_files(directory: Path) -> list[Path]:
        files = []
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in _SUPPORTED_EXTENSIONS:
                files.append(path)
        return files

    @staticmethod
    def _extract_text(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".txt" or suffix == ".md":
            return path.read_text(encoding="utf-8", errors="replace")
        if suffix == ".pdf":
            return _extract_pdf_text(path)
        if suffix == ".docx":
            return _extract_docx_text(path)
        raise ValueError(f"Unsupported extension: {suffix}")

    @staticmethod
    def _detect_type(path: Path) -> str:
        mapping = {".pdf": "PDF", ".txt": "TXT", ".md": "TXT", ".docx": "DOCX", ".html": "HTML"}
        return mapping.get(path.suffix.lower(), "TXT")

    def _write_stats(self, report: BuildReport) -> None:
        self.store.write_stats({
            "last_build": datetime.utcnow().isoformat(),
            "sources_scanned": report.sources_scanned,
            "sources_skipped": report.sources_skipped,
            "sources_rebuilt": report.sources_rebuilt,
            "evidence_objects": report.evidence_objects,
            "duplicates_merged": report.duplicates_merged,
            "embeddings_generated": report.embeddings_generated,
            "cache_hit_ratio": report.cache_hit_ratio,
            "extraction_run_id": report.extraction_run_id,
        })


# ---------------------------------------------------------------------------
# Text extraction helpers (mirrors loaders.py; independent of it)
# ---------------------------------------------------------------------------


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore[import]
    except ImportError:
        raise RuntimeError("Install pypdf to extract PDF sources: pip install pypdf") from None

    reader = PdfReader(str(path))
    chunks = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        if text.strip():
            chunks.append(f"[Page {i}]\n{text.strip()}")
    return "\n\n".join(chunks)


def _extract_docx_text(path: Path) -> str:
    try:
        import docx  # type: ignore[import]
    except ImportError:
        raise RuntimeError("Install python-docx to extract DOCX sources: pip install python-docx") from None

    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
