"""KnowledgeStore — persistent read/write layer for the Knowledge Base.

Layout:
  knowledge_store/
    sources/{domain}/{source_id}.json
    evidence/{domain}/evidence.jsonl
    evidence/{domain}/index.json          {evidence_id -> line_number}
    metadata/{domain}/metadata.jsonl
    extraction_runs/runs.jsonl
    manifests/manifest.json               {source_id -> SourceManifestEntry}
    contradictions/contradictions.jsonl
    embeddings/evidence/                  {evidence_id}.npy (optional)
    _meta/schema_version.json
    _meta/stats.json

All writes use write-then-rename to prevent partial writes.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .models import (
    Contradiction,
    Evidence,
    ExtractionRun,
    KnowledgeMetadata,
    Source,
    SourceManifestEntry,
)

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"
_DEFAULT_STORE = Path("knowledge_store")


class KnowledgeStore:
    """Filesystem-backed store for all canonical knowledge objects.

    Parameters
    ----------
    root:
        Root directory for the knowledge store.
        Defaults to ``knowledge_store/`` in the current working directory.
    """

    def __init__(self, root: str | Path = _DEFAULT_STORE) -> None:
        self.root = Path(root)
        self._ensure_layout()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _ensure_layout(self) -> None:
        for subdir in [
            "sources",
            "evidence",
            "metadata",
            "extraction_runs",
            "manifests",
            "contradictions",
            "embeddings/evidence",
            "_meta",
            "cache",
            "indexes",
        ]:
            (self.root / subdir).mkdir(parents=True, exist_ok=True)

        version_file = self.root / "_meta" / "schema_version.json"
        if not version_file.exists():
            self._atomic_write(version_file, {"schema_version": SCHEMA_VERSION, "created_at": datetime.utcnow().isoformat()})
            LOGGER.debug("knowledge_store: initialised schema version %s at %s", SCHEMA_VERSION, self.root)

    # ------------------------------------------------------------------
    # Atomic write helper
    # ------------------------------------------------------------------

    def _atomic_write(self, path: Path, data: dict | list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _atomic_append_jsonl(self, path: Path, record: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

    # ------------------------------------------------------------------
    # Source
    # ------------------------------------------------------------------

    def source_path(self, domain: str, source_id: str) -> Path:
        return self.root / "sources" / domain / f"{source_id}.json"

    def write_source(self, source: Source) -> None:
        path = self.source_path(source.domain, source.source_id)
        self._atomic_write(path, source.model_dump())
        LOGGER.debug("knowledge_store: wrote source %s", source.source_id)

    def read_source(self, domain: str, source_id: str) -> Source | None:
        path = self.source_path(domain, source_id)
        if not path.exists():
            return None
        return Source.model_validate_json(path.read_text(encoding="utf-8"))

    def has_source(self, domain: str, source_id: str) -> bool:
        return self.source_path(domain, source_id).exists()

    def iter_sources(self, domain: str) -> Iterator[Source]:
        domain_dir = self.root / "sources" / domain
        if not domain_dir.exists():
            return
        for path in sorted(domain_dir.glob("*.json")):
            try:
                yield Source.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception as exc:
                LOGGER.warning("knowledge_store: corrupt source %s — %s", path, exc)

    def find_source(self, source_id: str) -> Source | None:
        """Locate a Source by ID without knowing its domain (uses manifest)."""
        manifest = self.load_manifest()
        entry = manifest.get(source_id)
        if not entry:
            return None
        return self.read_source(entry.domain, source_id)

    def available_domains(self) -> list[str]:
        """Return the list of domains that have at least one evidence file."""
        evidence_root = self.root / "evidence"
        if not evidence_root.exists():
            return []
        return sorted(
            d.name for d in evidence_root.iterdir()
            if d.is_dir() and (d / "evidence.jsonl").exists()
        )

    # ------------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------------

    def _evidence_path(self, domain: str) -> Path:
        return self.root / "evidence" / domain / "evidence.jsonl"

    def _evidence_index_path(self, domain: str) -> Path:
        return self.root / "evidence" / domain / "index.json"

    def write_evidence(self, evidence: Evidence, domain: str) -> None:
        path = self._evidence_path(domain)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_append_jsonl(path, evidence.model_dump())

        # Update index: {evidence_id -> line_count - 1}
        index = self._load_evidence_index(domain)
        line_count = sum(1 for _ in open(path, encoding="utf-8")) if path.exists() else 0
        index[evidence.evidence_id] = line_count - 1
        self._atomic_write(self._evidence_index_path(domain), index)

        LOGGER.debug("knowledge_store: wrote evidence %s", evidence.evidence_id)

    def write_evidence_batch(self, items: list[Evidence], domain: str) -> None:
        if not items:
            return
        path = self._evidence_path(domain)
        path.parent.mkdir(parents=True, exist_ok=True)
        index = self._load_evidence_index(domain)

        existing_lines = 0
        if path.exists():
            with open(path, encoding="utf-8") as f:
                existing_lines = sum(1 for _ in f)

        with open(path, "a", encoding="utf-8") as f:
            for i, ev in enumerate(items):
                f.write(json.dumps(ev.model_dump(), ensure_ascii=False, default=str) + "\n")
                index[ev.evidence_id] = existing_lines + i

        self._atomic_write(self._evidence_index_path(domain), index)
        LOGGER.debug("knowledge_store: wrote %d evidence items to domain=%s", len(items), domain)

    def _load_evidence_index(self, domain: str) -> dict[str, int]:
        path = self._evidence_index_path(domain)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def read_evidence(self, domain: str, evidence_id: str) -> Evidence | None:
        index = self._load_evidence_index(domain)
        if evidence_id not in index:
            return None
        path = self._evidence_path(domain)
        if not path.exists():
            return None
        line_num = index[evidence_id]
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == line_num:
                    return Evidence.model_validate_json(line.strip())
        return None

    def iter_evidence(self, domain: str) -> Iterator[Evidence]:
        path = self._evidence_path(domain)
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield Evidence.model_validate_json(line)
                except Exception as exc:
                    LOGGER.warning("knowledge_store: corrupt evidence line — %s", exc)

    def evidence_count(self, domain: str) -> int:
        return len(self._load_evidence_index(domain))

    def get_statement_fingerprints(self, domain: str) -> set[str]:
        """Return the set of statement_fingerprint values for deduplication."""
        return {ev.statement_fingerprint for ev in self.iter_evidence(domain)}

    # ------------------------------------------------------------------
    # KnowledgeMetadata
    # ------------------------------------------------------------------

    def _metadata_path(self, domain: str) -> Path:
        return self.root / "metadata" / domain / "metadata.jsonl"

    def write_metadata(self, meta: KnowledgeMetadata, domain: str) -> None:
        path = self._metadata_path(domain)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_append_jsonl(path, meta.model_dump())

    def write_metadata_batch(self, items: list[KnowledgeMetadata], domain: str) -> None:
        if not items:
            return
        path = self._metadata_path(domain)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for m in items:
                f.write(json.dumps(m.model_dump(), ensure_ascii=False, default=str) + "\n")

    def iter_metadata(self, domain: str) -> Iterator[KnowledgeMetadata]:
        path = self._metadata_path(domain)
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield KnowledgeMetadata.model_validate_json(line)
                except Exception as exc:
                    LOGGER.warning("knowledge_store: corrupt metadata line — %s", exc)

    # ------------------------------------------------------------------
    # ExtractionRun
    # ------------------------------------------------------------------

    def _runs_path(self) -> Path:
        return self.root / "extraction_runs" / "runs.jsonl"

    def write_extraction_run(self, run: ExtractionRun) -> None:
        self._atomic_append_jsonl(self._runs_path(), run.model_dump())
        LOGGER.debug("knowledge_store: wrote extraction_run %s", run.run_id)

    def iter_extraction_runs(self) -> Iterator[ExtractionRun]:
        path = self._runs_path()
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield ExtractionRun.model_validate_json(line)
                except Exception as exc:
                    LOGGER.warning("knowledge_store: corrupt run line — %s", exc)

    def latest_extraction_run(self) -> ExtractionRun | None:
        last = None
        for run in self.iter_extraction_runs():
            last = run
        return last

    # ------------------------------------------------------------------
    # Contradiction
    # ------------------------------------------------------------------

    def _contradictions_path(self) -> Path:
        return self.root / "contradictions" / "contradictions.jsonl"

    def write_contradiction(self, c: Contradiction) -> None:
        self._atomic_append_jsonl(self._contradictions_path(), c.model_dump())

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _manifest_path(self) -> Path:
        return self.root / "manifests" / "manifest.json"

    def load_manifest(self) -> dict[str, SourceManifestEntry]:
        path = self._manifest_path()
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return {k: SourceManifestEntry.model_validate(v) for k, v in raw.items()}
        except Exception as exc:
            LOGGER.warning("knowledge_store: corrupt manifest — %s; returning empty", exc)
            return {}

    def save_manifest(self, manifest: dict[str, SourceManifestEntry]) -> None:
        data = {k: v.model_dump() for k, v in manifest.items()}
        self._atomic_write(self._manifest_path(), data)

    def update_manifest_entry(self, entry: SourceManifestEntry) -> None:
        manifest = self.load_manifest()
        manifest[entry.source_id] = entry
        self.save_manifest(manifest)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embedding_path(self, evidence_id: str) -> Path:
        return self.root / "embeddings" / "evidence" / f"{evidence_id}.npy"

    def has_embedding(self, evidence_id: str) -> bool:
        npy = self.embedding_path(evidence_id)
        return npy.exists() or npy.with_suffix(".json").exists()

    def write_embedding(self, evidence_id: str, vector: list[float]) -> None:
        try:
            import numpy as np  # type: ignore[import]
            path = self.embedding_path(evidence_id)
            arr = np.array(vector, dtype=np.float32)
            # Use suffix=".npy" so numpy does not append a second ".npy" extension
            # (numpy 2.x appends ".npy" when the path doesn't already end with it).
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npy")
            os.close(fd)
            np.save(tmp, arr)
            os.replace(tmp, path)
        except ImportError:
            # Store as JSON fallback if numpy is not available
            path = self.embedding_path(evidence_id).with_suffix(".json")
            self._atomic_write(path, {"evidence_id": evidence_id, "vector": vector})

    def read_embedding(self, evidence_id: str) -> list[float] | None:
        npy_path = self.embedding_path(evidence_id)
        if npy_path.exists():
            try:
                import numpy as np  # type: ignore[import]
                return np.load(str(npy_path)).tolist()
            except Exception:
                return None
        json_path = npy_path.with_suffix(".json")
        if json_path.exists():
            try:
                return json.loads(json_path.read_text())["vector"]
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def write_stats(self, stats: dict) -> None:
        self._atomic_write(self.root / "_meta" / "stats.json", stats)

    def read_stats(self) -> dict:
        path = self.root / "_meta" / "stats.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
