"""Knowledge Store Health Validation (J8.7).

Validates a KnowledgeStore before runtime use:
  - manifest exists
  - domain(s) have evidence files
  - evidence count > 0
  - index exists and is consistent with evidence count
  - embeddings directory exists and count matches (where possible)

Usage:
    from knowledge.health import check_store_health
    report = check_store_health(store)
    if not report.runtime_ready:
        for issue in report.issues:
            logger.warning(issue)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import KnowledgeStore

LOGGER = logging.getLogger(__name__)


@dataclass
class DomainHealth:
    """Health status for a single domain."""

    domain: str
    evidence_file_exists: bool = False
    evidence_count: int = 0
    index_exists: bool = False
    index_count: int = 0
    count_consistent: bool = False
    runtime_ready: bool = False
    issues: list[str] = field(default_factory=list)


@dataclass
class HealthReport:
    """Health status for an entire knowledge store."""

    store_path: Path
    manifest_exists: bool = False
    manifest_source_count: int = 0
    total_embeddings: int = 0
    domains: list[DomainHealth] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    runtime_ready: bool = False

    def print(self) -> None:
        ready = "READY" if self.runtime_ready else "NOT READY"
        print(f"\n=== Knowledge Store Health: {ready} ===")
        print(f"Store:              {self.store_path}")
        print(f"Manifest:           {'OK' if self.manifest_exists else 'MISSING'} ({self.manifest_source_count} sources)")
        print(f"Embeddings (total): {self.total_embeddings} files")
        print(f"Domains checked:    {len(self.domains)}")

        for dh in self.domains:
            dom_ready = "READY" if dh.runtime_ready else "NOT READY"
            print(f"\n  [{dh.domain}] {dom_ready}")
            print(f"    evidence.jsonl:   {'OK' if dh.evidence_file_exists else 'MISSING'} ({dh.evidence_count} items)")
            print(f"    index.json:       {'OK' if dh.index_exists else 'MISSING'} ({dh.index_count} entries)")
            print(f"    count_consistent: {'YES' if dh.count_consistent else 'NO'}")
            for issue in dh.issues:
                print(f"    ISSUE: {issue}")

        if self.issues:
            print(f"\nStore-level issues:")
            for issue in self.issues:
                print(f"  {issue}")


def _count_evidence_lines(evidence_path: Path) -> int:
    """Count non-empty lines in evidence.jsonl."""
    if not evidence_path.exists():
        return 0
    count = 0
    with open(evidence_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _count_index_entries(index_path: Path) -> int:
    """Count entries in index.json."""
    if not index_path.exists():
        return 0
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
        return len(data) if isinstance(data, dict) else 0
    except Exception:
        return 0


def count_total_embeddings(store_root: Path) -> int:
    """Count all .npy files in embeddings/evidence/ (store-wide, not per-domain)."""
    emb_dir = store_root / "embeddings" / "evidence"
    if not emb_dir.exists():
        return 0
    return sum(1 for f in emb_dir.iterdir() if f.suffix == ".npy")


def check_domain_health(store_root: Path, domain: str) -> DomainHealth:
    """Validate a single domain within the store."""
    dh = DomainHealth(domain=domain)

    evidence_path = store_root / "evidence" / domain / "evidence.jsonl"
    index_path = store_root / "evidence" / domain / "index.json"

    dh.evidence_file_exists = evidence_path.exists()
    if dh.evidence_file_exists:
        dh.evidence_count = _count_evidence_lines(evidence_path)
    else:
        dh.issues.append(f"evidence.jsonl missing at {evidence_path}")

    dh.index_exists = index_path.exists()
    if dh.index_exists:
        dh.index_count = _count_index_entries(index_path)
    else:
        dh.issues.append(f"index.json missing at {index_path}")

    if dh.evidence_count == 0:
        dh.issues.append(
            f"evidence count is 0 — store may be partially built. "
            f"Run: python3 -m knowledge build --sources <dirs>"
        )

    if dh.evidence_count > 0 and dh.index_count > 0:
        ratio = abs(dh.evidence_count - dh.index_count) / max(dh.evidence_count, dh.index_count)
        dh.count_consistent = ratio < 0.05
        if not dh.count_consistent:
            dh.issues.append(
                f"count mismatch: evidence_count={dh.evidence_count} vs index_count={dh.index_count}"
            )
    elif dh.evidence_count > 0 and dh.index_count == 0:
        dh.count_consistent = False
        dh.issues.append("index.json is empty — re-run build to regenerate index")
    else:
        dh.count_consistent = dh.evidence_count == 0 and dh.index_count == 0

    dh.runtime_ready = (
        dh.evidence_file_exists
        and dh.evidence_count > 0
        and dh.index_exists
        and len(dh.issues) == 0
    )

    return dh


def check_store_health(
    store: "KnowledgeStore",
    domain: str | None = None,
) -> HealthReport:
    """Validate a KnowledgeStore for runtime readiness.

    Parameters
    ----------
    store:
        KnowledgeStore instance to validate.
    domain:
        If provided, validate only this domain. Otherwise validates all domains
        found in the evidence directory.

    Returns
    -------
    HealthReport
        Contains per-domain health status and an overall ``runtime_ready`` flag.
    """
    report = HealthReport(store_path=store.root)

    # Check manifest
    manifest_path = store.root / "manifests" / "manifest.json"
    report.manifest_exists = manifest_path.exists()
    if report.manifest_exists:
        try:
            manifest = store.load_manifest()
            report.manifest_source_count = len(manifest)
        except Exception as exc:
            report.issues.append(f"manifest read error: {exc}")
    else:
        report.issues.append(f"manifest.json missing at {manifest_path}")

    # Count total embeddings (store-wide, not per-domain)
    report.total_embeddings = count_total_embeddings(store.root)

    # Determine domains to check
    evidence_root = store.root / "evidence"
    if domain:
        domains_to_check = [domain]
    elif evidence_root.exists():
        domains_to_check = sorted(
            d.name for d in evidence_root.iterdir() if d.is_dir()
        )
    else:
        domains_to_check = []

    if not domains_to_check:
        report.issues.append(
            "No evidence domains found — knowledge store is empty. "
            "Run: python3 -m knowledge build --sources <dirs>"
        )

    for dom in domains_to_check:
        dh = check_domain_health(store.root, dom)
        report.domains.append(dh)

    # Overall readiness: at least one domain must be runtime_ready
    report.runtime_ready = any(dh.runtime_ready for dh in report.domains)

    if not report.runtime_ready and not report.issues:
        report.issues.append("No domain is ready for retrieval — check per-domain issues above")

    return report
