"""EvidenceAgent – runs the research engine and captures evidence (J5.0b)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)


class EvidenceAgent(FunctionalAgent):
    """Runs the research_agent engine and stores the memo on the context."""

    def __init__(
        self,
        *,
        sources_dir: str | Path = "sources",
        client: Any = None,
        top_evidence: int = 50,
        top_chunks: int = 20,
        domain_profile: Any = None,
    ) -> None:
        self._sources_dir = Path(sources_dir)
        self._client = client
        self._top_evidence = top_evidence
        self._top_chunks = top_chunks
        self._domain_profile = domain_profile

    def _execute(self, context: AgentContext) -> AgentContext:
        from research_agent.agent import DcPowerAgent
        from research_agent.loaders import load_sources

        collection = load_sources(self._sources_dir)
        if collection.errors:
            for err in collection.errors:
                LOGGER.warning("Source load error: %s — %s", err.path.name, err.message)

        agent = DcPowerAgent(
            client=self._client,
            top_evidence=self._top_evidence,
            top_chunks=self._top_chunks,
            profile=self._domain_profile,
        )
        memo = agent.analyze(context.question, collection.documents)

        evidence_count = len(memo.source_notes or memo.evidence)
        fact_count = len(memo.confirmed_facts or [])

        # Detailed note on evidence_notes list
        context.evidence_notes.append(
            self._make_note(
                status="success",
                summary=f"Retrieved and analysed {evidence_count} evidence items.",
                evidence_count=evidence_count,
                confirmed_facts=fact_count,
                source_count=len(collection.documents),
            )
        )

        # Unified history entry
        self._record(
            context,
            status="success",
            summary=f"Retrieved {evidence_count} evidence items, confirmed {fact_count} facts.",
            evidence_count=evidence_count,
            confirmed_facts=fact_count,
        )

        # Stash memo and documents for downstream agents
        context.trace["_memo"] = memo
        context.trace["_documents"] = collection.documents
        return context
