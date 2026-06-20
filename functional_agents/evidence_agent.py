"""EvidenceAgent – runs the research engine and captures evidence (J5.0a.4/5)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)


class EvidenceAgent(FunctionalAgent):
    """Runs the research_agent engine and stores the memo on the context.

    The engine call happens here so downstream agents (QA, Report) can
    inspect the memo before the report is written.
    """

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

    def _execute(self, ctx: AgentContext) -> AgentContext:
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
        memo = agent.analyze(ctx.question, collection.documents)

        # Store memo on context for downstream agents
        ctx.evidence_notes.append(
            self._make_note(
                status="completed",
                summary=f"Retrieved and analysed {len(memo.source_notes or memo.evidence)} evidence items.",
                evidence_count=len(memo.source_notes or memo.evidence),
                confirmed_facts=len(memo.confirmed_facts or []),
                source_count=len(collection.documents),
            )
        )
        ctx.record_agent({"agent": self.name})

        # Stash the full memo so Report and QA agents can use it
        ctx.trace["_memo"] = memo
        ctx.trace["_documents"] = collection.documents
        return ctx
