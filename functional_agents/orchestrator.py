"""Orchestrator – creates AgentContext, validates it, then runs agents (J5.0b)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from research_agent.profile import DomainProfile, load_profile
from research_agent.research_object import create_research_object

from .context import AgentContext, ContextValidationError
from .planner_agent import PlannerAgent
from .evidence_agent import EvidenceAgent
from .qa_agent import QAAgent
from .report_agent import ReportAgent

LOGGER = logging.getLogger(__name__)


class Orchestrator:
    """Runs the functional agent pipeline end-to-end.

    Profiles:
        The first profile in *profile_names* is the execution profile passed
        to the research engine.  All profiles are recorded in the context.

    Context lifecycle (J5.0b.2):
        1. Build AgentContext with all required fields.
        2. Validate context — raise ContextValidationError on missing fields.
        3. Pass context through agents in sequence.
        4. Return final context.
    """

    def __init__(
        self,
        *,
        profile_names: list[str],
        sources_dir: str | Path = "sources",
        out_path: Path,
        client: Any = None,
        top_evidence: int = 50,
        top_chunks: int = 20,
    ) -> None:
        self._profile_names = profile_names
        self._sources_dir = Path(sources_dir)
        self._out_path = out_path
        self._client = client
        self._top_evidence = top_evidence
        self._top_chunks = top_chunks

        from research_agent.log import PROGRESS

        # Load execution profile (first in list)
        self._domain_profile: DomainProfile | None = None
        if profile_names:
            try:
                self._domain_profile = load_profile(profile_names[0])
                LOGGER.log(PROGRESS, "Execution profile loaded: %s", profile_names[0])
            except FileNotFoundError as exc:
                LOGGER.warning("Could not load profile %r: %s", profile_names[0], exc)

        # Verify all profiles exist (warn on missing; don't abort)
        for name in profile_names[1:]:
            try:
                load_profile(name)
                LOGGER.log(PROGRESS, "Supporting profile loaded: %s", name)
            except FileNotFoundError:
                LOGGER.warning("Supporting profile not found: %r", name)

    def run(self, question: str) -> AgentContext:
        """Build, validate, and execute the full functional agent pipeline."""

        execution_profile = self._profile_names[0] if self._profile_names else ""
        mock_mode = self._client is not None and getattr(self._client, "is_mock", False)

        # Create Research Object before building context (J5.0b — RO is prerequisite)
        research_object = create_research_object(
            question=question,
            profile_name=execution_profile or None,
            profile_source="cli_argument",
            sources_dir=self._sources_dir,
            web_search=False,
            mock_mode=mock_mode,
        )

        # Build context with all required fields populated (J5.0b.1)
        ctx = AgentContext(
            question=question,
            profiles=self._profile_names,
            execution_profile=execution_profile,
            research_object=research_object,
        )

        # Validate before running any agent (J5.0b.7)
        try:
            ctx.validate()
        except ContextValidationError as exc:
            LOGGER.error("Context validation failed: %s", exc)
            raise

        from research_agent.log import PROGRESS
        LOGGER.log(PROGRESS, "AgentContext validated — profiles=%s execution=%s",
                   ctx.profiles, ctx.execution_profile)

        # Build and run agents in sequence (J5.0b.2)
        agents = [
            PlannerAgent(),
            EvidenceAgent(
                sources_dir=self._sources_dir,
                client=self._client,
                top_evidence=self._top_evidence,
                top_chunks=self._top_chunks,
                domain_profile=self._domain_profile,
            ),
            QAAgent(),
            ReportAgent(
                out_path=self._out_path,
                domain_profile=self._domain_profile,
            ),
        ]

        for agent in agents:
            ctx = agent.run(ctx)

        return ctx
