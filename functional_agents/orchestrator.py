"""Orchestrator – drives agents in sequence and owns the context lifecycle (J5.0a)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from research_agent.profile import load_profile, DomainProfile
from research_agent.research_object import create_research_object

from .context import AgentContext
from .planner_agent import PlannerAgent
from .evidence_agent import EvidenceAgent
from .qa_agent import QAAgent
from .report_agent import ReportAgent

LOGGER = logging.getLogger(__name__)


class Orchestrator:
    """Runs the functional agent pipeline end-to-end.

    Profiles:
        The first profile in *profile_names* is the execution profile passed
        to the research engine.  All profiles are recorded in the trace.
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

        # Load execution profile (first in list)
        self._domain_profile: DomainProfile | None = None
        if profile_names:
            try:
                self._domain_profile = load_profile(profile_names[0])
                LOGGER.info("Execution profile: %s", profile_names[0])
            except FileNotFoundError as exc:
                LOGGER.warning("Could not load profile %r: %s", profile_names[0], exc)

        # Load all profiles (for context record — not used in engine yet)
        self._all_profiles: list[DomainProfile] = []
        for name in profile_names:
            try:
                self._all_profiles.append(load_profile(name))
            except FileNotFoundError:
                LOGGER.warning("Profile not found: %r", name)

    def run(self, question: str) -> AgentContext:
        """Execute the full functional agent pipeline and return the final context."""

        # Create initial context
        ctx = AgentContext(
            question=question,
            profiles=self._profile_names,
        )

        # Create research object before pipeline starts
        ctx.research_object = create_research_object(
            question=question,
            profile_name=self._profile_names[0] if self._profile_names else None,
            profile_source="cli_argument",
            sources_dir=self._sources_dir,
            web_search=False,
            mock_mode=self._client is not None and getattr(self._client, "is_mock", False),
        )

        LOGGER.info("Profiles loaded: %s", self._profile_names)
        LOGGER.info("Execution profile: %s", ctx.execution_profile)

        # Build and run agents in sequence
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
