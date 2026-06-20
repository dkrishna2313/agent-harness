"""QAAgent – quality checks on the memo before report generation (J5.0b)."""

from __future__ import annotations

from .base import FunctionalAgent
from .context import AgentContext


class QAAgent(FunctionalAgent):
    """Skeleton QA agent: checks fact count and flags evaluation warnings."""

    def _execute(self, context: AgentContext) -> AgentContext:
        memo = context.trace.get("_memo")
        issues: list[str] = []

        if memo is not None:
            facts = memo.confirmed_facts or []
            warnings = memo.evaluation_warnings or []
            if len(facts) < 3:
                issues.append(f"Low confirmed fact count: {len(facts)}")
            if warnings:
                issues.append(f"{len(warnings)} evaluation warning(s) from engine")
        else:
            issues.append("No memo available for QA")

        status = "success" if not issues else "warning"
        summary = "QA checks passed." if not issues else f"QA found {len(issues)} issue(s)."

        # Detailed note
        context.qa_notes.append(
            self._make_note(status=status, summary=summary, issues=issues)
        )

        # Unified history entry
        self._record(context, status=status, summary=summary, issues=issues)
        return context
