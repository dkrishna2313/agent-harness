"""Reasoning Target — the compatibility seam for Decision Domains (J10.1).

A ``ReasoningTarget`` is the unit a downstream agent reasons over. Today the
pipeline is operationally centered on ``context.question`` (a single research
question derived from ``decision_model.research_questions[0]``). J10.1 introduces
this abstraction WITHOUT changing behavior: the accessor returns exactly one
target derived from ``context.question``.

Later milestones (J10.2/J10.3) will let the accessor return one target per
Decision Domain / decision stream. Downstream agents that read through the
accessor will then move to Decision Domains with no further changes at the call
site. This module has no dependencies on AgentContext, so it can be imported
freely without cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Target kinds. Only the legacy kind is emitted today; the decision-domain kind
# is reserved for J10.2+ so consumers can branch on it ahead of the flip.
KIND_RESEARCH_QUESTION = "research_question"
KIND_DECISION_DOMAIN = "decision_domain"


@dataclass
class ReasoningTarget:
    """A single thing a downstream agent reasons over (J10.1).

    In legacy mode there is exactly one, derived from ``context.question``. The
    decision-domain fields are present now (nullable) so future targeting has a
    stable shape to populate rather than a schema change later.
    """

    id: str
    title: str
    kind: str = KIND_RESEARCH_QUESTION
    question: str = ""
    decision_domain_id: str | None = None
    decision_domain_title: str | None = None
    evidence_requirements: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "question": self.question,
            "decision_domain_id": self.decision_domain_id,
            "decision_domain_title": self.decision_domain_title,
            "evidence_requirements": list(self.evidence_requirements),
        }


def reasoning_targets_diagnostics(
    targets: list[ReasoningTarget], *, source: str
) -> dict[str, Any]:
    """Lightweight trace summary for the reasoning targets (J10.1 / J10.3).

    J10.3 adds a ``kinds`` histogram so multi-target (Decision Domain) runs are
    visible. Existing fields (count, primary_kind, source) are unchanged.
    """
    kinds: dict[str, int] = {}
    for t in targets:
        kinds[t.kind] = kinds.get(t.kind, 0) + 1
    return {
        "count": len(targets),
        "primary_kind": targets[0].kind if targets else None,
        "source": source,
        "kinds": kinds,
    }
