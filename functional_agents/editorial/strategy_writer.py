"""StrategyWriter — PH11.4 Editorial Writer.

Consumes EditorialBrief.strategy_narrative (when present) and populates
EditorialManuscript.strategic_direction with presentation-ready content.

Design constraints:
- Optional: section is skipped when brief.strategy_narrative is None.
- Improves communication, never reasoning.
- Does not invent facts or draw conclusions not in strategy_narrative.
- Does not read StrategyTrace, AgentContext, or other reasoning artifacts directly.
- All other manuscript sections remain untouched.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from .editorial_brief import EditorialBrief
from .editorial_manuscript import EditorialManuscript
from .editorial_writer import EditorialWriter

LOGGER = logging.getLogger(__name__)


class StrategyWriter(EditorialWriter):
    """Writer for EditorialManuscript.strategic_direction (PH11.4).

    No-ops when brief.strategy_narrative is None (missing-trace path).
    When brief.strategy_narrative is present, populates:
      - paragraphs: winning_position, winning_mechanism
      - bullet_groups[0]: evaluation criteria with scores
      - bullet_groups[1]: key assumptions
      - bullet_groups[2]: success conditions
      - bullet_groups[3]: failure modes
      - tables[0]: alternatives considered table
    """

    section_name: ClassVar[str] = "strategic_direction"
    optional: ClassVar[bool] = True

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    def write(
        self,
        brief: EditorialBrief,
        manuscript: EditorialManuscript,
    ) -> EditorialManuscript:
        """Populate manuscript.strategic_direction from brief.strategy_narrative.

        Returns manuscript unchanged when brief.strategy_narrative is None.
        """
        sn = brief.strategy_narrative
        if sn is None:
            LOGGER.debug("[StrategyWriter] strategy_narrative absent — section skipped")
            return manuscript

        sec = manuscript.strategic_direction
        if sec is None:
            LOGGER.warning(
                "[StrategyWriter] manuscript.strategic_direction is None — section skipped"
            )
            return manuscript

        # Paragraphs: winning position and mechanism (authoritative theory prose)
        paragraphs: list[str] = []
        if sn.winning_position:
            paragraphs.append(sn.winning_position)
        if sn.winning_mechanism:
            paragraphs.append(sn.winning_mechanism)

        # Bullet group 0: evaluation criteria with per-criterion scores
        criteria_bullets: list[str] = []
        for crit in sn.evaluation_criteria:
            score = sn.criterion_scores.get(crit, 0.0)
            criteria_bullets.append(f"{crit}: {score:.2f}")
        # Append evaluation strengths to criteria group
        for s in sn.winner_evaluation_strengths[:4]:
            criteria_bullets.append(f"+ {s[:180]}")

        # Bullet group 1: key assumptions (up to 8)
        assumption_bullets = [a[:200] for a in sn.assumptions[:8]]

        # Bullet group 2: success conditions (up to 6)
        condition_bullets = [c[:200] for c in sn.success_conditions[:6]]

        # Bullet group 3: failure modes (up to 6)
        failure_bullets = [fm[:200] for fm in sn.failure_modes[:6]]

        # Bullet group 4: strategic choices (up to 6), if any
        choices_bullets = [sc[:200] for sc in sn.winner_strategic_choices[:6]]

        bullet_groups: list[list[str]] = [
            criteria_bullets,
            assumption_bullets,
            condition_bullets,
            failure_bullets,
            choices_bullets,
        ]

        # Table: alternatives considered — no title (renderer heading already labels this)
        tables: list[dict[str, Any]] = []
        if sn.alternatives:
            rows = []
            for alt in sn.alternatives:
                label = alt.recommended_option_title or alt.theory_id
                # Prefer weaknesses; fall back to residual_risks
                negatives = alt.weaknesses[:2] or alt.residual_risks[:2]
                negatives_str = "; ".join(negatives) or "—"
                rows.append([
                    alt.theory_id,
                    label,
                    f"{alt.score:.2f}",
                    alt.confidence or "—",
                    negatives_str,
                ])
            tables.append({
                "title": "",  # heading already rendered by _s_strategic_direction
                "headers": ["Theory ID", "Option", "Score", "Confidence", "Key Weaknesses"],
                "rows": rows,
                "notes": "",
            })

        # Subtitle: key selection facts
        parts: list[str] = []
        if sn.winner_option_title:
            parts.append(sn.winner_option_title)
        parts.append(f"Score: {sn.winner_score:.2f}")
        if sn.overall_confidence:
            parts.append(f"Confidence: {sn.overall_confidence}")

        sec.paragraphs = paragraphs
        sec.bullet_groups = bullet_groups
        sec.tables = tables
        sec.subtitle = " | ".join(parts)

        return manuscript
