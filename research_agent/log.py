"""Custom PROGRESS log level (J5.0b+).

PROGRESS sits between INFO (20) and WARNING (30) at level 25.
Use it for high-value milestone lines that operators want to see
without the per-chunk / per-evidence noise of INFO.

Examples of PROGRESS messages:
  - Agent starting / completing
  - Profile loaded
  - Web search query fired
  - Evidence count summary
  - Report written

Import and use:
    from research_agent.log import PROGRESS
    LOGGER.log(PROGRESS, "message %s", value)

Or use the module-level helper:
    from research_agent.log import progress
    progress(LOGGER, "message %s", value)
"""

from __future__ import annotations

import logging

PROGRESS = 25
logging.addLevelName(PROGRESS, "PROGRESS")


def progress(logger: logging.Logger, msg: str, *args: object) -> None:
    """Emit a PROGRESS-level log message."""
    logger.log(PROGRESS, msg, *args)
