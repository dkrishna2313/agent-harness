"""Entry point for developer debug utilities.

Invoked as:
    python3 -m functional_agents.debug <command> [OPTIONS]

Currently registered commands:
    extract-experiment  — controlled extraction strategy comparison
"""

from .cli import app

app()
