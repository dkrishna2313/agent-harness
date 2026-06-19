Build a Python CLI application named `dc_power_agent`.

Purpose:
The application analyzes local documents related to AI data center infrastructure and produces a Markdown research memo.

Requirements:

Technology:

* Python 3.12
* Typer for CLI
* Pydantic v2 for schemas
* Anthropic SDK for Claude API access
* pypdf for PDF extraction
* pytest for testing

Project structure:

dc_power_agent/
├── dc_power_agent/
│   ├── cli.py
│   ├── agent.py
│   ├── claude_client.py
│   ├── loaders.py
│   ├── prompts.py
│   ├── schemas.py
│   ├── evaluator.py
│   └── markdown.py
├── tests/
├── sources/
├── outputs/
├── README.md
├── AGENTS.md
└── pyproject.toml

CLI Example:

python -m dc_power_agent.cli 
"Explain NVIDIA Rubin architecture and implications for AI data centers" 
--sources ./sources 
--out ./outputs/rubin.md

Capabilities:

1. Load source documents from a directory.
2. Support:

   * .pdf
   * .md
   * .txt
3. Extract text from documents.
4. Pass extracted text to the agent workflow.
5. Write final output as Markdown.

Create all files and tests but use mock LLM calls initially.
