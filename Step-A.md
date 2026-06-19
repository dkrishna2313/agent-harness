Add observability and trace generation to the existing research_agent project.

Goal:
Before integrating Claude, improve visibility into each harness step.

Requirements:

1. Add a --debug flag to the CLI.

Example:

python -m research_agent.cli \
  "Explain NVIDIA Rubin architecture and implications for AI data centers" \
  --sources ./sources \
  --out ./outputs/rubin.md \
  --debug

When --debug is used, print a concise run summary to the terminal:
- question
- source directory
- output path
- number of documents loaded
- each document name and extracted character count
- number of evidence items generated per document
- total evidence items
- memo sections generated
- evaluation warning count
- trace file path if written

2. Add trace JSON output.

For every run, create a trace file next to the Markdown output.

Example:
- Markdown output: ./outputs/rubin.md
- Trace output: ./outputs/rubin.trace.json

Trace JSON should include:
- timestamp
- question
- source_directory
- output_path
- documents_loaded
- total_characters_extracted
- documents:
  - filename
  - path
  - character_count
  - evidence_item_count
- evidence_items:
  - claim
  - source_document
  - evidence_snippet
  - category
  - relevance
  - confidence
- memo_sections
- evaluation_warnings
- mock_mode: true

3. Add additional warning-mode sensors in evaluator.py.

Add warnings for:
- zero documents loaded
- any document has zero extracted characters
- fewer than 3 documents loaded
- fewer than 10 total evidence items
- fewer than 3 documents have evidence items
- missing important topic coverage:
  - architecture
  - power
  - cooling
  - networking
  - rack architecture

These should remain warnings only. Do not block output.

4. Add or update tests.

Tests should cover:
- trace JSON file is created
- trace JSON includes question, documents, evidence items, warnings
- --debug prints expected summary fields
- evaluator warns on low evidence count
- evaluator warns on missing topic coverage

5. Keep the implementation simple.

Do not add Claude integration yet.
Do not add web search.
Do not add vector databases.
Do not change the existing memo format unless necessary.
Preserve the current working behavior.