Upgrade the research harness so Source Notes contain actual extracted evidence, not just filenames and character counts.

Current behavior:
- Source Notes only list document names and extracted character counts.

Required behavior:
- For each loaded source document, generate 3-8 source notes.
- Each source note should include:
  - source document name
  - short quoted or paraphrased evidence snippet
  - topic category
  - relevance to the user question
  - confidence: high / medium / low

Update schemas.py:
- Add or revise EvidenceItem with fields:
  - claim: str
  - source_document: str
  - evidence_snippet: str
  - category: str
  - relevance: str
  - confidence: Literal["high", "medium", "low"]

Update agent.py:
- Add an evidence extraction step after document loading.
- In mock mode, create evidence items from document excerpts so the workflow can be tested without Claude.
- Use the first meaningful chunks of each document for mock evidence extraction.
- Preserve warning-mode evaluation.

Update markdown.py:
- Source Notes should render grouped by document.
- Each note should show claim, evidence snippet, category, relevance, and confidence.

Update evaluator.py:
- Warn if zero evidence items are produced.
- Warn if fewer than 3 source documents have evidence items.
- Warn if evidence snippets are empty.

Add tests:
- EvidenceItem schema validation.
- Source Notes rendering includes evidence snippets.
- Evaluator warns on missing evidence.
- Evaluator passes when at least 3 documents have evidence items.

Keep the implementation simple.
Do not add Claude integration yet.
Do not add vector databases.
Do not add web search.