Implement Step C: source-grounded citations.

Goal:
Make the Markdown memo cite source documents for major claims.

Requirements:

1. Evidence IDs
- Add a stable evidence_id to each EvidenceItem.
- Format can be simple, e.g. E001, E002, E003.
- Preserve source_document and evidence_snippet.

2. Citation format
In memo sections, cite claims using:
[Source: filename.pdf, Evidence: E001]

Example:
Rubin is positioned as a rack-scale AI platform. [Source: nvidia_rubin.pdf, Evidence: E001]

3. Source Notes
In the Source Notes section, render evidence items grouped by document:
- Evidence ID
- Claim
- Evidence snippet
- Category
- Relevance
- Confidence

4. Evaluator sensors
Add warning-mode checks for:
- Confirmed Facts section has no source citations
- Power Implications has no source citations
- Cooling Implications has no source citations
- Networking Implications has no source citations, if present
- Rack Architecture Implications has no source citations, if present
- citations reference unknown evidence IDs

5. Claude prompting
Update prompts so Claude is instructed to use evidence IDs when writing the memo.
Claude should not invent source names or evidence IDs.
It should only cite evidence items provided by the harness.

6. Trace
Add evidence IDs to trace JSON.

7. Tests
Add tests for:
- evidence IDs are generated
- markdown includes evidence IDs
- evaluator warns on missing citations
- evaluator warns on unknown evidence IDs
- trace includes evidence IDs

Keep warning mode only.
Do not add web search.
Do not add vector databases.
Do not change CLI flags unless necessary.