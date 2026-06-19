Integrate Anthropic Claude into the existing research harness.

Goal:
Replace mock planning, evidence extraction, and memo synthesis with real Claude calls while preserving the existing workflow, schemas, evaluations, warnings, trace generation, and markdown output.

Requirements:

Environment:

* Read ANTHROPIC_API_KEY from environment.
* Use the Anthropic Python SDK.
* Default model: Claude Sonnet.

Create or update:

* claude_client.py

Functions:

create_research_plan(question, source_texts)

extract_evidence(question, source_texts)

synthesize_memo(question, evidence_items)

Behavior:

1. Research Plan

Claude should produce:

* research_questions
* key_topics
* source_priorities

Return data matching the existing ResearchPlan schema.

2. Evidence Extraction

Claude should extract evidence from loaded source text.

Each EvidenceItem must include:

* claim
* source_document
* evidence_snippet
* category
* relevance
* confidence

Categories should use:

* architecture
* power
* cooling
* networking
* rack architecture
* operations
* other

Evidence extraction should be source-grounded.

3. Memo Synthesis

Claude should generate content for:

* Executive Summary
* Confirmed Facts
* Inferences
* Power Implications
* Cooling Implications
* Networking Implications
* Rack Architecture Implications
* Open Questions
* Source Notes

Preserve the current markdown structure.

4. Error Handling

If Claude fails:

* log the error
* generate an evaluation warning
* continue gracefully
* do not crash

5. Trace Support

Add to trace JSON:

* model name
* Claude request timestamp
* Claude response success/failure
* token usage if available

6. Preserve Existing Features

Keep:

* --debug
* --show-sources
* evaluation warnings
* trace generation
* warning mode

Do not add:

* web search
* vector databases
* memory
* multi-agent architecture

7. Tests

Add mocked tests for:

* successful Claude responses
* Claude failures
* schema validation
* trace generation
* warning generation

The application must continue to run end-to-end and produce a markdown memo using Claude-generated content.
