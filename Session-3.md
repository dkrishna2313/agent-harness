Replace mock LLM calls with Anthropic Claude.

Requirements:

Environment variable:
ANTHROPIC_API_KEY

Create claude_client.py.

Functions:

create_research_plan()

extract_evidence()

synthesize_memo()

evaluate_memo()

Use Claude Sonnet as the default model.

Store prompts in prompts.py.

Implement robust error handling.

If Claude fails:

* log error
* continue gracefully
* return useful diagnostic messages

Add unit tests using mocked Anthropic responses.
