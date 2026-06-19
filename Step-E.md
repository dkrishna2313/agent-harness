Add a lightweight regression evaluation suite.

Create evals/questions.yaml with 5 sample data center infrastructure questions.

Add a CLI command or script:
python -m dc_power_agent.eval_runner --sources ./sources --evals ./evals/questions.yaml --out ./outputs/eval_report.md

For each question:
- run the agent
- capture warnings
- capture detected topics
- capture evidence count
- capture citation count
- write a Markdown eval report

Do not block output. Keep this simple.