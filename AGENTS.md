# Project Purpose

Build a local AI research harness focused on AI data center infrastructure.

Primary research domains:

- NVIDIA Rubin
- NVIDIA Vera
- NVIDIA Blackwell
- AI factories
- Data center power systems
- Power distribution
- Cooling infrastructure
- Networking infrastructure
- Rack architecture

# Current Version

Version: v1

Goals:

- Local files only
- Claude API
- Markdown output
- Warning-mode evaluation

Non-goals:

- Web search
- Vector databases
- Multi-agent systems
- Memory systems
- Dashboards

# Workflow

Question
→ Research Plan
→ Evidence Extraction
→ Synthesis
→ Evaluation
→ Markdown Memo

# Output Requirements

Every memo must contain:

- Executive Summary
- Confirmed Facts
- Inferences
- Power Implications
- Cooling Implications
- Open Questions
- Source Notes
- Evaluation Warnings

# Engineering Principles

- Keep implementation simple.
- Prefer readable code over clever code.
- Use Pydantic schemas.
- Add tests for core logic.
- Fail gracefully.
- Log useful diagnostics.

# Domain Guidance

For infrastructure questions, prioritize:

1. NVIDIA primary sources
2. OCP specifications
3. Infrastructure vendor documentation
4. Independent technical analysis

Clearly distinguish:
- Confirmed facts
- Reasoned inferences
- Speculation