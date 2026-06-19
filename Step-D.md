Implement Step D: question-specific evaluation sensors.

Goal:
Make evaluator.py adapt warning checks based on the user's question.

Requirements:

1. Add a function classify_question_topics(question: str) -> set[str]

Use simple keyword rules for v1.

Topic detection examples:
- power: power, electrical, voltage, utility, grid, UPS, BBU, busway, PDU
- cooling: cooling, liquid, thermal, CDU, heat, chilled water
- networking: networking, network, NVLink, InfiniBand, Ethernet, Spectrum, ConnectX
- rack architecture: rack, NVL72, rack-scale, tray, shelf, cabinet
- backup/resiliency: backup, battery, BBU, UPS, resiliency, redundancy
- operations: operations, commissioning, maintenance, monitoring

2. Based on detected topics, require corresponding memo sections or citations.

Examples:
- If question mentions power, warn if Power Implications is missing or uncited.
- If question mentions cooling, warn if Cooling Implications is missing or uncited.
- If question mentions networking, warn if Networking Implications is missing or uncited.
- If question mentions rack/NVL72, warn if Rack Architecture Implications is missing or uncited.

3. Keep default baseline checks:
- evidence exists
- at least 3 source documents
- citations reference known evidence IDs
- confirmed facts have citations

4. Do not block output.
Warnings only.

5. Add trace field:
question_topics_detected

6. Add debug output:
Question topics detected: power, cooling, networking, etc.

7. Add tests:
- topic classifier detects power terms
- topic classifier detects cooling terms
- topic classifier detects networking terms
- evaluator warns when a required topic section is missing
- evaluator does not warn when irrelevant topic is absent
- trace includes detected topics

Keep implementation simple.
Do not add web search.
Do not add vector databases.
Do not change citation format.