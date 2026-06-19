Implement J1.4: Contradiction Normalization.

Problem:

The contradiction engine is comparing values that represent different metrics.

Example:

300 GW capacity target

vs

13 GW/year licensing throughput

These are not contradictory.

Requirements:

Before contradiction detection:

Extract:

- quantity
- unit
- metric_type

Examples:

300 GW
metric_type = capacity_target

13 GW/year
metric_type = licensing_throughput

Only compare evidence if:

metric_type matches
OR
comparison scope is equivalent.

Examples:

capacity_target
vs
capacity_target

allowed

licensing_throughput
vs
capacity_target

not allowed

Add trace fields:

metric_type_a
metric_type_b

If metric types differ:

skip contradiction generation.

Success criteria:

Current SMR contradiction C001 disappears.