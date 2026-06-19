Implement J1.5: Profile-Aware Contradiction Topics.

Problem:

SMR contradiction analysis currently produces:

topic = rack power

This indicates AI data center terminology is still hard-coded.

Requirements:

Move contradiction topic labels into domain profiles.

Example:

AI Data Centers:

- rack power
- cooling
- networking
- resiliency

SMR:

- licensing
- economics
- fuel cycle
- construction
- safety
- grid integration

Contradiction labels should be generated from profile topics.

No AI data center topic should appear in SMR mode.

Success criteria:

SMR traces contain only SMR-related contradiction topics.