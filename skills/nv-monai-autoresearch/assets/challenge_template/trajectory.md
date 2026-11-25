<!-- miagent-template-source: assets/challenge_template/trajectory.md -->
<!-- miagent-template-contract: active; keep the miagent-template-source loaded for creation/material updates; reload after resume or compaction -->
# Decision Trajectory

Append-only index of decisions that changed the challenge trajectory. Record
consequential agent decisions and concise references to consequential human
steering. Keep the full human guidance in the artifact where it applies and
link that owning artifact here.

- Do not store verbatim chain-of-thought, chat, scheduler history, or heartbeat
  logs.
- Record an inspectable summary: decision, decisive evidence/alternatives, and
  resulting action.
- Omit routine status, retries, and repairs that do not change a decision.

Human steering may be intuitive or preference-based. Use a faithful short
summary and `not stated` where no reason was supplied; do not invent a formal
justification. Agent decisions should link the evidence that supports them.

- Use increasing IDs: `T0001`, `T0002`, ...
- Preserve prior rows; correct a factual error in place only when explicit.
- For an append, read linked entries and the table tail needed for the next ID.
- Read the full trace only for reconstruction or audit.

## Decision Trace

| ID | Time | Actor | Decision / Steering | Reasoning Summary | Evidence / Owning Artifact | Applied Action |
|---|---|---|---|---|---|---|
| T0001 | YYYY-MM-DD HH:MM TZ | human / agent / joint |  | Concise rationale and material alternatives considered | Markdown links |  |
