# Card authoring prompt (round 0 — self-bootstrap)

This is the one-time "compilation" prompt. The driver runs it automatically when
`cards/` is empty. Fill in the `<...>` placeholders for your workflow, or embed
it in the driver like the AutoML example does.

In the study, this session took ~10 minutes and wrote 4 cards (20.6k chars)
from three skills. Every later session runs the cards and never opens a skill.

---

You are setting up an efficient repeated-execution harness for a staged
workflow. Read the relevant skills under `<path to your skill bank>`, inspect
the workspace inputs, then author the stage cards. Do NOT run the workflow or
any docker/pip command in this session; only read and write cards.

Run parameters:
<Everything the workflow needs, made concrete for THIS machine: dataset paths,
container images (and whether pulls are allowed), GPU count, batch size,
metrics, output directory convention, anything pre-approved. Be exhaustive —
whatever you leave out, the execution sessions will burn tokens rediscovering.>

Card contract:
Write EXACTLY these N files into `<kit>/cards/` (terse markdown; every command
copy-paste executable for THIS host; a fresh session must run them without
reading any skill):

- `00-<first-stage>.md`: <what it does>. Append '<stage> ok' to $RD/progress.log.
- `10-<second-stage>.md`: <what it does>. Launch long jobs DETACHED
  (nohup ... &), then end turn. Idempotent: when the job's output already
  exists, record its result in $RD/state.json and append the ok line.
- ...
- `90-<final-stage>.md`: collect results, write $RD/results.md, append
  'done ok', and write DONE to $RD/DONE.marker.

Rules for every card:
- Sessions are fresh (no memory), so cards carry ALL knowledge: exact paths,
  exact flags, known quirks and their fixes.
- Check $RD/commands.log before deriving any command; reuse what worked.
- Detached launches end the turn (the driver waits on the job, not the agent).
- Include a failure branch: what the known failure modes are and how to
  recover, so a later session doesn't re-investigate.
- If a long-running job is still working when your card is invoked, append the
  previous stage's ok line again and end your turn (tells the driver to keep
  waiting).
- Be terse.
