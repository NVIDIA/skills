# Card authoring contract v2 (framework template — workflow-agnostic)

You are compiling a skill into stage cards for a repeated-execution harness.
Read the workflow's skill (and only the references its execution actually
needs), inspect the workspace inputs, then author the cards. Do NOT run the
workflow in the authoring session; only read and write cards.

The v1 contract still applies (fresh sessions have no memory; cards carry all
knowledge; check commands.log before deriving; detached launches end the turn;
be terse; every command copy-paste executable for THIS host). v2 adds the
following rules, derived from measured small-model execution failures. They
are framework rules — none of them may encode workflow-specific content; the
workflow specifics always come from the skill being compiled.

## v2 card-authoring rules

1. **One command per step.** Every step is exactly one copy-paste shell
   command. Never require the executor to assemble a command from fragments,
   substitute `<placeholders>`, or carry a shell variable from one step to the
   next (each tool call runs in a fresh shell; only harness-exported constants
   survive). Multi-action logic goes into a single `bash -c` line or a script.

2. **Prefer the skill's own contract tooling.** If the skill ships stage-commit,
   validation, audit, or state-init scripts, cards MUST route all state/log
   writes and completion claims through them and NEVER hand-roll the same
   effect (no inline python/jq/heredocs into state files). If the skill has no
   such tooling, fall back to the v1 append-a-line contract.

3. **Mechanical state gate.** Each card opens with a single state-inspection
   command (the skill's audit/next-action tool when present, else a fixed
   log-tail command) and a short table mapping its possible outputs to card
   steps. The executor follows the table; it does not reason about where it is.

4. **One kind of work per card.** Launching a long job and analyzing its
   results are different cards. If a stage contains both, split it. A card
   that ends with a detached launch says so explicitly and ends the turn.

5. **Exact artifact paths, including tool quirks.** When a tool nests or
   renames outputs (e.g. writes into an extra subdirectory), the card states
   the exact final path. The executor must never search for outputs.

6. **Explicit termination.** Every card ends with: print exactly
   `STAGE_DONE <card-id>` as the final message and stop. No verification
   loops, no goodbye commands, no summaries after the token.

7. **Failure branches name exact fixes.** Each known failure mode lists the
   exact error signature and the exact corrective command or spec field+value.
   "Investigate the error" is not a failure branch. Unknown failures: commit
   status=error through the skill's tooling (rule 2) and end the turn.

8. **Self-repair stays bounded.** If reality contradicts the card, fix the
   blocker, update the card file (bounded edit), continue. Never edit state
   files to match the card.
