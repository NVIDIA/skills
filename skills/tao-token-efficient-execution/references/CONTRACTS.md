# Kit contracts — the on-disk state that replaces chat history

Every contract here exists so a fresh session (or the driver, which is plain
bash) can answer "where are we?" without any conversation memory.

## Run directory (`$RD`)

One directory per run, created under `$WS/results/`. Everything a session
learns that later sessions need MUST land in `$RD`. The driver selects the
newest run dir matching the pack's glob that is newer than its launch marker —
runs never share state.

## Progress marks — how the driver routes

Two equivalent conventions; a pack uses exactly one:

- **Simple packs** (`progress.log`): each completed stage appends one line,
  `<stage> ok` (failures append `<stage> FAIL`). The driver routes on the last
  `ok` line only — error lines never advance routing, and a trailing `FAIL`
  halts the driver (no auto-retry; an operator decides).
- **Skill-tooling packs** (`loop_log.jsonl`): when the application skill ships
  its own commit machinery (e.g. DEFT's `commit_stage.py`, which validates
  evidence before writing), ALL state writes go through it and the driver
  routes on the last `status=="ok"` entry. Cards never hand-roll state writes
  that the skill's tooling owns.

## `STAGE_DONE <card-id>` — explicit termination

Every card ends with: print exactly `STAGE_DONE <card-id>` as the final
message and stop. Measured failure this prevents: models that cannot end a
turn burn hundreds of messages per session. The harness adapter enforces a
hard tool-call budget (`PI_KIT_TURN_BUDGET`) as the backstop.

## `commands.log` — reuse, don't re-derive

The recorder adapter appends every substantive executed command (with a
timestamp) to `$RD/commands.log`. Guard-blocked commands are never recorded.
The driver injects the tail into each session's prompt; cards instruct the
executor to reuse recorded commands instead of re-deriving them. This is the
single biggest defense against small-model drift: the correct command exists,
so the model's job is to run it, not to reinvent it.

## Environment contract

Cards reference ONLY driver-exported constants (`$WS`, `$RD`, `$ITER`,
container images, mount strings, script paths). Two hard rules, both from
measured failures:

- The executor's shell is fresh per tool call — variables set in one call do
  not survive to the next. Anything that must persist is either a driver
  export or a file in `$RD`.
- Cards never contain `<placeholders>` the executor must fill by searching.
  Exact paths, exact flags, one copy-paste command per step.

Per-host values (workspace, venv, model, images) come from
`~/.tao-kit/kit.env`, sourced by every pack driver. Credentials are never
written anywhere — drivers check presence of the API key variable and abort
with the export instruction if missing. Pi keeps provider credentials in the
harness process but strips them and arbitrary startup hooks from bash tool
subprocesses. See the adapter README for the isolated-worker requirement.

## Driver obligations

1. **Route** on the last committed ok mark (see above), mapping stage → card.
2. **Wait** on real activity (container CPU%, runner processes), not on the
   agent — cards launch long jobs detached and end the turn. Idle zombie
   containers must not count as work (check CPU, not existence).
3. **Halt** on committed errors and on N consecutive no-progress rounds.
   Check failure status before any completion marker. Never auto-retry a
   committed failure. Exhausting the round cap must return nonzero; only
   verified completion returns zero. For DEFT, loop_stop alone is insufficient:
   the driver must prepare the inference handoff and pass the completion audit.
4. **Snapshot** state into the prompt (progress tail, commands.log tail) so
   the fresh session starts with everything it needs and nothing more.
5. **Write nothing into the skill bank checkout** — sessions and driver logs
   go to `~/.tao-kit/<pack>/`.

## Guards — block dead ends before tokens burn

Guards are advisory-with-teeth: they block a tool call and return a reason
that tells the model WHY and WHAT TO DO INSTEAD. Framework guards (turn
budget, near-duplicate-command loop breaker, no-fabricated-results
verification, destructive-command safety net) ship in the adapters and are
env-configurable. Workflow guards (GPU quirks, disk headroom, container CLI
quirks) are added one per discovered failure mode — see the adapters' README
for real examples and the pattern to copy.
