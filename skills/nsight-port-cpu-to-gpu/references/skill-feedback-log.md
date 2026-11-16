# Skill Feedback Log

Append-only log of user corrections and skill gaps. Used later
to improve this skill — not for runtime porting state.

**File:** `{source_dir}/.nsight-ai-port/skill_feedback.json`

Create it in Phase 0 from
[assets/skill_feedback-template.json](../assets/skill_feedback-template.json).
Git-commit it with the rest of `.nsight-ai-port/` so feedback
survives across sessions.

## When to Append an Entry

Log **after** you apply the user's correction (or acknowledge
the mistake). Append when any of these occur:

| Trigger | Example |
|---------|---------|
| `user_correction` | "Don't port both — only the k-driver" |
| `user_rejection` | High-engagement reject with comments |
| `plan_override` | User edits targets/order in the plan |
| `skill_gap` | You followed the skill but the user says the guidance led you wrong for this codebase |
| `repeat_failure` | Same class of mistake after a prior log entry |

Do **not** log routine clarifications (paths, names, tolerances)
unless they reveal missing or misleading skill guidance.

## Entry Schema

Append one object to the `entries` array:

```json
{
  "id": "20250603-143022-001",
  "timestamp": "2025-06-03T14:30:22Z",
  "phase": "plan",
  "trigger": "user_correction",
  "context": "Selected driver_k and column_work as separate
    translation targets for the solver's continuity update.",
  "user_feedback": "Only target the k-loop driver; column_work
    should be a device routine, not a kernel.",
  "skill_section": "planning-from-trace.md — Loop Hierarchy",
  "action_taken": "Removed column_work from functions[];
    added to device_utilities[]; added strided refactor task.",
  "lesson": "Split k vs (j,i) loops across call stack → one
    target + device routine, never two kernels.",
  "tags": ["loop-hierarchy", "continuity", "planning"]
}
```

### Field guide

| Field | Required | Notes |
|-------|----------|-------|
| `id` | yes | `{YYYYMMDD-HHMMSS}-{seq}` — seq increments per day |
| `timestamp` | yes | ISO 8601 UTC |
| `phase` | yes | `setup`, `plan`, `refactor`, `device_utils`, `translate:{function}`, `integrate:{function}`, `profile:round_N`, `wrap-up` |
| `trigger` | yes | See table above |
| `context` | yes | What you did or assumed before correction |
| `user_feedback` | yes | User's correction, quoted or paraphrased |
| `skill_section` | no | Skill file/section that misled or was silent |
| `action_taken` | yes | What you changed after the feedback |
| `lesson` | yes | One sentence — written for a skill maintainer |
| `tags` | no | Short strings for later filtering |

## How to Append

1. Read `.nsight-ai-port/skill_feedback.json`.
2. Build the new entry (generate `id` from current time + seq).
3. Append to `entries` — never edit or delete prior entries.
4. Write the file back.
5. Continue the porting work.

On first correction in a project, set `project` to a short label
(e.g. `solver-continuity` or the repo directory name).

## Usage & Cost Logging (Periodic Snapshots)

The wrap-up retrospective should report **real** runtime, token, and
agent-turn totals — not guesses. Your harness tracks these live, but
that in-memory counter is lost if the session crashes and is
restarted. So persist periodic snapshots to disk; wrap-up aggregates
them.

**Where:** append-only `usage_log` array in
`.nsight-ai-port/skill_feedback.json`.

**When to append a snapshot:**

- Immediately **before each `git commit`** in the porting workflow —
  the existing per-step cadence (setup, refactor, device utils, each
  translate chunk, verify, integrate, nvtx, profile rounds). Most of
  these commits stage `.nsight-ai-port/`, so the snapshot is committed
  in that same commit and survives a later crash.
- Once more at **wrap-up**, alongside `session_summary`.

**Snapshot schema** (append one object per snapshot):

```json
{
  "id": "20250603-143022-001",
  "timestamp": "2025-06-03T14:30:22Z",
  "run_segment": 1,
  "phase": "translate:src/foo.f90:bar",
  "commit": "pending",
  "counter_basis": "segment-cumulative",
  "elapsed_wall_clock_sec": null,
  "input_tokens": null,
  "output_tokens": null,
  "cache_read_tokens": null,
  "total_tokens": null,
  "agent_turns": null,
  "cost_usd": null
}
```

**Field rules:**

| Field | Required | Notes |
|-------|----------|-------|
| `id` | yes | `{YYYYMMDD-HHMMSS}-{seq}` — same scheme as `entries[]` |
| `timestamp` | yes | ISO 8601 UTC |
| `run_segment` | yes | See crash-resilience note below |
| `phase` | yes | Same vocabulary as `entries[]` `phase` |
| `commit` | no | `"pending"` when written; optionally backfill short SHA later |
| `counter_basis` | yes | `"segment-cumulative"` or `"delta"` (see below) |
| usage fields | no | Numeric fields in the schema above (tokens, turns, wall-clock, cost) |

- **Only fill what you can actually observe** about your own usage.
  If your harness does not expose a field, leave it `null`. **Never
  estimate or fabricate** runtime, tokens, turns, or cost — a missing
  number is better than a wrong one.
- `run_segment` distinguishes continuous runs. On a fresh start with
  an empty `usage_log`, use `1`. On **resume** (session was
  restarted), read the existing `usage_log`, take the max
  `run_segment`, and use max+1. This matters because a restarted
  harness usually resets its own counters to zero, so numbers are only
  cumulative *within* a segment.
- `counter_basis` — `"segment-cumulative"` if your numbers accumulate
  from the start of this `run_segment` (typical), or `"delta"` if each
  snapshot reports only the change since the previous one. Set it so
  wrap-up can aggregate correctly.

**How to append:** same as `entries[]` — read the file, append to
`usage_log`, write it back. Never edit or delete prior snapshots.

### Aggregating at wrap-up

When writing `session_summary.stats.usage` (and presenting totals to
the user):

- For `"segment-cumulative"` snapshots: per `run_segment`, take the
  **last** snapshot (highest cumulative values); **sum across
  segments** for the totals. This correctly stitches together a run
  that crashed and resumed.
- For `"delta"` snapshots: sum all snapshots directly.
- If a field was `null` in every snapshot, report it as `null` /
  "not available" in the retrospective — do **not** estimate.

```bash
# Inspect raw snapshots:
jq '.usage_log[] | {run_segment, phase, total_tokens, agent_turns,
  elapsed_wall_clock_sec}' .nsight-ai-port/skill_feedback.json
```

## Retrieval (for skill maintainers)

To review feedback across a porting effort:

```bash
cat .nsight-ai-port/skill_feedback.json
# or filter lessons:
jq '.entries[] | {phase, lesson, tags}' .nsight-ai-port/skill_feedback.json
```

Maintainers aggregate `lesson` and `skill_section` fields from
many projects to decide what to change in SKILL.md and the
reference guides.

---

## Session Retrospective (Wrap-up — Required)

At **every successful Phase 6 wrap-up**, write `session_summary` to
`.nsight-ai-port/skill_feedback.json` **before** telling the user the
session is complete.

This is separate from incremental `entries[]` (corrections mid-run).
`session_summary` may be **written or replaced once at wrap-up** — it
is the only field besides `project` that is not append-only.

Read the schema and placeholders in
[session-retrospective-template.md](session-retrospective-template.md).

### What to cover (answer honestly for this project)

1. **What happened** — one paragraph: setup, flavor,
   problem, targets ported, verification, outcome.
2. **What worked well in the skill** — concrete mechanisms (state
   files, non-negotiables, guides) that helped.
3. **What can be improved in the skill** — gaps, misleading guidance,
   missing branches. Cross-reference any `entries[]` already logged
   (e.g. library hiccups).
4. **Discretionary choices** — places two agents could diverge without
   doing anything wrong (NVTX names, compiler choice, test inputs,
   Makefile wiring, etc.). For each: what you chose, why the skill
   is loose, what stricter guidance would improve repeatability.
5. **Nsys extraction** — how you
   pulled relevant-but-not-overwhelming data from nsys outputs; whether
   the approach scales to very large codes; specific skill gaps for
   the profiling/triage step.

Also populate `stats` (commits, functions ported, profile rounds, and
`usage` aggregated from `usage_log` per the rules above).

### User-requested retrospective

If the user asks for a summary after the fact (e.g. "what happened,
what worked, what should improve in the skill, where did you make
choices, how does nsys extraction scale?"), **refresh or write**
`session_summary` with the same structure, present it in chat, and
save to `skill_feedback.json`.

### Git

Commit the updated `skill_feedback.json` with the final porting commit
or immediately after wrap-up:

```bash
git add .nsight-ai-port/skill_feedback.json
git commit -m "nsight-port: session retrospective"
```

### Tell the user

After writing `session_summary`, say where it lives and give a short
markdown summary in chat. If `entries` is non-empty, mention both:

> Session retrospective written to
> `.nsight-ai-port/skill_feedback.json` (`session_summary`).
> N incremental feedback entries also logged.

Do **not** skip the retrospective because the port was small or
everything passed — small runs still produce useful skill gaps.
