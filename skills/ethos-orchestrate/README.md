<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ethos.md

`ethos.md` is the **intent layer** for a production agent: a Markdown snapshot
of what the agent is for, what it must not do, and what success looks like.
Developers, reviewers, evaluators, and the coding agents working on the agent
read it to detect and resolve divergence between developer intent and agent
execution.

Code shows implementation. ethos.md records mission, constraints, and what
success and failure mean. Creating it grants no permission to optimize or
change the agent's application, and implies no follow-up work. Permission for
an actual change belongs to the workflow performing that change when requested.

These three coding-assistant skills create and maintain an `ethos.md`:

| Skill | What it does |
| --- | --- |
| [`ethos-orchestrate`](SKILL.md) | Entry point. Finds any existing ethos.md and routes the request to explore or write. |
| [`ethos-explore`](../ethos-explore/SKILL.md) | Reads the agent's code and docs, fills in what they establish, then asks one intent question at a time for what they cannot. Hands its answers to `ethos-write`. |
| [`ethos-write`](../ethos-write/SKILL.md) | Writes `ethos.md` from those answers using the [schema-v1 template](../ethos-write/references/templates/ethos.md), validates the front matter and required sections, and reviews the result with you. |

## Why it exists

Critical agent context lives outside the codebase: business objectives,
developer intent, success and failure criteria, and hard constraints. It is
rarely written down. An agent or evaluator that sees only code and traces
cannot tell a real failure from a trade-off you already accepted.

Without an ethos.md, every new agent, review, or eval has to rediscover that
context, and changes align with whatever the traces happen to show. With one,
the same file can steer a new build, guide a review of logs and evals, reveal
divergence from intent, and keep changes inside human-set bounds.

## Scattered context, one contract

Intent arrives in pieces:

- **Codebase:** files, docs, configs, and logic. What the agent does today.
- **Traces and logs:** telemetry, errors, and metrics. How a run looked, not
  whether that was acceptable.
- **Business goals:** objectives, value, and the result the agent is accountable for.
- **Developer intent:** tone, priorities, and judgment calls that never made it
  into a ticket.
- **Constraints:** operating limits, policies, guardrails, and required
  approvals for actions the agent takes at runtime.

`ethos.md` is the portable snapshot of that contract. Write the intended state,
not only the implemented one; where they differ, say so.

## What it unlocks

| It unlocks | Meaning |
| --- | --- |
| **Portable context** | Every developer, reviewer, and agent working on this agent starts from the same file instead of reconstructing intent from the repo. |
| **Shared contract** | Humans and agents agree on purpose, bounds, and what counts as divergence. |
| **Spec-driven creation** | Purpose, goals, and expected behavior are explicit before code exists. |
| **Informed changes** | Anyone changing the agent can assess the proposed behavior against its purpose, priorities, and operating limits. The workflow performing the change handles authorization. |
| **Recursive updates** | Intent clarified while reviewing a change, eval, or finding goes back into the file. A stale ethos.md steers the next change the wrong way. |

Ethos creation does not collect optimization or application-change permissions.
Runtime action approvals still belong in `Behavior` or `Constraints` when they
are part of the agent's intended behavior.

## ethos.md is not `AGENTS.md`

| | `AGENTS.md` | `ethos.md` |
| --- | --- | --- |
| When it matters | Any repository a coding agent should navigate | When the repository, or a package in it, *is* an agent |
| Job | How to contribute: layout, conventions, commands | Intent, goals, and constraints |
| Location | Repository root | Next to the agent |

## What's in the file

YAML front matter (`schema_version`, `name`, `created_timestamp`, `author`, and
optional `owner` and `updated_timestamp`) followed by fourteen required sections.
Section bodies stay human-readable; the headings are the outline other agents
parse. A section with nothing to say contains `_(none)_`.

| Section | Captures |
| --- | --- |
| `Role` | What the agent does, in one concrete sentence. |
| `Purpose & Outcomes` | Why it exists, and the result it is judged by. |
| `Scope` | Who it serves, in-scope work, and boundaries. |
| `Tools` | APIs, tools, and knowledge sources it can use. |
| `Harness` | How the agent actually runs. |
| `Behavior` | Rules, tone, refusals, and policies. |
| `Principles` | How to decide when no rule in `Behavior` covers the case. |
| `Success Criteria` | What good production behavior looks like. |
| `Trade-offs` | Behavioral hard gates, priority order, and unacceptable compromises. |
| `Constraints` | Operating limits and runtime action approval requirements. |
| `Evaluation Setup` | How it is tested and measured. |
| `Metric Semantics` | What metric names mean, and the claims they do not support. |
| `Vision` | Where the agent is headed, beyond today's scope. |
| `Open Questions` | Unknowns to resolve. |

## Create an ethos.md

Install the three skills together:

```bash
npx skills add https://github.com/NVIDIA-NeMo/eval-author/tree/main/team-skills/ethos --agent claude-code
```

Replace `claude-code` with your assistant's identifier. Then, in your agent's
repository, ask your coding assistant:

```text
Use ethos-orchestrate to write an ethos.md for my agent.
```

`ethos-explore` scans the codebase first and asks at least three intent
questions, covering purpose and outcomes, principles, and vision. `ethos-write`
writes `ethos.md` at the repository root, or at a path you name, validates it,
and asks you to review it. Commit it with the agent's code. You can also write
the file by hand from the [template](../ethos-write/references/templates/ethos.md)
and check it with the bundled validator. From the `ethos-orchestrate/` directory
in either a source checkout or an installed set of the three skills:

```bash
python3 ../ethos-write/scripts/validate_ethos.py /path/to/ethos.md
```

To revise an existing ethos.md, ask for the edit directly; `ethos-write` updates the
relevant sections in place and sets `updated_timestamp`. Both skills also find
an `ETHOS.md` written by Eval Author 0.1.0; `ethos-write` renames it to `ethos.md`
when it writes.

## Using ethos.md with Eval Author

Eval Author's first evaluations and coverage audits use ethos.md as their target
and hand off to these skills to create or revise it. Installing Eval Author
(`npx skills add NVIDIA-NeMo/eval-author --skill '*'`) does not install
them; add them with the command above.
