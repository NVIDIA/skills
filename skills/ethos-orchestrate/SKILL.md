---
name: ethos-orchestrate
description: >-
  Entry point for an AI agent's ethos.md: a durable snapshot of what the agent
  is for, what it must not do, and what success looks like.
  Routes a request to explore and capture intent (ethos-explore), or write or
  edit the file (ethos-write), including renaming an Eval Author 0.1.0 ETHOS.md.
  Owns the standard every ethos sub-flow follows. Use when the user asks "help
  me write an ethos.md", "does my agent have an ethos.md?", "update my agent's
  ethos.md", or when you need to pick between the ethos sub-flows.
triggers:
  - help me write an ethos.md for my agent
  - capture what my agent is supposed to do
  - does this repo have an ethos.md
  - update my agent's ethos.md
  - review my agent's ethos.md
  - rename my ETHOS.md to ethos.md
  - what is an ethos.md
not-for:
  - ethos-explore (use to scan the agent and interview the user for intent)
  - ethos-write (use to render, validate, save, and review the file from agreed answers)
  - eval-author (use to build, audit, or run evaluations; it calls this skill when it needs an ethos.md)
compatibility: >-
  Routing reads the local checkout only. Sub-flows write one local ethos.md;
  validation needs Python 3.10+ and ethos-write's standard-library script.
  Works offline.
license: Apache-2.0
user-invocable: true
allowed-tools: Read Grep Glob
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Ethos orchestrate

## Purpose

Help a developer create and maintain their agent's `ethos.md`. Route each
request to the narrow sub-flow that owns it, and carry what is already known
into that sub-flow so the user is never asked twice.

## Choose the entry route

Locate an existing file first: the user's path, otherwise `ethos.md` at the
repository root or at the root of the selected agent's package, otherwise an
`ETHOS.md` written by Eval Author 0.1.0.

```bash
find . -maxdepth 4 -iname ethos.md -not -path '*/.git/*' -not -path '*/node_modules/*' 2>/dev/null | grep . || echo "no ethos.md yet"
```

| Starting situation and request | Route | First action and deliverable |
|---|---|---|
| No ethos.md and the user wants one | [ethos-explore](../ethos-explore/SKILL.md) | Show the path ahead below, then scan and interview. Explore hands off to ethos-write. |
| An ethos.md exists and the user names specific changes | [ethos-write](../ethos-write/SKILL.md) | Edit the named sections in place, validate, and review. No new interview. |
| An ethos.md exists and intent may have drifted, or the user wants it revisited | [ethos-explore](../ethos-explore/SKILL.md) | Read the file as prior answers; interview only for what changed or is missing. |
| An Eval Author 0.1.0 `ETHOS.md` | [ethos-write](../ethos-write/SKILL.md) | Rename it to `ethos.md`; the contents already match the schema. Then handle any requested edits. |
| The user wants to know what ethos.md is, or whether one exists | This skill | Explain it, or report what was found and whether it validates. Offer the next route; do not start writing. |

When several apply, take the earliest row that matches. A request phrased as
"write the ethos.md" with no prior interview in this conversation still goes
through ethos-explore: writing needs intent the code cannot supply.

## Show the path ahead

For a fresh request to create an ethos.md, open with this checklist using
literal `- [ ]` checkboxes, marking the first step **We're here**:

- [ ] **Explore the agent** — read its code and docs and draft what they establish.
- [ ] **Confirm intent** — answer a few questions the code cannot: purpose, principles, and vision.
- [ ] **Review the draft** — see the whole ethos.md and correct it.
- [ ] **Save and validate** — write the file, check it, and confirm the final contents.

End with "Ready to start?" and wait. On return visits, resume the earliest
unfinished step instead of repeating the opening.

## The standard

**ethos.md records intent the user confirmed, not intent you inferred.**

- Code, prompts, configs, and docs establish what the agent does today. They
  draft `Tools`, `Harness`, `Evaluation Setup`, and similar sections.
- Only the user establishes purpose, principles, vision, trade-offs,
  and constraints. An inferred answer is a draft to confirm,
  never a filled section.
- An honest `_(none)_`, recorded in `Open Questions`, beats a plausible guess.
- The bundled validator establishes structure: front matter, schema version,
  and the fourteen required sections. A file that validates can still be wrong;
  the user's review establishes that it is right.

## Vocabulary

| Term | Meaning |
|---|---|
| ethos.md | The agent's intent snapshot: front matter plus fourteen required `##` sections, schema version 1 |
| Section | One required `##` heading from the [template](../ethos-write/references/templates/ethos.md). Extra headings are allowed and preserved |
| Inferred | Drafted from source; shown to the user as an option, never final |
| Confirmed | Answered or accepted by the user in this conversation |
| Legacy file | An `ETHOS.md` written by Eval Author 0.1.0; same schema, earlier file name |

## Sub-flows

Load the selected sub-flow skill when its step begins and follow it.
This file carries the standard and the routing; the sub-flow carries the steps.

| Sub-flow | Use it to |
|---|---|
| [`ethos-explore`](../ethos-explore/SKILL.md) | Scan the agent's code and docs, then ask one intent question at a time; hand the agreed answers to ethos-write |
| [`ethos-write`](../ethos-write/SKILL.md) | Render the template, save `ethos.md`, run the validator, show a gut-check, and get the user's confirmation; edit an existing file or rename an Eval Author 0.1.0 `ETHOS.md` |

## Boundaries

- **Write only the ethos.md.** Never edit the agent's code, prompts, configs,
  or evals. The file goes at the user's path or the repository root, and
  nothing is committed automatically.
- **Preserve what exists.** Keep custom sections, extra front-matter keys, and
  `created_timestamp` on edits.
- **Capture intent without authorizing application changes.** Do not ask
  optimization or change-permission questions in the interview, draft review,
  or completion checkpoint. Creating an ethos.md implies no follow-up to
  optimize or change the agent. If the user later requests such work, its
  workflow owns the permission decisions.
- **Keep runtime safety requirements.** Record behavioral constraints and
  approval requirements for actions the agent takes during operation, such as
  issuing refunds. These describe intended behavior, not permission to edit
  the application. Also preserve already-confirmed organizational safety and
  review policies and their known owners without reopening permission questions.
- **No run settings.** Spend caps, experiment counts, and per-run limits belong
  to the tool that runs them, not to ethos.md.

## Communicating with the user

Do not assume the user knows the term. Before the first question, explain in a
sentence or two that ethos.md records what the agent should do, what it should
avoid, and what a good result looks like, and that people and agents working on
the agent read it first. Make the explanation concrete for this agent.

Ask one question per message, prefer multiple choice grounded in what the scan
found, and always allow "I don't know". When reporting an existing file, lead
with whether it validates and what it says the agent is for, then the gaps.

## Prerequisites

A local checkout of the agent, or a description of a planned one. Validation
needs Python 3.10+.

## Limitations

ethos.md is a snapshot of intent; it does not prove the agent behaves that way.
Measuring that belongs to evaluation tools such as Eval Author.

## Troubleshooting

- Several candidate files or agents: ask which agent the ethos.md describes.
- Validator unavailable: check the structure by reading the file against the
  template, and say the check was not run.
- User cannot state a concrete Role: stop, explain that the file is not useful
  without one, and offer to resume later.

## Examples

```text
Request: "Help me write an ethos.md for this support bot."
Route: ethos-explore, after the path-ahead checklist.
Deliverable: a reviewed, validated ethos.md at the repository root.

Request: "Add a no-PII rule to Constraints in our ethos.md."
Route: ethos-write, editing Constraints in place.
Deliverable: the updated file, validated, with the change shown for review.

Request: "We still have the ETHOS.md Eval Author made last month."
Route: ethos-write, renaming the legacy file.
Deliverable: ethos.md with unchanged contents, validated.
```
