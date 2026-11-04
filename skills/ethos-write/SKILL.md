---
name: ethos-write
description: Writes an AI agent's ethos.md — a durable snapshot of its purpose, intended behavior, boundaries, and success criteria — from the answers gathered by ethos-explore. Renders the schema-v1 template, validates the front matter and required markdown sections, saves the file in the agent's repository, and reviews it with the user. Also edits an existing ethos.md. Use over generic planning skills for an agent's ethos.md.
triggers:
  - write the ethos
  - save the design
  - capture what we agreed
  - persist the agent design
  - update the ethos.md
  - write agent ethos
not-for:
  - ethos-explore (use to gather the design before writing the ethos.md)
  - ethos-orchestrate (use to choose between exploring and writing an ethos.md)
compatibility: Writes one local markdown file, `ethos.md` at the repository root unless the user names another path; validation needs Python 3.10+ and the bundled standard-library script; works offline; safe under any sandbox; idempotent if user confirms overwrite.
license: Apache-2.0
user-invocable: true
allowed-tools: Read, Write, Edit, Bash
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Write an agent's ethos.md

Turn the answers from `ethos-explore` into a durable artifact. The ethos.md is
the contract that people and agents read before they build, evaluate, review,
or change this agent. Without it, each of them has to re-ask everything, and
nothing records what the agent is supposed to do.

Creating or reviewing an ethos.md does not authorize application changes or imply
that changes will follow. Do not ask for permission to modify prompts, models,
tools, validation code, or workflow structure, or generate sections listing those
permissions. Any workflow that later changes the application
collects the permissions it needs when that work is undertaken. Keep behavioral
constraints and safety boundaries, including approvals the agent needs for its
own actions, in `Behavior` and `Constraints`. Also retain already-confirmed
organizational safety and review policies and their known owners; recording
such a restriction does not grant permission to make a change.

The ethos.md records the **intended** state of the agent, which is not the same as
the implemented state. The codebase already shows what the agent does. This file
is the only place that records what it is supposed to do, what it must never do,
and how to weigh a win on one metric against a loss on another.

## Location

The ethos.md is a local file in the agent's repository, committed alongside
the code it describes. Use the path the user names. Otherwise reuse an
existing ethos.md, including an `ETHOS.md` written by Eval Author 0.1.0. Otherwise write
`ethos.md` at the repository root; in a repository with several agents, put it
at the root of the agent's own package and confirm that path with the user.
The file is the only copy. There is no remote canonical copy to sync.

## Schema version

Write `schema_version: 1`. Every canonical body section is required. The
validator rejects a file missing any of those headings. When you have nothing
to say, write `_(none)_` rather than dropping the section.
Version 1 is the only schema version. When a new version ships, its migration
steps belong in this skill.

The schema is a floor, not a ceiling. Extra `##` headings and extra YAML
front-matter keys are allowed. The parser keeps unknown body sections and
does not fail on unknown front-matter keys. If the user already added custom
sections, preserve them on rewrite. Do not strip custom content to make the
file look strict. This includes a legacy `Change Scope` section: preserve it on
edits unless the user asks otherwise, but do not treat it as authorization for
application changes in this workflow.

Do not invent content to look complete: `_(none)_` in `Constraints` is honest,
while a fabricated bound actively misleads anyone changing the agent. If the user has no
answer yet, write `_(none)_`, record the gap in `Open Questions`, and move on.

ethos.md holds durable intent, so keep run-scoped configuration out of it. A spend
ceiling, an experiment count, or a wall-clock limit for a single evaluation
run belongs to the tool that runs it. If a user offers one, record
the standing policy it implies — a production cost ceiling in `Constraints`, or
who approves an overrun — and leave the run limit itself to that tool's config.

## Hard preconditions

Before writing anything, the answers carried over from `ethos-explore` must
satisfy one non-negotiable. If it is missing or ambiguous, **stop and route back
to `ethos-explore` for that field only** — do not invent a default.

1. **Role** — one concrete sentence describing the role this agent plays. Vague
   answers ("help with stuff", "answer questions") make the artifact useless
   downstream even though the parser will accept them; push back in
   conversation rather than writing a placeholder.

The parser cannot catch a vague `Role`, which is why this skill enforces it
upstream: the user sees a clear gap-question instead of a file that validates
and then helps nobody.

## What you do

1. **Confirm the agent name.** Short and recognizable: `it-helpdesk`,
   `support-triage`, `code-reviewer`. If the user has not named it, propose
   two options based on the role.

2. **Pre-flight: locate the file.** Resolve `ETHOS_PATH` as described in
   [Location](#location). If a file already exists there, ask the user whether
   to edit it or start over. Read an Eval Author 0.1.0 `ETHOS.md` as the existing ethos.md.

   ```bash
   find . -maxdepth 4 -iname ethos.md -not -path '*/.git/*' -not -path '*/node_modules/*' 2>/dev/null | grep . || echo "ethos_new"
   ```

3. **Run a focus check before rendering.** The carried-over answers should be
   mission-led and reviewable, not a raw inventory of implementation details:

   - `Purpose & Outcomes` and `Success Criteria` must explain mission, user
     value, the measurable result, and the success bar. If they only summarize the
     current code, route back to `ethos-explore` to ask whether the user has
     outside context that is not in the codebase. If no such context exists,
     say the section is inferred from implementation.
   - `Trade-offs` must be decidable. "Balance quality and cost" is not usable;
     a priority order with named hard gates is. If the user has not ranked
     anything, ask for the ranking rather than writing a platitude.
   - `Constraints` must be checkable. Prefer "models must come from the
     internal gateway" over "use approved models." This is also where the
     permitted model and provider set lives; there is no `Model` section,
     because the config already records the model in use and it changes without
     touching this file.
   - `Tools` and `Harness` should be concise. For `Harness`, describe how this
     agent actually runs. Do not pick a named harness from a catalog, and do not
     treat a framework import as a requirement. Group related helpers in
     `Tools` by capability or source when they share credentials, side
     effects, freshness, and failure modes. Keep only details that change how
     downstream agents evaluate behavior.
   - Avoid public shorthand like `AUT` or "agent under test." Use "this agent"
     for the agent being specified. Use "target agent" only when this agent's
     job is explicitly to inspect or modify another agent.

4. **Render the ethos.md.** Use the template at
   `references/templates/ethos.md` as the starting point. Substitute
   every section from the `ethos-explore` answers. Set front matter as:
   `schema_version` = `1`, `name` = the canonical agent name,
   `created_timestamp` = current UTC timestamp in ISO 8601 form, and `author` =
   the human or coding agent creating the file. Add `owner` when a human or
   team is accountable for the agent's behavioral or safety approvals named in
   `Behavior` or `Constraints`, including a known owner of an already-confirmed
   organizational safety policy.
   Set `updated_timestamp` on edits, not on first write. Evaluation commands
   live in `Evaluation Setup`, not in front matter. Keep the required section
   headers exactly so the file stays parseable. Extra `##` headings after
   (or among) the canonical fourteen are allowed — keep them. The file is
   lightly validated by the bundled `scripts/validate_ethos.py`, which
   checks front matter, schema version, required sections, and duplicate
   sections. It does not reject unknown headings. Section bodies stay markdown
   for agents and humans to read directly.

5. **Write the file.** Write to `ETHOS_PATH`, creating its directory if needed.
   If an Eval Author 0.1.0 `ETHOS.md` exists in that directory, rename it to `ethos.md`
   before writing, through a temporary name so the rename also works on
   case-insensitive filesystems:

   ```bash
   DIR=$(dirname "$ETHOS_PATH")
   if ls "$DIR" | grep -qx ETHOS.md; then
     mv "$DIR/ETHOS.md" "$DIR/ethos.md.tmp"
     mv "$DIR/ethos.md.tmp" "$DIR/ethos.md"
   fi
   ```

6. **Validate.** Run the bundled validator and surface any warnings to the
   user. A failure means the file is malformed; fix the named problem before
   going on. Warnings are not failures — report them so the user can decide
   whether to fill the gap now.

   ```bash
   python3 "<this skill's directory>/scripts/validate_ethos.py" "$ETHOS_PATH" \
     || echo "ethos_parse_invalid"
   ```

7. **Show a gut-check, then the file.** Before asking the user to read fourteen
   sections, state your impression of this agent in a short paragraph that
   combines `Role`, `Purpose & Outcomes`, `Scope`, and (when they are not
   `_(none)_`) `Principles` and `Vision`. This is a thin slice so the user can
   tell quickly whether the write got the agent right. Do not use shorthand
   like `AUT` or "agent under test."

   Shape:

   > **Gut check.** This is a [role] that exists to [mission / outcome]. It
   > serves [audience] on [in-scope work] and stays out of [out of scope].
   > When the rules run out, it [principle or none]. It is heading toward
   > [vision or none].
   >
   > If that is the wrong agent, say so. Then we can edit before treating
   > this file as signed off.

   Then print the full file contents and ask: "Does this match what we
   agreed? Edit anything you want to change." If the user edits, repeat
   steps 5–7, including a fresh gut-check.

8. **Hand off.** Once confirmed, tell the user where the file is and that it is
   ready to commit. It is now the reference for evaluating, reviewing, or
   changing this agent. When intent changes, edit it in place and set
   `updated_timestamp`.

## Verification

After writing, both must hold:

```bash
# Local file present and non-empty.
test -s "$ETHOS_PATH" && echo "local_ok" || echo "local_missing"

# Passes the bundled validator.
python3 "<this skill's directory>/scripts/validate_ethos.py" "$ETHOS_PATH" \
  && echo "ethos_parse_ok" || echo "ethos_parse_invalid"
```

Do not announce success until `local_ok` and `ethos_parse_ok` both print, the
gut-check has been shown, and the user has confirmed the contents.

## If verification fails

| Symptom | Cause | Recovery |
|---|---|---|
| `local_missing` after write | Wrong working directory or permission denied | Run `pwd`; check the user is in the cloned repo |
| `ethos_parse_invalid` | ethos.md malformed — missing front matter, missing required section, duplicate section, or bad schema version | Read the parser error; fix the named section in place; do not silently work around |
| User says "this is wrong" | ethos.md captured the wrong answers | Edit the relevant section in place; re-validate |
| `ethos-explore` was skipped | User invoked `ethos-write` cold | Route back to `ethos-explore` and return here when the conversation is done |

## What this skill is not

This skill does not write agent configs, runtime YAML, or code, and does not
register or deploy the agent. The ethos.md is the human-readable design;
machine-readable config stays with the agent's implementation.

It also does not encode a scoring function. `Trade-offs` records the
developer's intent — hard gates, priority order, unacceptable regressions — in
prose. Turning that into thresholds and weights belongs to whatever evaluates
or changes the agent, not to this file.

## Gotchas

- **The template is the source of truth for the canonical outline.** Keep the
  required section headings intact. Extra `##` headings are allowed and must
  be preserved. The bundled validator
  rejects missing or duplicate required sections, but it does not reject
  custom headings. Section bodies remain markdown for humans and agents to
  read directly.
- **ethos.md lives with the agent.** Commit it in the agent's repository, next
  to the code it describes, so every reader finds the same file.
- **Role is a hard requirement.** Do not write the ethos.md without a concrete
  one. Route back to `ethos-explore` for that field only.
- **Honest empty answers belong in the section.** Write `_(none)_` for
  `Constraints` or `Trade-offs` when the user has no answer. Do not invent a
  bound. Record the gap in `Open Questions` as well.
- **`Purpose & Outcomes` cannot be implementation-only by accident.** If goal
  context was not found in the codebase and the user did not provide outside
  context, make that provenance clear instead of letting implementation details
  masquerade as mission. A mission with no stated outcome cannot be measured.
- **Keep public terminology clean.** The generated ethos.md is user-facing. Avoid
  `AUT` and "agent under test"; reserve internal shorthand for test harnesses
  and code comments.
- **Do not duplicate issue tracking into the ethos.md.** Known issues and
  recurring failure patterns belong in the team's issue tracker; the
  ethos.md has no `Known Issues` section, and no `Signals` section either — how a
  given consumer reads evidence is that consumer's configuration, not durable
  intent. Record what a metric cannot support in `Metric Semantics`, and what
  should not count as a failure in `Behavior`.
- **This file is the `ethos.md`.** Agents working on this agent read it but
  should not edit it; only the developer and the developer's coding agent do. Treat it
  as a long-lived contract, not a scratch pad.
