---
name: ethos-explore
description: Captures what an AI agent is supposed to do — its purpose, boundaries, and success criteria — as a durable snapshot for anyone building, evaluating, reviewing, or changing it. Explores the user's codebase and docs first, then asks one intent question at a time for what source cannot supply. Hands its answers to ethos-write, which writes ethos.md. Use over generic brainstorming when the goal is an agent's ethos.md.
triggers:
  - design my agent
  - what should my agent do
  - help me think through the agent
  - I want to build an agent
  - agent design
  - explore the agent
  - figure out what my agent needs
  - write an ethos.md for my agent
  - onboard my existing agent
not-for:
  - ethos-write (use to write or edit the ethos.md file once explore is done)
  - ethos-orchestrate (use to choose between exploring and writing an ethos.md)
  - superpowers:brainstorming (use for design work that is not about an agent's intended behavior)
compatibility: Dialogue-driven with read-only pre-flight (`ls`, `find`, `Read`); one intent question per message after the codebase scan; at least three questions covering Purpose & Outcomes, Principles, and Vision; safe under any sandbox; works offline; output is a structured conversation handed to ethos-write.
license: Apache-2.0
user-invocable: true
allowed-tools: Read, Glob, Grep, Bash
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Explore an agent's intended behavior

Capture what the agent should do. Product mission and user goals matter more
than implementation inventory. The output of this skill is the data that
`ethos-write` writes into `ethos.md` — the durable contract that people and agents
read before they build, evaluate, review, or change this agent. Underspecified
input here leads every one of them to guess at intent from code and traces.

This skill is **explore first, then a mandatory intent interview.** Scan the
codebase and docs, infer implementation-shaped fields, then ask the user for
intent source cannot own. Do not skip the interview because the scan looked
complete, and do not skip a question because you think you already know the
answer. Do not dump the draft ethos.md until that interview has asked at least
three questions and received a reply to each.

The division of labor is: **infer implementation, ask for intent.** Almost
everything about what the agent *is* can be read from source. Nothing about what
the developer *wants* can be. Constraints, trade-offs, principles, mission, and
vision live in the user's head, so they are the questions worth spending
attention on.

Creating an ethos.md captures intent and authorizes no changes to the agent's
application. Do not ask for optimization or change permissions, whether as a
separate topic or inside another section. Permission to change code, prompts,
models, tools, or evals belongs to the workflow performing that work if the user
requests it. Creating an ethos.md implies no such follow-up. Preserve already
confirmed organizational safety policies and their known owners, including
required security review before shipping a tool with write access. Recording
an existing restriction does not grant permission or require a new permission
interview.

## The schema you are filling

The ethos.md has five base front-matter fields, plus optional `updated_timestamp`,
and fourteen body sections. Every body section is required. One field is a
quality gate for handoff: `ethos-write` is blocked until `Role` is concrete.
For any other canonical section with nothing to say, write `_(none)_` rather
than dropping the heading.

The fourteen headings are a floor, not a ceiling. Extra `##` headings and extra
YAML front-matter keys are allowed. Keep them if the user adds them. Do not
strip custom sections to make the file look "strict."

**Front matter**

| Field | Required | Guidance |
| :---- | :---- | :---- |
| `schema_version` | yes | Always `1` for new files. `ethos-write` fills this at write time. |
| `name` | yes | Canonical agent name. Use the directory or workflow name if obvious; ask if not. |
| `created_timestamp` | yes | ISO 8601 timestamp for when the ethos.md is created. `ethos-write` fills this at write time. |
| `author` | yes | Human or agent that created the ethos.md. `ethos-write` fills this from the current author context when known; ask only if ambiguous. |
| `owner` | optional | Accountable human or team for runtime action approvals named in `Behavior` or `Constraints`; ask only when such an approval needs an owner. Also retain a known owner of an already-confirmed organizational safety policy, without opening a change-permission interview. |
| `updated_timestamp` | conditional | Set on edits. Omit on first write. `ethos-write` fills this. |

**Body sections** (in canonical order)

| # | Section | What "good" looks like |
| :---- | :---- | :---- |
| 1 | Role | One concrete sentence describing the role this agent plays. Example: "answer IT helpdesk questions about VPN, password reset, and software access." Vague answers ("help with stuff") are useless downstream. |
| 2 | Purpose & Outcomes | Two labeled parts. **Mission:** why the agent exists, what user value it provides, and the product or workflow context it serves — not a restatement of implementation mechanics. **Outcome:** the external result it is accountable for, with the measurable target and who owns that number. A mission with no outcome cannot be measured; an outcome with no mission gets improved in the wrong direction. Say so plainly when the agent is internal tooling with no business metric. |
| 3 | Scope | Audience, 3-6 task categories, expected in-scope work, and explicit out-of-scope work/non-goals. |
| 4 | Tools | Tools, APIs, and knowledge sources the agent can use, or "Prompt-only." Group related helpers by capability or source. Capture only behaviorally important purpose, credentials/scopes, side effects, freshness, and expected failures. |
| 5 | Harness | How this agent actually runs: the loop, tool use, and runtime. Write what is true of this agent. Do not pick a named harness from a catalog, and do not treat a framework import as a requirement. Write `_(none)_` if you cannot describe how it runs. |
| 6 | Behavior | Behavioral rules and boundaries: refusal/escalation policy, tone, safety/compliance requirements, accepted limitations, and known non-goals. Hard external limits belong in `Constraints`. |
| 7 | Principles | How the agent should decide when no rule in `Behavior` covers the case: which way to err on an ambiguous request, what it protects even at some cost to the answer, and whose interest wins when the user and the business disagree. Two or three concrete judgment calls. "Helpful, harmless, and honest" is not an answer — it is true of every agent. Write `_(none)_` if there is no judgment call beyond `Behavior`. |
| 8 | Success Criteria | What good production behavior looks like, independent of current evals: mission-level outcomes, quality standards, escalation quality, accuracy expectations, and examples of success. Rank them when some matter more. |
| 9 | Trade-offs | How to choose when desired outcomes conflict. Needs three things: hard gates never traded away, a priority order over the rest (quality, latency, cost, reliability), and unacceptable compromises. "Balance quality and cost" is not an answer. Write `_(none)_` if the user cannot rank them. These are behavioral priorities, not permission to optimize the application. |
| 10 | Constraints | Hard external bounds on the agent's operation: approved providers/models/regions, data residency and handling, compliance obligations, production cost ceilings and latency SLOs, and runtime actions that need human sign-off. Retain already-confirmed organizational safety and review policies. Give the current measured figure next to a ceiling when you know it. Usually organizational, which is why the code cannot supply them. Write `_(none)_` if unconstrained. Do not collect application-change permissions here. |
| 11 | Evaluation Setup | Current validation setup: how to run it, what datasets/checks it uses, what scorers/metrics measure, pass/fail thresholds, and known coverage gaps relative to the success criteria. If no eval suite exists, say so explicitly. |
| 12 | Metric Semantics | What ambiguous or load-bearing metric and telemetry field names actually mean, and which claims they do not support. Write `_(none)_` when every name means exactly what it says. |
| 13 | Vision | Where the agent is headed: an intention beyond today's job, plus one or two concrete use cases it should grow into but does not serve yet. That last part marks what `Scope` excludes *for now* rather than on principle. Write `_(none)_` rather than pasting a dated backlog. |
| 14 | Open Questions | Open facts that affect safe use or evaluation of the agent. Write `_(none)_` when there are none. Remove items once answered. Missing application-change permissions are not an Ethos gap. |

The ethos.md records durable intent, not a running list of observed failures.
Track known issues and recurring failure patterns where the team tracks bugs.

## Pre-flight

Check whether an ethos.md already exists for this agent. Look for `ethos.md`,
including an `ETHOS.md` written by Eval Author 0.1.0, at the repository root and next to
the agent's package. If one is present, ask the user whether they want to edit
the existing ethos.md or start over. If they want to edit, route to `ethos-write`
directly.

```bash
find . -maxdepth 4 -iname ethos.md -not -path '*/.git/*' -not -path '*/node_modules/*' 2>/dev/null | grep . || echo "no ethos.md yet"
```

If the repository has an earlier agent spec or design doc, read it as prior
answers. Scan the codebase, then still run the intent interview for anything
it never answered.

## Step 1 — Explore the codebase

Time-box this to ~5 minutes of tool use. Read first, ask second. Greenfield
projects will turn up nothing here, which is fine — move to the intent
interview and ask for Role first.

1. **Find agent definitions and entry points.** Look for agent configs,
   workflow YAMLs, Python agent builders, system prompts, skills, and tool
   definitions:

   ```bash
   find . -maxdepth 5 -type f \( -name "agent.yaml" -o -name "agent.yml" \) 2>/dev/null
   find . -maxdepth 4 -type f \( -name "*.workflow.yaml" -o -name "*.workflow.yml" \) 2>/dev/null
   find . -maxdepth 4 -type d -name "agents" 2>/dev/null
   ```

   Then use `Glob` / `Grep` to find `langgraph`, `StateGraph`,
   `create_react_agent`, `system_prompt`, skills, MCP servers, and tool
   definitions. Treat any harness or framework name you find as a clue about
   how the agent runs, not as a value the ethos.md must pick.

2. **Find design context.** Look for `README.md`, `AGENTS.md`,
   product/design/planning docs, launch notes, and anything in `docs/`. Read
   docs that describe goals and user value before implementation details when
   they look agent-relevant.

3. **Map findings to schema fields.** As you scan, hold a running mental
   table of what you can fill from the code/docs. Be honest about confidence:
   "inferred from system prompt" is different from "confirmed by the user."

4. **Note the model only as a constraint.** The ethos.md has no `Model`
   section. The agent's config already carries the model in use, and it
   changes without touching the ethos.md. If the scan shows a provider
   allowlist or pinned model family, carry it into `Constraints` as a draft to
   confirm.

   Typical inferences per field:

   - **name** — directory name, workflow name, or top-level package name.
   - **Role** — first paragraph of README, system prompt preamble, or
     top-level docstring. Often partial; usually needs user confirmation.
   - **Purpose & Outcomes** — mission from product docs, README motivation,
     system prompt preamble, or workflow context. Treat that as a draft to
     confirm, never as a skip. The outcome (the business objective) is rarely
     in code; even when a README names a metric, ask the user to confirm it.
   - **Scope** — audience from docs or prompts; categories from enumerated
     capabilities or named tool clusters; in/out boundaries from prompt rules.
   - **Tools** — from `@tool` decorators, tool registries,
     `create_react_agent(tools=[...])`, retrieval/corpus config, or API clients.
     Group low-level helpers when they share credentials, side effects,
     freshness, and failure modes.
   - **Harness** — infer from how the agent actually runs: adapter
     configuration, workflow YAML, service entrypoints, CLI commands,
     Dockerfiles, notebooks, or deployment configs. Capture behaviorally
     relevant capabilities, not a catalog name. If you cannot see how it runs,
     write `_(none)_`.
   - **Behavior** — system prompt rules ("never give medical advice"),
     refusal/escalation policy, tone, accepted limitations, and non-goals.
   - **Success Criteria** — desired production outcomes, product goals,
     quality standards, escalation quality, accuracy expectations, and examples
     of successful behavior.
   - **Trade-offs** — not in the code; ask in the intent interview.
   - **Constraints** — mostly not in the code, but scan for partial evidence
     worth confirming: a pinned gateway base URL or provider allowlist, region
     settings, redaction or PII middleware, timeout and token ceilings, and
     compliance notes in docs. Treat findings as a starting draft to confirm,
     never as the complete list.
   - **Evaluation Setup** — Makefile targets, scripts, CI config, eval YAMLs,
     metric definitions, thresholds, and coverage notes.
   - **Metric Semantics** — from scorer definitions, metric names in eval
     configs, and telemetry field names. Fill only the entries whose meaning is
     genuinely ambiguous from the name. Write `_(none)_` when every name is
     obvious.
   - **Principles** — not in the code; ask in the intent interview even if a
     prompt hints at one. A prompt is the implementation, not the intent
     behind it.
   - **Vision** — not in the code. Roadmap docs, design notes, or a README's
     future-work section are speculative until the user confirms them. Ask.
     Write `_(none)_` rather than guessing.
   - **Open Questions** — TODOs / FIXMEs in agent-adjacent code that
     affect safe use or evaluation.

## Step 2 — Ask intent questions

The scan told you what the agent is. It cannot tell you what the developer
wants. Run a real interview for that intent before you show a draft ethos.md.

This step is a hard gate. Do not present the full draft, and do not hand off
to `ethos-write`, until you have asked at least three questions and received a
reply to each. "The scan filled everything" and "asking feels unnecessary"
are not reasons to skip it.

### What to ask

After the scan, split fields into two piles:

- **Inferred** — implementation-shaped fields you can draft from source:
  `Tools`, `Harness`, `Evaluation Setup`, `Behavior` copied from prompts,
  `Metric Semantics` from scorer names.
- **Intent** — fields source cannot own: `Purpose & Outcomes`, `Principles`,
  and `Vision` always; then `Constraints`, `Trade-offs`, and
  a ranked `Success Criteria` when the repo only has eval wiring.

Walk the intent pile. Skip a topic only when **this conversation** already
answered it. A README, prompt, or roadmap is not a substitute for a user
reply on the always-ask topics below. Keep every other unanswered intent
topic on the list. Greenfield work adds `Role`, `Scope`, and whether the
agent is prompt-only before that intent list.

Question count is at least three, then however many remaining intent topics
still need a reply. Prefer fewer sharp questions over a long checklist, but
never go below three.

### Always ask these three

Code and docs can look like they already answered them. They did not, until
the user says so. Ask even when you have a high-confidence draft. Confirming
an inference counts; silently filling the section does not.

1. **`Purpose & Outcomes`** (business objective). Confirm the mission and the
   result the agent is accountable for. If you inferred a metric from a
   README, show it and ask whether that is the target.
2. **`Principles`**. The judgment call when `Behavior` runs out. Reject
   generic virtues. Ask what this agent should do that a careless version
   of it would not.
3. **`Vision`**. Where the agent is headed, or an explicit `_(none)_`. Do
   not copy a future-work section into Vision without asking. Speculative
   vision is the usual failure.

If `Role` is missing or vague, ask that first so the rest of the interview
is grounded. It does not replace one of the three always-ask topics.

Ask about remaining topics in this order, skipping any this conversation
already answered:

1. `Constraints` — hard bounds on the agent's operation, including runtime
   action approvals where needed.
2. `Trade-offs` — behavioral hard gates, then a ranking of desired outcomes.
3. Ranked `Success Criteria` when the repo only has eval wiring.

Do not ask which parts may be optimized or changed, including in a follow-up,
checkpoint, or draft review. Keep runtime action safety requirements: for
example, an agent may need approval before issuing a refund. That describes
its intended behavior; it does not grant permission to change its code.

Do not collect run limits. A spend cap or experiment count for one evaluation
run belongs to the tool that runs it. If the user volunteers
one, keep the standing policy and drop the number.

### How to ask

Follow this Q&A pattern:

- **One question per message.** If a topic needs more depth, ask a
  follow-up in the next message. Do not batch four intent questions.
- **Prefer multiple choice.** Ground the options in what you found.
  Always include a way to reject the list (`Something else` or `I don't
  know`). Open-ended is fine when a lettered list would fake certainty.
- **Wait for the reply** before the next question.
- **Accept "I don't know"** and move on. Write `_(none)_` and record the
  gap in `Open Questions`. A fabricated constraint is worse than a missing
  one.
- **Confirm inferred bounds.** If the scan found a pinned gateway or
  redaction middleware, show it and ask whether it is a real boundary.
- **Confirm inferred intent.** If the scan produced a plausible `Purpose &
  Outcomes`, `Principles`, or `Vision`, show the draft as an option. Do not
  treat that draft as the answer.

Example (one message, then stop):

> The README says this agent exists to "cut ticket volume." Is that the
> business objective I should record in `Purpose & Outcomes`?
>
> A. Yes — ticket volume is the outcome (tell me the target if you have one)
> B. Close, but the real objective is … (tell me)
> C. No business metric — internal tooling
> D. I don't know

A later Constraints example:

> The scan shows a pinned model gateway and no cost ceiling in docs.
>
> Which hard bounds should `Constraints` record?
>
> A. The agent must use that gateway; no other bounds
> B. That gateway is required, plus a production cost or latency ceiling (tell me the number)
> C. No hard bounds — write `_(none)_`
> D. Something else

Reject generic virtues for `Principles`. "Helpful, harmless, and honest"
gives a downstream reader nothing. Ask for the judgment call: what does
this agent do that a careless version of it would not? If nothing comes
back, write `_(none)_`.

Push once on a non-answer for `Trade-offs`. "Balance quality and cost" is
not decidable. One concrete follow-up usually produces a ranking. If it
does not, record the ambiguity and move on.

If the user provides outside context, read it and update the inferred
draft before the next question.

### Red flags — stop and ask

These mean you skipped the interview:

- You are about to paste the full ethos.md and have asked fewer than three
  questions
- You filled `Purpose & Outcomes`, `Principles`, or `Vision` from the scan
  without a user reply
- Several intent questions in one message
- Handing off to `ethos-write` with fewer than three Q&A replies in this
  conversation

| Excuse | Reality |
| --- | --- |
| The scan filled everything | Code never owns intent. Ask at least three questions. |
| I already know Purpose / Principles / Vision | Those answers belong to the user. Confirm the draft. |
| Asking feels unnecessary | Unnecessary-looking questions are the ones that catch speculative Vision. |
| Batching questions saves turns | One question gets a real answer. A dump gets shallow ones. |
| The draft review can collect intent | Review is for corrections. Ask intent first. |
| The user looks busy | One short multiple-choice question is enough to start. |

## Step 3 — Present the draft ethos.md

After the interview, present the entire ethos.md at once — every field, with
inferred values shown inline and remaining gaps as `_(none)_`. Then ask
one question:

> "Here's the full ethos.md I'd write. Tell me what to change, and I need a
> concrete Role before I can hand off to `ethos-write`."

Show the rendered ethos.md inline in markdown (one `##` section per field,
same shape as the on-disk file). For fields you defaulted, note the
default in parentheses so the user knows they can override:

- `Tools: Prompt-only.` *(default — say so if the agent needs tools)*
- `Purpose & Outcomes` / `Success Criteria` inferred from implementation
  *(say so if there is outside context to incorporate)*

When the user could not answer a section, write `_(none)_` and add the
gap to `Open Questions`. Say "`Constraints`: _(none)_ — operational limits
remain unspecified" so the consequence is visible while it is still cheap to
fix. An unspecified limit grants no application-change permission.

Do not walk the schema field by field in this review. Do not restart the
intent interview here unless the reply surfaces a contradiction.

Do not use public-facing shorthand like `AUT` or "agent under test" in the
rendered ethos.md. Use "this agent" for the agent being specified. Use
"target agent" only where the agent's purpose is explicitly to inspect or
modify another agent.

Allowed follow-ups after the draft:

1. The **hard-required** `Role` is missing or vague — ask for it before
   handoff.
2. The user's reply surfaces a contradiction that needs one targeted
   clarification (for example they say "drop the search tool" but the
   codebase shows the agent depends on it).

## Step 4 — Hand off

After the user's reply, apply the corrections and check the one hard
precondition:

1. **Role** is a concrete one-sentence answer (not "help with stuff").

If it is still unresolved, ask for it in one final message and stop until the
user provides it. Do not hand off with `Role` blank — the artifact is useless
downstream even though the parser accepts it.

Every body section heading must be present before handoff. Write `_(none)_`
for honest gaps, and note them in `Open Questions` when they affect safe use
or evaluation. Do not drop a heading.

If the Role quality gate is satisfied, announce the handoff in one line
("Handing off to `ethos-write` to write `ethos.md`") and trigger it.

## If the user pushes back

- **They want to change one or two fields.** Apply the edits, re-show the
  changed sections only, ask "good now?", proceed.
- **They want to redo the whole thing.** That usually means the codebase
  scan got something fundamentally wrong. Re-scan with their correction in
  mind, then re-present once.
- **They keep changing their mind on Role.** Stop. Tell them the agent will
  not be useful until they can write one concrete sentence and offer to
  come back later. Do not loop on rewording.
- **They skip remaining intent questions.** You already asked at least
  three, including `Purpose & Outcomes`, `Principles`, and `Vision`. Treat
  skipped topics as `I don't know`: write `_(none)_`, record them in
  `Open Questions`, and say once what it costs. Then present the draft. Do
  not re-open the full interview.
- **They say "just write it" before three questions.** Ask the next
  always-ask question anyway. After they answer or decline each of the
  three, continue.

## Gotchas

- **"You decide" means commit to the default and announce it.** Example:
  "I'll record `Tools: Prompt-only.` Tell me to change it if not." Never
  silently fill in.
- **Tool over-spec is the most common error.** Users ask for a search tool
  when prompt-only would work. Probe: "Do you have evidence the model alone
  fails on these?" If no, drop the tool.
- **Tool and harness inventory should be compressed.** Do not create one row
  per helper method when several helpers share the same source, credential,
  side effect, freshness, and failure mode. Group them and call out only the
  differences an evaluator or reviewer needs to know.
- **Mission before mechanics.** An ethos.md that only says how the current code is
  wired is not good enough. If goal context cannot be found in the codebase or
  docs, say the mission is inferred from implementation and still ask the user
  to confirm `Purpose & Outcomes`. A guessed business objective is worse than
  `_(none)_`.
- **Never infer a constraint you cannot verify.** A guessed provider allowlist
  or cost ceiling misstates the agent's operating limits. Ask, or write
  `_(none)_` and record the gap in `Open Questions`.
- **Trade-offs make competing outcomes decidable.** A priority order tells
  the agent and its evaluators which outcome matters when quality, latency,
  cost, or reliability conflict. It grants no optimization permission.
- **"No behavior constraints" usually means "I haven't thought about it."** Probe
  once: "Anything that should never appear — names, phone numbers, competitor
  mentions?" One probe, then move on.
- **Do not skip the codebase scan even when the user seems eager to
  answer questions.** Spending the first five minutes reading makes the
  interview shorter and sharper. Asking something the codebase already
  answers loses trust immediately.
- **Do not skip the intent interview even when the scan looks complete.**
  Ask at least three questions. Always ask `Purpose & Outcomes`,
  `Principles`, and `Vision`, even when you think you already know. One at
  a time. Multiple choice when you can.
- **Do not invent Vision from a backlog.** A future-work bullet is
  speculative until the user confirms it. Ask, or write `_(none)_`.
- **A framework import is not a harness.** Describe how the agent runs. Do not
  map an import onto a harness name. Imports such as `langchain`,
  `langgraph`, `crewai`, `autogen`, or `pydantic_ai` say nothing about that.
- **Record intent, not runtime config.** Describe the desired harness behavior
  and artifacts without asking the user to design a raw runtime config. Config
  files stay with the agent's implementation.
- **Do not invent Known Issues fields.** Known issues and recurring failure
  patterns belong in the team's issue tracker, not in the ethos.md.
