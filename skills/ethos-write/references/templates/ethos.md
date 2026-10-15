---
schema_version: 1
name: <canonical-agent-name>
created_timestamp: <ISO 8601 creation timestamp, e.g. 2026-06-02T20:00:00Z>
updated_timestamp: <ISO 8601 timestamp of the last edit; omit on first write>
author: <human or agent that created this ethos.md>
owner: <accountable human or team for the agent's behavioral or safety approvals; omit if unowned>
---

# ethos.md: <name>

> This file is the agent's ethos.md — a snapshot of the intended behavior,
> capabilities, validation setup, and boundaries for this agent. People and
> agents read it before they build, evaluate, review, or change this agent.
> Keep it accurate; stale entries here lead them to the wrong conclusions.
> Creating or reviewing this file does not authorize application changes or
> imply that changes will follow. A workflow that changes the application
> collects any required permissions when that work is undertaken.
>
> The layout below is lightly parseable by `ethos-write`: front matter and the
> required `##` section headers are machine-checked, while section bodies stay
> markdown for humans and agents to read directly.
>
> Write the intended state, not just the implemented state. Where the two differ,
> say so. The code already shows what the agent does; this file is the only place
> that records what it is supposed to do.
>
> Every canonical body section is required. Parsing fails if a required `##`
> heading is missing. When you have nothing to say, write `_(none)_` rather
> than dropping the section. An honest empty answer is better than a
> fabricated one.
>
> Extra `##` headings and extra YAML front-matter keys are allowed. Add them
> when your use case needs a field this outline does not cover. The parser
> does not reject custom keys. Do not delete them to look strict.
>
> This file records durable intent, so keep run-specific settings out of it. A
> spend ceiling or experiment count for one evaluation run belongs in that
> tool's own config, not here.
>
> Section rules:
>
> - **Bullet sections** (`Open Questions`): list items only. If the list
>   is empty, write `_(none)_` instead of leaving the section blank.
> - **Labeled-bullet sections** (`Scope`):
>   `- Label: value` lines only. No prose, no blank-line-separated paragraphs.
>   For list-valued labels inside `Scope`, separate items with semicolons, or
>   write `_(none)_`.
> - **Free-form sections** (`Role`, `Purpose & Outcomes`, `Tools`, `Harness`,
>   `Behavior`, `Principles`, `Success Criteria`, `Trade-offs`, `Constraints`,
>   `Evaluation Setup`, `Metric Semantics`, `Vision`): any markdown. `Tools`
>   accepts a markdown table or the literal string `Prompt-only.`

## Role

<one concrete sentence describing the role this agent plays for its users; this
is the fast, human-readable one-liner another agent should remember>

## Purpose & Outcomes

<the mission and the result it is judged by, in that order. Keeping them in one
section is deliberate: a purpose with no outcome cannot be measured, and an
outcome with no purpose gets improved in the wrong direction.

**Mission.** One or two short paragraphs: why the agent exists, what user value
it provides, which product or workflow it belongs to, and the decision or
business context it supports. Use context from outside the codebase when the
user provides it, and say so plainly when this is inferred from code alone.

**Outcome.** The external result the agent is accountable for, stated so someone
outside the team could tell whether it worked. Give the measurable target where
one exists and who owns that number. This is what the agent earns its keep by
improving, and what any change to the agent weighs against cost and latency.

Example: "Ships inside the Acme onboarding flow to get a new developer from
signup to a working API call without human help. Success is cutting median
time-to-first-successful-request from 40 minutes to under 15, without raising
support ticket volume. Owner: growth team."

When the agent is internal tooling with no business metric, say that outright
rather than inventing one.>

## Scope

- Audience: <who talks to it — internal employees, external customers, developers, etc.>
- Categories: <3-6 task buckets, separated by semicolons; e.g. VPN; password reset; software access>
- In scope: <capabilities, user intents, or situations the agent is expected to handle; semicolon-separated or `_(none)_`>
- Out of scope: <capabilities, user intents, or situations the agent should not handle; semicolon-separated or `_(none)_`>

## Tools

<tools, APIs, and knowledge sources the agent can use, or the literal string
`Prompt-only.` if none. Group related helpers by capability or source instead
of listing every low-level method. Capture only behaviorally important purpose,
credentials or scopes, side effects, data freshness, expected failures, and
anything an agent reviewing a trace should know when deciding whether it shows
bad agent behavior or a normal tool/source limitation.>

| Tool or source | Purpose | Credentials/scopes | Side effects | Freshness / expected failures |
|---|---|---|---|---|
| current_datetime | clock for time-sensitive answers | none | none | current at call time |

## Harness

<how this agent actually runs. Describe the loop, how it calls tools, and where
it executes. Write what is true of this agent. Do not pick a name from a
harness catalog, and do not treat a framework import as a requirement.

Useful things to cover when they apply:

- How model turns, tool calls, and stop conditions are orchestrated
- How tools are chosen, executed, and returned
- How context, memory, and session state are kept
- Safety, sandbox, or permission controls around actions
- How you observe and verify a run
- The runtime, such as a service, CLI, notebook, hosted agent, or workflow

Write `_(none)_` if you cannot describe how this agent runs.>

## Behavior

<behavioral rules and boundaries: constraints, refusal and escalation policy,
tone, safety/compliance requirements, accepted limitations, and known non-goals.
Use this to tell agents working on this agent what counts as divergence and
what should not be filed as a failure.>

## Principles

<how this agent should decide when no rule in `Behavior` covers the case. Rules
run out; this section is what a downstream agent falls back on.

Write the judgment calls, not the personality. "Prefer saying I don't know over
a confident guess, because a wrong answer here costs a user their afternoon" is
usable. "Helpful, harmless, and honest" is not — it is true of every agent and
tells a reader nothing.

Two or three of these is plenty. Good candidates:

- Which way to err when the request is ambiguous: ask, assume, or refuse.
- What the agent protects even when it makes the answer worse — a source
  citation, an audit trail, an explicit uncertainty flag.
- Whose interest wins when the user's request and the business goal disagree.

Keep concrete rules in `Behavior` and metric weighting in `Trade-offs`. This
section is only for judgment where neither applies. Write `_(none)_` if there
is no judgment call beyond those two.>

## Success Criteria

<what good production behavior looks like for this agent, independent of the
current eval suite. Lead with mission-level outcomes and user goals, then
capture quality standards, escalation quality, accuracy expectations, and
representative examples of successful behavior. Rank them if some matter more
than others.

Keep this section about quality and outcomes. Latency and cost belong
elsewhere: a production ceiling is a `Constraints` entry, and how to weigh speed
against quality is a `Trade-offs` entry.>

## Trade-offs

<how the agent should prioritize when desired outcomes conflict. State the
behavioral priorities that guide the agent and its evaluators.

State three things:

1. **Hard gates.** Qualities that are never traded away, at any gain elsewhere.
   Correctness and safety usually live here.
2. **Priority order.** Rank the remaining goals — task quality, latency, cost,
   reliability, coverage — so competing outcomes can be judged.
3. **Unacceptable compromises.** Outcomes that must not get worse even when
   another improves, and how much variation is acceptable if some slack is fine.

Example: "Answer correctness is a hard gate; never trade it. Then, in order:
p95 latency under 10s, cost per session, then token efficiency. A 5% cost
increase is acceptable for a 2-point correctness gain. Escalation accuracy must
never regress; it is how users trust the agent."

Keep this at the level of intent; do not write a scoring function or grant
permission to optimize the application here.>

## Constraints

<hard external bounds the agent must respect in operation. These are usually
organizational rather than technical, which is why source alone cannot
establish them. Record behavioral and safety boundaries here; permissions to
modify the application belong to the workflow undertaking those changes.

Cover whichever apply:

- **Approved providers and models:** which vendors, endpoints, regions, and
  model families the agent may use. Name the deployment mode where it is fixed,
  such as self-hosted only.
  State the permitted set rather than the model in use today, which the agent
  config already carries and which changes without touching this file.
- **Data handling:** residency, retention, redaction, and what may not leave a
  boundary or enter a prompt.
- **Compliance and policy:** regimes the agent operates under and the audit or
  disclosure obligations that follow.
- **Production ceilings:** maximum cost per run or session and latency SLOs the
  deployed agent must respect. Where you know the current measured figure, give
  it alongside the ceiling — "p95 is 4.1s today against a 6s SLO" tells a
  reader how much headroom there is, where the ceiling alone does not. Write
  `_(unmeasured)_` rather than guessing.
- **Action approvals:** actions the agent must escalate for human sign-off
  during operation, and who signs.

Retain already-confirmed organizational safety and review policies, such as
security review before shipping a tool with write access, and their known
owners. Do not turn these restrictions into a new change-permission interview
or infer permission to modify the application from them.

Example: "Models must come from the internal inference gateway; no direct
third-party API calls. Customer data must not leave the EU. Cost must stay under
$0.08 per session. The agent must obtain human approval before issuing a refund."

Write `_(none)_` only if you are genuinely unconstrained.>

## Evaluation Setup

<the current validation setup. Include how to run it, what datasets or checks it
uses, what scorers or metrics measure, pass/fail thresholds, and known coverage
gaps relative to the success criteria. If no eval suite is wired yet, say that
explicitly and describe any partial/manual validation that exists.>

## Metric Semantics

<what your metric and telemetry field names actually mean, and the claims they
do not support. A downstream agent that guesses a field's meaning from its name
will produce confident, wrong findings.

Use a row per field that is ambiguous, easy to misread, or load-bearing in a
decision. Skip the obvious ones. A known-broken or known-noisy field belongs
here too: saying what it cannot support is the cheapest way to stop an analyst
re-reporting the same non-issue.>

| Field or signal | Meaning | How consumers may use it |
|---|---|---|
| <e.g. `score`> | <what the number actually measures, and its source> | <the claim it supports, and the claim it does not> |

Use `_(none)_` if every metric name means exactly what it says.

## Vision

<where this agent is headed, so a change that fits the direction is not judged
only against today's scope.

**Intention.** <what the agent is ultimately for, beyond the job it does today.
One or two sentences.>

**Target use cases.** <one or two concrete scenarios the agent should grow into
but does not serve yet. These are the cases `Scope` calls out of scope *for
now*, which is different from out of scope on principle — the distinction stops
a change from hardcoding around a direction you already know is coming.>

- <future use case>

Keep this durable. A dated backlog belongs in your tracker, because a roadmap
that goes stale makes the whole file less trusted. Write `_(none)_` rather than
filling this with next quarter's tickets.>

## Open Questions

- <unresolved fact that affects safe use or evaluation of the agent; remove once answered>
