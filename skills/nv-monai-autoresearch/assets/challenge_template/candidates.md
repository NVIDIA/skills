<!-- miagent-template-source: assets/challenge_template/candidates.md -->
<!-- miagent-template-contract: active; keep the miagent-template-source loaded for creation/material updates; reload after resume or compaction -->
<!-- miagent-template-schema: candidate-selection-v10 -->
# Candidates

Record the main agent's post-search scientific decision and the serious external
alternatives. This file is an external-loop queue, not an idea generator,
literature checklist, or numerical ranking exercise.

- Create it only after `literature.md` contains both the preserved Independent
  Scientific Thesis and the Revised Scientific Direction.
- Include the pre-search direction unless decisive evidence replaced or
  rejected it; record that disposition in Selected Workflow.
- Use the agent's judgment to choose the complete pipeline most likely to
  maximize the objective. Evidence challenges and improves that judgment but
  does not restrict the solution to published pipelines.
- Name the scientifically strongest direction before considering implementation
  convenience. Select it unless its row contains a verified hard blocker;
  untested setup, unfamiliarity, or a desire to get a baseline running are not
  blockers.
- Each direction is one committed end-to-end composition. Do not select a
  control with advanced components deferred until the control succeeds.
- A simpler control may run diagnostically, but it cannot replace the selected
  direction without a new evidence-backed selection decision.
- Difficulty is not a rejection reason. Hard blockers require verified
  rule/license/access conflict, unavailable required assets, demonstrated
  interface incompatibility, or measured minimum compute beyond budget.
- Add, combine, or replace directions whenever model intuition or new evidence
  reveals a stronger possibility. Preserve consequential changes in
  `trajectory.md`.
- `Next Decisive Evidence` is the focused research, probe, or comparable
  experiment that could change the direction's status.
- During the external loop, keep this as a current exploration map. Rows need
  not all be experimentally closed before freeze: `state.md` must instead record
  why no remaining direction is likely to produce a large gain. Do not turn an
  untried difficult direction into a scientific rejection.

## Candidate Comparison

| Candidate / Direction | Committed End-to-End Composition | Why It May Win | Main Uncertainty | Next Decisive Evidence | Status / Evidence |
|---|---|---|---|---|---|
| `baseline001_<direction>` | <Mandatory scientific components and their interfaces; cite useful L001-style evidence without treating it as the solution boundary> | <Agent's causal performance judgment> | <Scientific uncertainty, not implementation inconvenience> | <Focused evidence that could promote, reject, or reshape it> | open / selected / active / `supported: <comparable experiment link>` / `rejected: <comparable experiment link>` / `blocked: hard blocker: <reason and evidence link>` |
| `baseline002_<direction>` |  |  |  |  |  |

## Selected Workflow

Record the outcome of the workflow's baseline-selection gate.

| Field | Value |
|---|---|
| pre-search thesis direction |  |
| pre-search thesis disposition | verify / refine / replace / reject: <why, with decisive literature evidence when applicable> |
| scientifically strongest direction before feasibility |  |
| selected baseline |  |
| selection reason | <Concise causal comparison of the selected direction and strongest serious alternative> |
| trajectory ID | `TXXXX` after the gate outcome is recorded; `not applicable: proposal awaiting gate` before approval |
| approval status | proposed / approved / revise |

## Human Steering

Record only human commands that change baseline selection, candidate preference,
or candidate approval. Do not record general task goals, budget constraints,
Codex selection rationale, or chat transcript text here.

| Date | Human Decision | Reason | Applied Action |
|---|---|---|---|
| YYYY-MM-DD | Example: choose `baseline002_direction` over `baseline001_direction` | The human prefers the second scientific direction | Mark `baseline002_direction` as selected |
