<!-- miagent-template-source: assets/challenge_template/challenge.md -->
<!-- miagent-template-contract: active; keep the miagent-template-source loaded for creation/material updates; reload after resume or compaction -->
# Challenge

- Record stable task, data, rule, metric, and submission facts from the
  challenge website.
- Replace every example when initializing a challenge.
- For an unknown required fact, write `unknown` and add a question below; never
  leave it blank or guess.
- Policy fields state what is allowed, prohibited, or unresolved—not a current
  method choice such as “not planned.”

## Snapshot

| Field | Value |
|---|---|
| MIAgent harness version | Example: `v3.5` / commit `<sha>` when this challenge folder was created |
| challenge name |  |
| URL |  |
| objective/core problem class |  |
| input / modality |  |
| anatomy / object / target structure |  |
| output / labels |  |
| metrics, metric meaning, and aggregation | What each metric measures and how the final ranking is calculated |
| deadline / submission limit |  |

## Data And Splits

| Field | Value |
|---|---|
| data source / access |  |
| data manifests |  |
| available official train / validation / test counts |  |
| official hidden data counts for leaderboard |  |

## Rules And Evaluation

| Field | Value |
|---|---|
| external data policy |  |
| pretrained model policy |  |
| runtime / hardware limits |  |
| evaluator / proxy evaluator |  |
| submission format |  |
| packaging requirements |  |
| official / starter benchmark |  |

## Human Questions And Assumptions

- Ask when important information is missing or requires human-only access.
- Mark a question blocking only when it prevents meaningful progress, such as
  restricted pages/data or unclear task, metric, or submission rules.
- Record an assumption only when work can safely proceed without the answer.

| Status | Question / assumption | Why it matters | Default if unanswered | Answer / source |
|---|---|---|---|---|
| open / answered / assumed / blocked |  |  |  |  |
