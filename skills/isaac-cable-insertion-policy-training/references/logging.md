## Session records

Use the committed supervisor to create one invocation directory under the skill's
ignored `logs/`. Keep one `report.md`; do not create separate diaries, custom
reports, or copies of native checkpoints and videos. When setup is needed, link
its `setup.log` and bounded smoke session as described in [Setup](setup.md).

```text
logs/<timestamp>__<name>/
├── report.md
├── session.json
├── events.jsonl
└── runs/
    └── <timestamp>__rNN__<run-id>/
        ├── train.log
        ├── evaluation.log
        ├── evaluations.jsonl
        └── params/
            ├── agent.yaml
            └── env.yaml
```

Native Isaac Lab output holds TensorBoard data, checkpoints, videos and exports;
link to these artifacts. Evaluation files appear only when evaluation runs.
The supervisor keeps nonempty `supervisor.log` and removes empty logs at close.

### Names and provenance

Choose a descriptive plan `name` and unique logical run `id`. The supervisor adds
UTC timestamps and attempt numbers, including failures. Queue/evaluate commands
use the logical ID; retries receive a new ID and `retry_of` link. Never rename
completed runs based on their scores.

- **report.md:** generated result, comparison, findings, costs, issues and artifact
  links. Use the supervisor's `note` command at idle decision points for analysis.
- **session.json:** immutable normalized plan and mutable state, including every
  attempt's parameters, argv, parent, seed, timing, exit status, metrics and artifact
  paths/hashes. Unavailable exit codes and lineage totals remain unavailable.
- **events.jsonl:** append-only progress, decisions, failures, repairs and notes.
- **params/:** each attempt's resolved `agent.yaml` and `env.yaml`, both required for
  successful verification. Failed attempts retain whichever files were produced.
- **train.log / evaluation.log:** original process output; evaluations append each
  checkpoint/seed command and its outcome.
- **evaluations.jsonl:** durable per-seed episode results and process status. Identity
  includes checkpoint hash, task, run spec and evaluation configuration. Recovery
  skips committed successful seeds and reconciles live evaluators before relaunch.

The supervisor records script hashes, repository commit/ref/dirty status, lockfile
hash, installed Isaac Lab commit and simulator version. These records do not archive
uncommitted changes or the full environment. Do not copy scripts, project files,
or source patches into logs. Preserve completed records and original commands;
see [Supervisor](supervisor.md) for compatible resume and historical sessions.

`max_iterations` means additional updates; resumed labels can repeat. Record
`parent_total_updates` and `parent_training_seconds` only when verified. Session
cost includes failed attempts; inherited training is reported separately.

### Evaluation and delivery

Use the originating run and the evaluation `group` (`screening`, `tuning`, or
`independent`). Select candidates using tuning results; use disjoint independent
seeds to validate the selected policy and report those results separately.
A training video shows a rollout, not necessarily the final saved checkpoint.
Smoke runs are not policy-success evidence.

Deliver the report to the user when the series completes or stops early, including
any blocker. A file on disk does not notify the user; an active agent must deliver it.
Generated logs and the user-scoped EULA receipt remain outside Git.
