<!-- miagent-template-source: assets/challenge_template/dashboard.md -->
<!-- miagent-template-contract: active; keep the miagent-template-source loaded for creation/material updates; reload after resume or compaction -->
<!-- miagent-template-schema: dashboard-v6 -->
# Dashboard

- This file is the shared scheduler and result index.
- Keep scientific understanding/direction in `state.md`, valid result history
  in experiment Outcomes, and command/retry history in `jobs/`.
- Update authorization, stop condition, usage, loop, active baseline, heartbeat,
  monitoring time, queue, and blockers when live state changes.
- Index every viewer-bearing and other decision-relevant completed run; details
  remain in run folders.
- Verify scheduler/process state; an old row is not proof of activity.

## Snapshot

| Field | Value |
|---|---|
| Challenge Name | Example: MSD Task09 Spleen |
| Challenge leaderboard link | |
| Challenge top numerical results | |
| MIAgent working directory | Example: `/lustre/.../MIAgent/msd_spleen` |
| Target / stop condition | Scientific target or budgeted-maximization criterion; a one-shot evaluation limit is not pre-freeze completion |
| Compute Budget | Example: 400 GPU-hours, 32 H100 max parallel |
| Run authorization | not entered / active: human / active: `:/auto` / paused / stopped / completed |
| Active usage | baseline001, 4 active jobs, 16 GPUs running |
| GPU efficiency / evidence | utilization, peak memory, throughput, input wait/bottleneck, and profile/log link for the active setup |
| Run state | planning / waiting for approval / active / paused / blocked / stopped / completed |
| Current loop | planning / internal baseline optimization / external research / monitoring / stopped |
| Active baseline | One direction at a time |
| Heartbeat | active in current session / unavailable: `<reason>` / stopped; owner/session, cadence, and next wake |
| Last monitoring pass / next wake | Example: `2026-07-23 10:00 EDT`; heartbeat `every 30 min`, next `10:30 EDT` |
| Latest trajectory decision | `TXXXX` in `trajectory.md` |
| Final freeze readiness | not checked / not ready: `<reason>` / ready: `[TXXXX](trajectory.md)`; final-path canary: `<train-only experiment link>` / `not applicable: unchanged validated checkpoint and inference path` |

## Result Table

- Show only the strongest official result, our final/leaderboard result, and
  our current best local result when available.
- Only record important finished runs that may reach target performance. Smoke tests are not important.
- Name the exact model/checkpoint and evaluation data.
- Distinguish self-split validation, held-out local test, official evaluation,
  and hidden leaderboard results.

| Result Type | Model / Checkpoint | Training Data | Evaluation Data | Result | Evidence |
|---|---|---|---|---|---|
| top leaderboard / our leaderboard / best local | `baseline001/.../best.ckpt` | dataset and split | hidden leaderboard / internal validation / held-out test | exact metrics | submission ID, evaluator output, or experiment Outcome |

## Challenge Readiness Checklist

Use `not checked`, `blocked: <reason>`, `not applicable: <reason>`, or
`done: <evidence link>`. This is challenge-wide readiness; baseline-specific
implementation acceptance belongs in each `baseline.md`.

| Gate | Status / Evidence |
|---|---|
| raw data downloaded and integrity checked | Example: done, 61 cases under `/abs/dataset_root/msd_spleen` |
| official preprocessing available or reproduced | Example: done, `data/official_preprocessing_manifest.json` |
| legal split manifest fixed | Example: done, `data/split.json` with first 80% train / last 20% held out |
| metric and evaluator parity verified | Example: done, `shared/evaluation/parity.md` |
| submission and packaging contract verified | Example: done, `baselines/baseline001_nnunet/runs/exp001_smoke/experiment.md` |
| scheduler or process status checker verified | Example: done, SLURM `squeue`/`sacct` evidence in the smoke experiment record |
| shared visualization implemented and smoke-tested | Example: done, `baselines/baseline001_nnunet/runs/exp001_smoke/result_viewer.html` |

## baseline001_nnunet

### Active Jobs

Use one row per experiment. Group its active training, resume, inference,
evaluation, visualization, packaging, and repair jobs in that row.

| Experiment | Active Stage / Jobs | State | Purpose / Evidence |
|---|---|---|---|
| `exp006_lr_sweep` | train `123456_[0-3]` | running | reduce LR from `1e-4` to `5e-5`; experiment record `baselines/baseline001_nnunet/runs/exp006_lr_sweep/experiment.md` |
| `exp007_long_train` | resume local PID `91234`; evaluation waits on resume | running | test whether the current model is undertrained; experiment record `baselines/baseline001_nnunet/runs/exp007_long_train/experiment.md` |

### Important Completed Runs

- Include every completed run with `result_viewer.html` and every other
  decision-relevant run. Smoke tests are not decision-relevant.
- Keep rows concise and link the Outcome; do not copy model, data, provenance,
  interpretation, or configuration details.

| Run | Result | Meaning | Experiment Outcome | Viewer |
|---|---|---|---|---|
| `exp004_full_train` | Dice `0.984`, HD95 `2.1`, val loss `0.018` | Current best after 1000 epochs with default nnU-Net | `baselines/baseline001_nnunet/runs/exp004_full_train/experiment.md#outcome` | `baselines/baseline001_nnunet/runs/exp004_full_train/result_viewer.html` |

### Planned Jobs

Each row is a planned scientific experiment, not an inference, evaluation,
visualization, packaging, or repair command for an existing experiment. Every
row must link an existing `experiment.md` Intent and address the current focus or
bottleneck in `state.md`. Give fair independent screens the same launch group
and run them concurrently when resources permit; keep dependent promotions in a
later group. Parallel groups stay within the one active baseline; independent
external directions require a recorded switch. Do not parallelize work that
would invalidate the comparison.

| Launch Group | Experiment | Experiment Record | Purpose |
|---|---|---|---|
| screen-A | `exp008_deep_supervision_off` | `baselines/baseline001_nnunet/runs/exp008_deep_supervision_off/experiment.md#intent` | test whether small-organ Dice improves |

## baseline002_swin_unetr

Use the same format as `baseline001_nnunet`. Create another baseline section
only when the challenge changes its active solution direction. Keep at most one
baseline's Active Jobs or Planned Jobs nonempty.

## Human Queue Steering

Record only human commands that change dashboard-visible priority or resource
use. The `:/auto` authorization belongs in Snapshot and its initial trajectory
row, not in this table.

| Date | Human Decision | Reason | Applied Action |
|---|---|---|---|
| YYYY-MM-DD | Example: submit more parallel jobs to use the approved compute budget | GPU resources are idle | Submit additional ready experiments |
