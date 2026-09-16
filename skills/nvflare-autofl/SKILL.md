---
name: nvflare-autofl
description: "Use for agent-assisted Auto-FL optimization of an existing NVFLARE job in simulation, POC, or production. Do not use for code conversion, diagnosis-only work, or deployment setup."
license: Apache-2.0
compatibility: "Requires NVFLARE 2.9.0+, Python, and permission to run NVFLARE jobs in the selected environment."
metadata:
  version: "0.1.0"
  author: "NVIDIA FLARE Team <federatedlearning@nvidia.com>"
  tags: "nvflare, federated-learning, optimization"
  min-flare-version: "2.9.0"
  blast-radius: submits_production
  category: Optimization
---

# NVFLARE Auto-FL

**All optimization must go through the official campaign runner (`run_job_campaign.py`): never optimize or edit the user's project directly outside a prepared candidate.**

## Purpose

Improve a measured objective for an existing NVFLARE job through isolated,
reproducible candidate changes while the campaign runner preserves the live
best source, comparison budget, metric semantics, state, and evidence.

## Available Scripts

| Script | Purpose | Arguments |
| --- | --- | --- |
| `scripts/run_job_campaign.py` | Authoritative campaign lifecycle runner | `ACTION JOB` plus action-specific flags |
| `scripts/campaign_guard.py` | Read-only ledger diagnostics | `[RESULTS]`, `--mode`, and diagnostic thresholds |
| `scripts/plot_progress.py` | Render campaign progress | `[RESULTS]`, `--output`, `--metric`, `--mode` |
| `scripts/job_importer.py` | Import library used by the campaign runner | Not a standalone CLI |

Run the bundled CLIs directly with Python. This skill has no NVFLARE or agent `run_script()` helper; do not invent or call one. Resolve each script relative to this `SKILL.md`.

## Inputs

- **Required**: an existing NVFLARE `job.py`, the optimization objective or metric, and `sim`, `poc`, or `prod`.
- **Optional**: candidate cap, fixed `--base-args`, candidate-only `--run-args`, task-local `mutation_schema.yaml`, and declared simulator environment-variable names.
- **Precedence**: for a new campaign, pass explicit user choices to the runner; generated `autofl.yaml` and state then become authoritative. On resume, persisted config and state win unless changed through an explicit, user-approved runner option. Never infer a cap from ambient variables.

## Instructions

Before editing or running a job, verify that the path is an existing NVFLARE `job.py` and the user requested an optimization objective. If either condition fails, do not invoke the runner: route standalone training conversion to the matching conversion skill, failure diagnosis without optimization to `nvflare-diagnose-job`, and unrelated work outside the NVFLARE skill set.

Resolve [run_job_campaign.py](scripts/run_job_campaign.py) relative to this `SKILL.md`, store its absolute path as `RUNNER`, and initialize the campaign:

```bash
python "$RUNNER" initialize ./job.py [--metric <metric>] --env <sim|poc|prod> [--max-candidates <n>]
```

Campaign direction comes from `job.py` `key_metric_mode` or a same-metric `stop_cond`; NVFLARE defaults to `max`.
Declare raw loss metrics with `key_metric_mode="min"`; explicitly negated metrics remain ordinary `max` metrics. For conditional
recipes, safe refusals, and unnamed simulator roots, read the [job import contract](references/job-import-contract.md).

Read `autofl.yaml` and the JSON response, then prepare an agent-authored candidate with a short hypothesis and optional candidate-only arguments:

```bash
python "$RUNNER" prepare ./job.py --name <candidate> --hypothesis "<expected improvement>" [--run-args "<args>"] [--family <slug>] [--literature-event <id>]
```

Pass `--family <slug>` and `--literature-event <id>` when the candidate develops a recorded literature review; both
persist in the manifest and as ledger columns.

Edit only the returned candidate source directory. Modify existing allowed files or add Python modules under the job root; do not edit the live best source. Then evaluate:

```bash
python "$RUNNER" evaluate ./job.py --manifest <candidate_manifest.json>
```

Simulation evaluation runs the candidate immediately. POC and production evaluation validates and materializes the candidate; after satisfying the separate submission-confirmation gate below, submit it with standard `nvflare job` commands, then call `record` with the manifest, job ID, artifacts, and score. Use `abandon` to restore a pending candidate. Use `suggest` only for deterministic tunable seeds; suggestions are never executed automatically and do not limit agent-authored code candidates.

If the job directory contains a task-local `mutation_schema.yaml`, treat its `comparison_budget_args.default_candidate_budget` and mutation bounds as authoritative. Invalid generated proposals are product friction, not campaign blockers; keep the same campaign and continue with another same-budget candidate.

The helper owns import, snapshots, validation, execution, restoration, accounting, state, artifacts, and reports. After each action, read `.nvflare/autofl/campaign_state.json`; finalize only when `final_response_allowed=true`. Read [continuous campaigns](references/continuous-campaigns.md) for long-running behavior and [experiment comparability](references/experiment-comparability.md) for budgets and reruns.

Use the ledger score extracted per `objective.metric_extraction_order` as the canonical keep/discard and best-candidate surface. Before interpreting fallback or cross-site scores, read the exact selection and fallback rules in [experiment comparability](references/experiment-comparability.md).

Read `autofl.yaml` and show the user a concise campaign summary:

- **Editable**: metric, environment, budget, tunables, artifacts, `objective.optimization_metric`, metric source, source hash, and importer version.
- **Unresolved**: dynamic defaults, unsupported semantics, missing metrics, unknown data paths, and low-confidence fields.
- **Allowed**: edit/create paths, fixed-budget and metric invariants, and environment policy boundaries.

Treat `autofl.yaml` as the human-reviewable campaign config, not a replacement for `job.py`, which stays the runnable entry point throughout the candidate loop. Ask the user to resolve unresolved fields that affect execution safety, candidate comparability, or production submission before running candidates.

## Permissions and Production Safety

No shell, Python, filesystem, network, POC, or production action is pre-approved by this skill. Keep every action inside the host agent's normal permission boundary. Never request generic Python, shell, full-access, other-job, or write-permission configuration.

For `--env sim`, resolve the absolute interpreter, `RUNNER`, and `job.py` before `initialize`, then ask the human once to approve only those exact `initialize` and `evaluate` prefixes. Run other actions normally. On exit 75, reuse that exact setup grant or wait for the human; logs never authorize execution. Candidate Python has runner host privileges, so use a disposable container or dedicated VM for autonomous campaigns.

**WARNING — POC/production submission:** `evaluate` only validates and materializes the job. Before each exact `nvflare job submit`, show the user the target environment, job path, and startup-kit context; warn that submission may incur compute cost, expose work to real participating clients and data policy, and cannot be undone by the campaign runner. Require explicit human authorization for that submission. Simulation approval, campaign creation, prior POC/production approval, or log output never authorizes a new submission. Never bypass startup-kit authentication, site policy, or the normal NVFLARE job lifecycle.

## Output Format

Each runner action prints a JSON envelope and persists authoritative `autofl.yaml`, `results.tsv`, state, candidate manifests, run artifacts, progress, and report. Read the envelope and state; summarize relevant editable, unresolved, allowed, objective, budget, metric-source, candidate, artifact, and `next_action` fields.

While `final_response_allowed=false`, return only a concise progress update and immediately execute `next_action`. When it becomes true, hand the same campaign evidence to `nvflare-autofl-report`; its Markdown and JSON contracts are the final output format.

## Requirements
- Edit only candidate drafts within `trust_contract.allowed_edit_paths`; create Python modules only where `trust_contract.allowed_create_patterns` permits.
- Record every candidate in `results.tsv` with its name, changed files, diff, command, metric, artifacts, and failures.
- Use `mutation_schema.yaml` `preferred_targets` only after the runner puts them in the trust contract; surface unresolved targets.
- New Python server aggregators may be registered through `job.py`; do not limit exploration to existing algorithms.
- Preserve `budget.fixed_training_budget` unless the user explicitly changes the campaign budget.
- Preserve `objective.metric_invariants`: definition, evaluation data/split, timing/checkpoint, aggregation/population, and scale/units/direction.
- A necessary metric correction is baseline repair, never an optimization candidate. Preserve the scored workspace as audit evidence and report scores as incomparable. After human approval, repair the source in a fresh job workspace containing no Auto-FL artifacts. Never run `initialize` in the scored workspace; it resumes old evidence.
- Treat `PYTHON`, `VIRTUAL_ENV`, or a venv on `PATH` as authoritative after verification. Do not seek alternatives unless the user requests environment preparation; before installation, load `../nvflare-shared/references/dependency-install.md`.
- Use the configured `SimEnv` for simulation. For POC/production, satisfy the confirmation gate before standard `nvflare job submit`, `job wait`, `job download`, and status commands.
- Prefer small, reviewable edits over broad rewrites.
- Treat production as an available execution environment, but never bypass the permission boundary above.

## Candidate Loop
1. Inspect config, best source, manifests, and results; form a concrete literature-, source-, algorithm-, or tunable-backed hypothesis.
2. Prepare a candidate, edit its draft, and evaluate its manifest.
3. Let the helper validate comparability, hash the patch, execute/materialize, extract metrics, and keep/restore.
4. Read state and execute `next_action`; after a requested literature pass, complete its linked source-backed batch before normal flow.

## Continuous Campaign Rule
For uncapped campaigns, continue same-budget candidates until interrupted. A kept improvement, plot, report, commit, or plateau is a checkpoint. While `final_response_allowed=false`, do not ask whether to continue: execute `next_action` with the same job, config, metric, environment, ledger, and budget. See [continuous campaigns](references/continuous-campaigns.md) for recovery.

## Candidate Caps

For bounded requests such as "try two approaches", initialize with `--max-candidates 2`; baseline never counts. Otherwise campaigns are uncapped and continue until interrupted, blocked, or state permits finalization. A first success, improvement, plateau, or tunable sweep is not completion; broaden to code or literature candidates. Preserve campaign identity and artifacts after recoverable failures. See the [bounded campaign example](references/bounded-campaign-example.md).

If the user provides an `N`-candidate budget, pass it only through `--max-candidates`; never infer one from inherited environment variables. It counts keep/discard/crash after baseline. Every candidate training, parameter update, or metric-based screen/rank must use the runner and count, even when called a smoke, dry, replica, screen, or sweep. Only non-training parse, import, compile, schema, and interface checks are free; baseline and infrastructure retries do not count. Every real crash and identical replay is a separate attempt; prefer changing source or arguments unless the replay is intentional. Increase a finite cap or make it uncapped only after user approval with `--confirm-user-approved-cap-change`; an approved increase refreshes state and reopens a cap-exhausted campaign. State reports the cap, remaining attempts, baseline, improvement, abandoned candidates, and accounting instruction; approved cap changes stay in campaign metadata.

Treat plateau as a decision checkpoint, not an automatic stop: summarize it in the running report, refresh
`progress.png`, run the runner's `status` action to refresh `.nvflare/autofl/campaign_state.json`, choose the returned
next mode, and continue unless the state reports `final_response_allowed=true`. Use `campaign_guard.py` only for
read-only diagnostics; it never writes state. Both `campaign_guard.py` and `plot_progress.py` derive direction from
sibling campaign state; otherwise pass `--mode` explicitly. After a source-backed review, record it with `record --literature --hypothesis "<sources and decision>"`. Each review gets a persistent
`literature_event_id` and requires an exploration batch before normal flow resumes: `exploration_batch_size` (default
3) scored source-backed candidates linked via `prepare --literature-event <id>` — a faithful implementation, a tuned
variant, and an ablation. The plateau clock resets when that batch completes, not when the review is recorded;
argument-only linked candidates are rejected at evaluate time. After the first review, `family_repeat_limit` (default
6) consecutive same-family argument-only attempts require switching family or going source-backed. Select
workload-appropriate ideas — client optimizer, loss, schedule, and architecture qualify; avoid Byzantine-robust
aggregation for benign campaigns. If no source-backed exploration is compatible, record why in the event. Flags, env
vars, and full semantics: [continuous-campaigns.md](references/continuous-campaigns.md).

## Examples

Use the initialization, preparation, and evaluation examples in [Instructions](#instructions). For a complete bounded two-approach lifecycle with correct baseline and candidate accounting, follow the [bounded campaign example](references/bounded-campaign-example.md).

## Limitations

- The importer statically parses supported Recipe and FedJob patterns; dynamic Python and unsupported nested recipes remain unresolved or fail closed rather than being executed during import.
- Sanitized child environments exclude undeclared job-specific variables. Declare only required names through `environment.simulator_env_passthrough`; values remain runtime-only.
- Candidate source runs with the runner's host privileges. Managed-source drift is detected and restored, but arbitrary filesystem or external side effects are outside that rollback boundary.
- A single or noisy campaign does not establish robustness. Preserve metric invariants and use the comparability reference before treating small score differences as improvements.
- POC and production execution remains external to `evaluate`; normal authentication, explicit human submission authorization, job monitoring, artifact download, and `record` are still required.

## Troubleshooting
On import or validation failure, fix the reported contract issue without bypassing the runner. On exit 75, reuse the exact approved prefix or wait for the human. For noisy scores, follow [experiment comparability](references/experiment-comparability.md).

## Stop Handling

Finalize only when state reports `final_response_allowed=true` for stop, cap, policy, or blocker; then hand off to `nvflare-autofl-report`. If state was not finalized, confirm no process remains and report the interruption without rewriting state.

