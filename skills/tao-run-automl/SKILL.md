---
name: tao-run-automl
description: Run AutoML / hyperparameter optimization (HPO) for NVIDIA TAO networks using AutoMLRunner. Handles algorithm
  selection (bayesian, hyperband, asha, bohb, llm, hybrid, autoresearch), WandB experiment tracking, job execution on any TAO SDK
  platform, result interpretation, and per-rec custom evaluation hooks. Use when the user mentions TAO AutoML, hyperparameter
  optimization, HPO, automl, automl_settings, AutoMLRunner, tao_automl, bayesian search, hyperband, ASHA, LLM-guided search,
  autoresearch, or wants to tune training hyperparameters for any TAO network. Platform-agnostic — runs on any SDK (Brev,
  SLURM, Kubernetes, Docker).
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit. Workflows declare additional requirements.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash Write
tags:
- automl
- hpo
- workflow
- training
- optimization
- llm
---

# TAO AutoML

Run automated hyperparameter optimization for a TAO model by combining:

1. The selected model skill under `skills/models/<network>/`.
2. The selected platform skill under `skills/platform/<platform>/`.
3. `AutoMLRunner`, which generates recommendations, launches train jobs,
   extracts metrics, and feeds results back to the optimizer.

Do not launch until model metadata, platform preflight, data visibility,
credentials, image choice, and compute shape are all proven.

## Reference Map

- `references/skill_info.yaml`: this workflow's structured metadata.
- Split detailed references: `automl-preflight-concepts.md` for prerequisites
  and support checks; `automl-intent-algorithms.md` for search policy;
  `automl-runner-configuration.md` for runner/API/WandB details;
  `automl-advanced-monitoring.md` for hooks, resume, and pitfalls; and
  `automl-examples.md` for conversation examples. `detailed-guide.md` is only
  the map.
- `skills/models/<network>/SKILL.md`: model-specific dataset requirements, metrics,
  HPO notes, checkpoint handoff, and known failures.
- `skills/models/<network>/references/skill_info.yaml`: train action contract,
  container image, inputs, outputs, upload exclusions, and `mode`.
- `skills/platform/<platform>/SKILL.md`: selected platform preflight, credentials,
  resource shape, monitoring, and cancellation.
- `skills/core/tao-launch-workflow/SKILL.md`: shared intake pattern for platform,
  credentials, dataset visibility, image confirmation, and user confirmation.

## Preflight

1. Run the shared launch intake. If the user has not chosen a platform, ask;
   Brev, SLURM, Kubernetes, and Docker are equal peers.
2. Run the selected platform skill's preflight before generating runner files.
3. Verify `nvidia-tao-automl` imports:

```bash
python -c "import tao_automl; from tao_automl.runner import AutoMLRunner; print('OK')"
```

If missing, show the exact install command from `versions.yaml` and ask before
installing:

```bash
SB="${TAO_SKILL_BANK_PATH:-~/tao-skills-external}"
pip install "$($SB/scripts/resolve_versions_key.py wheels.tao_automl_<platform>)"
```

Valid platform wheel keys are `tao_automl_brev`, `tao_automl_slurm`,
`tao_automl_kubernetes`, `tao_automl_docker`, and `tao_automl_all`. Use
`all` only for development machines that need every backend. Add `,llm` only
when the user requests LLM-guided algorithms.

## Model Support Gate

Before every run:

1. Read the model `SKILL.md` and `references/skill_info.yaml`.
2. Confirm `automl_enabled: true` for the model or that the model skill
   explicitly routes train-stage requests to AutoML.
3. Confirm `<skill_dir>/schemas/train.schema.json` exists and parses. This is
   the AutoML search-space gate.
4. For non-TAO-Core models such as Cosmos-RL and CLIP, also require
   `references/spec_template_train.yaml`; otherwise the runner has no complete
   train defaults.
5. If any gate fails, do not improvise a search space. Report the missing
   package artifact.

## Inputs

Collect these before runner construction:

| Input | Requirement |
|---|---|
| `model_skill` | Resolved model skill directory under `skills/models/`. Accept user aliases such as `network_arch` only after resolving them to the packaged skill directory. |
| `platform` | One of the supported TAO platform skills. |
| `train_dataset` / `eval_dataset` | Use model-specific spec keys and dataset layout. |
| `results_root` | Local, Lustre, or S3 path appropriate for the platform. |
| `gpu_count`, `num_nodes` | Respect model and platform limits. |
| `container_image` | Resolve through model metadata and `versions.yaml`; show it to the user. |
| `automl_algorithm` | Default `bayesian` unless user asks for another algorithm or the model skill recommends one. |
| `metric`, `direction` | Prefer the model skill's validation/task metric. |
| `automl_budget` | Recommendation count, max epochs/rungs, concurrency, or population size as required by the algorithm. |

Never ask for secret values. Verify required env vars with
`[ -n "$VAR_NAME" ] && echo SET || echo UNSET`.

## Pre-Launch Review Gate

Before launching any recommendation jobs, show a concrete launch review and get
user confirmation. This gate applies to every AutoML run for every
AutoML-supported model/network; it is not Cosmos-specific and must not be
scoped to a single model skill. This applies even when platform and image
preflight already passed. The review must include:

- model/network, platform, image, GPU/node shape, and result/workspace root
- dataset mode and concrete spec keys, including train/eval sample counts when
  they can be read cheaply
- algorithm, budget, max concurrent jobs, metric, and direction
- searchable parameters and ranges, including default values when the user did
  not provide an explicit search space
- exact generated recommendation configs for the initial launch batch, produced
  in a review-only step before any recommendation job is submitted
- estimated runtime per recommendation and total expected wall time, with the
  assumptions used
- the automatic baseline eval job id, metric value, and result path from the
  post-preflight eval job, or an explicit blocker if the model has no runnable
  evaluate action or validation data
- the post-AutoML final evaluation plan for the selected best checkpoint/model,
  including metric, dataset, and record path

If the estimate is longer than the user's stated limit or materially longer
than a normal interactive run, ask whether to reduce recommendations, epochs,
dataset size, validation frequency, or search space before launch. Do not hide
multi-day estimates in logs.

## Automatic Baseline Eval Job

After platform, image, credential, data, and model preflight pass, run the
model's evaluate action once on the selected validation/eval data before
submitting any AutoML recommendation jobs. This is required AutoML setup, not an
optional "pretrained eval" question for the user. Use the same base model or
checkpoint that the AutoML training run starts from, the model skill's evaluate
spec/template, and the selected platform's normal job submission path. If the
model skill recommends a smaller shape for evaluation than training, use that
shape and call it out in the launch review.

Share the eval metric number with the user in the launch review before asking
for confirmation to launch recommendations. If the model has no packaged
evaluate action, the eval dataset is missing, or the eval job fails, stop and
report the blocker instead of silently falling back to a training-loss-only
AutoML run. Continue without this baseline only when the user explicitly accepts
that the run will optimize a proxy metric and will not have an impact baseline.

The AutoML runner owns final evaluation of the selected best checkpoint/model.
When a runnable evaluate action and validation/eval data exist, pass a
`final_eval_fn(best_rec, train_job_id)` callback to `AutoMLRunner.run`. The
callback must evaluate the selected best checkpoint/model with the same metric,
dataset, and direction used for the baseline, store a structured record under
the workspace, and return the measured metric or a dict containing
`metric_value` and metadata such as `record_path` and `job_id`. Do not run final
evaluation as an agent-side step after `runner.run`; the returned result should
contain `result["final_evaluation"]` with a concrete status and reason.

## Dependency And Data Preflight

If the selected workflow needs object storage or a platform CLI and the tool is
missing, report the missing dependency and offer the exact install command
before continuing. After user approval, rerun
`scripts/check_tao_launch_preflight.py` with `--install-missing-tools` so it
installs the smallest needed package and immediately retries path verification.
For S3 paths, verify both credentials and path readability from the launch
platform before creating runner artifacts. Do not wait for the first training
container to discover a missing AWS CLI, S3 client, or unreadable URI.

For models that read large media archives or directories during every training
trial, stage or extract the dataset once to storage visible from the execution
platform, then point all recommendation specs at that staged path. Record the
source URI, staged path, byte/file-count evidence when available, and timestamp
in `<workspace>/evaluations/data_staging.json`. If staging is not possible,
include the repeated S3 I/O risk in the pre-launch review and ask before
spending a long AutoML budget on it.

When the model skill defines sample-count-sensitive constraints, enforce them
before launch. Reject or cap every batch-size recommendation that would create
zero training steps for the selected dataset and GPU shard count. Use
`scripts/check_tao_launch_preflight.py --effective-batch-limit
train_annotation=<batch_size>,<shard_count>` for each generated recommendation
before submitting it. If a recommendation later fails because the data is too
small for the effective batch size, classify it as an invalid configuration,
replace or adjust it only when remaining budget exists, and report the
correction in the final summary.
When train sample count is known from an annotation file or cheap manifest read,
pass it as `automl_settings["train_sample_count"]` to `AutoMLRunner.run` so the
runner can cap impossible recommendations before submitting a job and record the
adjustment in `result["history"][i]["adjustments"]`.

## Algorithm Policy

| Algorithm | Good fit | Required knobs |
|---|---|---|
| `bayesian` | Default for small/medium budgets and few parameters. | `num_recommendations`, metric, direction |
| `hyperband`, `asha` | Many configs with cheap early rungs; ASHA supports parallelism. | `max_epochs`, `reduction_factor`, optional `max_concurrent` |
| `bohb`, `dehb` | Mixed Bayesian/evolutionary search with multi-fidelity budgets. | same rung budget fields as Hyperband |
| `pbt` | Long training where schedules should mutate during training. | population and generation budget |
| `llm`, `hybrid`, `autoresearch` | User explicitly wants LLM-guided search and has an endpoint configured. | LLM endpoint config plus budget |

Prefer the model skill's recommendation over generic defaults. Avoid ASHA or
Hyperband when the model skill says startup, validation, or checkpoint cost
dominates short trials.

## Spec And Search Space

Build specs as nested dictionaries. If a model skill lists paths in dotted
notation for readability, walk the path and assign the nested leaf; do not store
flat dotted strings as spec keys.

Use the packaged train schema for:

- `automl_default_parameters`
- `automl_disabled_parameters`
- valid min/max ranges
- enums, option weights, conditions, dependencies, and popular parameters

User-provided search spaces must stay inside schema constraints. For integer
knobs with discrete choices, include the schema's required integer option shape
instead of a loose list if the model skill calls that out.

Data source overrides are mandatory unless the model skill says the launcher can
derive them. Preserve exact user-provided spec keys when the dataset uses direct
annotation/media paths.

## Metric Policy

Training loss is cheap but can be misleading. Prefer the model skill's task
metric. Use one of these:

- Log metric: `metric=<name>`, `direction=maximize|minimize`.
- `metric_extractor(logs, metric_name)`: parse the model's logs when the
  default resolver is ambiguous.
- `eval_fn(rec, train_job_id)`: run the model's evaluate action after each
  recommendation when the user wants a downstream task metric.

Do not map `kpi` to a metric unless the model skill explicitly defines that
mapping.

For every AutoML run with a runnable evaluate action and validation/eval data,
run the automatic baseline eval job after preflight and before recommendations.
The final report must compare that baseline metric, each recommendation's
metric, and the selected best metric so users can see the impact of tuning. For
model skills that require an `eval_fn` to compute the real task metric, use
that evaluator instead of optimizing a convenient training loss unless the user
explicitly accepts the proxy metric.

## Runner Construction

Use the selected platform SDK only after its preflight passes. Construct SDKs
without embedding credentials in code.

```python
from pathlib import Path
from tao_automl.runner import AutoMLRunner

skill_bank = Path("<absolute-tao-skill-bank>")
model_skill = "<resolved-model-skill-directory>"
skill_dir = skill_bank / "skills" / "models" / model_skill

runner = AutoMLRunner(
    skill_dir=str(skill_dir),
    platform_sdk=sdk,
    workspace_dir="<automl_workspace>",
)

result = runner.run(
    automl_algorithm=algorithm,
    automl_settings=automl_settings,
    spec_overrides=spec_overrides,
    automl_hyperparameters=automl_hyperparameters,
    custom_param_ranges=custom_param_ranges,
    metric_extractor=metric_extractor,  # optional
    eval_fn=eval_fn,                    # optional
    final_eval_fn=final_eval_fn,        # optional but required when final eval is runnable
)
```

Only resume an existing workspace when the user explicitly asks to resume,
continue, recover, or inspect an existing experiment. Treat a plain "run
AutoML" request as a fresh run.

## Monitoring

Use `runner` status output and the platform SDK's `get_job_status`,
`get_job_logs`, and `get_failure_analysis`. For active jobs, report:

- recommendation id / trial id
- platform job id
- status
- current metric
- best metric so far
- selected hyperparameters for the current/best recommendation
- elapsed time and updated ETA when enough timing data exists

On failure, classify whether it is infrastructure, data visibility, image,
credential, spec/schema, or model-code failure. Fix only the minimal cause and
do not silently spend additional budget on repeated invalid recommendations.
If a blocker is fixed during run setup, continue from the original task after
showing the updated preflight/launch review instead of leaving the user to
restate the request.

For LLM-based algorithms, inspect the brain logs before calling the run valid.
Verify that LLM calls succeeded, proposals were generated, prior metrics were
used to choose later parameter changes, and logs show keep/discard or
equivalent algorithm decisions. If the brain falls back to random sampling,
classify the LLM workflow as failed or blocked instead of treating it as a
valid LLM-guided run.

## Result Handoff

At completion:

1. Identify the best recommendation by the selected metric and direction.
2. Return the best train child job id and its result path.
3. Resolve the model checkpoint using the model skill's checkpoint metadata and
   SDK helpers; do not guess filenames such as `latest`.
4. Report the exact search space, algorithm, budget, metric, and platform.
5. Report the automatic baseline eval job id/result path/metric, all
   recommendation metrics, final evaluation status/result path/metric, failed
   recommendations and root causes, elapsed time, and final runtime notes.
6. If this feeds a workflow such as AutoML + DEFT, pass the winning spec
   overrides and checkpoint through the workflow's declared handoff fields.

## Common Pitfalls

- Do not expect `~/tao-core` at runtime. Schemas and templates must be packaged
  inside the model skill.
- Do not infer dataset URIs from previous runs.
- Do not precompute SDK-managed output paths; non-URI output values are routed
  by the SDK.
- For SLURM, stage large datasets on Lustre rather than burning GPU allocation
  time on large S3 downloads.
- For gated HuggingFace models, verify `HF_TOKEN` is set without reading it.
- If all recommendations fail, stop and summarize the shared root cause instead
  of launching more trials.
