---
name: tao-run-automl
description: Run container-backed AutoML / hyperparameter optimization (HPO) for NVIDIA TAO networks using AutoMLRunner. Handles algorithm
  selection (bayesian, hyperband, asha, bohb, llm, hybrid, autoresearch), WandB experiment tracking, job execution on any TAO SDK
  platform, result interpretation, and per-rec custom evaluation hooks. Use when the user mentions TAO AutoML, hyperparameter
  optimization, HPO, automl, automl_settings, AutoMLRunner, tao_automl, bayesian search, hyperband, ASHA, LLM-guided search,
  autoresearch, or wants to tune train/evaluate/inference/distill/prune/quantize for a TAO network. Model actions use the resolved
  image; venv training requires an explicit request. Platform-agnostic — runs on any SDK (Brev,
  SLURM, Kubernetes, Docker). Do not use generic "keep improving" language alone to override a matching domain-specific
  DEFT workflow; attribute-labelled CLIP / SigLIP image-retrieval loops belong to tao-run-deft-pas unless HPO is explicit.
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit. Workflows declare additional requirements.
metadata:
  author: NVIDIA Corporation
  version: "0.1.1"
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

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run `tao-setup` first (host preflight, credentials, cross-skill discovery).

Run automated hyperparameter optimization for a TAO model by combining:

1. The selected model skill under `skills/models/<model_skill>/`.
2. The selected platform skill under `skills/platform/<platform>/`.
3. `AutoMLRunner`, which generates recommendations, launches selected action jobs,
   extracts metrics, and feeds results back to the optimizer.

Do not launch until model metadata, platform preflight, data visibility,
credentials, image choice, and compute shape are all proven.

## Execution Runtime — Hard Gate

Every recommendation, baseline evaluation, per-recommendation evaluation, and
final evaluation runs in the selected model action's resolved
`container_image` by default. Resolve it from the model skill before any
training-environment setup. A local checkpoint or Hugging Face model ID does not
change this rule.

Use venv-based **model execution** only when explicitly requested. Never infer
venv mode from `local-docker`, local GPUs, Python, or `pyproject.toml`. If
absent, execution is container-backed. A host/controller venv for `tao_automl`,
TAO SDK, or a platform adapter is
control-plane-only; keep child model actions in the resolved container image.

## Reference Map

- `references/skill_info.yaml`: this workflow's structured metadata.
- Split detailed references: `automl-preflight-concepts.md` for prerequisites
  and support checks; `automl-intent-algorithms.md` for search policy;
  `automl-compression-literature.md` for distill/prune/quantize algorithm
  sufficiency and future compression-search roadmap;
  `automl-runner-configuration.md` for runner/API/WandB details;
  `automl-advanced-monitoring.md` for hooks, resume, and pitfalls; and
  `automl-examples.md` for conversation examples; and
  `automl-common-pitfalls.md` for recurring safety checks. `detailed-guide.md`
  is only the map.
- `skills/models/<network>/SKILL.md`: model-specific dataset requirements, metrics,
  HPO notes, checkpoint handoff, and known failures.
- `skills/models/<network>/references/skill_info.yaml`: action contract,
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

Then verify the selected platform's SDK constructs — importing `tao_automl` does
not prove the platform backend is installed (e.g. `DockerSDK()` raises
`CredentialError` without the `docker` package). See
`automl-preflight-concepts.md`.

If missing, show the exact install command from `versions.yaml` and ask before
installing:

```bash
SB="${TAO_SKILL_BANK_PATH:-~/tao-skill-bank}"
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
   explicitly routes the selected action to AutoML.
3. Confirm `<skill_dir>/schemas/<action>.schema.json` exists and parses. This
   is the AutoML search-space gate.
4. For non-TAO-Core models such as Cosmos-RL and CLIP, also require
   `references/spec_template_<action>.yaml`; otherwise the runner has no
   complete action defaults.
5. If any gate fails, do not improvise a search space. Report the missing
   package artifact.

## Inputs

Collect these before runner construction:

| Input | Requirement |
|---|---|
| `model_skill` | Resolved model skill directory under `skills/models/`. Resolve user aliases such as `network_arch` to the packaged skill directory first. |
| `network_arch` | Read from the resolved model skill metadata. |
| `action` | Action to optimize — `train`, `evaluate`, `inference`, `distill`, `prune`, or `quantize` with a packaged schema/template. |
| `platform` | One of the supported TAO platform skills. |
| `train_dataset` / `eval_dataset` / action inputs | Use model-specific spec keys and layout. Non-train actions may also need parent/teacher checkpoints, calibration data, or pruned artifacts. |
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

Share the eval metric number in the launch review before asking for confirmation. If a
starting checkpoint exists but the baseline cannot be
produced — no packaged evaluate action, missing eval dataset, failed eval job —
stop and report the blocker instead of silently falling back to a
training-loss-only run.

For training from scratch, record the baseline as unavailable and proceed; do
not evaluate an empty checkpoint.

The runner owns final evaluation. When eval is runnable, pass
`final_eval_fn(best_rec, train_job_id)` to `AutoMLRunner.run`; the result then
carries `result["final_evaluation"]`. See `automl-preflight-concepts.md` for
the callback, checkpoint, baseline, and from-scratch rules.

## Dependency And Data Preflight

If the selected workflow needs object storage or a platform CLI and the tool is
missing, report the missing dependency and offer the exact install command
before continuing. After user approval, rerun
`scripts/check_tao_launch_preflight.py` with `--install-missing-tools` so it
installs the smallest needed package and immediately retries path verification.
For S3 paths, verify both credentials and path readability from the launch
platform before creating runner artifacts.

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
| `hyperband`, `asha` | Many configs, cheap early rungs; ASHA is parallel-friendly. | `max_epochs`, `reduction_factor`, optional `max_concurrent` |
| `bohb`, `dehb` | Mixed Bayesian/evolutionary search with multi-fidelity budgets. | same rung budget fields as Hyperband |
| `pbt` | Long training where schedules should mutate during training. | population and generation budget |
| `llm`, `hybrid`, `autoresearch` | User explicitly wants LLM-guided search with a configured endpoint. | LLM endpoint config plus budget |

For `evaluate` or `inference`, default to Bayesian/BFBO-style search over the
selected action's prompt, decoding, preprocessing, or runtime config knobs.
Use a task metric from the action outputs/logs and set `direction` explicitly
when the metric name is ambiguous. Do not use training-loss assumptions for
actions that do not update weights.

For `distill`, use the same train-like policy when the distill action performs
epoch-based optimization and writes checkpoints. For single-shot `prune` and
`quantize`, default to `bayesian` or `bfbo` unless the action schema/model skill
declares an epoch-like or calibration-budget field that makes
`hyperband`/`asha`/`bohb`/`dehb` meaningful. Use `eval_fn` when the selected
metric must be computed by a follow-up evaluate/inference action after the
compression action completes.

Prefer the model skill's recommendation over generic defaults. Avoid ASHA or
Hyperband when the model skill says startup, validation, or checkpoint cost
dominates short trials.

## Spec And Search Space

Build specs as nested dictionaries. If a model skill lists paths in dotted
notation for readability, walk the path and assign the nested leaf; do not store
flat dotted strings as spec keys.

Use the packaged selected-action schema for:

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

The final report must compare the baseline metric, each recommendation's
metric, and the selected best metric so users can see the impact of tuning. For
model skills that require an `eval_fn` to compute the real task metric, use
that evaluator instead of optimizing a convenient training loss unless the user
explicitly accepts the proxy metric.

## Runner Construction

Use the selected platform SDK only after its preflight passes. Construct SDKs
without embedding credentials in code.

`sdk` is the platform SDK object; containerless venv models use
`VirtualEnvSDK(venv_path=..., work_dir=...)`. Always pass `work_dir` -- the
default `~/.tao_sdk/virtualenv` fills the home directory with trial
checkpoints. See `automl-runner-configuration.md`.

```python
import sys
from pathlib import Path
from tao_automl.runner import AutoMLRunner

skill_bank = Path("<absolute-tao-skill-bank>")
model_skill = "<resolved-model-skill-directory>"
skill_dir = skill_bank / "skills" / "models" / model_skill
sys.path.insert(
    0, str(skill_bank / "skills/applications/tao-run-automl/scripts")
)
from resolve_automl_session import validate_session_settings

runner = AutoMLRunner(
    sdk=sdk,
    skill_dir=str(skill_dir),
    action=action,                      # train, distill, prune, quantize, ...
)

workspace_path = Path("<automl_workspace>")
resume = False
# Mandatory fail-closed gate from this skill's bundled scripts directory.
validate_session_settings(
    automl_settings,
    resume=resume,
    workspace=workspace_path if resume else None,
)
result = runner.run(
    workspace_path=str(workspace_path),    # timestamp it to avoid collisions
    automl_settings=automl_settings,       # must contain an explicit session_id
    spec_overrides=spec_overrides,
    automl_hyperparameters=automl_hyperparameters,
    custom_param_ranges=custom_param_ranges,
    metric_extractor=metric_extractor,  # optional
    eval_fn=eval_fn,                    # optional
    final_eval_fn=final_eval_fn,        # optional but required when final eval is runnable
    resume=resume,
)
```

Set `automl_settings["session_id"]` explicitly and call
`validate_session_settings` before every run. Generate a fresh ID once with
`scripts/resolve_automl_session.py new`. Resume only when explicitly requested;
resolve its controller with
`scripts/resolve_automl_session.py resolve --workspace <full-run-path>`.
Missing or ambiguous state is a blocker. See the
resume section of `references/automl-advanced-monitoring.md` for the complete
fresh/resume pattern.

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
2. Return the best child job id and its result path.
3. Resolve the model checkpoint or action artifact using the model skill's
   checkpoint/artifact metadata and SDK helpers; do not guess filenames such as
   `latest`.
4. Report the exact search space, algorithm, budget, metric, and platform.
5. Report the automatic baseline eval job id/result path/metric, all
   recommendation metrics, final evaluation status/result path/metric, failed
   recommendations and root causes, elapsed time, and final runtime notes.
6. If this feeds a workflow such as AutoML + DEFT, pass the winning spec
   overrides and checkpoint through the workflow's declared handoff fields.
7. With the default retention policy, verify that cleanup-supported, safely
   prunable terminal trial artifacts were deleted and that the winning
   training artifacts remain. Report protected promotion/resume parents or
   conservative Hybrid results explicitly. A remote bind, named volume, or
   other output route the SDK cannot reclaim must fail retention preflight
   before the first trial rather than be silently retained.

## Common Pitfalls

See `references/automl-common-pitfalls.md` before launching or recovering an
AutoML run.
