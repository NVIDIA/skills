# TAO AutoML Intent And Algorithms

User intent parsing, train-vs-AutoML policy, algorithm selection, and the algorithm decision tree.

Load this file only when the compact `SKILL.md` points here for the current task. If this reference conflicts with `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the compact/current source wins.

## Contents

- network_arch is NOT a runner.run() arg anymore; resolve the user's model
- request to a packaged skill_dir, then read network_arch from that skill's
- metadata.
- Step 2: Select Algorithm
- Classical Algorithms
- LLM/Agentic Algorithms (NEW)
- Quick Reference: Algorithm Decision Tree

## Step 1: Parse User Intent

Default to a quick-start run unless the user explicitly asks to customize AutoML or agrees to a customization offer. Do not present algorithm, budget, or search-space choices as required inputs for a normal "run AutoML" request.

Any workflow/application that reaches a train-capable model skill must consult
the selected model's `automl_enabled` metadata. If it is `true`, use this
AutoML workflow as the default training path unless the run/workflow setting
has `automl_policy: off` or the user explicitly asks for a plain single
training run. Treat `automl_policy: on` as the default enabled state. This keeps
AutoML enablement scalable across tao-train-single-step, DEFT,
and future workflows without duplicating allowlists in each application skill.

Extract these fields for a default run:

| Field | Required | Example | How to get it |
|---|---|---|---|
| `requested_model` | Yes | `"cosmos-rl"` or `"tao-finetune-cosmos-reason"` | User states the model, model family, or network alias. |
| `model_skill` | Yes | `"tao-finetune-cosmos-reason"` | Resolve `requested_model` to a packaged directory under `skills/models/` by using model metadata. Do not assume `network_arch` is the directory name. |
| `network_arch` | Yes | `"cosmos-rl"` | Read from `<skill_dir>/references/skill_info.yaml` after resolving `model_skill`. |
| `platform` | Yes | `"brev"`, `"slurm"`, `"local-docker"`, `"kubernetes"` | After the user confirms they want AutoML, run `scripts/list_tao_platforms.py --format text` and ask them to choose from that output. |
| `train_dataset_uri` or direct train spec paths | Yes | `"s3://bucket/data/subset"`, `"/lustre/fsw/tao_datasets/<model>/train"`, or `custom.train_dataset.annotation_path=/...` | User provides a root URI/path, exact spec-key paths, or the model skill declares a default profile for this exact network/use case. |
| `eval_dataset_uri` or direct eval spec paths | Model-dependent | `"s3://bucket/data/eval"`, `"/lustre/fsw/tao_datasets/<model>/eval"`, or `custom.val_dataset.media_path=/...` | Ask only if the model skill's Per-Action Dataset Requirements require an eval/validation source and no default profile supplies it. |
| `image` | Yes | `"nvcr.io/..."` | Resolve the default with `scripts/resolve_tao_image.py --model <requested_model> --action train`, show it to the user, and require confirmation or `image=<override>` before creating the AutoML runner. |
| `metric` | No | `"<metric_name>"` | Use the model skill recommendation or ask if unclear. Do not choose model-specific metrics from this AutoML skill. |
| `direction` | No | `"minimize"` or `"maximize"` | **Only needed if your metric name doesn't contain `"loss"` AND you want to minimize, or contains `"loss"` AND you want to maximize.** Otherwise the implicit "contains 'loss' → minimize, else maximize" rule applies. |
| `skill_dir` | Yes | `"<bank-root>/skills/models/tao-train-dino"` | Absolute path to the resolved model skill directory in the skill bank. Passed explicitly to `AutoMLRunner(skill_dir=...)` — no env-var fallback. |
| `long_running_enabled` | Yes | `true` | Ask during launch intake. If enabled, keep the agent attached and emit status until completion. Default: enabled. |
| `status_interval_minutes` | Yes | `5` | Ask during launch intake. Default: 5 minutes. |
| required credentials | Platform/model-dependent | `SLURM_USER`, `SLURM_HOSTNAME`, `SSH_KEY_PATH` or `SSH_AUTH_SOCK`, `HF_TOKEN` | First filter platform credentials with `scripts/list_tao_platforms.py --platform <platform>`, satisfy required credential groups, then add selected-model credentials. Do not ask for unrelated platform credentials. |
| compute shape | Model-dependent | `num_gpus=4`, `num_nodes=1` | Ask only for model-required hardware fields that are not provided by the platform/default profile. |
| `llm_endpoint` / `base_url` | **Yes** (for `llm`/`hybrid`/`autoresearch`) | `"https://inference-api.nvidia.com"` | Resolve from user input or `AUTOML_LLM_ENDPOINT`; pass explicitly in `automl_settings`. |
| `llm_model` / `model` | **Yes** (for `llm`/`hybrid`/`autoresearch`) | `"gcp/google/gemini-3.1-pro-preview"` | Resolve from user input or `AUTOML_LLM_MODEL`; pass explicitly in `automl_settings`. |
| `llm_api_key` / `api_key` | **Yes** (for `llm`/`hybrid`/`autoresearch`) | `"nvapi-..."` or `"sk-..."` | Resolve from `AUTOML_LLM_API_KEY` or `NVIDIA_API_KEY`. Do not print or log the value; if unavailable, stop for credential setup instead of falling back silently. |

Use these quick-start AutoML defaults without asking:

| Field | Default |
|---|---|
| `algorithm` | `bayesian`, unless the user/model default profile explicitly selects another algorithm |
| `automl_max_recommendations` | model/workflow default if declared, otherwise `10` |
| `automl_hyperparameters` | `None` so AutoML uses dataclass-schema params with `automl_enabled=true` |
| `custom_param_ranges` | `None` so ranges/options/defaults come from the generated dataclass schema |
| `long_running_enabled` | `true` |
| `status_interval_minutes` | `5` |

If any required field is missing, ask the user. Do NOT guess dataset paths, skill bank paths, credentials, or hardware that the model skill marks as required.

When asking for missing AutoML launch inputs, use a first-time-user friendly
prompt. Do not say only "train dataset root" / "eval dataset root", and do not
say "attached monitoring every 5 minutes" without explaining it. Include:

- platform choices;
- root-mode dataset examples for the selected platform;
- direct spec-parameter mode as an equal option;
- model-required spec keys from the model skill's Per-Action Dataset
  Requirements table;
- resolved train container image and the option to override it with
  `image=<override>`;
- monitoring meaning and cadence choices.

Before generating an AutoML script, verify platform access and dataset
visibility using the shared launch preflight. For SLURM, that means
passwordless SSH to at least one login host and remote `test -e` checks for
each required annotation/media path. If preflight fails, stop with remediation
steps instead of creating a runner that will immediately fail.

If the selected model skill's Per-Action Dataset Requirements or Typical Spec
Overrides show train/evaluate/inference inputs that come from a prior
`dataset_convert` action, run that conversion before calling
`AutoMLRunner.run`. Use the `dataset_convert` action's own `container_image`
when it overrides the model default, persist the conversion output under the
current run's results root, verify every required converted artifact named by
the model skill exists, and pass those current-run converted paths in
`spec_overrides`. Do not reuse stale conversion paths from another AutoML
algorithm/run folder.

Also verify container image confirmation using the shared launch preflight.
AutoML launches real train jobs for each recommendation, so the confirmed train
image must be passed into `AutoMLRunner.run(..., image=chosen_image, ...)` or
into the SDK adapter's `create_job(..., image=chosen_image, ...)`. Do not rely
on an implicit default after the user has chosen a platform and dataset.

Also run any model-specific annotation content checks documented by the model
skill. Missing required annotation fields are a preflight failure, not an
AutoML recommendation failure.

Before submitting any recommendation job, generate the initial recommendation
batch in a review-only step and show the exact configs to the user together
with metric, direction, expected runtime, and effective-batch checks. Do not
replace concrete configs with only search bounds.

After all platform, image, credential, data, and model preflight checks pass,
submit the model's evaluate action once against the selected validation/eval
data before submitting AutoML recommendations. This is the automatic baseline
eval job for the run. Show its job id, result path, and metric value in the
launch review, then ask for confirmation before spending training budget on
recommendations. If the eval job cannot run because the model has no packaged
evaluate action, validation data is missing, or the eval job fails, stop and
surface the blocker. Proceed without this baseline only when the user
explicitly accepts a proxy-metric run with no impact baseline.

**Customization gate:** After the required quick-start fields are resolved, you may briefly offer customization. If the user declines or does not ask for it, proceed with the defaults above. If the user chooses customization, then present the additional options below.

Customization-only fields:

| Field | Example | Notes |
|---|---|---|
| `algorithm` | `bayesian`, `asha`, `hyperband`, `bohb`, `llm`, `hybrid`, `autoresearch` | Present the algorithm guide only in customization mode or when the user names an algorithm. |
| `max_recommendations` | `5`, `10`, `20` | Explain that each recommendation is a real training job. |
| `long_running_enabled` | `false` | Only use false when the user explicitly does not want the agent to keep monitoring. |
| `status_interval_minutes` | `5`, `10`, `15` | Already asked during launch intake; customize only if the user wants a different cadence. |
| `automl_hyperparameters` | `["train.optm_lr", "train.epoch"]` | List choices from the generated schema JSON, not from hand-written guesses. |
| `custom_param_ranges` | `{"train.optm_lr": {"valid_min": 1e-6, "valid_max": 1e-4}}` | Validate against schema type/range/options before using. |
| `llm_endpoint`, `llm_model`, `llm_api_key` | `https://inference-api.nvidia.com`, `gcp/google/gemini-3.1-pro-preview`, `nvapi-...` | Required only when the selected algorithm is `llm`, `hybrid`, or `autoresearch`. Resolve credentials from env/secret files; do not echo keys. |

**MANDATORY: Read the generated dataclass schema before configuring AutoML.**

For the selected model/action, read:

- `${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/skills/models/<model_skill>/schemas/train.schema.json`
- `${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/skills/models/<model_skill>/schemas/manifest.json`

AutoML is enabled by the model skill, but it can run only when
`schemas/train.schema.json` is packaged with the plugin and valid for the
selected model. Do not fall back to hand-written model
notes, old runner scripts, or a local `~/tao-core` checkout for AutoML
parameter metadata. If the train schema is missing, stop and report that AutoML
is enabled for that model but not runnable until the schema is generated and
shipped in the skill bank.

Use the schema JSON as the source of truth for `automl_default_parameters`,
`automl_disabled_parameters`, per-parameter defaults, ranges, enums,
`option_weights`, `math_cond`, `depends_on`, `parent_param`, and `popular`.

When `automl_hyperparameters=None`, the runner automatically discovers all
params marked `automl_enabled=True` in the network's generated schema. Each
network has its own set; never hardcode them in this workflow skill.

Quick-start runner shape:

```python
# network_arch is NOT a runner.run() arg anymore. It is read from the model
# metadata at skill_dir, which was passed to AutoMLRunner(...) at construction.
result = runner.run(
    train_dataset_uri=TRAIN_DATASET_URI,
    automl_settings={
        "algorithm": "bayesian",
        "metric": metric,
        "automl_max_recommendations": 10,
    },
    automl_hyperparameters=None,  # use schema params marked automl_enabled=true
    custom_param_ranges=None,     # use schema ranges/options/defaults
    spec_overrides={...},         # from model skill + dataset requirements
    workspace_path=f"./automl/{TIMESTAMP}",
)
```

Customization runner additions:

```python
result = runner.run(
    ...,
    automl_hyperparameters=selected_param_names,
    custom_param_ranges={
        "<param_name>": {"valid_min": min_value, "valid_max": max_value},
        "<categorical_param>": {
            "valid_options": ["option_a", "option_b"],
            "option_weights": [0.7, 0.3],
        },
    },
)
```

**MANDATORY LLM configuration for LLM-based algorithms (`llm`, `hybrid`, `autoresearch`):**

When the user requests or customizes into an LLM-powered algorithm, resolve ALL THREE of the following before generating the script. Do not ask for these on default `bayesian` quick-start runs.

1. **`llm_endpoint`** — user input -> `AUTOML_LLM_ENDPOINT` -> `https://inference-api.nvidia.com`
2. **`llm_model`** — user input -> `AUTOML_LLM_MODEL` -> `gcp/google/gemini-3.1-pro-preview`
3. **`llm_api_key`** — `AUTOML_LLM_API_KEY` -> `NVIDIA_API_KEY` -> declared local secret file when allowed. Do not print the value.

If the runner does not receive valid LLM settings, the LLM brain may silently fall back to random sampling — wasting GPU budget on random configs instead of intelligent ones. There is no error message; the only clue is "LLM call failed... Falling back to random" in the logs.

**MANDATORY: Read the model skill before generating the script.**

AutoML runs training. Before generating any AutoML script, read `<bank-root>/skills/models/<model_skill>/SKILL.md` (where `<bank-root>` is wherever the agent loaded this SKILL.md from). The model skill contains all model-specific knowledge:

- **Training Requirements** — dataset type, formats, monitoring metric, required dataset URIs to prompt for, required user prompts (data format, num_classes, etc.), and mandatory `spec_overrides`. Prompt the user for every required field. Apply mandatory spec_overrides exactly.
- **Per-Action Dataset Requirements** — table mapping each action to its spec keys, data source, expected files, and whether the field is a list. Use this table to construct the correct data source `spec_overrides` for the requested action. If the model's Typical Spec Overrides mark data sources as "mandatory", construct them from this table and the user's dataset URIs.
- **Typical Spec Overrides** — per-action override suggestions (train, evaluate, export, inference, etc.) extracted from SDK notebooks. Use these as the starting point for `spec_overrides` and suggest them to the user. When overrides are marked "mandatory data sources", they MUST be included — the runner cannot auto-resolve them. Merge with any other mandatory overrides from Training Requirements.
- **AutoML / HPO Notes** — metric, direction, model-specific constraints, and any guidance that narrows or overrides the generated schema. Hyperparameter names/ranges/defaults come first from `schemas/train.schema.json`.
- **Error Patterns** — common training failure modes that apply to AutoML recs too.

Do NOT hardcode model-specific knowledge in the AutoML script without reading the model skill first. Each network has different requirements.

**MANDATORY: No model-specific constants in this AutoML skill.**

The AutoML skill must not define model-specific hyperparameter names, ranges, defaults, metric names, dataset layouts, archive names, class-count rules, spec override keys, container images, checkpoint quirks, or custom metric regexes. Hyperparameter metadata belongs in `<bank-root>/skills/models/<model_skill>/schemas/train.schema.json`; model-specific runtime guidance belongs in the model skill's **Training Requirements**, **Typical Spec Overrides**, **AutoML / HPO Notes**, and **Error Patterns** sections. This skill may describe how to read and apply those sources, but not the concrete per-model values.

**MANDATORY: Timestamped workspace folders.**

ALWAYS generate `workspace_path` with a timestamp suffix. Running the same script twice without a timestamp overwrites the previous experiment. Pattern:

```python
from datetime import datetime
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
workspace_path = f"./experiment_name/{TIMESTAMP}"
```

Do NOT use a flat path like `workspace_path="./my_experiment"`. The user should never have to manually delete old workspace folders.

**MANDATORY: Fresh runner per new AutoML request, after preflight passes.**

Every new user request to run AutoML MUST create a new runner script and launch a new AutoML job, even if an older runner script for the same network/algorithm already exists. This freshness rule starts only after platform and dataset preflight passes. Existing runner files and logs may be read only as references for dataset URIs, credentials patterns, and proven fixes; do not reuse them as the execution target for a new request.

Use a unique timestamp in the new runner filename, log filename, PID filename, SDK `state_file`, and `workspace_path`. Derive path components from the requested `network_arch` and `algorithm`; do not hardcode any model or algorithm name unless it is the actual requested value.

```python
import re

def slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_").lower()

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"{slug(network_arch)}_{slug(algorithm)}"
runner_path = f"automl_runs/run_{RUN_NAME}_{TIMESTAMP}.py"
log_path = f"automl_runs/{RUN_NAME}_{TIMESTAMP}.log"
pid_path = f"automl_runs/{RUN_NAME}_{TIMESTAMP}.pid"
state_file = f"tao_session_state_{RUN_NAME}_{TIMESTAMP}.json"
workspace_path = f"./automl_runs/{RUN_NAME}/{TIMESTAMP}"
```

Only resume an existing runner/workspace when the user explicitly asks to resume, continue, recover, or inspect an existing experiment. If the user says "run automl" or asks for a new AutoML run, treat it as a fresh job.

**Best-practice on metric choice**:

- Training loss is cheap, but can overfit on small fine-tuning datasets. Prefer the model skill's recommended validation or task metric when available.
- If the model skill recommends a validation proxy, also apply the model skill's required validation-related `spec_overrides` so the metric is actually emitted.
- A real task metric via `eval_fn` is often the most honest but adds per-rec cost. Use it when the model skill says log-based metrics are insufficient or the user explicitly wants downstream evaluation.
- For AutoML runs with a runnable evaluate action and validation/eval data, run
  the automatic baseline eval job after preflight and before recommendations,
  then include that baseline metric in the final comparison.

**Checkpoint / resume behavior**:

- Resume-based algorithms (`hyperband`, `asha`, `bohb`, `dehb`, `pbt`, `hyperband_es`) must resume from the checkpoint for the stopped rung/generation epoch or step, not a generic `*_latest.*` file.
- The runner records the intended `resume_from_epoch` / `resume_from_step` on promoted recommendations and resolves the model-specific checkpoint path through the SDK checkpoint resolver. Do not patch runner scripts to guess names like `model_latest.pth`.
- If the intended checkpoint is epoch 1, prefer artifacts such as `epoch_1`, `epoch_001`, `model_epoch_001.pth`, `model_epoch_000_step_*.pth` when the trainer writes zero-indexed epoch files, or a checkpoint directory like `checkpoints/epoch_1`.
- Use a `latest` checkpoint only when the requested action explicitly has no epoch/step target. If no epoch/step-specific resume artifact exists, report the model as blocked and fix the model/skill checkpoint metadata rather than silently resuming from latest.

---

## Step 2: Select Algorithm

### Classical Algorithms

These require no external services — they use statistical/mathematical methods to pick hyperparameters.

| Algorithm | Use when | Typical budget | How it works |
|---|---|---|---|
| `bayesian` | **Default choice.** Small budgets, few parameters. | 5–20 recs | Builds a Gaussian Process model of metric vs. hyperparameters. Sequential — waits for each result before proposing the next, so it learns fast but can't parallelize. |
| `bfbo` | Alternative to bayesian with different acquisition function. | 5–20 recs | UCB-based Bayesian optimization with local penalization. Good when bayesian gets stuck. |
| `hyperband` | Large search spaces, many parameters. | 20–50+ recs | Trains many configs cheaply for a few epochs, keeps the best, trains longer. Requires `automl_max_epochs` and `automl_reduction_factor`. |
| `hyperband_es` | Hyperband + early stopping. | 20–50+ recs | Like hyperband but adds early-stop thresholds to halt clearly bad runs sooner. |
| `asha` | Async variant of hyperband, supports parallel execution. | 10–30 recs | Same successive-halving idea as hyperband, but trials run concurrently. Best when you have many GPUs. Uses `automl_max_concurrent`. |
| `bohb` | Best of both — Bayesian intelligence + Hyperband efficiency. | 15–40 recs | Combines KDE-based model (like Bayesian) with Hyperband's multi-fidelity scheduling. Good all-rounder for medium budgets. |
| `dehb` | Evolutionary + multi-fidelity. | 15–40 recs | Differential evolution mutations + hyperband scheduling. Good for complex search spaces with many interacting parameters. |
| `pbt` | Dynamic schedules — mutates hyperparameters during training. | population_size × generations | Population-Based Training. Starts N configs in parallel, periodically copies weights from winners and perturbs their hyperparameters. Best for long runs where hyperparameters should change over time (e.g. learning rate schedules). Final handoff selects the best member at the largest observed training budget; earlier lower-budget metrics are promotion evidence, not the final checkpoint choice. |

### LLM/Agentic Algorithms (NEW)

These use a large language model to reason about hyperparameter choices. They require an LLM endpoint (NVIDIA NIM, OpenAI, vLLM, Ollama, etc.) and the `openai` Python package.

| Algorithm | Use when | Typical budget | How it works |
|---|---|---|---|
| `llm` | Domain knowledge matters more than statistical rigor. | 5–20 recs | An LLM proposes hyperparameter configs based on the search space schema, experiment history, and its training knowledge. Falls back to random sampling on LLM failure. Sequential like bayesian. |
| `hybrid` | You want the LLM to orchestrate multi-phase optimization. | 10–50 recs | An LLM strategist plans optimization phases over model-skill parameters. Each phase uses a classical sub-algorithm. Stops when the strategist detects diminishing returns. |
| `autoresearch` | Fully autonomous agent loop. | 10–50 recs | The most powerful mode. Combines: (1) RAP knowledge retrieval about the network, (2) LLM-proposed spec modifications, (3) training-free pre-screening of candidates, (4) multi-stage verification (pre-launch + post-result), (5) keep/discard reasoning. Automatically stops on budget exhaustion or consecutive failures. |

**Default to `bayesian` unless** the user specifically asks for something else, has a large GPU budget, or needs early-stopping on cheap intermediate metrics (ASHA / hyperband).

**Use `llm` / `hybrid` / `autoresearch` when** the user wants LLM-guided search, has an API key for NVIDIA NIM or OpenAI, and wants richer reasoning about why certain hyperparameters are chosen.

**Caveat on ASHA with expensive checkpoints:** ASHA's whole point is running many configs cheaply for early rungs, then promoting survivors. If the model skill warns that checkpoints, validation, or startup cost dominate short trials, prefer the model skill's recommended algorithm instead of assuming ASHA will be cheaper.

---

## Quick Reference: Algorithm Decision Tree

```
Is your budget tiny (≤10 recs)?
  YES → bayesian
  NO  ↓

Do you have an LLM API key and want AI-guided search?
  YES → Do you want full autonomy? → autoresearch
        Just LLM proposals?        → llm
        LLM orchestrating phases?  → hybrid
  NO  ↓

Do you need parallel execution?
  YES → asha (or bohb for smarter sampling)
  NO  ↓

Is your search space large (10+ parameters)?
  YES → hyperband or dehb
  NO  ↓

Do hyperparameters need to change during training (schedules)?
  YES → pbt
  NO  → bayesian (safe default)
```

---
