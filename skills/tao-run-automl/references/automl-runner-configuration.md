# TAO AutoML Runner Configuration

Runner construction, SDK handoff examples, settings keys, metrics, custom parameter ranges, spec overrides, and WandB setup.

Load this file only when the compact `SKILL.md` points here for the current task. If this reference conflicts with `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the compact/current source wins.

## Contents

- Minimal Example
- Pick whichever SDK matches where you want trials to run. AutoMLRunner is
- platform-agnostic — none of the SDKs is a default; the user picks.
- from tao_sdk.platforms.slurm      import SlurmSDK      # SLURM cluster
- from tao_sdk.platforms.kubernetes import KubernetesSDK # K8s (EKS / GKE / on-prem)
- from tao_sdk.platforms.docker     import DockerSDK     # local Docker daemon
- Full Example (all options)
- LLM-Powered Algorithm Example
- Programmatic API (without runner)
- `automl_settings` keys
- `kpi` metric resolution
- `custom_param_ranges` format
- Model-specific search-space rules
- LLM Analyzer (server-side range narrowing)
- `spec_overrides`
- WandB Experiment Tracking
- Setup
- or (when reinstalling tao-run-automl with the wandb extra):
- `pip install "$(${TAO_SKILL_BANK_PATH:?}/scripts/resolve_versions_key.py wheels.tao_automl_brev | sed 's/]/,wandb]/')"`
- How it works
- Minimal WandB setup
- Option 1: via config dict
- Option 2: environment variable (simpler)
- export WANDB_API_KEY=your-key
- Dashboard features

## Step 3: Configure and Run

### Minimal Example

```python
from datetime import datetime
from pathlib import Path

# Pick whichever SDK matches where you want trials to run. AutoMLRunner is
# platform-agnostic — none of the SDKs is a default; the user picks.
from tao_sdk.platforms.brev       import BrevSDK       # Brev GPU instances
# from tao_sdk.platforms.slurm      import SlurmSDK      # SLURM cluster
# from tao_sdk.platforms.kubernetes import KubernetesSDK # K8s (EKS / GKE / on-prem)
# from tao_sdk.platforms.docker     import DockerSDK     # local Docker daemon
from tao_automl.runner import AutoMLRunner

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

sdk = BrevSDK()                                  # reads platform credentials from env
runner = AutoMLRunner(
    sdk=sdk,
    skill_dir=SKILL_BANK / "skills" / "models" / model_skill, # resolved skill dir
    action="train",
)
result = runner.run(
    train_dataset_uri=train_dataset_uri,
    automl_settings={
        "algorithm": algorithm,
        "metric": metric,
        "automl_max_recommendations": max_recommendations,
    },
    workspace_path=f"./automl_workspace/{TIMESTAMP}",  # timestamped to avoid collisions
    # Platform-specific create_job kwargs go here as **platform_kwargs.
    # See each platform's SKILL.md for the kwargs each accepts.
    gpu_count=8,
    num_nodes=1,
    gpu_type="H100",                             # Brev-specific
)
```

### Full Example (all options)

```python
def my_eval(rec, train_job_id):
    """Optional post-training evaluator. Return a float (the real metric)
    or None to fall back to the log-based extractor."""
    # e.g. read a results file uploaded by the container and compute the requested metric
    ...
    return 0.71

result = runner.run(
    # --- Required ---
    train_dataset_uri=train_dataset_uri,

    # --- Dataset + resources ---
    eval_dataset_uri=eval_dataset_uri,
    base_checkpoint="",
    image=image,                                      # only set to override skill_info's container_image

    # --- AutoML config ---
    automl_settings={
        "algorithm": algorithm,
        "metric": metric,
        "direction": direction,                       # explicit when needed
        "automl_max_recommendations": max_recommendations,
    },
    automl_hyperparameters=automl_hyperparameters,    # from model skill / schema
    custom_param_ranges=custom_param_ranges,          # from model skill / user constraints

    # --- Per-rec spec overrides ---
    spec_overrides=spec_overrides,                    # mandatory model-specific overrides from model skill

    # --- State + durability ---
    workspace_path=f"./my_experiment/{TIMESTAMP}",   # ALWAYS timestamp to avoid collisions
    resume=False,                                    # True → recovers in-flight jobs

    # --- Hooks (all optional, opt-in) ---
    metric_extractor=None,                           # custom log→metric parser
    eval_fn=my_eval,                                 # post-training real-metric eval
    on_recommendation=lambda r: print(f"launching rec {r.id}: {r.specs}"),
    on_result=lambda r, metric, status: print(f"rec {r.id} {status} → {metric}"),

    # --- Platform create_job kwargs (forwarded as **platform_kwargs) ---
    # SLURM:      partition, account, num_nodes, gpu_count
    # Kubernetes: namespace, node_selector, tolerations, num_nodes, gpu_count
    # Docker:     mounts, gpu_count
    # Brev:       instance_id, gpu_type, gpu_count
    gpu_count=8,
    num_nodes=1,
    gpu_type="H100",
)
```

### LLM-Powered Algorithm Example

For `llm`, `hybrid`, or `autoresearch`, use the same generic runner shape as above, plus the required LLM endpoint, model, and key in `automl_settings`. All model-specific hyperparameters, metric extractors, and `spec_overrides` must still come from the model skill.

**LLM endpoint configuration** (in order of precedence):
1. `automl_settings` keys: `llm_endpoint`, `llm_model`, `llm_api_key`; aliases `base_url`, `model`, and `api_key` are also accepted.
2. Environment variables: `AUTOML_LLM_ENDPOINT`, `AUTOML_LLM_MODEL`, `AUTOML_LLM_API_KEY`
3. Fallback env var for API key: `NVIDIA_API_KEY`
4. Defaults: NVIDIA inference endpoint (`https://inference-api.nvidia.com`) with `gcp/google/gemini-3.1-pro-preview`. Always pass the endpoint, model, and key explicitly for reproducible LLM-based testing.

### Programmatic API (without runner)

For tighter control, use the `AutoML` class directly:

```python
from tao_automl import AutoML

automl = AutoML(
    workspace="/tmp/my_experiment",
    network=network_arch,
    train_specs=my_train_spec_dict,
    settings={
        "algorithm": "bayesian",
        "metric": "loss",
        "automl_max_recommendations": 10,
    },
    wandb_config={"enabled": True, "project": "my-project"},
)

while not automl.is_complete():
    recs = automl.next_recommendation()
    for rec in recs:
        metric_value = train_model(rec.specs)    # your training function
        automl.report_result(rec.id, metric_value)

automl.finish()   # close WandB run
print("Best:", automl.get_best().specs)
```

### `automl_settings` keys

| Key | Type | Default | Description |
|---|---|---|---|
| `algorithm` | str | **required** | `bayesian`, `hyperband`, `bohb`, `asha`, `bfbo`, `dehb`, `pbt`, `hyperband_es`, `llm`, `hybrid`, `autoresearch` |
| `metric` | str | `"loss"` | Metric name. The implicit rule for direction is "contains `'loss'` → minimize, else maximize". Override with `direction`. |
| `direction` | `"minimize"` \| `"maximize"` | inferred | Explicit direction. Required only when it disagrees with the implicit rule. The runner transparently inverts reported values so callers always see their metric in its original scale. |
| `automl_max_recommendations` | int | 20 | Max trials (bayesian, bfbo, llm) |
| `automl_max_epochs` | int | 27 | Epoch budget (hyperband, bohb, asha, dehb) |
| `automl_reduction_factor` | int | 3 | Halving factor (hyperband variants) |
| `automl_max_concurrent` | int | 4 | Max parallel configs (asha only) |
| `automl_population_size` | int | 10 | Population size (pbt only) |
| `automl_max_experiments` | int | 50 | Max experiments (autoresearch only) |
| `llm_endpoint` | str | `https://inference-api.nvidia.com` | OpenAI-compatible API endpoint (llm, hybrid, autoresearch) |
| `llm_model` | str | `gcp/google/gemini-3.1-pro-preview` | LLM model name (llm, hybrid, autoresearch) |
| `llm_api_key` | str | from env | API key for the LLM endpoint |
| `research_program` | str | None | Free-text research directives for the autoresearch agent |
| `automl_delete_intermediate_ckpt` | bool | False | Delete non-best checkpoints to save storage. Hyperband-family algorithms defer deletion until bracket completion for safety. |
| `override_automl_disabled_params` | bool | False | Include params whose schema `automl_enabled` is False. For advanced users who want to search over params the network author didn't flag for AutoML. |

### `kpi` metric resolution

When `metric="kpi"`, the controller resolves the actual metric key from the network config's `metrics.monitoring_metric` field. Whether `kpi` is appropriate, and whether a custom `metric_extractor` is needed, is model-specific. Follow the model skill's **AutoML / HPO Notes**.

### `custom_param_ranges` format

Each entry can include:

| Field | Type | Description |
|---|---|---|
| `valid_min` | float/int/list | Min value. For list-valued parameters, pass the list shape required by the schema. |
| `valid_max` | float/int/list | Max value. Same list rules as min. |
| `valid_options` | list[str] | For categorical/ordered params: restrict to these values |
| `option_weights` | list[float] | Sampling weights for `valid_options`. Must match length. Higher weight = more likely to be sampled. |
| `disable_list` | bool | For params that can be float OR list: `True` keeps it as a single float for optimization, bypassing network list helpers. Use only when supported by the schema/model skill. |

Example with all features:

```python
custom_param_ranges={
    "<float_param>": {"valid_min": min_value, "valid_max": max_value, "disable_list": True},
    "<categorical_param>": {
        "valid_options": ["option_a", "option_b"],
        "option_weights": [0.7, 0.3],
    },
    "<list_param>": {"valid_min": [min_a, min_b], "valid_max": [max_a, max_b]},
}
```

### Model-specific search-space rules

Some networks have built-in search-space exclusions or algorithm restrictions. Do not document them here; read the model skill's **AutoML / HPO Notes** and let schema validation report unsupported combinations.

### LLM Analyzer (server-side range narrowing)

The controller supports automatic range narrowing via the LLM analyzer. Enable via environment variables before launching:

```python
os.environ["AUTOML_LLM_ANALYZER_ENABLED"] = "true"
os.environ["AUTOML_LLM_ANALYZER_INTERVAL"] = "5"        # analyze every 5 completed recs
os.environ["AUTOML_LLM_ANALYZER_NARROW_RANGES"] = "true" # auto-tighten custom_param_ranges
```

When enabled, after every N completed experiments the analyzer reviews patterns, assesses convergence, and optionally narrows search ranges to focus on promising regions. This happens server-side and persists the narrowed ranges.

### `spec_overrides`

`spec_overrides` keys are model-specific. Read the model skill's **Training Requirements**, **Per-Action Dataset Requirements**, and **Typical Spec Overrides** sections, then pass only the keys required or recommended there. Do not infer override keys from examples in this AutoML skill.

Every key you pass is validated against the skill's spec schema. Typos that look like existing keys raise `ValueError` with a suggestion; genuinely-new keys are accepted with a warning.

---

## WandB Experiment Tracking

AutoML optionally integrates with [Weights & Biases](https://wandb.ai) to track all experiments in a single dashboard.

### Setup

```bash
pip install wandb
# or (when reinstalling tao-run-automl with the wandb extra):
#   pip install "$("${TAO_SKILL_BANK_PATH:?}/scripts/resolve_versions_key.py" wheels.tao_automl_brev | sed 's/]/,wandb]/')"
```

### How it works

When `wandb_config={"enabled": True}` is passed:

1. The controller creates a WandB **run** named `automl_brain` in the specified project.
2. All recommendations are grouped under a WandB **group** (e.g. `automl_abc123`) so parent + child training runs appear together in the dashboard.
3. After every result, a **WandB table** (`automl_experiments`) is logged containing:
   - `experiment_id`, `job_id`, `status`, metric value, `best_epoch_number`
   - All varying hyperparameter values
4. Call `automl.finish()` (or let `runner.run()` complete) to finalize the WandB run.

### Minimal WandB setup

```python
# Option 1: via config dict
result = runner.run(
    ...,
    wandb_config={
        "enabled": True,
        "project": "tao-hpo",
        "api_key": "your-key",  # or set WANDB_API_KEY env var
    },
)

# Option 2: environment variable (simpler)
# export WANDB_API_KEY=your-key
result = runner.run(
    ...,
    wandb_config={"enabled": True, "project": "tao-hpo"},
)
```

### Dashboard features

Once tracking is active, you can:
- **Compare all trials** side-by-side in the WandB table view
- **Sort by metric** to find the best config instantly
- **Group by hyperparameter** to see which values correlate with good results
- **Link to child training runs** if the compute backend also logs to WandB (group name is available via `automl.wandb_group`)

---
