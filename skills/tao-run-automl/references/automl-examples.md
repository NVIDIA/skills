# TAO AutoML Example Conversations

Example user requests and the correct high-level agent response shape.

Load this file only when the compact `SKILL.md` points here for the current task. If this reference conflicts with `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the compact/current source wins.

## Contents

- User: "Optimize `<network_arch>` training on my dataset"
- User: "I want the real task metric, not the default proxy"
- User: "Use the LLM to figure out the best hyperparameters"
- User: "Run a fully autonomous search — let the AI figure everything out"
- User: "Resume the experiment from yesterday"
- User: "Switch to ASHA with 30 trials and track in WandB"
- User: "I don't know what parameters to tune for `<network_arch>`"

## Example Conversations

### User: "Optimize `<network_arch>` training on my dataset"

```
Agent: I'll read the model skill for `<network_arch>` first, then generate a fresh timestamped runner using its required datasets, spec overrides, metric, and recommended search space. I need:
1. The dataset URI or URIs required by that model skill
2. How many trials? (default: 10)

User: s3://bucket/data/train, 10 recs

Agent: Launch review before I spend GPU budget:
- model skill, network_arch, platform, image, GPU/node shape, workspace, dataset paths/counts, metric/direction, algorithm, budget, search bounds, estimated runtime, and automatic baseline eval job id/result path/metric
- exact generated recommendation configs for the initial launch batch, created in a review-only step before any recommendation job is submitted

User: confirmed

[executes runner.run(...)]
```

### User: "I want the real task metric, not the default proxy"

```
Agent: For a real task metric, I'll use the eval_fn hook described by the model skill. This adds per-rec cost, so I’ll adjust the budget if needed.
[shows launch review with eval_fn cost and metric/direction]
User: confirmed
[executes runner.run(metric=task_metric, direction=direction, eval_fn=model_specific_eval, ...)]
```

### User: "Use the LLM to figure out the best hyperparameters"

```
Agent: I'll use the LLM algorithm — it reasons about your network architecture and learns from each experiment.
I need three things for the LLM brain:
1. LLM endpoint URL (default: https://inference-api.nvidia.com)
2. LLM model name (default: gcp/google/gemini-3.1-pro-preview)
3. API key available in `AUTOML_LLM_API_KEY` or `NVIDIA_API_KEY`

User: endpoint "https://inference-api.nvidia.com", model "gcp/google/gemini-3.1-pro-preview"

Agent: I verified the LLM endpoint/key env var is available, the selected train image is confirmed, and platform/data preflight passed. Launch review: 10 LLM-guided recs, metric/direction, search bounds, workspace, GPU/node shape, and estimated runtime.
User: confirmed
[executes runner.run(automl_settings={
    "algorithm": "llm",
    "llm_endpoint": "https://inference-api.nvidia.com",
    "llm_model": "gcp/google/gemini-3.1-pro-preview",
    "llm_api_key": os.environ["AUTOML_LLM_API_KEY"],
    ...
})]
```

### User: "Run a fully autonomous search — let the AI figure everything out"

```
Agent: I'll use the autoresearch algorithm. It will:
1. Retrieve tuning knowledge for your network
2. Propose modifications, pre-screen them, verify before launch
3. Make keep/discard decisions with reasoning after each experiment
4. Automatically stop when it detects diminishing returns

I'll cap it at 30 experiments. I need:
- Dataset URI
- LLM endpoint (default: https://inference-api.nvidia.com)
- LLM model (default: gcp/google/gemini-3.1-pro-preview)
- LLM API key in `AUTOML_LLM_API_KEY` or `NVIDIA_API_KEY`

User: dataset s3://bucket/data, endpoint https://inference-api.nvidia.com, model gcp/google/gemini-3.1-pro-preview

[shows launch review with 30-experiment cap, LLM config source, data/image/platform checks, exact generated configs for the initial launch batch, and estimated runtime]
User: confirmed

[executes runner.run(automl_settings={
    "algorithm": "autoresearch", "automl_max_experiments": 30,
    "llm_endpoint": "https://inference-api.nvidia.com",
    "llm_model": "gcp/google/gemini-3.1-pro-preview",
    "llm_api_key": os.environ["AUTOML_LLM_API_KEY"],
    ...
})]
```

### User: "Resume the experiment from yesterday"

```
Agent: Resuming. The runner will poll any jobs that were in flight when we stopped, report their results, then show a resume review before submitting any new recommendation jobs.
[shows resume review with completed/failed/in-flight recs, remaining budget, metric state, image/platform/data checks, and ETA]
User: confirmed
[executes runner.run(..., workspace_path="./my_experiment", resume=True)]
```

### User: "Switch to ASHA with 30 trials and track in WandB"

```
Agent: Switching to ASHA — parallel + early-stopping. I’ll first check the model skill for any warning that short trials are dominated by startup, validation, or checkpoint cost, then generate a fresh runner with WandB tracking.
[shows launch review with ASHA rungs, max concurrent jobs, WandB project, search bounds, and estimated runtime]
User: confirmed
[executes runner.run(
    automl_settings={"algorithm": "asha", "automl_max_epochs": 9, ...},
    wandb_config={"enabled": True, "project": "tao-hpo"},
)]
```

### User: "I don't know what parameters to tune for `<network_arch>`"

```
Agent: I can generate a configuration from your goal description. What are you trying to achieve?

User: Maximize my task metric on a small custom dataset with 1 GPU.

Agent: Based on your constraints, I recommend:
- Algorithm: bayesian (sample-efficient for small budgets)
- Parameters: from the model skill's AutoML / HPO Notes and the generated schema
- Budget: 12 recs
- Ranges: from the model skill and user constraints
[uses NLConfigGenerator, shows launch review with generated config and estimated runtime]
User: confirmed
[executes runner.run with the generated config]
```
