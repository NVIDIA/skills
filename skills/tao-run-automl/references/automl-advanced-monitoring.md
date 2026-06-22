# TAO AutoML Advanced Monitoring

LLM/agentic options, optional hooks, progress monitoring, resume, result interpretation, pitfalls, and status queries.

Load this file only when the compact `SKILL.md` points here for the current task. If this reference conflicts with `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the compact/current source wins.

## Contents

- Natural Language Configuration
- config = {
- "automl_algorithm": "bayesian",
- "automl_hyperparameters": ["<param_from_model_schema>", ...],
- "algorithm_specific_params": {"automl_max_recommendations": 15},
- "metric": "<metric_from_model_skill_or_user_request>",
- "reasoning": "..."
- }
- LLM Analyzer (works with ANY algorithm)
- After every 5 completed experiments, call:
- analysis = {
- "patterns": ["..."],
- "convergence_assessment": "improving",
- "recommendations": ["..."],
- "suggested_ranges": {"<param_name>": {"min": ..., "max": ...}},
- }
- Autoresearch Agent Components
- Research Programs
- Validate before running
- Advanced hooks (opt-in)
- `metric_extractor(logs: str, metric_name: str) → float | None`
- `eval_fn(rec, train_job_id: str) → float | None`
- Step 4: Monitor Progress
- Resume after interruption
- Step 5: Interpret Results
- How to report to the user
- If all recs failed
- Model-Specific Notes
- Common Pitfalls
- Querying Experiment Status
- Progress summary
- Best config
- Per-rec details
- In-flight jobs

## LLM/Agentic Features Deep Dive

### Natural Language Configuration

Don't know which algorithm or parameters to use? The `NLConfigGenerator` translates plain English into a valid AutoML configuration:

```python
from tao_automl.brain.nl_config import NLConfigGenerator

generator = NLConfigGenerator()   # uses NVIDIA NIM by default
config = generator.generate_config(
    user_prompt=user_goal,
    network=network_arch,
    available_parameters=param_records,  # from generate_hyperparams_to_search()
    hardware_info=hardware_info,
)
# config = {
#   "automl_algorithm": "bayesian",
#   "automl_hyperparameters": ["<param_from_model_schema>", ...],
#   "algorithm_specific_params": {"automl_max_recommendations": 15},
#   "metric": "<metric_from_model_skill_or_user_request>",
#   "reasoning": "..."
# }
```

### LLM Analyzer (works with ANY algorithm)

The `LLMAnalyzer` can be used alongside any classical algorithm to provide periodic analysis of experiment results:

```python
from tao_automl.brain.llm_analyzer import LLMAnalyzer

analyzer = LLMAnalyzer(analysis_interval=5, narrow_ranges=True)

# After every 5 completed experiments, call:
analysis = analyzer.analyze(
    experiments=experiment_history,
    parameters=param_records,
    network=network_arch,
    metric_name=metric,
    metric_direction=direction,
    best_metric=best_metric,
)
# analysis = {
#   "patterns": ["..."],
#   "convergence_assessment": "improving",
#   "recommendations": ["..."],
#   "suggested_ranges": {"<param_name>": {"min": ..., "max": ...}},
# }
```

When `narrow_ranges=True`, the analyzer suggests tighter search bounds based on observed patterns. These can be applied to dynamically focus the search.

### Autoresearch Agent Components

The `autoresearch` algorithm integrates five AutoML-Agent concepts:

| Component | What it does | When it runs |
|---|---|---|
| **KnowledgeRetriever** (RAP) | Retrieves built-in tuning knowledge for the requested network and optionally web-searched papers/benchmarks | Once at initialization |
| **SpecPrescreener** | LLM predicts which of N candidate configs are worth running, WITHOUT training. Saves GPU budget by filtering unlikely-to-improve configs. | Before each trial — proposes 3 candidates, pre-screens to pick the best 1 |
| **MultiStageVerifier** | Pre-launch: validates proposed changes won't crash/OOM. Post-result: checks metrics are plausible (not NaN, not anomalous). | Before launch + after result |
| **ExperimentTracker** | Tracks full history with keep/discard decisions and reasoning | After each result |
| **LLMAnalyzer** | Periodic pattern detection, convergence assessment, and optional range narrowing | Every N completed experiments |

### Research Programs

For complex multi-phase optimization, define a research program:

```python
from tao_automl.brain.research_program import ResearchProgram, ResearchPhase

program = ResearchProgram(
    objective=objective,
    network=network_arch,
    phases=[
        ResearchPhase(
            name="Phase 1",
            algorithm="bayesian",
            parameters=["<param_from_model_schema>", "..."],
            trials=8,
        ),
        ResearchPhase(
            name="Phase 2",
            algorithm="asha",
            parameters=["<another_param_from_model_schema>", "..."],
            trials=15,
            carry_forward="best",   # best values carry into this phase
        ),
    ],
)

# Validate before running
issues = program.validate(
    available_parameters=available_parameters,
    available_algorithms=["bayesian", "asha"],
)
```

---

## Advanced hooks (opt-in)

Both hooks are optional. If neither is provided, the runner uses its built-in log regex extractor.

### `metric_extractor(logs: str, metric_name: str) → float | None`

Called on every poll of the training container's logs. Return the most recent/final metric value seen, or `None` if the metric isn't yet present.

Use it when:
- Your container emits the metric in a non-standard log format the built-in regex misses.
- You want to parse values from log lines instead of using the generic patterns.
- Your metric needs derivation from multiple log fields.

```python
import re

def extract_custom_metric(logs: str, metric_name: str):
    m = re.search(rf"{re.escape(metric_name)}:\s*([0-9.]+)", logs)
    return float(m.group(1)) if m else None

runner.run(..., metric_extractor=extract_custom_metric)
```

Exceptions raised inside the extractor are caught and logged; the runner continues polling.

### `eval_fn(rec, train_job_id: str) → float | None`

Called once after a rec's training job reaches a terminal state, before the result is reported to the brain. Whatever it returns **overrides** any value captured by `metric_extractor` and becomes what the brain optimizes on.

Use it when:
- The real task metric lives outside the training logs.
- You want a true-test-metric sweep without building surrounding plumbing yourself.
- Per-rec cost is acceptable relative to `metric_extractor`.

```python
def eval_on_held_out(rec, train_job_id):
    # Implement the model-specific evaluation flow documented in the model skill.
    metric_value = run_model_specific_eval(rec, train_job_id)
    return metric_value

runner.run(
    ...,
    automl_settings={"metric": task_metric, "direction": direction, ...},
    eval_fn=eval_on_held_out,
)
```

Exceptions from `eval_fn` are caught and logged — the runner falls back to the log-extracted metric for that rec.

---

## Step 4: Monitor Progress

`runner.run()` blocks until all recommendations complete. Use callbacks for
live launch/result events from the process that owns the runner:

```python
def on_rec(rec):
    print(f"Rec {rec.id}: trying {rec.specs}")

def on_result(rec, metric, status):
    print(f"Rec {rec.id}: {status}, metric={metric}")

result = runner.run(..., on_recommendation=on_rec, on_result=on_result)
```

For status questions from a separate shell, agent turn, or detached monitor,
default to the structured AutoML state instead of reading launcher logs:

```python
from tao_automl import query_status

status = query_status("<full_workspace_path>/run_<timestamp>")
```

Use `status["progress"]`, `status["best"]`, `status["recommendations"]`, and
`status["active_jobs"]` as the source of truth for recommendation ids, specs,
metrics, success/failure/pending counts, and active TAO job ids. The state comes
from `<workspace>/.automl/controller/`, `<workspace>/.automl/best_rec/`,
`<workspace>/.automl/brain/`, and `<workspace>/active_jobs.json`.

Do not comb through launcher logs for normal AutoML status. Logs are fallback
debug material only: use them when the user explicitly asks for logs, when
`query_status()` reports no usable state, when diagnosing a failed child job, or
when checking metric-extractor / LLM-client warnings that are not represented in
the structured state. If live backend queue state is needed, join the active TAO
job ids from `query_status()["active_jobs"]` with the selected SDK's job status
API or the platform scheduler (`squeue` for SLURM); do not derive recommendation
results from logs.

Each rec takes 10–90 minutes depending on model size, dataset, epochs, and checkpoint save cost. Don't assume failure during long uploads.

### Resume after interruption

If the orchestrator dies mid-run (network timeout, machine sleep, Ctrl-C), re-run with `resume=True` and the **full suffixed path** (including the `run_<timestamp>` directory):

```python
result = runner.run(
    ...,
    workspace_path="./my_experiment/run_20260423_183015",   # full suffixed path
    resume=True,
)
```

When `resume=True`, the runner does NOT append a new timestamp suffix — it reuses the path as-is.

Behaviour on resume:
1. **Brain state** is reloaded from `<workspace>/.automl/*` — all completed rec results are already registered.
2. **Any in-flight jobs** recorded in `<workspace>/active_jobs.json` (persisted after each submission) are polled to terminal, their metrics extracted, and reported to the brain — *before* the main propose-new-rec loop starts. No duplicate submissions; no leaked GPU work from the previous orchestrator.
3. After recovery, the loop continues normally until `automl.is_complete()`.

---

## Step 5: Interpret Results

The result is a plain dict:

```python
{
    "best": {
        "rec_id": 4,
        "specs": {"<param_name>": "<value>", "...": "..."},
        "metric_value": 0.7077,
    },
    "progress": {
        "completed": 8, "total": 8,
        "best_metric": 0.7077, "best_rec_id": 4,
        "algorithm": "bayesian",
    },
    "history": [
        {"rec_id": 0, "metric": 0.6308, "status": "success"},
        {"rec_id": 1, "metric": 0.7077, "status": "success"},
        ...
    ],
}
```

Metric values in `best` and `history` are always in the original scale the user provided — direction inversion (if any) is undone before the dict is returned.

For multi-fidelity algorithms (`hyperband`, `bohb`, `asha`, `dehb`, `hyperband_es`, and `pbt`), `best` is selected from the largest observed training budget once those candidates exist. Report any lower-budget metric that beats the selected final-budget metric as promotion context rather than treating it as the downstream checkpoint.

### How to report to the user

1. **Best config** — show the winning hyperparameters and metric value.
2. **Comparison table** — rank recs by metric and highlight the best. For multi-fidelity algorithms, rank the largest-budget candidates first and include lower-budget winners as context when they differ from the selected final checkpoint.
3. **Insights** — call out what the optimizer learned from the requested parameters and metric.
4. **WandB link** — if tracking was enabled, provide the dashboard URL.
5. **Next steps** — suggest:
   - More recs (re-run with `resume=True` + higher `automl_max_recommendations`).
   - Train longer with the best config using `sdk.create_job(specs=result["best"]["specs"])`.
   - Run a downstream evaluation on the best checkpoint.
   - Run the model skill's recommended export/deploy workflow for the best model.

### If all recs failed

Check common issues:
- **Dataset path wrong** — verify the URI points to the layout required by the model skill.
- **Metric never appears** — verify the model skill's required metric-related overrides and custom extractor are present.
- **Checkpoint or eval artifact missing** — verify the model skill's checkpoint/export/eval requirements.
- **Model or data download timeout** — inspect backend logs and model-skill error patterns.
- **OOM** — reduce the model-specific batch, resolution, sequence length, or memory-heavy knobs recommended by the model skill.
- **Cached data corruption** — inspect the model skill's dataset/cache error patterns and clear only the affected cache path if documented.
- **LLM endpoint unreachable** (llm/hybrid/autoresearch only) — the brain falls back to random sampling. Check `AUTOML_LLM_ENDPOINT` and `AUTOML_LLM_API_KEY`. Verify with: `curl -s $AUTOML_LLM_ENDPOINT/models -H "Authorization: Bearer $AUTOML_LLM_API_KEY"`.

If the runner reports no new recommendations and there are no pending/running
child jobs, treat the AutoML run as exhausted instead of continuing to poll
forever. Inspect the failed child job logs, fix the model skill/config/setup
issue, then relaunch from a fresh runner or resume only after the failed cause
is corrected.

---

## Model-Specific Notes

Model-specific notes do not belong in this AutoML skill. For every requested model, resolve the packaged `model_skill`, read `<bank-root>/skills/models/<model_skill>/SKILL.md`, and use its **Training Requirements**, **Per-Action Dataset Requirements**, **Typical Spec Overrides**, **AutoML / HPO Notes**, and **Error Patterns** sections as the source of truth.

---

## Common Pitfalls

1. **`skill_dir` not passed (or wrong path).** `AutoMLRunner(skill_dir=...)` requires an absolute path to a model directory inside the skill bank. The runner raises `FileNotFoundError: skill_info.yaml not found at <skill_dir>/references/skill_info.yaml` if the path is wrong. Use the same bank root the agent loaded this SKILL.md from; combine with `skills/models/<model_skill>/` after resolving aliases.
2. **Wrong LLM endpoint (404).** Use `https://inference-api.nvidia.com` for NVIDIA inference API testing and pass it explicitly in `automl_settings`. The LLM brain falls back to random sampling on LLM failure, so check logs for "LLM call failed" before trusting the run as LLM-guided.
3. **Model-specific training failures (data format, missing datasets, invalid params).** Each network has unique training requirements. ALWAYS read `<bank-root>/skills/models/<model_skill>/SKILL.md` — the "Training Requirements" and "Error Patterns" sections document model-specific failure modes that apply to AutoML recs too.
4. **Workspace path collisions.** Running the same script twice overwrites the previous experiment. Always include a timestamp: `workspace_path=f"./automl_workspace/{TIMESTAMP}"` where `TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")`.
5. **Using a weak proxy metric.** The brain can optimize a metric that does not reflect real task quality. Use the metric recommended by the model skill or provide `eval_fn`.
6. **Implicit direction trap.** If the metric name does not imply the desired direction, set `direction` explicitly.
7. **Spec-override typos.** `save_freq_in_epochs` (plural) used to silently do nothing; now raises `ValueError` with suggestion. If you see that error, it's the fix working.
8. **Orchestrator dies mid-sweep.** Relaunch with the same `workspace_path` and `resume=True`. In-flight jobs are recovered from `active_jobs.json`.
9. **Rec never reports a metric.** Check the model skill's metric-emission requirements and custom extractor guidance.
10. **Parallel Bayesian arms.** Bayesian is inherently sequential. If you want parallelism, use `asha`. If you use multiple `AutoMLRunner` instances, give each its own `<SDK>(state_file=...)` (e.g., `BrevSDK(state_file=...)`, `KubernetesSDK(state_file=...)`) to avoid SQLite write races on the SDK's job store.
11. **LLM brain returning random configs.** If every LLM recommendation looks random, the LLM endpoint is probably failing silently. Check the logs for "LLM call failed" warnings. Verify your API key and endpoint are correct. Common cause: using the wrong endpoint URL (see pitfall #2).
12. **`openai` package not installed.** The `llm`, `hybrid`, and `autoresearch` algorithms require the `openai` Python package. Install with `pip install openai` or reinstall tao-run-automl with the `llm` extra by resolving the platform wheel key from `versions.yaml` and appending `,llm` to the extra.
13. **WandB not logging.** Ensure `wandb_config={"enabled": True}` is passed and either `api_key` is in the config or `WANDB_API_KEY` is set in the environment. Check logs for "WandB initialized" confirmation.
14. **`No default train specs found` for a network.** The skill bank model directory is missing `references/spec_template_train.yaml`, or the packaged AutoML support check is missing `schemas/train.schema.json`. Generate both during skill-bank maintenance and ship them with the plugin; do not expect `~/tao-core` to exist on the runtime machine.
15. **`conda run` buffers output.** When running AutoML via `conda run -n tao_sdk python script.py`, all output is buffered until completion. Use `PYTHONUNBUFFERED=1 ~/miniconda3/envs/tao_sdk/bin/python script.py` for real-time output.

---

## Querying Experiment Status

Use `query_status()` to check experiment progress from a separate process. This
is the default status path for AutoML; do not parse launcher logs unless the user
asks for log output or you are debugging missing/failed structured state.

```python
from tao_automl import query_status

status = query_status("./my_experiment")

# Progress summary
p = status["progress"]
print(f"{p['completed']}/{p['total']} recs done, "
      f"{p['succeeded']} succeeded, {p['failed']} failed")

# Best config
if status["best"]:
    print(f"Best: rec {status['best']['rec_id']}, "
          f"metric={status['best']['metric_value']}, "
          f"specs={status['best']['specs']}")

# Per-rec details
for rec in status["recommendations"]:
    print(f"  Rec {rec['rec_id']}: {rec['status']} "
          f"metric={rec['metric_value']} specs={rec['specs']}")

# In-flight jobs
for job in status["active_jobs"]:
    print(f"  Active: rec {job['rec_id']} job {job['job_id']}")
```

The function reads from the persisted state store (`<workspace>/.automl/`) and `active_jobs.json`. It is safe to call while the runner is active — no locking conflicts.

The `AutoML` class also exposes `get_status()` for in-process queries:

```python
automl = AutoML(workspace=..., ...)
status = automl.get_status()
```

---
