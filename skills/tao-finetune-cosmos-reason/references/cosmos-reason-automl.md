# Cosmos-RL AutoML / HPO

Load this only when `SKILL.md` points here for an AutoML/HPO task. If this conflicts with `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the current/compact source wins.

The base model URI or local path is required at runtime. Apply it
base model consistently to train (`policy.model_name_or_path`) and
post-training evaluation (`model.base_model_path`) unless the user explicitly
provides a different HuggingFace model id, `hf_model://...` URI, or
cluster-local snapshot, or converted `Cosmos3-Nano-VLM` directory. If the
conversion helper was required for the selected image, treat the converted
directory as the PTM for the whole run.

Do not hardcode dataset paths in this reusable model skill. Dataset locations
must come from the user's current request, a selected dataset profile, or direct
spec overrides for that run. For a user-provided Cosmos-RL train/eval root, map
the run inputs to concrete spec keys:

```text
custom.train_dataset.annotation_path=<train_root>/annotations.json
custom.train_dataset.media_path=<train_root>
custom.val_dataset.annotation_path=<eval_root>/annotations.json
custom.val_dataset.media_path=<eval_root>
```

When annotation `video` values are relative to a `videos/` subdirectory, use
direct spec mode for `media_path` rather than plain dataset-root mode. If media
is packaged as `videos.tar.gz`, use the extracted `videos/` directory when
present, or the archive only if the selected runtime extracts it before dataset
lookup. Do not edit or patch the user's source annotation files unless the user
explicitly asks for a dataset repair.

If the user's objective names `accuracy` or an accuracy target such as
`>=90%`, optimize an evaluation metric, not `val/avg_loss`. Use AutoMLRunner's
`eval_fn` to run the model skill's `evaluate` action on the validation dataset
after each recommendation, with `task=""`, `model.enable_lora=false`, and
`model.base_model_path` set to the same base model used for training. Use true
only when the user explicitly requested a PEFT/LoRA search and the train spec
contains a LoRA table. Return
the evaluator's task metric and set `direction="maximize"`. Use `accuracy` for
constrained classification prompts and BERTScore F1 for free-form
summarization/answering prompts when the user asks for semantic text quality.
Use `val/avg_loss` only when the user accepts a proxy metric or no task metric
is available.

`val/avg_loss` is a training-only metric; the `evaluate` action does not emit
it. The mandatory AutoML baseline, per-recommendation evaluation, and final
evaluation must therefore use one consistent evaluator task metric through
`eval_fn` and `final_eval_fn`. If the user explicitly chooses training loss as
a proxy, obtain approval to skip the evaluation-backed baseline comparison and
state that limitation in the report.

Before launching AutoML for an accuracy objective, run the model's evaluate
action once after preflight and before recommendation jobs on the same
validation subset. Use the selected base model or starting checkpoint,
`task=""`, and the same prompt/metric setup planned for per-recommendation
evaluation. Report that eval job id, result path, and accuracy in the launch
review before asking for confirmation to start recommendations. The final
AutoML summary must compare this baseline accuracy, every recommendation's
accuracy, and the selected best recommendation.

## Zero-shot Evaluate AutoML / Auto-Prompter

For Metropolis/VSS-style zero-shot video QA, use `action="evaluate"` when the
user asks to optimize prompts, inference config, or Auto-Prompter behavior
without training. This launches multiple real Cosmos-RL evaluate jobs over the
same eval dataset/model and compares the requested task metric. It is useful
before fine-tuning or DEFT because it can identify whether prompt wording,
frame sampling, or generation settings recover enough accuracy without changing
weights.

The packaged Cosmos evaluate schema exposes this dataset-neutral joint
prompt/config search space:

```text
dataset.system_prompt
vision.num_frames
generation.max_tokens
generation.temperature
generation.repetition_penalty
generation.presence_penalty
generation.frequency_penalty
```

These are parameter names, not packaged experiment values. Resolve the
baseline with `scripts/evaluation_workflow.py` first. The system-prompt seed,
frame value/range, generation length/range, dataset, checkpoint, and results
path must come from the selected training plan or explicit current-run intake.

There are three distinct operating modes:

1. **Bounded fallback:** use `algorithm="bayesian"` and require the user to
   supply the exact categorical prompt seeds and numeric candidate ranges for
   this run. This is useful for a cheap prompt ablation, but it is not a
   reflective Auto-Prompter.
2. **Generic reflective fallback:** use `algorithm="autoresearch"`, provide the LLM
   endpoint/model/key, and set
   `evolvable_text_parameters=["dataset.system_prompt"]`. The four packaged
   prompts become seeds; the agent may write new prompts and jointly change the
   declared frame-sampling and generation settings. Supply `feedback_fn` so the
   next proposal sees compact, leak-free training failures instead of only an
   aggregate score. This is TAO autoresearch, not GEPA, and must not be reported
   as the Metropolis Auto-Prompter result.
3. **TAO GEPA Auto-Prompter:** install `nvidia-tao-automl[autoprompter]` and use
   `tao_automl.GEPAutoPrompter`, `TAOGEPAAdapter`, and
   `TAOActionBatchRunner`. TAO owns this integration; the original Auto-Tuner
   and VLMEvalKit repositories are read-only references/metric providers and
   must not receive TAO feature changes. A TAO evaluate callback launches one
   action job for `run_batch(candidate, items) -> outputs`, preserving GEPA's
   aligned per-example scores. Keep fixed inference settings in the base action
   spec; only user-supplied prompt components being evolved belong in GEPA's
   seed. Use the
   generic TAO autoresearch mode above when prompt and bounded config knobs must
   be explored jointly.

For GEPA, distinguish the metric-call budget from any proposal-count stopper.
Each proposal scores both its parent and new prompt on the reflection minibatch,
and an accepted proposal is then scored on complete validation. Do not leave a
small `MaxCandidateProposalsStopper` in place when raising the metric budget: a
four-proposal stopper ends search after four attempts even if substantial metric
budget remains. Prefer the metric budget plus a wall-time guard, or set a
proposal cap that cannot become the unintended limiting condition.

For a strict budget comparison, checkpoint one seeded GEPA trajectory by metric
calls instead of comparing independent stochastic searches. In the full reference video-QA
Nano run, the best prompt available at 1,124 calls scored 1,721/2,676 (64.31%)
in a fresh full evaluation; the best prompt through 3,002 calls scored
1,754/2,676 (65.55%). The same fresh default run scored 1,449/2,676 (54.15%).
All three specs used the same model, data, order, 8 frames, decoding settings,
seed, and exact-match metric. Treat these as integration evidence, not as a
VANTAGE benchmark.

Use the canonical set-level `macro_f1` metric for VANTAGE binary event
verification. GEPA still needs decomposable per-item feedback for reflection
and proposal gating, but TAO's `GEPAutoPrompter` reranks every accepted
candidate with `binary_aggregate` on the complete validation set and selects on
true Macro-F1 before running test. Pass VLMEvalKit's read-only `binary_metric`
and `binary_aggregate` callbacks; no scorer changes are required. Use `accuracy`
only for tasks whose official metric is accuracy. Use BERTScore F1 or another
model-skill-supported semantic metric only for free-form answers where exact
matching is not meaningful.

Example reflective evaluate AutoML setup:

```python
action = "evaluate"
automl_settings = {
    "algorithm": "autoresearch",
    "metric": "macro_f1",
    "direction": "maximize",
    "automl_max_experiments": 20,
    "llm_endpoint": llm_endpoint,
    "llm_model": llm_model,
    "llm_api_key": llm_api_key,
    "evolvable_text_parameters": ["dataset.system_prompt"],
    "research_program": (
        "Optimize zero-shot event verification. Use visible temporal evidence; "
        "learn from the supplied false-positive and false-negative examples."
    ),
}
automl_hyperparameters = [
    "dataset.system_prompt",
    "vision.num_frames",
    "generation.max_tokens",
    "generation.temperature",
    "generation.repetition_penalty",
    "generation.presence_penalty",
    "generation.frequency_penalty",
]
custom_param_ranges = {
    "vision.num_frames": {
        "value_type": "ordered_int",
        "valid_options": [4, 8, 16],
    },
    "generation.max_tokens": {
        "value_type": "ordered_int",
        "valid_options": [256, 512, 1024],
    },
    "generation.temperature": {"valid_min": 0.0, "valid_max": 0.4},
}
```

For the generic TAO fallback, `eval_fn` returns the validation score while
`feedback_fn` reads training-split artifacts and returns a bounded, leak-free
payload, for example:

```python
{
    "failures": [
        {
            "query": "Did a second person cross during the same gate cycle?",
            "generated_output": "No",
            "feedback": "The response did not track the complete gate-open interval."
        }
    ]
}
```

Do not include gold/expected labels, sample or video IDs, media paths, or the
full result corpus in `feedback_fn`. Select representative training failures,
describe the failure mode without revealing the answer, and keep the payload
compact enough for the reflection model. TAO removes common identifier and
ground-truth fields, but callers must also avoid leaking answers in free-form
text.

### Dataset and reporting protocol

- Use three disjoint roles: reflect on train, select candidates on validation,
  and report once on untouched test. The current deterministic split of the 163
  VANTAGE event-verification items is approximately 66 train / 32 validation /
  65 test. The older 98/65 protocol and its reported gains are historical
  two-way results, not validation of the current implementation.
- For generic TAO autoresearch, select the best prompt/config on validation
  Macro-F1, then use `final_eval_fn` for the untouched test. For TAO GEPA, pass
  `aggregate_metric_fn=binary_aggregate` and
  `aggregate_metric_key="macro_f1"`; the returned candidate is then selected on
  true validation Macro-F1. In either mode, report zero-shot baseline, tuned
  result, absolute percentage-point lift, and job/result paths, plus
  full-dataset results when required.
- For Metropolis alert verification, report VK/model accuracy and AB/end-to-end
  Alerts accuracy side by side. If both are available for every recommendation,
  return both metrics and configure `automl_settings["objectives"]`; otherwise
  optimize VK in-loop and run AB as final system validation.
- Never describe a four-static-prompt run, a subset-only run, a reference video-QA
  run, or the historical two-way VANTAGE result as validation of the current
  three-way GEPA/VLM implementation. Those remain useful ablations or historical
  evidence and must be labeled as such.

If the eval spec uses `vision.num_frames`, do not also search `vision.fps` by
default. Search `vision.fps` only when the user explicitly requests FPS-based
sampling and the spec/runtime has been switched away from frame-count sampling.
Sixteen frames is a higher-cost option intended for detail- or coverage-bound
failures on sufficiently provisioned local evaluation. The current Cosmos
evaluator materializes processed video inputs before inference; split a large
corpus into complete, video-disjoint execution shards when one monolithic action
would exhaust host memory. Use the repository evaluator's final metric; do not
add a second prediction-ID matching gate after evaluation.
Validate every referenced media path and the resolved decoder contract before
launching the baseline. Do not invent frame rates or patch source annotations
unless the user explicitly requests dataset repair.

For the evaluator prompt "search over learning rate, batch size, number of
epochs, weight decay, warmup ratio", map the requested knobs to:

```text
learning rate     -> train.optm_lr
batch size        -> train.train_batch_per_replica
number of epochs  -> fixed train.epoch=2 by default; do not include in search unless explicitly requested
weight decay      -> train.optm_weight_decay
warmup ratio      -> fixed train.optm_warmup_epochs=0 by default; do not include in search unless explicitly requested
```

The schema exposes `train.optm_warmup_epochs`, not a native warmup-ratio field.
If the evaluator requires a ratio to be preserved exactly, stop and report that
the current Cosmos-RL schema needs a first-class warmup-ratio parameter.

Example custom ranges for the Cosmos Reason 3 AutoML evaluation prompt:

```python
automl_hyperparameters=[
    "train.optm_lr",
    "train.train_batch_per_replica",
    "train.optm_weight_decay",
]
custom_param_ranges={
    "train.optm_lr": {"valid_min": 1e-5, "valid_max": 1e-3},
    "train.train_batch_per_replica": {
        "value_type": "ordered_int",
        "valid_options": [8, 16, 32],
    },
    "train.optm_weight_decay": {"valid_min": 0.0, "valid_max": 0.1},
}
```

Keep `train.train_policy.mini_batch=1` unless the user explicitly changes it,
so all listed batch sizes remain divisible by the micro-batch size. For small
datasets, cap `train.train_batch_per_replica` so it does not exceed
`floor(num_train_samples / policy.parallelism.dp_shard_size)`. When the
annotation count is known, pass it as `automl_settings["train_sample_count"]`;
current `AutoMLRunner` versions use that to cap invalid batch-size
recommendations before launch and record the adjustment in AutoML history.
For integer knobs with discrete choices, include `value_type: "ordered_int"`
with `valid_options`; integer `valid_options` alone are ignored by the current
Bayesian sampler.

Before launching recommendation jobs, show the user the exact number of
recommendations, search parameters, ranges/defaults, planned dataset subset
size, expected runtime per recommendation, and total expected runtime. If the
first sampled recommendation is available before launch, include its concrete
config. If the estimate exceeds the user's time limit, reduce budget or search
space only after user confirmation.
