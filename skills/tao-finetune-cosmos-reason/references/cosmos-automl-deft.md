# Cosmos-Reason AutoML And DEFT Notes

Cosmos-Reason AutoML/HPO policy, search-space guidance, and DEFT support notes.

Load this file only when the compact `SKILL.md` points here for the current task. If this reference conflicts with `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the compact/current source wins.

## Contents

- DEFT Support
- Gap Analysis (`scripts/analyze_gaps.py`)

## AutoML / HPO Notes

Requests for "Cosmos Reason 3", "Cosmos3 Nano Reasoner", or
`nvidia/Cosmos3-Nano` are handled by this skill. The packaged default base
model is `hf_model://nvidia/Cosmos3-Nano`; override it only when the user
explicitly provides a different HuggingFace model id, `hf_model://...` URI, or
cluster-local snapshot. Apply the same base model consistently to train
(`policy.model_name_or_path`) and post-training evaluation
(`model.base_model_path`).

Do not hardcode dataset paths in this reusable model skill. Dataset locations
must come from the user's current request, a selected dataset profile, or direct
spec overrides for that run. For a user-provided Cosmos-RL train/eval root, map
the run inputs to concrete spec keys:

```text
custom.train_dataset.annotation_path=<train_root>/annotations.json
custom.train_dataset.media_path=<train_root>/videos
custom.val_dataset.annotation_path=<eval_root>/annotations.json
custom.val_dataset.media_path=<eval_root>/videos
```

When annotation `video` values are relative to a `videos/` subdirectory, use
direct spec mode for `media_path` rather than plain dataset-root mode. If media
is packaged as `videos.tar.gz`, use the extracted `videos/` directory when
present, or the archive only if the selected runtime extracts it before dataset
lookup. Do not patch optional annotation fields or edit the user's source
dataset unless the user explicitly asks for that dataset mutation.

If the user's objective names `accuracy` or an accuracy target such as
`>=90%`, optimize an evaluation metric, not `val/avg_loss`. Use AutoMLRunner's
`eval_fn` to run the model skill's `evaluate` action on the validation dataset
after each recommendation, with `task=""`, `model.enable_lora=true`, and
`model.base_model_path` set to the same base model used for training. Return
the evaluator's `accuracy` value and set `direction="maximize"`. Use
`val/avg_loss` only when the user accepts a proxy metric or no task metric is
available.

Before launching Cosmos-Reason AutoML for an accuracy objective, run the
evaluate action once after preflight and before recommendation jobs on the same
validation subset. Use the selected base model or starting checkpoint,
`task=""`, and the same prompt/metric setup planned for per-recommendation
evaluation. Report that eval job id, result path, and accuracy in the launch
review before asking for confirmation to start recommendations.

For the evaluator prompt "search over learning rate, batch size, number of
epochs, weight decay, warmup ratio", map the requested knobs to:

```text
learning rate     -> train.optm_lr
batch size        -> train.train_batch_per_replica
number of epochs  -> train.epoch
weight decay      -> train.optm_weight_decay
warmup ratio      -> train.optm_warmup_epochs, computed as round(train.epoch * ratio)
```

The schema exposes `train.optm_warmup_epochs`, not a native warmup-ratio field.
If the evaluator requires a ratio to be preserved exactly, stop and report that
the current Cosmos-RL schema needs a first-class warmup-ratio parameter.

Example custom ranges for the Cosmos Reason 3 AutoML evaluation prompt:

```python
automl_hyperparameters=[
    "train.optm_lr",
    "train.train_batch_per_replica",
    "train.epoch",
    "train.optm_weight_decay",
    "train.optm_warmup_epochs",
]
custom_param_ranges={
    "train.optm_lr": {"valid_min": 1e-5, "valid_max": 1e-3},
    "train.train_batch_per_replica": {
        "value_type": "ordered_int",
        "valid_options": [8, 16, 32],
    },
    "train.epoch": {
        "value_type": "ordered_int",
        "valid_options": [3, 5, 10],
    },
    "train.optm_weight_decay": {"valid_min": 0.0, "valid_max": 0.1},
    "train.optm_warmup_epochs": {
        "value_type": "ordered_int",
        "valid_options": [0, 1, 2, 3, 4, 5],
    },
}
```

Keep `train.train_policy.mini_batch=1` unless the user explicitly changes it,
so all listed batch sizes remain divisible by the micro-batch size. For small
datasets, also cap `train.train_batch_per_replica` so it does not exceed
`num_train_samples / policy.parallelism.dp_shard_size`.
For integer knobs with discrete choices, include `value_type: "ordered_int"`
with `valid_options`; integer `valid_options` alone are ignored by the current
Bayesian sampler.

## DEFT Support

Cosmos-RL implements the DEFT workflow contract for video QA tasks. Use the
packaged model metadata and `workflow/deft/deft.md` for the pipeline overview;
this skill does not package a `config.json`.

### Gap Analysis (`scripts/analyze_gaps.py`)

Model-specific script that identifies failure cases from cosmos-rl evaluation output.

- **Eval output format:** `results.json` with fields: `video_id`, `response`, `question`, `gt`
- **Comparison:** exact string match after `.lower().strip()` — requires eval prompts that force short constrained answers (e.g., yes/no)
- **Output:** parquet with `video_id` (full path), `question`, `ground_truth`

**Limitation:** Brittle exact match. If the model responds with full sentences instead of constrained answers, mismatches will be over-reported. The eval prompt design must account for this.
