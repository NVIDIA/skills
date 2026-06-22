# Cosmos-RL AutoML / HPO

Load this only when `SKILL.md` points here for an AutoML/HPO task. If this conflicts with `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the current/compact source wins.

The packaged default base model is `hf_model://nvidia/Cosmos3-Nano`. Apply this
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
after each recommendation, with `task=""`, `model.enable_lora=true`, and
`model.base_model_path` set to the same base model used for training. Return
the evaluator's task metric and set `direction="maximize"`. Use `accuracy` for
constrained classification prompts and BERTScore F1 for free-form
summarization/answering prompts when the user asks for semantic text quality.
Use `val/avg_loss` only when the user accepts a proxy metric or no task metric
is available.

Before launching AutoML for an accuracy objective, run the model's evaluate
action once after preflight and before recommendation jobs on the same
validation subset. Use the selected base model or starting checkpoint,
`task=""`, and the same prompt/metric setup planned for per-recommendation
evaluation. Report that eval job id, result path, and accuracy in the launch
review before asking for confirmation to start recommendations. The final
AutoML summary must compare this baseline accuracy, every recommendation's
accuracy, and the selected best recommendation.

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
