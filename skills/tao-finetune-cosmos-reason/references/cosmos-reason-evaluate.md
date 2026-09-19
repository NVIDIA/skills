# Cosmos evaluation

Load this only when `SKILL.md` points here. The repository evaluator is shared
by Cosmos-RL checkpoints and Framework checkpoints after the mandatory native
Framework export.

## Resolution contract

Never submit `references/spec_template_evaluate.yaml` directly. It is a
dataset-neutral shape template whose zero and empty semantic values are
deliberately unresolved. Run `scripts/evaluation_workflow.py` first. The
helper verifies a sealed fine-tuning plan, records field-level provenance,
returns all missing user inputs in one bounded list, and emits runtime TOML
only when the request is complete.

Resolve fields in this order:

1. Use an explicit current evaluation override when the user is deliberately
   changing the validation corpus or evaluation semantics.
2. Apply a packaged fingerprint-locked evaluator profile when the exact
   annotation bytes match one.
3. Otherwise inherit exact values from the selected fine-tuning plan.
4. Run deterministic checkpoint pre-actions owned by the backend.
5. Ask the user only for fields that remain absent or ambiguous.

Do not use a template value, nearby directory, historical run, checkpoint
mtime, or filename convention as an additional source.

### Fingerprint-locked evaluator profiles

The packaged profiles below were verified by full-split evaluation and are
selected only when the single validation annotation SHA256 matches exactly.
All use maximum 1,024 generated tokens, temperature zero, repetition penalty
one, and zero presence/frequency penalties.

| Annotation SHA256 | Profile | Answer/batch/seed |
|---|---|---|
| `c33afc26f979cbdb488b8f1aefdc65604992cd7552d5e75ea782e4565fdc21e1` | `VALIDATION_C33AFC26` | letter / 1 / 42 |
| `6a30babb1921af59155dfe45cf766465597b57cafa1e0e83663a159d89289b6a` | `VALIDATION_6A30BABB` | freeform / 1 / 42 |
| `f828a63f1bbdd45197e1f3393fb94f76ebfdfc785402617aa8c1397b0b47c555` | `VALIDATION_F828A63F` | letter / 1 / 42 |
| `f120ca66f28e3e5b5a01a3ace93d16c856cf13098faf61b44263a4afc449c709` | `PEFT_HPO_VALIDATION_F120CA66` | freeform / 8 / 1 |

The runtime implementation is backend-owned: Cosmos-RL retains strict
`pynvvideocodec`; Framework uses its sealed
`torchcodec-cuda-on-demand` profile. Report that identity separately from the
protocol anchor.

Never activate this profile from a filename, directory, record count, or only
part of its metadata. Explicit current-run overrides remain authoritative and
their provenance must replace the packaged value in the evaluation plan.

### Inherit from fine-tuning

For the original validation split, inherit these without asking again:

- annotation manifest, media root, and dataset fingerprint;
- system prompt, including an explicitly empty prompt;
- complete frame-sampling, clip-time, resize, and pixel-budget configuration,
  plus precision, seed, and validation batch size;
- training sequence limit and evaluator tensor-parallel degree (one model replica per rank);
- task/answer/metric semantics when validation inspection proved them;
- backend, training mode, base-model identity/fingerprint, and GPU count;
- dense versus PEFT behavior and the prepared base model required to merge a
  Cosmos-RL adapter.

The fine-tuning planner stores these in `evaluation_contract`. Dataset
inspection stores `evaluation_profile`, including task semantics, declared
metrics, answer type, normalization version, and any fields that could not be
inferred safely.

### Ask only when unresolved

Prompt once for the remaining fields reported in
`required_user_inputs`. Common cases are:

- the new user-owned evaluation results directory;
- exact checkpoint or checkpoint epoch when more than one checkpoint event is
  present and the training plan did not record a selection;
- generation maximum tokens, because it is not a fine-tuning parameter;
- maximum video pixels when a non-Nano training plan did not record a usable
  budget. For Nano, a sealed native default remains an omitted evaluator
  override so preprocessing does not silently change;
- task type or metric names when annotation targets/metadata were ambiguous;
- exact annotation and media paths when evaluating a different corpus.

An empty system prompt is valid. The flow must distinguish “recorded empty” or
“user supplied empty” from “missing”. For a different evaluation corpus,
fingerprint and inspect that exact manifest/media pair on the selected compute
frame before launch; do not inherit the old dataset's prompt or scoring
semantics automatically.

### Resolver usage

First run with the selected training artifacts and whatever evaluation inputs
are already known:

```bash
python scripts/evaluation_workflow.py \
  --training-plan <sealed-training-plan.json> \
  --training-status <structured-training-status.json-or-jsonl> \
  --results-dir <new-evaluation-results-dir> \
  --plan-output <evaluation-plan.json> \
  --config-output <evaluation.toml>
```

Exit code `3` means the JSON plan was written but contains unresolved intake or
an automated pre-action. Ask only for entries in `required_user_inputs`; do
not ask for entries in `automated_actions`. Multiple recorded validation
manifests/media roots are an automated deterministic materialization step, not
a reason to ask the user to select a subset; preserve the sealed fingerprint
and full validation selection. Rerun with user inputs such as
`--checkpoint-epoch`, `--checkpoint`, `--generation-max-tokens`,
`--evaluation-batch-size`, `--evaluation-seed`,
`--max-video-pixels`,
`--task-type`, `--answer-type`, `--metric`, `--validation-annotation`, or
`--validation-media-root`.

The plan and TOML contain SHA256 values. Persist both in the evaluation job
record. Validate all inherited fingerprints and paths from the target compute
frame before submit.

## Exact checkpoint selection

Use a checkpoint recorded by the selected training job. A single structured
checkpoint event is unambiguous. With multiple events, require an exact path,
an exact epoch, or a checkpoint selection already sealed in the training plan.
Never choose “latest” by directory order or mtime. Require terminal successful
training status before consuming a checkpoint.

For Cosmos-RL, a status event normally names the native
`checkpoints/epoch_N/policy` artifact. That path is never evaluator-loadable.
Run `scripts/cosmos_rl_checkpoint_action.py` on the target compute frame; it
derives and validates the exact sibling `safetensors/epoch_N` export and emits
a binding manifest with path, size, and SHA256 for every config, index, shard,
or adapter file. Rerun the resolver with both `--action-model-path` and
`--action-model-manifest`. A missing, truncated, wrong-epoch, or wrong-kind
export blocks launch automatically and is not a user question.
Each emitted checkpoint pre-action carries its checksum-closed
`supporting_files`; the selected platform stages that declared set directly.

For Cosmos-RL dense training, the verified HF export becomes
`model.model_name`. For Cosmos-RL PEFT, the verified adapter becomes
`model.model_name`, `model.enable_lora=true`, and
`model.base_model_path` is inherited from the fine-tuning model-preparation
record. Do not ask the user to repeat that base-model path.

## Framework DCP pre-action

When `backend=cosmos-framework`, never pass native DCP directly to the shared
evaluator and never ask the user to export it. `evaluation_workflow.py` emits a
`framework_checkpoint_pre_action` entry. Run
`scripts/framework_checkpoint_action.py plan`, then `prepare` in the clean
repository-derived Framework action image. The helper validates the saved
Framework config, DCP metadata, base-model identity/revision, exact exported
keys, indexed weights, and export manifest. Run `verify` from the target
compute frame and pass its `action_model_path` and terminal action JSON back to
`evaluation_workflow.py --action-model-path ... --action-model-manifest ...`.

Framework PEFT is reconstructed and merged by the native exporter, so the
shared evaluation config keeps `model.enable_lora=false`. Export failure or
provenance mismatch blocks evaluation; it never selects another checkpoint or
export.

## Task and metric semantics

The generic evaluator supports structurally detected conversation and
task-aware records. Validation inspection classifies complete `yes`/`no`
targets as binary and complete `A`-through-`D` targets as multiple choice.
Explicit task metadata such as `bcq`, `binary`, `binary_choice`, `mcq`, and
`multiple_choice` is canonicalized to the repository evaluator semantics.
Ambiguous `A`/`B` targets require the user to choose binary or multiple choice.

Accuracy-defined tasks use the shared task-aware scorer and `metrics.names=[]`.
Generative tasks use only metrics declared by the annotation metadata or
explicitly selected by the user. Do not turn text metrics on for a
classification task, and do not relabel generation NLL or validation loss as
answer accuracy.

Preserve the resolved prompt, frame sampling, pixel budget, generation,
parsing, normalization, and evaluator version in metadata. Report the final
metric emitted by the repository evaluator, including correct/total and
per-task values when it provides them.

## Decoder and execution

Keep decoder contracts isolated by backend. Cosmos-RL retains its strict
`pynvvideocodec` path and any validated override artifact. Framework inherits
the sealed `torchcodec-cuda-on-demand` cache/thread/device profile and requires
an explicit `min_pixels=max_pixels` bound when the sealed maximum is present,
so Qwen preserves that lower Framework pixel budget for predecoded frames
instead of rejecting it against its larger runtime default minimum. This
normalization is Framework-only; it does not change Cosmos-RL's registered
reader normalization. Framework also requires
the selected SQSH to attest the Framework preprocessing implementation before
the evaluator child starts. Never copy one backend's decoder settings into the
other, invent FPS metadata, rewrite annotations, or silently fall back to CPU
decoding.

Use `torchrun` data parallelism according to the resolved GPU count. Keep one
model replica per rank unless the selected backend contract explicitly
requires another topology. Full evaluation uses `limit=-1`; the repository
evaluator owns its rank-aware outputs and scoring.

The READY evaluation plan contains a validated `spec_bundle`. Its
`execution` lifecycle owns backend CLI selection, non-secret runtime
environment, and Framework in-image capability attestation. It does not add a
post-evaluation prediction-ID, annotation-envelope, or rank-shard gate after
the evaluator returns its metric.

Pass that bundle unchanged to the selected platform. On SLURM,
`tao-run-on-slurm` owns the standard template, persistent results mount,
job-record placeholder binding, distributed rendezvous, no-requeue/exclusive
directives, timeout, and child-exit propagation. Do not create a Cosmos-only
SLURM renderer, copy these semantics into an application skill, or treat
per-rank shards as a complete evaluation.
Run `scripts/framework_evaluation_image_preflight.py` against a selected
Framework SQSH before opening/submitting the evaluation record. A missing
baked Framework preprocessor is an immutable-image incompatibility, not a
reason to stage source or select Cosmos-RL.

## Completion and results

Treat scheduler completion as provisional. Require child exit zero, terminal
TAO `SUCCESS`, and the final metric emitted by the repository evaluator. Do not
turn a successful evaluation into a failure by comparing prediction IDs with
annotation IDs or by imposing a second annotation-format assumption.
Persist the selected checkpoint, Framework export when applicable, resolved
config and SHA256, evaluation plan and provenance, stdout/stderr, status,
results, any emitted per-task metrics, normalization/evaluator version, and
duration in the job record.
