# Hugging Face Trainer Conversion

## Canonical Path

Use this path for a standard Trainer conversion:

1. Confirm Hugging Face Trainer ownership with `nvflare agent inspect source`.
2. Run `nvflare recipe show` and apply the shared PyTorch-family construction
   profile.
3. Adapt `../assets/client_with_eval.py` into `client.py`, preserving the
   source Trainer factory and metric behavior.
4. Adapt `../assets/server_model.py` and `../assets/job.py` for the source model
   factory, client arguments, selected metric, and capability-gated options.
   Hugging Face Trainer is part of the PyTorch family, so
   `../../nvflare-shared/references/pytorch-model-exchange.md` owns the
   tensor-payload and state-dict compatibility rules for the server model and
   every client; load it before pinning shared constructor values.
5. Follow the shared validation ladder with the HF-specific validation delta.

The maintained assets are the canonical standard path. The client asset owns
the patch and round-loop shape. The server asset returns the source model
directly so no wrapper prefix changes its state-dict keys. The job asset owns
the required FedAvg constructor, server/client packaging, `SimEnv`, and
`recipe.execute()` shape. Adapt these surfaces rather than drafting
replacements or inspecting NVFLARE implementation source.

The generated `client.py` entry point is FL-only: it always reaches
`flare.init()` and `flare.patch(trainer)`. Do not infer FL launch from
`CLIENT_API_TYPE` or another environment variable; the launcher does not expose
a reliable branch signal to the trainer. If preserving a standalone CLI is
required, factor shared setup into an explicit function parameter and have the
generated client call that function with `federated=True`; keep the standalone
path behind an entry point that passes `federated=False` explicitly.

When the source has no valid evaluation dataset or metric and neither per-round
evaluation nor best-model selection is requested, adapt the asset with
`evaluate_before_train=False` and `key_metric=""` so model selection is disabled.
Do not omit `key_metric`: `FedAvgRecipe` otherwise activates its documented
default selector. See the Best-Model Metric section of
`../../nvflare-shared/references/pytorch-family-recipe-construction.md`.

Import the Client API as `import nvflare.client.hf as flare`, as the asset does,
so every `flare.*` call below resolves to `nvflare.client.hf`.

Follow the framework-neutral Client API initialization and rank contract in
`../../nvflare-shared/references/conversion-common.md`. The standard
asset always calls rankless `nvflare.client.hf.init()`. The product resolves an
initialized process-group rank or global `RANK` and rejects an unresolved
multi-process launch before creating a Client API context; do not reimplement
that logic in generated code. For a preserved multi-GPU launch, load
`huggingface-state-and-distributed.md`. Initialize before
`flare.get_site_name()`, `flare.get_config()`, or other Client API context access
that occurs before `patch()`.

Do not add manual model loading from `flare.receive()` or model sending through
`flare.send()`. The patch wraps `train()` and `evaluate()`, loads the received
global parameters, and sends the result from rank 0.

## Preserve Trainer Semantics

- Construct the model, tokenizer/processor, datasets, collator, Trainer
  arguments, callbacks, and Trainer once before the federated loop.
- Preserve `compute_metrics` and evaluation datasets.
- Keep mid-training evaluation controlled by the Trainer's own evaluation
  strategy; it is not a separate FL task.
- Resolve exactly one per-round budget and encode it in one place:
  1. user-requested local steps become generated `TrainingArguments.max_steps`;
  2. user-requested local epochs become generated
     `TrainingArguments.num_train_epochs`, with a conflicting positive
     `max_steps` cleared because Hugging Face gives steps precedence;
  3. when the user explicitly asks to preserve the source budget, leave its
     positive `max_steps` or `num_train_epochs` unchanged;
  4. otherwise use the bounded generated default `max_steps=10` and report it.
- Let `flare.patch(trainer)` infer that Trainer budget. Do not also pass the same
  value through patch `local_steps` or `local_epochs`; that duplicate expression
  is misleading even though the patch option owns the runtime budget. Use a
  patch budget option only when the Trainer's argument surface cannot express
  the requirement, and then do not encode the same budget in Trainer arguments.
  A length-less iterable dataset requires a positive
  `TrainingArguments.max_steps` budget; let the patch infer it without also
  setting patch `local_steps`.
- With `restore_state=True`, the patch sets the Trainer's cumulative
  `max_steps` target to `per_round_budget_steps * total_rounds` so optimizer and
  scheduler progress continues across rounds. Hugging Face may display that
  whole-job target in its progress bar, while the NVFLARE callback still stops
  each `trainer.train()` call after one per-round budget. Do not interpret the
  progress-bar denominator as work for the current round.
- When the generated or preserved client uses `HfArgumentParser`, construct it
  with `allow_abbrev=False`, using the actual project and framework dataclass
  types. Other client parsers retain their own strict-parser mechanism.
- Keep local-only callbacks and reporting. Leave network trackers disabled
  during validation unless explicitly requested.

## Recipe Integration

Use the PyTorch recipe family. Run `nvflare recipe show <recipe-name> --format
json`, then load
`../../nvflare-shared/references/pytorch-family-recipe-construction.md` before
constructing the recipe. That shared reference owns capability checks,
tensor-native transport and decomposer registration, disk offload, external
process selection, and common metric-selection policy.

For FedAvg, adapt `../assets/job.py`. Its required structure matches the plain
PyTorch fast path: `FedAvgRecipe(...)`, explicit importable model config,
`SimEnv(...)`, and `recipe.execute(...)`. It uses
`{"class_path": "server_model.ServerModel", ...}` at recipe construction time.
`class_path` is the public recipe key; `path` is the normalized key in exported
job configuration. Do not inspect `PTModel`, persistors, class loaders, or
Recipe source to reconcile those representations.

Copy the adapted `job.py`, `client.py`, and `server_model.py` into the same
writable source directory as packaged project-local modules such as `model.py`.
Keep the asset's `SOURCE_DIR = Path(__file__).resolve().parent` resolution for
`add_server_file()` and `add_client_file()` so construction works from any
caller working directory. Keep `train_script="client.py"` as the portable
app-local runtime path, and explicitly package its source with
`recipe.add_client_file(str(SOURCE_DIR / "client.py"))`. The maintained asset
enters `SOURCE_DIR` only while the Recipe validates and records those resources,
then restores the caller's working directory before returning. Do not move
generated files into a child package and refer back with `../model.py`:
NVFLARE rejects parent-traversal external-script paths. For an exceptional
non-co-located module, pass its existing resolved absolute source path to the
packaging API.

`SOURCE_DIR` identifies the generated-file directory for packaging; it is the
source-project root only in the colocated case. If a read-only source root puts
generated files elsewhere, pass the inspected original root explicitly to
`resolve_source_local_path(..., source_root=...)` for source-relative data. Do
not resolve those data paths against the writable generated-file directory.

Add optional recipe arguments and decomposers only as directed by the selected
recipe's capability profile and the shared construction reference. Do not copy
the FedAvg constructor shape to another recipe. When sites genuinely need
different non-entrypoint settings, such as complete `train_args` or
source-backed launcher values, pass the resolved site mapping through the
asset's `per_site_config` argument. It calls
`set_per_site_config(recipe, per_site_config)` immediately after construction
and before any client file or configuration is added. Do not pass the deprecated
Recipe constructor `per_site_config` option through `recipe_options`.
Every site uses the same packaged `client.py`; do not include `train_script` in
the per-site mapping. The asset rejects both relative and absolute site-specific
script overrides because it cannot package them while preserving one portable
app-local runtime path.
Pass that same mapping to the asset's `build_sim_env(...)`, which implements the
single-topology-owner rule from
`../../nvflare-shared/references/conversion-common.md`. Leave the mapping unset
for ordinary generated partitions, and derive their indices from the initialized
`site-N` name.
Follow the shared construction reference's client-argument transport rule.
`train_args` is not necessarily shell parsed: use unquoted whitespace-free
tokens for the default in-process executor, and use shell quoting only for a
documented POSIX-tokenizing launcher. Ask or fail closed when a required value
cannot be represented by the selected public argument surface. Do not probe
internal command-splitting helpers.

`train_script` names the primary client entry point in the runtime config; it
does not provide a separate source root for caller-cwd-independent packaging.
Use its portable app-local name in the constructor and add the resolved source
once with `recipe.add_client_file(...)`. For an exported-artifact target, export
and inspect that job before running it with the simulator CLI. For a local
`python job.py` target, do not export; after the run, inspect the materialized
simulation workspace for the same config and packaging evidence. Reject
absolute `task_script_path` values in generated configs because runtime apps
must launch their packaged client script portably.

For an exported-artifact target, exported app layout is owned by
`../../nvflare-shared/references/conversion-workflow.md`: inspect the exported
job root and enumerate the app directories it actually contains before asserting
any path.

Preserve the job asset's explicit packaging of `server_model.py` and the source
model module into the server app, and package source modules imported by the
client. A client import is not enough when an export separates server and
client apps. Verify the required files under the discovered layout:
`app/custom` for a unified export, or `app_server/custom` and each
`app_<site>/custom` for a per-site export. Do not reuse a path assumption from
another export. Installed NVFLARE, framework, and third-party class paths stay
runtime dependencies validated through requirements installation plus
import/preflight checks.

## Data And Model Selection

Follow `../../nvflare-shared/references/site-data-and-paths.md` when generated
site partitions, relative-path resolution, or per-site data locations are
involved. Pass data roots through client arguments or per-site configuration —
never a path hardcoded in the generated client, and never copy private site data
into the job.

Prefer preserving source metric names in the client metrics output. If the
generated evaluation call emits `accuracy`, configure `key_metric="accuracy"`.
Trainer commonly prefixes `compute_metrics` output, so `{"accuracy": ...}` can
become `eval_accuracy`; when that is the returned client key, configure the
server with `key_metric="eval_accuracy"` and report the mapping from source
metric name to server metric key.

`FedAvgRecipe` does not expose a lower-is-better direction flag, so the shared
rule in `../../nvflare-shared/references/pytorch-family-recipe-construction.md`
applies unchanged: when best-model selection is requested, every
lower-is-better metric is delivered as an explicitly negated companion and
selected by that key. This holds for loss exactly as it does for any other
metric.

- For a metric returned by `compute_metrics`, preserve the original and add the
  negated companion in the same dict — `{"wer": wer, "neg_wer": -wer}` — then
  select the prefixed key `eval_neg_wer`.
- For Trainer-generated `eval_loss`, which `compute_metrics` never sees, add the
  companion with a small `TrainerCallback` whose `on_evaluate` inserts
  `metrics["eval_neg_loss"] = -metrics["eval_loss"]`, then select
  `key_metric="eval_neg_loss"`. Register that callback on the Trainer **before**
  `flare.patch(trainer)`: the patch appends its own callback, and Transformers
  passes the same metrics dict to callbacks in registration order, so the
  companion is present when FLARE captures the metrics. Guard the insertion so a
  missing or non-finite `eval_loss` is skipped rather than raising.

Never select raw loss as though increasing values were improvements. Ask or fail
closed only when the metric direction itself is unclear, not merely because the
sole available metric is a loss.
