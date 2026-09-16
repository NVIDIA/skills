---
name: nvflare-convert-huggingface
description: "Convert existing Hugging Face Transformers Trainer or TRL SFTTrainer training code into an NVFLARE federated job using flare.patch(trainer), local validation, and job export; use when the user names Hugging Face or preliminary source inspection identifies one Hugging Face owner, and not for manual PyTorch loops, Lightning, inference-only pipelines, deployment, or experiment workflows."
license: Apache-2.0
metadata:
  version: "0.1.0"
  author: "NVIDIA FLARE Team <federatedlearning@nvidia.com>"
  min-flare-version: "2.9.0"
  blast-radius: runs_simulator
  category: Conversion
  tags: "nvflare, federated-learning, huggingface, transformers, trl, peft, conversion"
  languages: "python"
  frameworks: "huggingface, transformers, trl, pytorch, nvflare"
  domain: ml
---

# NVFLARE Convert Hugging Face

## Use When

Use when converting training code built around `transformers.Trainer`, `Seq2SeqTrainer`,
TRL `SFTTrainer`, or another Trainer subclass. Support full-model and PEFT/LoRA
fine-tuning, datasets/tokenizers, Trainer callbacks and metrics, checkpoint continuity,
and replicated `torch.distributed` training.

## Do Not Use When

Do not use for an `AutoModel` driven by a manual PyTorch loop without a
Hugging Face Trainer (route to `nvflare-convert-pytorch`), PyTorch Lightning
(route to `nvflare-convert-lightning`, including Lightning modules that contain
Transformers models), inference-only pipelines, model serving, failed jobs
(route to `nvflare-diagnose-job`), or federated statistics without training
(route to `nvflare-fed-stats`). Route a project with active Lightning and
Hugging Face Trainer entrypoints to `nvflare-orient` to select one training-loop
owner or separate jobs. Route unresolved Trainer ownership, such as a Trainer
factory without a bound owner call, to `nvflare-orient`; do not patch either Trainer.
Out of scope: DeepSpeed, FSDP, production/POC deployment, controller rewrites,
experiment search, and privacy-protection requests such as HE, encrypted
aggregation, differential privacy, or privacy filters; never substitute an
unprotected recipe or present a disclaimer as implementation.
If a request combines federated statistics and model-training conversion, treat it as two independent jobs and
workflows: do not merge or automatically chain them, do not route the combination to `nvflare-orient`, and ask which
workflow to run first before generating or running either job. Recommend `nvflare-fed-stats` first only when the
user's purpose is to understand data distribution; handle conversion later as a separate request.

## Available Scripts

| Script | Purpose | Arguments |
| --- | --- | --- |
| `scripts/resolve_model_snapshot.py` | Resolve a local or Hub model/dataset snapshot and emit JSON evidence. | Identifier; required `--source`; optional source-root, download, revision, cache, and repository-type flags. |

Use `run_script()` only during the validation phase after loading `references/huggingface-validation.md`. For example:

- `run_script("scripts/resolve_model_snapshot.py", ["--source", "local", "--source-root", "<absolute-source-root>", "<configured-path>"])`
- `run_script("scripts/resolve_model_snapshot.py", ["--source", "hub", "<org/model>"])`

## Workflow

1. Load `../nvflare-shared/references/conversion-common.md` and apply it for the
   whole conversion; this SKILL.md states only the framework-specific deltas.
   Load `../nvflare-shared/references/conversion-workflow.md` only for a non-standard
   rerun, authorization, or missing-semantics case; it no longer holds the
   data-location or partitioning contracts, whose invariants `conversion-common.md`
   owns. Load `../nvflare-shared/references/site-data-and-paths.md` for generated
   partitions, relative paths, or per-site data locations.
2. Inspect before editing with `nvflare agent inspect source <path> --format json`
   plus direct source reading. Load `references/huggingface-detection.md` during
   this phase. If inspect recommends `nvflare-orient` for unresolved Trainer
   ownership or active Lightning/Hugging Face owners, stop before editing.
   Extract the entrypoint, Trainer subclass, model constructor, tokenizer or
   processor, datasets and collator, Trainer arguments, `compute_metrics`,
   callbacks, checkpoint and PEFT settings, precision, local budget,
   distributed launcher, site/round counts, data location, and aggregation
   intent. Do not import or execute user training modules to discover them.
3. Apply the dependency-install ordering rule in `../nvflare-shared/references/conversion-common.md` before any Python command imports user, framework, NVFLARE, or declared dependencies. Keep dependency inventory single-purpose: check NVFLARE separately, inventory non-product packages separately, and do not append Hugging Face cache or filesystem discovery.
   Defer model and dataset availability checks to the maintained resolver during validation. If an optional path must be inspected, run it separately and report a missing directory with exit code zero.
4. Select the recipe from FL intent. For explicit FedAvg, run `nvflare recipe
   show fedavg-pt --format json`, then immediately load
   `../nvflare-shared/references/pytorch-family-recipe-construction.md` and use
   the returned module, class, and parameters with the required construction
   and execution shape in `assets/job.py`. Import `FedAvgRecipe` from
   `nvflare.app_opt.pt.recipes.fedavg`, never from `nvflare.recipe`. Treat
   `class_path` as the public recipe key and `path` as its normalized exported
   representation; do not inspect Recipe source or signatures to reconcile
   them. Do not guess adjacent symbols or add per-site recipe config unless
   sites genuinely differ. Load
   `../nvflare-shared/references/pytorch-family-recipe-selection.md` only for
   ambiguous, evaluation-only, or non-FedAvg requests.
5. Convert with `references/huggingface-conversion.md` and adapt
   `assets/client_with_eval.py` rather than drafting a new round loop. Preserve
   model, tokenizer/processor, datasets, collator, Trainer arguments,
   callbacks, and metrics. Apply the step-1 data-location rules to the client's
   data argument. Import the Client API as `import nvflare.client.hf as flare`,
   so `flare.init()`,
   `flare.patch()`, and `flare.is_running()` resolve to `nvflare.client.hf`. Keep
   `flare.patch(trainer)` simple with inferred `params_scope="auto"` and encode
   one per-round budget in
   Trainer arguments: requested steps use `max_steps`, requested epochs use
   `num_train_epochs`, and a silent prompt uses the reported default
   `max_steps=10` unless source-budget preservation was requested. Do not
   duplicate the budget in patch `local_steps`/`local_epochs`. When the client
   uses `HfArgumentParser`, construct it with `allow_abbrev=False`.
6. Adapt `assets/server_model.py` and `assets/job.py` instead of inventing
   server-model, packaging, export, or `SimEnv` wiring. Keep generated and
   packaged project-local modules in the same writable source directory. Never
   use `..` in `train_script`, `add_server_file()`, or `add_client_file()`; use
   an existing resolved absolute path when co-location is impossible. Keep the
   server and Trainer model factory and exchange keyspace identical. Use the
   recipe's documented `class_path` or `path` key plus complete `args` for
   required or overridden values; a direct zero-argument instance must not use
   `from_pretrained()`, downloads, or
   checkpoint loading during job construction. Apply only options confirmed by
   the construction reference. Preserve the job asset's recipe-before-parser
   ordering, `ArgumentParser(allow_abbrev=False)`, and strict `parse_args()`; do
   not use `parse_known_args()`.
7. Only after generated files exist, load `../nvflare-shared/references/validation-evidence.md`
   and `references/huggingface-validation.md`. Follow the shared compile,
   construction, simulation, and terminal-evidence ladder. Inspect export/package
   evidence only for an exported final target;
   inspect a local target's materialized evidence after its run. Apply only the
   standard HF Trainer checks and stop at the first failed rung. Review and exercise
   the maintained assets directly; do not inspect NVFLARE implementation source,
   improvise Recipe API probes, or write one-off AST programs to re-prove them. Use
   `references/huggingface-state-and-distributed.md`
   only when inspection found PEFT, DDP, checkpoint/restore overrides,
   auxiliary trainable models, or another non-default patch setting.
8. Report the recipe, source facts, parameter scope, data partition, changed
   files, validation status, and exact artifact paths. When validation produces
   metrics, load `../nvflare-shared/references/metrics-and-artifact-reporting.md`
   before the final response and report each observed primary scalar with its
   metric name, numeric value, and artifact or bounded-log source.

## Requirements

- Must use `flare.patch(trainer)` as the sole model-exchange owner. `receive()`
  inside a patched loop may inspect task metadata only; it must not load a
  second copy of the global model.
- Must follow the Client API initialization and conditional rank contract in
  `../nvflare-shared/references/conversion-common.md`. Keep the generated client
  rankless; `nvflare.client.hf.init()` owns distributed-rank resolution and
  rejection. Load `references/huggingface-state-and-distributed.md` only for a
  distributed multi-process source path.
- Must preserve source evaluation. When per-round global-model evaluation is
  required, call `trainer.evaluate()` before `trainer.train()` on every rank.
  Do not invent `compute_metrics`, label mappings, averaging denominators, or
  metric direction.
- Must follow the Best-Model Metric policy in
  `../nvflare-shared/references/pytorch-family-recipe-construction.md`; the
  Hugging Face delta is only how the delivered key is named and produced. Must
  preserve source metric names when practical: if the generated
  `trainer.evaluate()` emits `accuracy`, set `key_metric="accuracy"`; if Trainer
  emits a prefixed key such as `eval_accuracy`, set the server to that exact key
  and report the source-to-server mapping. When best-model selection is
  requested, every lower-is-better metric, including Trainer-generated
  `eval_loss`, is delivered as an explicitly negated companion and selected by
  that key — never as raw loss. When selection is not requested, use
  `key_metric=""`; do not omit it and accidentally activate the recipe default.
- Must preserve PEFT configuration exactly and verify adapter key compatibility
  between the server model and patched Trainer. Do not infer LoRA target
  modules, silently switch adapter/full-model scope, or solve key mismatches
  with non-strict loading.
- Must verify that `trainer.model` owns all federated trainable state for
  Trainer subclasses with reference, reward, value-head, or other auxiliary
  models. Ask or fail closed when `params_scope="auto"` would omit trainable
  state required by the algorithm.
- Must preserve model constructor values needed on both server and clients per
  `../nvflare-shared/references/pytorch-model-exchange.md` (State-Dict
  Compatibility). Ask one semantic question or fail closed when required values
  are not statically available.
- Must patch only one Trainer per Python process. Preserve a single Trainer
  lifecycle across rounds when `restore_state=True`.
- Must use a positive `TrainingArguments.max_steps` budget for a length-less
  iterable training dataset and let `flare.patch(trainer)` infer it.
- Must reject or report DeepSpeed, FSDP, `save_only_model=True` with
  `restore_state=True`, `load_best_model_at_end=True`, prebuilt
  optimizer/scheduler instances with `restore_state=False`, and checkpoint paths
  not visible to every distributed rank. Do not rewrite these settings silently.
  `launch_once` is a framework-neutral recipe parameter owned by
  `../nvflare-shared/references/pytorch-family-recipe-construction.md`; the
  Hugging Face delta is only that the product rejects explicit
  `launch_once=False` together with `restore_state=True`.
- Must initialize `torch.distributed` before patching when rank environment
  variables declare multiple ranks. All ranks must call patched Trainer methods
  in identical order.
- Must use the maintained HF validation resolver with an explicit local/Hub
  source. For authorized downloads it obtains or validates a full commit-SHA
  revision before downloading. Must not copy it into
  generated job code, set `trust_remote_code=True`, download model/data
  artifacts unless requested, or recover from a cache-only miss by going
  online. Cache misses, remote identifiers, and validation requests do not
  authorize online retries; see `../nvflare-shared/references/conversion-common.md`.
- Site partitioning, custom aggregation, the Source Of Truth Boundary, and user
  input/authorization follow `../nvflare-shared/references/conversion-common.md`.

Always read this converter SKILL.md together with `../nvflare-shared/references/conversion-common.md`. Complete each workflow phase before loading the next phase's reference.
Do not preload validation, state/DDP, broad workflow, dependency, or reporting references. The standard FedAvg path loads, in order:
`../nvflare-shared/references/conversion-common.md`, `references/huggingface-detection.md`, and `../nvflare-shared/references/site-data-and-paths.md` only for its stated triggers;
`../nvflare-shared/references/pytorch-family-recipe-construction.md`, `references/huggingface-conversion.md`, and `../nvflare-shared/references/pytorch-model-exchange.md`;
then `../nvflare-shared/references/validation-evidence.md` and `references/huggingface-validation.md`; load `references/huggingface-state-and-distributed.md` and other shared references only under the triggers above. Do not depend on repository examples.
