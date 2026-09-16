# Hugging Face Conversion Validation

Use `../../nvflare-shared/references/validation-evidence.md` for compilation,
Recipe construction, export, package inspection, simulation, terminal evidence,
and reporting. Run these Trainer-specific checks only after the shared ladder
reaches the applicable phase, and stop at the first failure.

## Generated Trainer Contract

- Adapt `../assets/client_with_eval.py`, `../assets/server_model.py`, and
  `../assets/job.py` rather than drafting replacements. Review the generated
  differences and exercise the asset functions directly; do not write one-off
  AST, class-loader, persistor, Recipe-source, or simulator-source probes.
- Confirm the generated client keeps one persistent Trainer, one
  `flare.patch(trainer)` call, no manual model receive/send path, and the
  source-prescribed evaluate/train sequence.
- Keep model, tokenizer/processor, datasets, collator, Trainer arguments, and
  callbacks outside the FL round loop. The generated FL client must always
  reach `flare.init()` and `flare.patch(trainer)`; preserve standalone behavior
  only behind an explicit non-client entry parameter.
- Pass the final generated `train_args` through the client entry's actual
  argument mechanism and actual dataclass types in parse-only mode; reject every
  unused argument. When the generated client uses `HfArgumentParser`, construct
  it with `allow_abbrev=False` and parse with the same project and framework
  dataclass types. When it preserves `argparse` or another parser, use that
  parser instead. Test the exact arguments produced by the generated job helper;
  do not append an argument only in preflight that the recipe will not deliver.
- Run intentional typo and abbreviation rejection cases through the shared
  assertion-wrapper rule. `HfArgumentParser.parse_args_into_dataclasses()` may
  raise `ValueError` for unused arguments rather than `SystemExit`; the wrapper
  must catch that exact `ValueError`, confirm its diagnostic names the rejected
  argument, and treat the rejection as success so the wrapper itself exits zero.
  A wrapper that catches only `SystemExit` is invalid.
- Version-check only fields claimed to belong to a framework
  `TrainingArguments` or `SFTConfig` base class. Preserve a project-defined
  subclass field when its source definition is verified and the actual parser
  accepts it.
- For the default in-process executor, require unquoted whitespace-free
  `train_args` values. Do not assume shell parsing or call internal command
  splitters. If a required value contains whitespace and no documented
  structured argument surface exists, fail closed.
- Only when source inspection found a source-project-relative local file or
  directory argument, change to a fresh working directory outside the project,
  construct the Recipe with that relative source value, and inspect the final
  `train_args`. Require the source argument name to be unchanged and its
  transmitted value to be absolute and equal to the expected location under
  the explicit inspected source-project root before running the full simulation.
  `SOURCE_DIR` is that root only when generated `job.py` is colocated with the
  source; a read-only-source flow must validate against the separate original
  root. Skip this path-specific check when the source has no local path argument.
  Do not pass absolute local paths, per-site paths, Hub identifiers, or URLs
  through the relative-path test; validate their classification using
  `../../nvflare-shared/references/site-data-and-paths.md` instead.

## Hugging Face Artifacts And Compatibility

- Before constructing a model or tokenizer/processor, run the maintained
  `../scripts/resolve_model_snapshot.py` with `--source local` for a configured
  path or `--source hub` for a Hub repository ID. Never infer the source type
  from slash syntax: `models/checkpoint` and `org/model` are intentionally
  distinguished by that required option. The cache-only Hub path catches
  `LocalEntryNotFoundError`, emits a structured `missing` result, and exits zero;
  report that result as a blocker.
- Use the identifier as the positional argument; do not invent a `--model`
  option. These are the canonical invocations:

  ```bash
  python <skill-dir>/scripts/resolve_model_snapshot.py --source local --source-root <absolute-source-root> <configured-path>
  python <skill-dir>/scripts/resolve_model_snapshot.py --source hub <org/model>
  python <skill-dir>/scripts/resolve_model_snapshot.py --source hub --repo-type dataset <org/dataset>
  ```

  A relative local identifier requires the absolute original source-project
  root so resolution never depends on the validation command's working
  directory. An absolute local identifier must omit `--source-root`; the option
  resolves relative identifiers and is not a sandbox boundary.
- Only when the user authorizes downloading the specific public Hub artifact if
  uncached, rerun its resolver once with the matching canonical invocation:

  ```bash
  python <skill-dir>/scripts/resolve_model_snapshot.py --source hub --allow-download <org/model>
  python <skill-dir>/scripts/resolve_model_snapshot.py --source hub --repo-type dataset --allow-download <org/dataset>
  ```

  In that authorized invocation, the resolver uses the matching public
  `HfApi().model_info(...).sha` or
  `HfApi().dataset_info(...).sha` result, validates the full immutable
  40-character commit SHA, and passes it to `snapshot_download`. An already
  audited full SHA may instead be supplied with `--revision`. For a model
  repository, use the returned immutable `revision` and `resolved_path` for the
  server and every client. For a dataset repository, pass `resolved_path` only
  to a source-compatible local dataset loader; if the source cannot load that
  snapshot layout, ask or fail closed rather than inventing a loader. Do not run
  a preceding `snapshot_download(..., local_files_only=True)` probe, copy
  resolver logic into generated `job.py`, or download without authorization.
  For a configured Hub dataset ID, use the same resolver with
  `--repo-type dataset`; for a source-prescribed local dataset path, use the
  existing local path without a Hub lookup.
- Never recover from an offline/cache-only miss by removing
  `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`, dropping `local_files_only=True`, or
  rerunning online unless the user explicitly requested the download. Do not
  substitute another local checkpoint. Keep the job as a draft and report
  full-model validation blocked when the required artifact is unavailable.
  A cache-only Hugging Face error can mention `huggingface.co` while reporting
  a local cache miss; that URL alone is not evidence that a network request was
  attempted. Judge the configured offline flags, `local_files_only` setting,
  and exception/result type instead.
- Pass the same resolved local model path and cache configuration from the
  resolver to the server model and every client. Do not validate with a cached
  Hub identifier and export a job that depends on an unverified online lookup.
- Do not call `flare.init()`, `flare.patch()`, `flare.is_running()`, or patched
  Trainer methods in a standalone preflight; they require the Client API context
  created by the recipe or simulator launcher. Construct the Trainer without
  patching, inspect the generated patch call and supported public signature
  statically, and let the first bounded simulation validate runtime patch
  acceptance. Reject or report DeepSpeed, FSDP, best-model-at-end,
  save-only-model, prebuilt optimizer/scheduler, and checkpoint/restore
  combinations identified in `huggingface-state-and-distributed.md`.

## Parameter Checks

- Instantiate the adapted `server_model.ServerModel` and the source Trainer
  model through the same importable factory path and with the same local
  constructor arguments. Always compare state-dict key sets, shapes, and dtypes
  without training. Do not inspect NVFLARE persistors or class loaders; the
  exported job and final simulation validate product-side construction.
- Before comparing values, classify checkpoint-loaded versus missing or newly
  initialized parameters from source evidence and the available model load
  report/loading information. Loading the same fixed checkpoint proves value
  determinism only for parameters actually loaded from it; a checkpoint that
  omits a task head or other exchanged parameters does not prove deterministic
  construction of the complete model.
- For checkpoint-loaded parameters, or when the factory explicitly resets all
  initialization seeds/state on every call, require exact per-tensor equality
  or an equivalent stable state hash between independently constructed source
  and server models. A single coincidentally equal run is not proof. For keys
  reported as missing/newly initialized without a deterministic reset, require
  matching keys, shapes, and dtypes, record their names and provenance, and do
  not require their independently initialized values to match. If per-key
  provenance is unavailable, treat the factory as nondeterministic rather than
  assuming the entire fixed-checkpoint construction is deterministic.
- Determine this comparison policy before asserting values. Do not first run an
  unconditional full-state equality assertion and then recover by excluding
  newly initialized keys after it fails.
- Require exact full-model key agreement after any documented prefix
  transformation. For PEFT or auxiliary trainable models, load
  `huggingface-state-and-distributed.md` and run its adapter/ownership checks.
- Confirm tensor-native exchange only when selected by the capability profile.

## Standard Execution Checks

- Follow the shared compile, construction, export, package-inspection, and one
  final foreground simulation path. Do not add separate Recipe-method,
  argument-transport, simulator-process, or export-dispatch probes when the
  maintained job asset compiles and the public validation rungs pass.
- Require terminal completion evidence, positive per-round training step
  counts, and finite source-backed evaluation metrics for the claimed stage.
- Confirm the requested budget appears once, metrics advance across requested
  rounds, and server artifacts contain the selected metric.

## Conditional Execution Checks

- When the real model may exceed host capacity, estimate server and concurrent
  client memory before simulation. Capability-check optional diagnostics such
  as `free` or `nvidia-smi`, run them separately from correctness checks, and
  make absence an exit-zero result; missing diagnostics mean capacity evidence
  is unavailable, not conversion failure. Use a reduced one-round topology
  smoke only when the real model cannot fit, keep the generated default
  unchanged, and report full-model validation blocked.
- After partial aggregation, unexplained disconnect, or exit `-9`, inspect logs
  and either reduce workload with a changed causal factor or report a resource
  blocker. Do not retry the same payload with guessed concurrency settings, and
  make at most one expensive real-model retry.
- For non-default checkpoint/restore behavior, PEFT, auxiliary trainable
  models, or DDP, load `huggingface-state-and-distributed.md` and run only its
  applicable checks.
- Do not report DDP validated from a single-process or rank-zero-only
  simulation. A DDP validation claim requires a two-process `torchrun` case in
  which each process records its launcher rank and product-resolved Client API
  rank, and the observed ranks are exactly `{0, 1}`.

## Report

Report commands, exit codes, terminal status, metric keys/values, parameter
scope, checkpoint mode, process/rank count, exact workspace/export/result paths,
and unresolved validation blockers with concrete reasons.
