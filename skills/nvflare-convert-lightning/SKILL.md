---
name: nvflare-convert-lightning
description: "Convert existing PyTorch Lightning training code into an NVFLARE federated job using the Lightning Client API patch, local validation, and job export; use only when the request names federated/NVFLARE conversion or asks multiple sites to train collaboratively while keeping each site's data local, and either names PyTorch Lightning or preliminary source inspection identifies one Lightning owner; do not use for non-federated Lightning work such as DDP, profiling, inference serving, or training-loop changes, nor for plain PyTorch, TensorFlow/Keras, other frameworks, deployment, POC/production lifecycle, or experiment workflows."
license: Apache-2.0
metadata:
  version: "0.1.0"
  author: "NVIDIA FLARE Team <federatedlearning@nvidia.com>"
  min-flare-version: "2.9.0"
  blast-radius: runs_simulator
  category: Conversion
  tags: "nvflare, federated-learning, pytorch-lightning, conversion"
  languages: "python"
  frameworks: "pytorch-lightning, pytorch, nvflare"
  domain: ml
---

# NVFLARE Convert PyTorch Lightning

## Use When

Use only when the user asks to convert PyTorch Lightning code into an NVFLARE federated training job; require both federation intent and Lightning ownership. Treat requests for multiple sites or institutions to train collaboratively while each site's data remains local as federation intent, even when the request does not say "federated" or "NVFLARE."
Lightning source evidence alone is not sufficient. Relevant source may contain a `LightningModule`, `LightningDataModule`, a `Trainer` fit/validate/test loop,
Lightning callbacks, checkpointing, or loggers.
Supported: the PyTorch recipe family with `flare.patch(trainer)` as the model
exchange integration, Lightning-native evaluation, custom aggregation through
the same recipe `aggregator=` hook, and local validation and export.

## Standard Path

Always read this converter SKILL.md with
`../nvflare-shared/references/conversion-common.md`. For an explicit-FedAvg
conversion, load only these references, in workflow order:

1. During inspection, `references/lightning-detection.md`.
2. For generated splits, relative paths, or per-site data locations,
   `../nvflare-shared/references/site-data-and-paths.md`.
3. After `nvflare recipe show fedavg-pt --format json`,
   `../nvflare-shared/references/pytorch-family-recipe-construction.md`.
4. During conversion, `references/lightning-conversion.md`, then
   `../nvflare-shared/references/pytorch-model-exchange.md`.
5. Only after generated files exist,
   `../nvflare-shared/references/validation-evidence.md`, then
   `references/lightning-validation.md`.

Complete each workflow phase before loading the next phase's reference. Do not
enumerate reference directories or preload validation, DDP/tracking, broad
workflow, dependency, runtime-output, or reporting references. Do not depend on
NVFLARE repository examples.

## Do Not Use When

Do not use for non-federated Lightning changes such as DDP-only configuration, profiling, inference serving, callbacks, early stopping, or schedulers; or for plain `torch.nn.Module` manual training loops without Lightning
(route to `nvflare-convert-pytorch`), Hugging Face Trainer (route to `nvflare-convert-huggingface`), TensorFlow,
XGBoost, scikit-learn, a failed job (route to `nvflare-diagnose-job`),
federated statistics without training (route to `nvflare-fed-stats`), or
generic Lightning debugging without FLARE intent; when the inspected project
actively contains both Lightning and Hugging Face Trainer entrypoints, route to
`nvflare-orient`. Out of conversion scope: production deployment, Kubernetes,
POC lifecycle, deployment privacy/security policy design, custom distributed
launch policies not expressible by product APIs, experiment tracking redesign,
and experiment search across recipes. Privacy-protection requests — homomorphic encryption (HE) /
encrypted aggregation, differential privacy, and privacy filters — are not
supported: they require provisioning or deployment policy beyond conversion
scope, so report such a request as unsupported and route it to
provisioning/deployment, never substituting an unprotected recipe or disclaimer.
If a request combines federated statistics and model-training conversion,
treat it as two independent jobs and workflows: do not merge or automatically
chain them, do not route the combination to `nvflare-orient`, and ask which
workflow to run first before generating or running either job. Recommend
`nvflare-fed-stats` first only when the user's purpose is to understand data
distribution; handle conversion later as a separate request.

## Workflow

1. Apply `../nvflare-shared/references/conversion-common.md` for the whole
   conversion; this SKILL.md states only the framework-specific deltas.
2. Inspect before editing with `nvflare agent inspect source <path> --format json`
   plus direct reading; fact extraction is static. Confirm Lightning versus plain
   PyTorch and hand off to `nvflare-convert-pytorch` when no Lightning evidence
   exists. If inspection recommends `nvflare-orient` for active Lightning and
   Hugging Face Trainer owners, stop and hand off before editing.
3. Apply the dependency-install ordering rule in
   `../nvflare-shared/references/conversion-common.md` before any Python command
   imports user, Lightning, NVFLARE, or declared dependency modules. Determine
   applicable dependencies from the selected execution path first. If required
   data artifacts already exist and static inspection shows that the selected
   path will not reach a download helper or its imports, treat its download-only
   requirements as inapplicable: do not install or import-probe them. Probe only
   modules the generated conversion and selected validation path will execute,
   and keep an optional probe separate and exit-zero when unavailable.
4. Identify the existing `LightningModule`, `LightningDataModule`, trainer
   construction, callbacks, checkpointing, `validation_step`/`test_step` and
   dataloaders, metrics, logger usage, source partition evidence, distributed
   process-spawning evidence, custom aggregation intent, and the concrete model
   constructor values that server and clients must share.
5. Reuse the PyTorch recipe family; Lightning is not a separate recipe family.
   For the standard case — the user explicitly requests FedAvg and inspection
   identifies Lightning — run `nvflare recipe show fedavg-pt --format json`
   directly and construct it. For `fedavg-pt`, import `FedAvgRecipe` only from
   `nvflare.app_opt.pt.recipes.fedavg`, never from `nvflare.recipe`. Use FedEval
   for evaluation-only. After every `recipe show`, derive construction capabilities
   from the construction reference. Then use that documented path: do not run
   exploratory NVFLARE imports or use `inspect`, `hasattr`, constant discovery,
   SDK source/docstring reads, or lifecycle probes. If a required detail is absent,
   report a skill gap or fail closed instead of guessing. Call `recipe.execute(SimEnv(...))`.
6. Convert the training entry point to the Lightning Client API: build the
   `Trainer`, call `flare.patch(trainer)`, and let the patched trainer own
   model exchange. Keep evaluation inside Lightning per
   `references/lightning-conversion.md` and use `self.log`. Derive
   `evaluate_only=True` only for FedEval; omit it for training recipes so its
   default stays `False`. Derive `evaluate_before_train = recipe_algorithm !=
   "cyclic"`: Cyclic persists only its final sequential model; every other
   algorithm uses explicit validation for server metrics and, for training,
   best-model selection. Verify the key in server evidence or fail closed.
7. Add or update `job.py` under the shared constructor-serialization rule. Use
   the recipe's `class_path` or `path` key plus complete `args` when values are
   needed; a permitted zero-argument instance is the complete module. Add
   requested `aggregator=` wiring and the metric, tensor-transport, server
   offload, and execution settings derived from the shared PyTorch-family
   construction profile. If sites need distinct `train_args`, make every site
   override the complete argument string; never split shared arguments and a
   site-specific data path across recipe-level and per-site values expecting a
   merge.
8. Immediately after generated files exist and before any preflight, smoke test,
   cleanup, validation, or execution command, load the two validation references
   in Standard Path order. Before executing a full run, select and record exactly
   one final validation target:
   - for a requested local or first-run simulation without an export claim, run
     `python job.py` and do not export or run the exported simulator afterward;
   - for a requested exported/deployable artifact, export first and run only the
     exported folder with the simulator CLI; do not first run `python job.py`.
   If the selected full-run target fails, diagnose it, apply a scoped fix, and
   rerun that same target. Change targets only when evidence shows the original
   target does not represent the requested artifact, and record that reason.
   Export inspection belongs only to the exported path. Keep cleanup, export,
   and simulation as separate tool calls; never combine recursive cleanup with
   execution. Stop at the first failed validation rung before diagnosing it;
   do not add speculative recovery probes. Use
   the environment and permission mechanisms supplied by the agent host; do not
   inspect or enforce its security boundary.
9. Report the recipe, changed files, selected validation target, validation
   status, metrics, and exact artifact paths.

## Non-standard Cases

Load only the reference matching an encountered case:

- `../nvflare-shared/references/conversion-workflow.md` for an unresolved
  non-standard rerun, authorization, or missing-semantics case; it no longer
  holds the data-location or partitioning contracts.
- `../nvflare-shared/references/pytorch-family-recipe-selection.md` for an
  ambiguous or non-FedAvg algorithm; use its catalog for FedAvg, FedOpt, FedProx,
  SCAFFOLD, Cyclic, Swarm, or FedEval, and reserve `nvflare recipe list` for these cases.
- `../nvflare-shared/references/dependency-install.md` when an applicable
  dependency is missing.
- `../nvflare-shared/references/runtime-output-guidance.md` for a read-only
  source root or user-chosen output destination.
- `references/lightning-ddp-and-tracking.md` when inspection finds its trigger.
- `../nvflare-shared/references/metrics-and-artifact-reporting.md` when normal
  metric artifacts are absent or inconsistent.

## Requirements

- Must integrate through `flare.patch(trainer)` and let the patched trainer own
  model exchange. Must not generate a manual `FLModel` send/receive path as the
  default Lightning exchange, and must not pass the received `input_model` into
  the `Trainer`.
- Must treat `flare.receive()` inside the patched loop as optional metadata or
  task-progression access only, not as a second model-load path.
- Must keep evaluation inside Lightning (`trainer.validate`/`trainer.test`,
  `validation_step`, `self.log`); must not generate a raw PyTorch
  `model.eval()` loop for ordinary Lightning conversion.
- Except for Cyclic, must run an explicit standalone `trainer.validate(...)`
  before `trainer.fit(...)` and rely on the patched callback to attach its finite
  scalar metrics; never populate `model.__fl_meta__[MetaKey.INITIAL_METRICS]`.
  Validation inside `trainer.fit(...)` is not a received-global-model metric. Cyclic must
  skip the pre-fit call and report its persisted final model, not a best model.
- Must audit model constructor arguments before writing `job.py` by reading the
  `LightningModule.__init__` signature and the selected recipe's `model`
  parameter from `nvflare recipe show <recipe-name> --format json`, not by
  reading NVFLARE library source. Emit the recipe-documented `class_path` or
  `path` key plus complete `args` for every required or overridden value. Direct
  `LightningModule` use is allowed only when unchanged zero-argument defaults reconstruct it. Values
  must be clear from source, configuration, or supplied metadata. Otherwise ask
  one semantic question when an answer channel exists or fail closed.
- Must use the PyTorch recipe family; must not invent a Lightning-only recipe.
  Apply the construction reference after `recipe show`; it is canonical for
  optional recipe parameters, model selection, tensor transport, server disk
  offload, and execution mode.
- Must preserve local-only callbacks and logger behavior where safe. Existing
  network-connected tracking, upload callbacks, and custom/unknown loggers are
  evidence, not a user request: keep them disabled during validation unless
  explicitly requested, and do not ask solely to enable them. This narrows
  `references/lightning-conversion.md`.
- Must not make non-PyTorch-family skills load
  `../nvflare-shared/references/pytorch-model-exchange.md`.
- Site partitioning, custom aggregation, the Source Of Truth Boundary, and user
  input/authorization follow `../nvflare-shared/references/conversion-common.md`.

