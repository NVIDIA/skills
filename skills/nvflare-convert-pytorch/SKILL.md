---
name: nvflare-convert-pytorch
description: "Convert existing plain or manual PyTorch training code into an NVFLARE federated job using Client API model exchange, local validation, and job export; use when the user names plain PyTorch or preliminary source inspection identifies one plain-PyTorch owner, and not for Lightning, other frameworks, deployment, POC/production lifecycle, or experiment workflows."
license: Apache-2.0
metadata:
  version: "0.1.0"
  author: "NVIDIA FLARE Team <federatedlearning@nvidia.com>"
  min-flare-version: "2.9.0"
  blast-radius: runs_simulator
  category: Conversion
  tags: "nvflare, federated-learning, pytorch, conversion"
  languages: "python"
  frameworks: "pytorch, nvflare"
  domain: ml
---

# NVFLARE Convert PyTorch

## Use When

Use when converting an existing plain PyTorch training script, `torch.nn.Module`,
manual training loop, `state_dict` workflow, data loader, checkpoint, or metric loop
into an NVFLARE federated training job. Supports horizontal FL, Client API model
exchange with `FLModel`, recipe `aggregator=` hooks, validation, and export.

## Do Not Use When

Do not use for PyTorch Lightning (route to `nvflare-convert-lightning`), Hugging Face Trainer (route to `nvflare-convert-huggingface`), TensorFlow, XGBoost,
scikit-learn, failed jobs (route to
`nvflare-diagnose-job`), federated statistics without training (route to
`nvflare-fed-stats`), or generic PyTorch debugging without FLARE intent. Out of
scope: production deployment, Kubernetes, POC lifecycle, privacy/security policy design,
controller/workflow rewrites outside recipe or Job APIs, experiment search, and
data distribution experiments beyond minimal validation setup. Privacy-protection
requests — HE/encrypted aggregation, differential privacy, and privacy filters — need provisioning/deployment
policy; route onward rather than substituting an unprotected recipe or adding only a disclaimer.
If a request combines federated statistics and model-training conversion,
treat it as two independent jobs and workflows: do not merge or automatically
chain them, do not route the combination to `nvflare-orient`, and ask which
workflow to run first before generating or running either job. Recommend
`nvflare-fed-stats` first only when the user's purpose is to understand data
distribution; handle conversion later as a separate request.

## Workflow

1. Load `../nvflare-shared/references/conversion-common.md` and apply it for the
   whole conversion; this SKILL.md states only the framework-specific deltas.
   Load `../nvflare-shared/references/conversion-workflow.md` only for a non-standard
   rerun, authorization, or missing-semantics case; it no longer holds the
   data-location or partitioning contracts, whose invariants `conversion-common.md`
   owns. Load `../nvflare-shared/references/site-data-and-paths.md` for generated
   partitions, relative paths, or per-site data locations.
2. Inspect before editing with `nvflare agent inspect source <path> --format json`
   plus direct reading. Fact extraction is static; do not import or execute
   user training modules to discover fields. Extract: training entrypoint,
   model class path and constructor args, checkpoint behavior, train/eval
   functions, data loading, metric names and denominators, local epochs/steps,
   requested client and round counts, source data split or partition evidence,
   tracking evidence, DDP evidence, and any custom aggregation intent.
3. Apply the dependency-install ordering rule in `../nvflare-shared/references/conversion-common.md` before
   any Python command imports user, PyTorch, NVFLARE, or declared dependency
   modules.
4. Select the recipe from the requested FL workflow, not from PyTorch alone. For
   the standard case — the user explicitly requests FedAvg and inspection
   identifies PyTorch — run `nvflare recipe show fedavg-pt --format json`
   directly and construct it; do not add per-site recipe config unless sites
   actually differ. Load
   `../nvflare-shared/references/pytorch-family-recipe-selection.md` (discovery,
   algorithm guide, catalog-based selection, HE-not-supported rule) only for
   ambiguous or non-FedAvg algorithms, reserving `nvflare recipe list` for those
   cases. Use the module, class, and parameters returned by `recipe show` for
   standard `job.py` construction; for `fedavg-pt`, import `FedAvgRecipe` from
   `nvflare.app_opt.pt.recipes.fedavg`, never from `nvflare.recipe`. After every
   `recipe show`, load
   `../nvflare-shared/references/pytorch-family-recipe-construction.md` and
   derive the recipe's construction capabilities. Load
   `references/recipe-selection.md` only when non-FedAvg or execution-mode
   details are needed.
5. Convert training and evaluation as a pair using
   `references/pytorch-client-api-conversion.md`: initialize FLARE, receive an
   `FLModel`, load `params`, evaluate the received global model, train, and
   send an `FLModel` with updated `params`, `metrics`, and the actual completed
   local optimizer-step count in `NUM_STEPS_CURRENT_ROUND`. Adapt the user's
   evaluation code into the packaged evaluation template; if evaluation is
   required but missing, ask or fail closed. Apply the step-1 data-location
   rules to the generated client's data argument.
6. Add or update `job.py` under the shared constructor-serialization rule:
   use explicit `class_path` (or documented `path` alias) plus complete `args`
   whenever reconstruction needs values. Add requested `aggregator=` wiring,
   metric, tensor-transport, server offload, and execution settings derived
   from the shared PyTorch-family construction profile.
7. Validate in a ladder per `../nvflare-shared/references/validation-evidence.md`:
   compile checks, recipe construction, one final full-run path chosen by the
   artifact being validated, with export and package inspection only for the
   selected exported-artifact path. For a local target, inspect the materialized
   configs and packaging evidence after that run. Use
   `references/job-validation.md` for PyTorch-specific failures. Stop at the
   first failed rung and report the product error. Use the environment and
   permission mechanisms supplied by the agent host; do not inspect or enforce
   its security boundary.
8. Report the recipe, changed files, validation status, metrics, and exact
   artifact paths. Load
   `../nvflare-shared/references/metrics-and-artifact-reporting.md` only when
   normal metric artifacts are absent or inconsistent.

## Requirements

- Must audit model constructor arguments before writing `job.py` by reading the
  model module's `__init__` and the selected recipe's `model` parameter from
  `nvflare recipe show <recipe-name> --format json`, not by reading NVFLARE
  library source. Emit the selected recipe's documented `class_path` or `path`
  key plus complete `args` for every required or overridden constructor value;
  a direct `torch.nn.Module` is allowed only when
  unchanged zero-argument defaults reconstruct it. Values must be statically
  clear from literal source, configuration, or supplied metadata. Otherwise ask
  one semantic question when an answer channel exists or fail closed.
- Must follow `../nvflare-shared/references/pytorch-model-exchange.md` and
  `references/pytorch-client-api-conversion.md` for the canonical plain-PyTorch
  payload and round-loop pattern.
- Must apply
  `../nvflare-shared/references/pytorch-family-recipe-construction.md` after
  `recipe show`; it is the canonical policy for optional recipe parameters,
  model selection, tensor transport, server disk offload, and execution mode.
  Never patch a framework-neutral runtime module or register FOBS handlers in
  `client.py`.
- Must convert source evaluation alongside training and return metrics through
  `FLModel.metrics`; must not synthesize metric semantics without source
  evidence.
- Must count completed local optimizer steps in each generated training round
  and send that positive value as `MetaKey.NUM_STEPS_CURRENT_ROUND`. This is the
  FedAvg aggregation weight; do not omit it, reuse a cumulative count, or
  invent a value when the source loop cannot establish it.
- Must load checkpoints with `torch.load(..., weights_only=True)`; a
  checkpoint that needs full unpickling is ask/fail, per
  `references/pytorch-client-api-conversion.md`.
- Must not make non-PyTorch-family skills load
  `../nvflare-shared/references/pytorch-model-exchange.md`; that reference is
  for plain PyTorch, PyTorch Lightning, and Hugging Face Trainer model/state-dict
  exchange only.
- Site partitioning, custom aggregation, the Source Of Truth Boundary, and user
  input/authorization follow `../nvflare-shared/references/conversion-common.md`.

Always read this converter SKILL.md together with
`../nvflare-shared/references/conversion-common.md`. The standard routing,
recipe selection, and reporting path is inline, so common FedAvg does not load
broad policy or algorithm-selection references. Load the client template,
model-exchange reference, validation reference, and aggregator asset only when
their phase needs them. Load other detailed references only for exceptions:

- `../nvflare-shared/references/conversion-workflow.md` for the full conversion
  contract when a case is non-standard;
- `../nvflare-shared/references/site-data-and-paths.md` only for generated site
  partitions, relative-path resolution, or per-site data locations;
- `../nvflare-shared/references/pytorch-family-recipe-selection.md` only for
  ambiguous or non-FedAvg algorithms, and `references/recipe-selection.md` only
  for non-FedAvg or execution-mode construction details not supplied by
  `recipe show`;
- `../nvflare-shared/references/pytorch-family-recipe-construction.md` after
  every `recipe show`;
- `../nvflare-shared/references/dependency-install.md` only when an install is
  needed;
- `../nvflare-shared/references/runtime-output-guidance.md` only for read-only
  source roots or user-chosen output destinations;
- `../nvflare-shared/references/metrics-and-artifact-reporting.md` only when
  metrics are absent or inconsistent;
- `../nvflare-shared/references/validation-evidence.md` before validation, and
  `../nvflare-shared/references/pytorch-model-exchange.md` only for PyTorch-family exchange;
- `references/pytorch-client-api-conversion.md` for Client API conversion, and
  `references/job-validation.md` for PyTorch-specific validation failures.

Do not load every reference preemptively, and do not depend on NVFLARE
repository examples being present in the user's environment.
