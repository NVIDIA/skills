# PyTorch Client API Conversion

This reference covers standard PyTorch training loops that already have a
`torch.nn.Module`, optimizer, data loaders, and metrics. Load
`../../nvflare-shared/references/pytorch-model-exchange.md` for PyTorch-family state-dict and
tensor payload rules before changing model exchange code.

## Canonical Path

Use this path for plain PyTorch conversion:

1. Confirm PyTorch routing with `nvflare agent inspect source`.
2. Select a PyTorch-family recipe with `nvflare recipe list/show`.
3. Generate `client.py` with `nvflare.client` `receive` / `send` and
   `FLModel(params=...)` as the model exchange path.
4. Generate `job.py` that builds the selected recipe and calls
   `recipe.execute(SimEnv(...))`.
5. Select exactly one final target: run `python job.py` and inspect its
   materialized simulation workspace for local validation, or, only when the
   user requests an exported/deployable artifact, export and run that folder
   with the simulator CLI. Do not run both targets after one succeeds.

HE is not supported at steps 4–5: follow the HE-not-supported rule in
`../../nvflare-shared/references/pytorch-family-recipe-selection.md`.

Follow the Source Of Truth Boundary and the generated-entry rule in
`../../nvflare-shared/references/conversion-workflow.md`: `client.py` is an
FL-only Client API entry point, not a standalone/FL auto-detecting launcher.

## Conversion Pattern

- Import `nvflare.client as flare`.
- Build the model, optimizer, loss, and data loaders once before the loop, not
  inside it, per the shared "Setup Outside The Round Loop" rule in
  `../../nvflare-shared/references/conversion-workflow.md`.
- Call `flare.init()` before setup hooks that need Client API context, such as
  `flare.get_config()` or `flare.get_site_name()`, while still keeping setup
  outside the federated round loop.
- Loop while `flare.is_running()`.
- Call `flare.receive()` to get the incoming `FLModel`.
- Load `input_model.params` into the PyTorch model with `load_state_dict`.
- Train or evaluate using the user's existing data loader and optimizer.
- Count only optimizer steps completed in the current round. The generated
  `train_one_round` helper must return this positive count; do not use an
  estimated, cumulative, or default value.
- Send the trained weights with the canonical plain-PyTorch payload pattern in
  `../../nvflare-shared/references/pytorch-model-exchange.md`. Do not call
  `model.cpu()`, which moves the persistent model off the training device.

## PyTorch Parameter Payload Type

Use the exact outbound payload contract and send snippet in
`../../nvflare-shared/references/pytorch-model-exchange.md`. Apply the separate
recipe-capability policy in
`../../nvflare-shared/references/pytorch-family-recipe-construction.md`; its
server disk-offload optimization does not change the client payload contract or
execution mode.

## Source Layout

Use the canonical FLARE source layout defined in
`../../nvflare-shared/references/conversion-workflow.md` ("Generated Job Layout"):
`client.py`, `model.py`, `job.py`, and the optional `aggregators.py`, data-setup,
and requirements files. Avoid ad hoc entry-point names such as `fl_train.py`
unless the user explicitly requests that naming, and use
`../../nvflare-shared/references/runtime-output-guidance.md` for runtime
workspaces, exported job directories, and validation output locations.
During export inspection, verify generated or project-local modules referenced
by server-side `class_path` config are packaged into the server app with
`recipe.add_server_file(...)` or an equivalent server-targeted API. The
`train_script` import closure packages client apps and is not enough for
per-site exports that create `app_server` separately. Installed NVFLARE,
framework, and third-party class paths stay runtime dependencies and are
validated through requirements installation plus import/preflight checks.

For standard FedAvg, package shared generated files for all clients. Do not
replace all-client deployment with explicit per-site deployment unless the
conversion has real per-site differences such as different scripts, arguments,
data-split settings, or launch behavior.

For multi-site conversion from a single-node PyTorch source, the generated
client or data-loader setup must filter the local training data by site. Do not
let every simulated site train on the full source training set unless the user
explicitly asks for shared training data or the source already provides
site-specific data that resolves to that behavior. Validation/test loaders may
remain shared only when that matches the source's validation/test semantics.
For generated Pandas partition code, follow
`../../nvflare-shared/references/site-data-and-paths.md`.

## Model Construction Consistency

Follow the shared model-config and construction-consistency rule in
`../../nvflare-shared/references/conversion-workflow.md` ("Recipe Model Config"):
same class and constructor args on server and client, explicit config whenever
reconstruction needs a constructor value, and derive-or-ask/fail-closed for it.

PyTorch-specific delta: the client loads `input_model.params` into the model
with `load_state_dict`, so the server-initial model and the client model must
have matching state-dict shapes (same parameter names and tensor shapes), not
only matching constructor args.

Acceptable patterns include:

- a shared `model_args` dict imported by both `job.py` and `client.py`;
- an explicit recipe model config such as
  `{"class_path": "model.ModelClass", "args": model_args}` (prefer
  `class_path`; `path` is the normalized job-config key);
- a small JSON/config file read by both sides;
- explicit CLI arguments passed through recipe `train_args` and parsed by
  `client.py`, with the same values used in `job.py`.

Before simulation, validate the generated model construction path when possible
by instantiating the server-side and client-side model with the same arguments
and checking that `load_state_dict` can accept the initial parameters. Treat a
state-dict key or tensor-shape mismatch as a conversion bug, not as a reason to
change the model architecture without user approval.

## Paired Evaluation Template

Training and evaluation are a pair: every converted training loop that has
source evaluation evidence must also convert that evaluation, and its metrics
must reach the server through `FLModel.metrics`. Adapt the user's existing
evaluation code into this template. Do not synthesize metric semantics,
validation loaders, label mappings, or averaging denominators from scratch
without source evidence; when evaluation is required but the source has none,
ask in interactive mode or fail closed in unattended mode.

Follow the "Best-Model Metric" contract in
`../../nvflare-shared/references/pytorch-family-recipe-construction.md`. It owns
metric-name matching and lower-is-better direction; this reference owns where
evaluation occurs in the plain-PyTorch round loop.

The self-contained runnable template ships at
`../assets/client_with_eval.py`; adapt it rather than duplicating its code here
or depending on repository `examples/`. It initializes setup once, receives and
loads the global model, evaluates that received model, handles evaluation-only
tasks, trains, and sends the canonical model payload, source-backed metric, and
the actual completed optimizer-step count for FedAvg weighting. If the source
loop cannot establish that count, ask in interactive mode or fail closed in
unattended mode instead of silently using equal client weights.

The round `FLModel.metrics` is this pre-training evaluation of the received
global model, not a post-training metric — see
`../../nvflare-shared/references/metrics-and-artifact-reporting.md`
("Received-Model Metric Ownership").

When the task is evaluation-only or cross-site evaluation, use
`flare.is_evaluate()` to send `flare.FLModel(metrics=...)` without local
training and without params.

## Checkpoint Loading Safety

Generated code that loads PyTorch checkpoint files must use safe weight-only
loading: `torch.load(..., weights_only=True)`. A checkpoint that requires full
pickle unpickling or custom executable deserialization is not statically safe;
ask in interactive mode or fail closed in unattended mode instead of loading
it.

## Scope Boundaries

- Keep user model architecture and loss function unless the user asks for a
  change.
- Keep data loading local to the site and do not add code that copies private
  data into generated artifacts.
- For checkpoints, preserve user checkpoint semantics and document what is
  federated versus site-local.
- For metrics, send scalar summaries in the `metrics` field. Use
  `../../nvflare-shared/references/metrics-and-artifact-reporting.md` for generic final metrics,
  round metrics, model artifact paths, and missing-evidence reporting.

## Job Pattern Reference

Load `recipe-selection.md` before creating or updating `job.py` so the selected
recipe matches the user's requested FL workflow. Do not assume NVFLARE
repository examples are available in the user's environment.
