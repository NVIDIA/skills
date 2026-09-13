# Lightning Client API Conversion

This reference covers converting a PyTorch Lightning `Trainer` workflow to the
NVFLARE Lightning Client API. Load `../../nvflare-shared/references/pytorch-model-exchange.md` for
PyTorch-family tensor/state-dict rules before changing model exchange code.

The Lightning integration owns model load and send through callbacks installed
by `flare.patch(trainer)`. Do not generate a manual `FLModel` send/receive path
for normal Lightning training.

## Canonical Path

Use this path for Lightning conversion:

1. Confirm Lightning routing with `nvflare agent inspect source`.
2. Select a PyTorch-family recipe with `nvflare recipe list/show`.
3. Generate `client.py` with `flare.patch(trainer)` as the model exchange path.
4. Generate `job.py` that builds the selected recipe and calls
   `recipe.execute(SimEnv(...))`.
5. Select exactly one final target per `lightning-validation.md`: run
   `python job.py` for local simulation without an export claim, or export and
   run the exported folder with the simulator CLI for a requested deployable
   artifact. Never run the local target and then export after it succeeds.

HE is not supported at steps 4–5: follow the HE-not-supported rule in
`../../nvflare-shared/references/pytorch-family-recipe-selection.md`.

Follow the Source Of Truth Boundary and the generated-entry rule in
`../../nvflare-shared/references/conversion-workflow.md`: `client.py` is an
FL-only Client API entry point, not a standalone/FL auto-detecting launcher.

## Conversion Pattern

- Import the Lightning client API: `import nvflare.client.lightning as flare`.
- Build the `LightningModule`, `LightningDataModule`, and `Trainer` as the
  source code already does.
- Call `flare.patch(trainer)` once, after the trainer is constructed.
- Loop while `flare.is_running()` and call `trainer.fit` / `trainer.validate` /
  `trainer.test` as the workflow requires.

```python
import nvflare.client.lightning as flare

flare.init()  # required before get_site_name(), get_config(), receive(), or other pre-patch context
trainer = Trainer(...)
flare.patch(trainer)

while flare.is_running():
    # Optional: call receive() only when round/site/task metadata is needed.
    # The patched trainer loads the global model internally.
    flare.receive()
    validate_global_model(trainer, model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)  # when test evidence is requested/available
```

Adapt `validate_global_model` from `../assets/lightning_client.py`; it keeps
evaluation Lightning-native and preserves training-result metrics on the
patched exchange.

For evaluation-only / FedEval conversions, run `trainer.validate(...)` (the
patched trainer sends the validation metrics) and **do not call
`trainer.fit(...)`** — training was not requested, and fitting after the metrics
are sent can train an unwanted round or block the task. The packaged
`../assets/lightning_client.py`
`main(..., evaluate_only=True, recipe_algorithm="fedeval")` skips `fit`.
Pass that complete pair only when the resolved recipe algorithm is FedEval.
For every workflow that also trains, omit the argument so its default remains
`False` and the received training task completes through `trainer.fit(...)`.

## Patch Ownership Rules

- `flare.patch(trainer)` installs callbacks that receive the global model into
  the Lightning model and send the updated model after fit. Let it own the
  exchange.
- Do not pass the received `input_model` into `Trainer` methods. The patched
  trainer loads the global model internally.
- Do not add a second manual `flare.send(FLModel(...))` for normal training; the
  patched trainer already sends the trained model.
- Use `flare.get_site_name()` or `flare.get_config()` only when the code needs
  pre-patch context such as site-specific logging or data paths, and call
  `flare.init()` before the first such Client API context access.
- Use `flare.receive()` in the patched loop only for FL task progression,
  round/site logging, or task metadata, never for manual model loading.
- During export inspection, verify generated or project-local modules
  referenced by server-side `class_path` config are packaged into the server app
  with `recipe.add_server_file(...)` or an equivalent server-targeted API. The
  `train_script` import closure packages client apps and is not enough for
  per-site exports that create `app_server` separately. Installed NVFLARE,
  framework, and third-party class paths stay runtime dependencies and are
  validated through requirements installation plus import/preflight checks.

## Lightning Evaluation Template

Keep evaluation inside Lightning; do not reuse the raw PyTorch
`model.eval()` / `torch.no_grad()` loop for normal Lightning conversion:

- Require or preserve `validation_step()` / `test_step()` and a
  validation/test dataloader or `LightningDataModule`.
- Log validation metrics from the `LightningModule` with `self.log(...)` so
  they are visible in the trainer callback metrics.
- After `flare.patch(trainer)` and `flare.receive()`, call
  `trainer.validate(model, datamodule=...)` before `trainer.fit(...)` when
  server-side model selection or round metrics need validation; keep this
  inside the `while flare.is_running()` loop.
- Use `trainer.test(...)` only when the source workflow already has test
  semantics or the user requests test reporting.
- Rely on Lightning's validate/test loops to set evaluation mode and disable
  gradients; generate a manual `model.eval()` loop only when the conversion
  intentionally routes to plain PyTorch.
- If the source project lacks validation/test steps or dataloaders, ask in
  interactive mode or fail closed in unattended mode instead of inventing
  metric semantics.

### Training-result metric delivery

For a training task, derive
`evaluate_before_train = recipe_algorithm != "cyclic"` from the normalized
`algorithm` returned by `nvflare recipe show`. When it is true, call an explicit
standalone `trainer.validate(...)` after `flare.patch(trainer)` and before
`trainer.fit(...)`. The patched callback captures finite scalar callback metrics
from that validation and attaches them to the outgoing training result,
regardless of whether the executor's `train_with_evaluation` setting is `False`
or unavailable through the selected recipe. That setting controls whether
missing pre-training metrics are an error: `True` requires them; `False` makes
them optional rather than suppressing metrics that Lightning supplies.

Only that explicit pre-fit validation scores the received global model.
Lightning sanity checks and validation performed inside `trainer.fit(...)` run
in the fitting lifecycle and are deliberately excluded from the global-model
score.

Best-model selection therefore depends on the pre-fit call. Omitting it from a
non-Cyclic recipe leaves the received global model unscored, so the selector has
nothing to compare and no best global model is persisted; with
`train_with_evaluation=True` the round fails on the missing required metrics.
Do not expose an independent skip flag: derive the value from the recipe
algorithm so selection and evaluation cannot silently diverge.

Cyclic is the intentional exception. Its clients update the model sequentially,
so their pre-fit validations would score different intermediate models rather
than one round-global candidate. Skip the standalone pre-fit call, preserve any
ordinary in-fit validation as local behavior, and report the persisted final
model without claiming a best-model artifact. For a non-Cyclic source without
validation semantics, ask or fail closed rather than silently disabling its
selector. When an application must keep explicit pre-fit metrics local, report
the need for an authorized custom task-result filter.

Use `../assets/lightning_client.py` as the copyable validate-before-fit loop.
Its finite-scalar check validates the values returned by `trainer.validate`,
while the patched callback owns delivery. Do not copy validation metrics into
`model.__fl_meta__[MetaKey.INITIAL_METRICS]`: that reserved metadata bypasses
the automatic capture contract and can replace the freshly captured callback
metrics. Other source-owned `__fl_meta__` entries remain unrelated custom
metadata.

Metric names are not remapped by NVFLARE. The key emitted through `self.log`,
the recipe's `key_metric`, and artifact reporting must match exactly. Prove the
key in client and aggregated `FLModel.metrics` or server artifacts before
claiming server-side model selection; local callback metrics alone remain
insufficient end-to-end validation evidence.

Metric names are selected with an explicit direction. When the source
establishes that lower `val_loss` is better, preserve its existing epoch-level
log:

```python
self.log("val_loss", loss, on_step=False, on_epoch=True)
```

The recipe then selects `key_metric="val_loss", key_metric_mode="min"`.
For DDP, preserve the source-backed distributed reduction behavior on the log.
Only choose a direction the source establishes; do not invent one. Use a
negated companion only when the resolved recipe lacks `key_metric_mode`; name
and document that compatibility metric explicitly. A `val_loss`-only module is
never a reason to fail closed or to skip requested best-model selection.

If a custom `ModelAggregator` is selected, it must also aggregate supported
client `FLModel.metrics` values and return them in the aggregated
`FLModel.metrics`. Follow the Custom Aggregation contract in
`../../nvflare-shared/references/conversion-workflow.md` and adapt
`../../nvflare-shared/assets/aggregator.py`; a parameters-only aggregate loses
the server-level metric even when clients delivered it.

This template is self-contained packaged guidance; do not depend on NVFLARE
repository `examples/` being present in the user's environment. The runnable
form ships at `../assets/lightning_client.py`; adapt it rather than inventing a
new structure.

## Preserve Lightning Behavior

- Preserve user callbacks, loggers, and checkpoint callbacks unless the user
  asks to change them.
- Repo-shipped checkpoint files (`.ckpt` passed to `load_from_checkpoint`,
  `Trainer.fit(ckpt_path=...)`, or resume logic) are untrusted executable input
  per `../../nvflare-shared/references/conversion-workflow.md`: full-unpickle loading of a
  repo-supplied checkpoint is ask/fail. Checkpoints produced by the current
  validation run (for example `ckpt_path="best"` from this run's checkpoint
  callback) may follow normal Lightning handling.
- Keep the `LightningModule`/`LightningDataModule` architecture and data logic;
  do not rewrite training_step/validation_step semantics.
- Avoid repeated expensive setup (model build, dataset download) inside the FL
  round loop; construct the model, datamodule, and trainer once before the loop
  when the source code allows it.

## Local Data And Loss Policy

Follow the training-policy distinction in
`../../nvflare-shared/references/pytorch-model-exchange.md` and the site split guidance in
`../../nvflare-shared/references/conversion-workflow.md`. Lightning-specific implication:
label/site-derived values that affect `training_step`, `LightningDataModule`
sampling, or validation/test decision logic remain local to each site partition
unless the user explicitly requests one global training policy. Do not move
those values into recipe `model` args just because architecture args must be
shared.

If such a local value is a persistent model parameter or buffer, recomputing it
before the round loop does not keep it local: the patched trainer sends its full
`state_dict()` and the received model can overwrite the value. The FedAvg recipe
`exclude_vars` argument is only an aggregation rule; it is not a bidirectional
payload filter and is not a locality or privacy control.

For a FedAvg `train` task, keep an explicitly named state key local only by
filtering both network directions with separate filters:

```python
from nvflare.app_common.filters import ExcludeVars

recipe.add_server_output_filter(ExcludeVars([local_key]), tasks=["train"])
recipe.add_client_output_filter(ExcludeVars([local_key]), tasks=["train"])
```

Pair those filters with
`flare.patch(trainer, load_state_dict_strict=False)`, because each received
global model intentionally omits the local key. Non-strict loading does not
suppress outbound state by itself. Apply the same two-direction rule to every
model-bearing task selected by a different recipe. Verify exported server and
client filter configuration, then verify that the exact key is absent from both
directions—including the initial and later server-to-client payloads—and the
site-computed value is not overwritten in the first or later rounds. If the
selected recipe does not expose supported filters for every
model-bearing direction, ask or fail closed. Do not substitute aggregation-only
`exclude_vars` or silently exchange the value.

Report the split policy, seed, and where local training-policy values are
computed.

For multi-site conversion from a single-node Lightning source, the generated
client or `LightningDataModule` must filter the local training split by site.
Do not let every simulated site train on the full source training set unless
the user explicitly asks for shared training data or the source already provides
site-specific data that resolves to that behavior. Validation/test splits may
remain shared only when that matches the source's validation/test semantics.

## Model Construction Consistency

Follow the shared model-config and construction-consistency rule in
`../../nvflare-shared/references/conversion-workflow.md` ("Recipe Model Config"):
same class and constructor args on server and client, explicit config whenever
reconstruction needs a constructor value, and derive-or-ask/fail-closed for it.

Lightning-specific delta: the exchanged unit is the whole `LightningModule`
managed by the patched trainer, so construct the identical `LightningModule` on
the server and on the client in `client.py`, not just the inner
`torch.nn.Module`. For explicit config, express shared arguments as a
`model_args` dict (prefer `class_path`; `path` is the normalized job-config
key).

## Source Layout

Use the canonical FLARE source layout defined in
`../../nvflare-shared/references/conversion-workflow.md` ("Generated Job Layout").
Lightning-specific delta: `client.py` patches the trainer as the model-exchange
path, and `model.py` holds the `LightningModule` (and `LightningDataModule`)
definition when a new file is needed. Avoid ad hoc names such as `fl_train.py`
unless the user requests them, and use
`../../nvflare-shared/references/runtime-output-guidance.md` for runtime
workspaces, exported job directories, and validation output locations.

If the Lightning entry point imports a model module or source file — whatever
its filename, for example `model.py`, `simple_model.py`, or a module loaded
through explicit import machinery — or defines the
`LightningModule`/`LightningDataModule` in `train.py`, preserve that source
structure. Generate `client.py`, `job.py`, and optional `aggregators.py`; do not
generate a new replacement model implementation that recreates the full
Lightning/data stack. If the generated package needs the canonical `model.py`
name, use a mechanical copy/rename or thin wrapper around the detected source
module rather than re-authoring it.

## Recipe Reuse

Lightning reuses the PyTorch recipe family. Follow
`../../nvflare-shared/references/pytorch-family-recipe-selection.md` for recipe
discovery, the algorithm guide, catalog-based selection rules, and the
HE-not-supported rule — the same catalog and rules apply to Lightning, including
non-FedAvg workflows such as FedOpt, FedProx, SCAFFOLD, Cyclic, Swarm, and
FedEval. Use FedAvg for standard horizontal training and FedEval for
evaluation-only.

The generated `job.py` should use the selected recipe's public parameters from
`recipe show`, construct the model through explicit `class_path` (or `path`) plus
`args` when constructor arguments are required, and call
`recipe.execute(SimEnv(...))`. HE is not supported: homomorphic-encryption
recipes reject `SimEnv` and require provisioned `PocEnv`/`ProdEnv` outside
conversion scope — follow the HE-not-supported rule in
`../../nvflare-shared/references/pytorch-family-recipe-selection.md` (report
unsupported, route to provisioning/deployment, ask or fail closed; do not
generate an HE job). Do not replace this with ad
hoc SDK-internal APIs based on local source or docstring inspection. Follow
`../../nvflare-shared/references/conversion-workflow.md` for export and
command-line behavior.
