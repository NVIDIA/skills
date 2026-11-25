# Lightning Job Validation Notes

Use `../../nvflare-shared/references/validation-evidence.md` for generic validation status,
commands, blockers, and evidence reporting. Use
`../../nvflare-shared/references/metrics-and-artifact-reporting.md` for final metrics, round
metrics, model artifact paths, and missing-evidence reporting. This file only
covers Lightning-specific validation checks.

## Validate In Order

1. If an applicable dependency is missing, load
   `../../nvflare-shared/references/dependency-install.md` and install it before
   importing the user's Lightning code. Do not load that reference for an
   already-satisfied environment.
2. Before a full run, select exactly one final validation target:
   - For local or first-run simulation without an export claim, run only
     `python job.py` and follow
     `../../nvflare-shared/references/runtime-output-guidance.md` for workspace
     location.
   - For a requested exported/deployable artifact, run the recipe's public
     export command, inspect the export per
     `../../nvflare-shared/references/conversion-workflow.md` ("Export"), and run
     only the exported folder with the simulator CLI. Do not first run
     `python job.py` as a local simulation.
   HE is not supported: homomorphic-encryption recipes reject `SimEnv` and
   require provisioned `PocEnv`/`ProdEnv` outside conversion scope, so refuse an
   HE request before this step. Report it as unsupported and route to
   provisioning/deployment per the HE-not-supported rule in
   `../../nvflare-shared/references/pytorch-family-recipe-selection.md` and
   `../../nvflare-shared/references/conversion-workflow.md` rather than
   generating an HE job.
3. Run the selected target to completion per the shared contract
   (`../../nvflare-shared/references/conversion-workflow.md` hard-stop and
   `../../nvflare-shared/references/validation-evidence.md` evidence contract).
   Never run the other target after success. After failure, diagnose, apply a
   scoped fix, and rerun the same target; change targets only when evidence shows
   that the original target does not represent the requested artifact.
4. Account for Lightning startup: the patched `Trainer`, callback setup, and
   logger flush make Lightning runs slower than plain PyTorch, and
   distributed-process jobs launch externally (see
   `lightning-ddp-and-tracking.md`). Observe their completion before reporting
   success; scheduled wakeups or progress logs are not success evidence. If the
   run times out, report it as blocked or timed out with current server/client
   log evidence.
5. Report the declared primary/global metric scalar when one exists.

## Lightning-Specific Checks

- Confirm `flare.patch(trainer)` is applied to the same `Trainer` instance used
  for fit/validate/test, and that the patched trainer, not manual `FLModel`
  code, performs model exchange.
- Confirm no `input_model` returned by `flare.receive()` is passed into
  `Trainer` methods.
- Confirm the `LightningModule` constructed on the client matches the recipe's
  server-side model constructor arguments and state-dict shapes.
- Confirm callbacks, loggers, and checkpoint callbacks still run after patching
  and do not break the FL round loop.
- Confirm validation metrics are exposed as scalars (for example through
  `self.log(...)` in the `LightningModule`) so aggregation recipes can write
  server-side metric artifacts.
- For every non-Cyclic training task,
  confirm an explicit standalone `trainer.validate(...)` runs before
  `trainer.fit(...)`, its exact logged key reaches client `FLModel.metrics`, and
  no `model.__fl_meta__[MetaKey.INITIAL_METRICS]` override is generated. Local
  callback metrics alone are insufficient end-to-end evidence.
- Confirm Lightning sanity checks and validation performed inside
  `trainer.fit(...)` are not reported as received-global-model metrics.
- Treat a non-Cyclic round that reports no server metrics as evidence that the
  pre-fit validation is missing. Without it the received global model is
  unscored, so best-model selection silently produces nothing and
  `train_with_evaluation=True` fails on the missing required metrics. For
  Cyclic, instead confirm the pre-fit call is absent, the final sequential model
  is persisted, and no best-model artifact is claimed or expected.
- When a custom `ModelAggregator` is used, confirm client models reach it with
  non-empty `FLModel.metrics` and its aggregated result returns non-empty
  `FLModel.metrics`.
- When server metrics were requested, require the expected metric key in the
  server metric artifact or bounded server logs. A terminal `Finished` state
  without that metric is incomplete validation, not a successful metric result.
- For data-prep changes, confirm the `LightningDataModule` receives the
  generated per-site path or arguments rather than hard-coded global paths.

## Known SimEnv Limitations

- SimEnv runs sites in a single local environment; accelerator scaling and
  distributed-process behavior are validated separately (see
  `lightning-ddp-and-tracking.md`). A single-process SimEnv run validates
  conversion structure, not distributed scaling.
- Treat synthetic or smoke-test data runs as structural validation, not as
  meaningful accuracy evidence, unless the user supplies expected metrics.
- Report Lightning-specific blockers such as a trainer that cannot be patched, a
  callback or logger that fails inside the round loop, checkpoint loading that
  conflicts with the patched model exchange, or metrics that are not logged as
  scalars.
