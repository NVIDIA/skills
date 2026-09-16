# Lightning DDP And Experiment Tracking

Use this reference only for Lightning distributed-process work or
experiment-tracking work. Single-process Lightning training does not need it.

## DDP Execution Model

Use the process-evidence criterion and capability checks in
`../../nvflare-shared/references/pytorch-family-recipe-construction.md`; that
reference owns the DDP/DataParallel-to-execution-mode decision. This reference
only covers Lightning behavior after that decision.

## Rank-Synchronized Round Loop

Before patching under DDP, apply the framework-neutral global-rank contract in
`../../nvflare-shared/references/conversion-common.md`: resolve the global
process rank from the initialized process group or global `RANK`, never
`LOCAL_RANK`, and pass it to `flare.init(rank=global_rank)`.

Under DDP, only rank 0 communicates with the FL server, so all ranks must agree
on whether to continue. Broadcast `flare.is_running()` from rank 0 before each
round, and let the patched trainer callback receive and broadcast the model
during `validate`/`fit`:

```python
flare.patch(trainer)

while True:
    is_running = flare.is_running()
    is_running = trainer.strategy.broadcast(is_running, src=0)
    if not is_running:
        break

    validate_global_model(trainer, model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)  # when test evidence is requested/available
```

Adapt `validate_global_model` from `../assets/lightning_client.py` when server
metrics are required.

Do not insert an unguarded `flare.receive()` on every rank for metadata. Outside
the patched callback, rank 0 can fetch the cached `FLModel`; non-zero ranks do
not get the task/model object. If user code outside the patched trainer needs
round/task metadata on all ranks, derive a small serializable value on rank 0 and
broadcast that value before using it:

```python
round_info = {}
if trainer.global_rank == 0:
    input_model = flare.receive()
    round_info = {
        "current_round": input_model.current_round if input_model else None,
        "meta": dict(input_model.meta or {}) if input_model and input_model.meta else {},
    }
round_info = trainer.strategy.broadcast(round_info, src=0)
```

As with single-process training, the patched trainer owns model exchange. Do not
pass `input_model` into `Trainer` methods and do not add a manual `flare.send`.
In DDP, rank-0 communication with the FL server is either handled by the patched
callback path or by explicit rank-0 guarded metadata code like the snippet above;
non-zero ranks should read broadcast values, not `input_model`.

### DDP validation metrics use the patched callback

DDP requires the external-process launch (`launch_external_process=True`), which
runs the script under `ClientAPIExecutor(execution_mode="external_process")`.
That executor defaults to `train_with_evaluation=False`, and the recipe's
`ScriptRunner` does not expose a switch to change it. The setting makes metrics
optional; it does not suppress finite scalar metrics captured from an explicit
standalone `trainer.validate(...)` before `trainer.fit(...)`.

When DDP training requires server metrics, use the canonical pre-fit validation
path in `lightning-conversion.md` and let the patched callback attach those
metrics to the outgoing training result. Ensure Lightning reduces distributed
validation metrics consistently (for example, preserve source `sync_dist`
behavior) and that the sending rank receives the scalar result. Do not populate
`model.__fl_meta__[MetaKey.INITIAL_METRICS]` and do not add a second manual
`flare.send(...)`. Sanity checks and in-fit validation remain local fitting
lifecycle metrics, not received-global-model scores.

Only pass a source-derived `key_metric` or claim server-side round metrics after
the generated execution path and validation evidence confirm that the exact
metric reaches client and aggregated `FLModel.metrics`. Otherwise surface the
limitation or fail closed instead of promising metrics from local Lightning
logs.

## GPU/CPU Fallback

- Keep the user's `accelerator`/`devices` settings; do not silently force CPU.
- When the validation environment has no GPU, report the limitation and validate
  conversion structure on CPU or a reduced device count instead of changing the
  user's training intent.

## Experiment Tracking

Enable remote tracking only when the user explicitly requests it. Existing
source logger or callback configuration is evidence to inspect, not a user
request, and the skill must not ask solely to enable an external effect.

- Preserve local-only loggers such as a local `TensorBoardLogger` when their
  output stays in the selected runtime directory.
- Treat remote `MLFlowLogger`, WandB/Comet-style clients, upload callbacks, and
  custom or unknown loggers as network-capable. Their source configuration is
  evidence to inspect, not a user request. Keep them disabled during validation
  unless the user explicitly requested remote tracking.
- Hand metrics to FLARE through `add_experiment_tracking` or the FLARE client
  logger when the workflow needs server-side or streamed tracking. The canonical
  client-facing shortcut is `flare.logger()` (with
  `import nvflare.client.lightning as flare`); the class is
  `nvflare.app_opt.lightning.loggers.client_logger.ClientLogger`. Do not import
  it as `nvflare.app_opt.lightning.loggers.ClientLogger`, which is not exported.
- The client logger streams metrics through the FL client; it is not a full
  replacement for a standalone tracking server. State this limitation rather
  than promising parity with a dedicated tracking backend.

Follow `../../nvflare-shared/references/metrics-and-artifact-reporting.md` for reporting metric
and artifact paths and for missing-evidence reporting.
