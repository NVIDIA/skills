# Hugging Face State, PEFT, And Distributed Training

## Parameter Scope

Keep `params_scope="auto"` unless source evidence requires an override:

- a normal Trainer exchanges the full model state;
- a PEFT `PeftModel` exchanges adapter state only;
- `"adapter"` requires an actual PEFT model;
- `"model"` forces full-model state and increases communication and memory.

The server model must expose exactly the exchanged keyspace. For PEFT, construct
the same base model and PEFT configuration and have its `state_dict()` /
`load_state_dict()` use `get_peft_model_state_dict()` and
`set_peft_model_state_dict()`. Preserve adapter name and every LoRA/configuration
value. Compare actual key sets before validation. Use `server_key_prefix` only
for a known wrapper namespace difference; do not use non-strict loading to hide
an unexplained mismatch.

Some Trainer subclasses own auxiliary reference, reward, critic, or value-head
models in addition to `trainer.model`. Before conversion, inspect which of those
objects are trainable and whether their state participates in aggregation.
Proceed only when the complete federated trainable state is represented by the
patched Trainer's exchanged parameter scope; otherwise ask or fail closed.

Use `ExchangeFormat.PYTORCH` for native tensor dtypes such as BF16. Keep dtype,
precision, quantization, and device-map behavior from source evidence rather
than model-family assumptions. Follow the authorization rules in
`../../nvflare-shared/references/conversion-common.md`, plus the `SKILL.md`
delta covering `trust_remote_code` and offline recovery, for remote code and
download effects.

## Checkpoint Continuity

`restore_state=True` is the default and preserves optimizer, scheduler, RNG, and
Trainer progress through standard Hugging Face checkpoints within one Trainer
process lifecycle. Preserve it unless the process is intentionally relaunched
for each task.

Reject or report:

- `save_only_model=True` with `restore_state=True`;
- `load_best_model_at_end=True`;
- explicit recipe `launch_once=False` with `restore_state=True` — the persistent
  Trainer cannot survive a per-task relaunch. `launch_once` itself is owned by
  `../../nvflare-shared/references/pytorch-family-recipe-construction.md`;
- prebuilt optimizer/scheduler instances with `restore_state=False`;
- checkpoint paths that are not visible to every distributed rank.

Do not promise crash recovery or cross-job checkpoint provenance. With
`restore_state=False`, each round is stateless and Trainer creates fresh
optimizer/scheduler state.

## Distributed Training

This section applies only to a preserved distributed multi-process launch.
Apply the framework-neutral global-rank contract in
`../../nvflare-shared/references/conversion-common.md`; do not add a rank
argument to a standard single-process conversion.

Initialize `torch.distributed` before `flare.patch(trainer)` whenever
`WORLD_SIZE` or `LOCAL_WORLD_SIZE` is greater than one. Resolve global rank from
the initialized process group or global `RANK`, not `LOCAL_RANK`.
`nvflare.client.hf.init()` owns this resolution and rejects a declared
multi-process launch when neither source is available, before creating a Client
API context. Keep the generated client rankless instead of duplicating product
logic.
`LOCAL_RANK` selects the local device and may differ from global `RANK` on
multi-node launches; do not pass it as the FLARE rank.

Every rank must execute the same generated sequence of patched methods. If the
source-backed loop evaluates before training, all ranks call
`evaluate()` then `train()`; if the valid loop is train-only, all ranks use that
same train-only sequence.

Rank 0 owns FLARE receive/send; the patch broadcasts tasks and parameters.
Default distributed parameter exchange is in-memory. File exchange is an
explicit optimization and requires a shared `output_dir`. Regardless of
parameter transport, `restore_state=True` requires all ranks to see the same
checkpoint paths.

Replicated DDP is supported. DeepSpeed and FSDP are not supported by this Client
API version. Preserve a source-provided `torchrun`/scheduler launcher; do not
invent node counts, ranks, rendezvous settings, or scheduler commands. When the
selected recipe exposes a command prefix, carry the observed `torchrun` command
through that product parameter; use per-site configuration when launcher values
differ by site. Before reporting DDP validated, run a two-process `torchrun`
conversion case and verify that both processes reach the patched lifecycle with
distinct global ranks `0` and `1`, and that each passes that same global rank to
`flare.init()`.
