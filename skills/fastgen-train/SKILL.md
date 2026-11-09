---
name: fastgen-train
description: >-
  Run, resume or troubleshoot NVIDIA FastGen training: SFT or distillation with train.py, bounded smoke tests, distributed launches, checkpoints, validation and W&B evidence. Use it whenever training is executed or a training run misbehaves.
license: Apache-2.0
metadata:
  author: "Julius Berner <jberner@nvidia.com>"
  domain: ml
---

# Train or distill with FastGen

Apply [fastgen-runtime](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-runtime/SKILL.md) first. The
[README](https://github.com/NVlabs/FastGen/blob/main/README.md#quick-start) and the [configuration README](https://github.com/NVlabs/FastGen/blob/main/fastgen/configs/README.md)
own launch commands, run directories and checkpoint storage; the
[troubleshooting guide](https://github.com/NVlabs/FastGen/blob/main/fern/docs/troubleshooting.mdx) owns failure diagnosis. Read
[run contracts](references/run-contract.md) to bound a run, read its logs or continue a
checkpoint.

1. **Goal and inputs.** Decide whether the deliverable is a smoke test, a bounded
   experiment, a quality target or a resumed run. Use
   [fastgen-configure](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-configure/SKILL.md) for the recipe and
   [fastgen-dataset](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-dataset/SKILL.md) for data; check every enabled
   loss, and the validation loader, against the real-data versus prompt-only contract.
2. **Weights versus state.** Distinguish external weights, warm starts and full resume;
   the [checkpoint table](https://github.com/NVlabs/FastGen/blob/main/fern/docs/troubleshooting.mdx#a-checkpoint-fails-to-load)
   lists the settings and their order. Resume is on by default, so independent runs need
   new run identities, and FSDP runs resume only from sharded checkpoints
   ([checkpoint storage](https://github.com/NVlabs/FastGen/blob/main/fastgen/configs/README.md#distributed-checkpoint-storage)).
3. **Bound and observe.** Before launching set `trainer.max_iter`,
   `trainer.save_ckpt_iter`, `trainer.validation_iter`, `trainer.logging_iter` and the
   W&B destination; keep the algorithmic callbacks. For a smoke test log scalars and media
   every iteration through the [W&B callback](https://github.com/NVlabs/FastGen/blob/main/fastgen/callbacks/README.md) and run
   long enough to cover a student update and every alternating optimizer phase.
4. **Launch.** Enable exactly one of `trainer.ddp`/`trainer.fsdp` for multi-GPU runs; a
   launcher alone distributes nothing. Budget memory for teacher, student, auxiliary
   networks, optimizer state and preprocessors
   ([CUDA out of memory](https://github.com/NVlabs/FastGen/blob/main/fern/docs/troubleshooting.mdx#cuda-runs-out-of-memory)),
   then confirm the [runtime startup checks](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-runtime/SKILL.md)
   and, when requested, online tracker uploads.
5. **Monitor.** Check finite losses and gradients, parameter updates and samples with
   [fastgen-validate](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-validate/SKILL.md); compare fixed
   held-out inputs before and after training with
   [fastgen-inference](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-inference/SKILL.md). Stop on non-finite updates,
   broken conditioning or degrading samples before spending the budget.
   Verify the run's saved checkpoint and readable media before reporting success; include
   these artifact checks in a review-only run plan. Scheduler requeue alone saves nothing;
   rely on periodic checkpoints or a `trainer.auto_resume` integration (default: no-op).

Report run URLs, the resolved config, executed iterations and optimizer phases,
checkpoint location, compute used and validation evidence. A submitted job is not a
successful run and a smoke test is not convergence. For stalls distinguish queueing,
downloads, empty loaders and distributed startup; for resume mismatches inspect run
identity and checkpoint contents first.
