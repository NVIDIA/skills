# Training, logging and resume contracts

Read when bounding a run, interpreting its logs or continuing a checkpoint. The
[configuration README](https://github.com/NVlabs/FastGen/blob/main/fastgen/configs/README.md) owns launch syntax, run directories
and checkpoint storage; the [troubleshooting guide](https://github.com/NVlabs/FastGen/blob/main/fern/docs/troubleshooting.mdx)
owns memory, NaN, resume-collision and checkpoint-loading problems; the module docstring
of [checkpointer.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/utils/checkpointer.py) owns the checkpoint formats.

## What an iteration does

Trace `setup` in [scripts.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/utils/scripts.py) and `Trainer.run`,
`train_step` and `validate` in [trainer.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/trainer.py).

- Sample accounting uses the data-parallel world size and integer accumulation rounds;
  `trainer.batch_size_global` replaces the requested accumulation and can change the
  effective global batch. Use startup's logged values, not the request or the dry run.
- An iteration is one optimizer phase after its accumulation microbatches. Alternating
  methods update only auxiliary networks on most iterations; check the method's phase
  schedule (`get_optimizers`) and actual parameter changes before claiming the student
  learned.
- With a validation loader, validation runs initially only when `iter_start == 0`, then
  every `trainer.validation_iter` and after the last step, once per `trainer.global_vars_val`
  profile. It calls the method's `single_train_step` under no-grad with the iteration's
  loss phase, so the batch contract applies. Bound finite or repeating loaders with the
  `MAX_VAL_STEPS` profile key and change the student NFE with `STUDENT_STEPS`
  ([global_vars.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/utils/global_vars.py)).
- Memory peaks after the first updates: optimizer state allocates lazily and FSDP can
  hold unsharded accumulated gradients
  ([CUDA out of memory](https://github.com/NVlabs/FastGen/blob/main/fern/docs/troubleshooting.mdx#cuda-runs-out-of-memory)).
  Measure every optimizer phase and the media generation. Inference can leave lazy state
  behind, so exercise generation followed by backward under the real wrapper.
- Do not drop algorithmic callbacks (EMA, consistency schedules, carry) to make a smoke
  test cheaper; the [callbacks README](https://github.com/NVlabs/FastGen/blob/main/fastgen/callbacks/README.md) explains their
  ordering.

## Interpret media and scalars correctly

Read `WandbCallback` and `_LossDictRecord` in [wandb.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/callbacks/wandb.py)
and `FastGenModel._get_outputs` in [model.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/methods/model.py).

- Scalar (`logging_iter`) and media (`sample_logging_iter`) cadences are independent, and
  explicit callback arguments override the trainer's interval; set both to one for a
  per-iteration integration test. Losses accumulate on every iteration and each key is
  averaged over the calls and ranks that reported it, so an alternating method's
  auxiliary losses appear at any cadence; media capture only the iteration they fire on.
- Step-end callbacks receive the **last** accumulation microbatch. A smooth chart is not
  proof of finite losses or gradients on every microbatch and rank; instrument the hook
  (before gradient clipping sanitises NaNs) when that claim matters.
- The student panel shows a fresh deferred generation, normally from the first EMA; the
  detached training estimate is a separate debug-level panel. Callback prompts and noise
  change over training, so use a fixed held-out inference set for before/after judgments.
- Media are generated for the whole local batch before `log_batchsize` and `max_ranks`
  select what is written, so neither reduces inference memory. Teacher panels
  (`trainer.visualize_teacher`) need a teacher on the method.
- Disabling validation media keeps validation losses; local media files do not prove
  tracker uploads. Decoding runs on every rank, so rank-zero-only custom sampling hangs
  sharded preprocessors.

## Resume restores state, not the exact trajectory

Trace `Trainer.load_pretrained_ckpt` and `resolve_sampler_start_idx` in `trainer.py`
together with [checkpointer.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/utils/checkpointer.py); the loading mechanisms
and their order are tabulated under
[a checkpoint fails to load](https://github.com/NVlabs/FastGen/blob/main/fern/docs/troubleshooting.mdx#a-checkpoint-fails-to-load).

- `Checkpointer.save`/`load` take an `nn.ModuleDict` of named roots (`model.model_dict`,
  or `nn.ModuleDict({"net": net})` for a weights-only check) with the optimizer and
  scheduler dicts, the grad scaler and the callbacks; validation walks module
  registrations, so a plain dict is not equivalent. A weights-only reload proves nothing
  about optimizer, scheduler or callback state.
- Warm starts through `trainer.checkpointer.pretrained_ckpt_path` map destination roots
  to checkpoint keys with `pretrained_ckpt_key_map` (default: `net` only). They do not
  advance the start iteration, but the saved iteration still offsets EMA and consistency
  schedules through `model.resume_iter`, and a later resume overwrites them.
- DCP optimizer loading may initialise state for parameters the checkpoint never trained;
  the coverage log names missing and extra parameters. Read it before claiming every
  optimizer entry came from the checkpoint.
- `latest_checkpoint` in [io_utils.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/utils/io_utils.py) ranks recognised
  names by iteration, with modification time only breaking ties; read the restored
  iteration in the startup log. `trainer.max_iter` includes restored iterations.
- With `sampler_start_idx=None` the trainer derives the data offset from the restored
  iteration and the **current** effective global batch; changing batch size, world size
  or the data stream changes its meaning, and shuffled loaders ignore it.
- RNG is reseeded from seed, iteration and data rank; it does not restore every
  interrupted generator or worker state. Carry slots are not checkpointed, so carried
  rollouts rebuild their phase after resume. Verify restored optimizer, scheduler,
  enabled gradient scaler and callback state and subsequent progress; do not promise
  bitwise replay.
- W&B reuses `wandb_id.txt` independently of checkpoint loading
  ([a new run resumes an earlier experiment](https://github.com/NVlabs/FastGen/blob/main/fern/docs/troubleshooting.mdx#a-new-run-resumes-an-earlier-experiment)).
