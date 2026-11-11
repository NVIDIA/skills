# Inference weight and sampling identity

Read when comparing teachers and students, evaluating checkpoints or diagnosing
unexpected samples. The [inference README](https://github.com/NVlabs/FastGen/blob/main/scripts/README.md#checkpoint-selection)
owns the student and teacher roles, the EMA choice and the behavior without a checkpoint;
trace `load_checkpoint`, `setup_inference_modules` and `default_inference_output_dir` in
[inference_utils.py](https://github.com/NVlabs/FastGen/blob/main/scripts/inference/inference_utils.py) for the details.

- `--ckpt_path` loads `net` and the configured EMA networks only, non-strictly: one
  matching parameter suffices. Read the reported missing and unexpected keys instead of
  trusting the success message. Check the teacher's weight provenance under
  [checkpoint selection](https://github.com/NVlabs/FastGen/blob/main/scripts/README.md#checkpoint-selection).
- Evaluation (`load_config(..., evaluation=True)`) sets `model.inference`, skips
  training-only modules at construction, enables the preprocessors, prepares only the
  selected sampling roots after loading, builds a teacher on demand for methods trained
  without one and discards the other roots. The standard scripts do not shard models
  whatever the launcher size.
- Teacher steps (`--sample_steps`, the network's own `sample()`) and student steps
  (`model.student_grid_cfg.steps`, the method's `generator_fn`) are different paths. The
  base `generator_fn` reads the grid from the **passed network's** scheduler, so teacher
  roots can carry different defaults, and distilled methods may own other grids or
  rollout rules. Count the effective network evaluations when comparing NFE across
  methods, and dispatch custom entry points to the method's generator rather than to
  `network.sample`.
- Seeding is rank-dependent, so the same seed differs across launch layouts, and prompt
  expansion changes the inputs. Persist the actual prompts and initial noise for parity.
- Default output directories encode the checkpoint identity; an explicit destination is
  caller-owned, and reusing it with the same prompt and seed filenames overwrites
  earlier comparisons. When calling [save_media](https://github.com/NVlabs/FastGen/blob/main/fastgen/utils/media_utils.py)
  directly, select the writer's format explicitly and verify the written file.
