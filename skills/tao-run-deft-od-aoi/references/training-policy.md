# RT-DETR training and selection policy

Read this before preparing, submitting, resuming, or selecting training.

## Frozen model contract

- RT-DETR with ResNet-50.
- Binary COCO category id 1 named `defect`.
- `dataset.num_classes: 2`, `eval_class_ids: [1]`, and
  `remap_mscoco_category: false`.
- Four GPUs, batch size eight, base LR `1e-4`, backbone LR `1e-5`, and
  validation/checkpointing every epoch by default.
- Every iteration starts from the same frozen base checkpoint.
- KPI `val_mAP50` selects configurations and checkpoints; test never selects.
- `automl_policy: off`; this application owns its probe policy.

## Epoch policy

Iterations 1–2 train for 36 epochs. They skip probes under the packaged
`probes_start_iteration: 3` default; freeze a lower positive start iteration
when an approved workflow needs the tuning path earlier.

When probes are enabled and the current iteration has reached
`probes_start_iteration`, run three independent ten-epoch probes from the
frozen base:

1. incumbent learning rates;
2. a data-growth-scaled candidate;
3. deterministic jitter using seed `4000 + iteration`.

All three must complete before selection. The main budget is:

```text
round(36 * sqrt(10000 / training_images))
```

clipped to 24–48 epochs. When probes are disabled, skip the bake-off but retain
the size-based main budget.

`prepare_deft_od_aoi_training.py` emits the immutable specs and manifest.
`select_deft_od_aoi_training.py probes` accepts exactly three structured
status files and writes `train.yaml`, the winner, history, and incumbent
optimizer settings.

## Checkpoint selection and extension

Select the maximum finite KPI `val_mAP50`; ties favor the earlier epoch. The
matching `model_epoch_NNN.pth` must exist.

If the best epoch is within the final three epochs, the first selection emits a
single 12-epoch `extension.yaml` that resumes from the terminal checkpoint,
not necessarily the KPI-best checkpoint. Submit it once, then reselect with
`--extension-applied`. Pass every status phase so selection covers the full
history. Selection counts each epoch once, keeping the latest row, and records
`epochs_covered` and `duplicate_epochs` in `checkpoint_selection.json`.

An operational resume preserves data, optimizer settings, target epoch, and
output identity. It is not the policy extension and never initializes the next
iteration. Platform-specific recovery belongs to the selected platform skill.

With the packaged `dataset.workers: 4`, an undersized container `/dev/shm` can
stall training instead of failing it: the log shows `unable to allocate shared
memory(shm)`, `status.json` stays `RUNNING`, and no new epoch or checkpoint
appears. Treat that as a failed run: stop it, relaunch with the platform
skill's shared-memory sizing, and resume from the last checkpoint.
