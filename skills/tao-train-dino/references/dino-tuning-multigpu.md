# DINO Tuning And Multi-GPU Notes

Full AutoML/HPO guidance and multi-GPU spec consistency rules. Load this file only when the compact `SKILL.md` points here for the current task. If this reference conflicts with `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the compact/current source wins.

## Multi-GPU Spec Consistency

When increasing `train.num_gpus`, also set `train.gpu_ids` to the same visible
device range. For example, an 8-GPU single-node Slurm run must include both
`"train.num_gpus": 8` and `"train.gpu_ids": [0, 1, 2, 3, 4, 5, 6, 7]`.
Leaving the template default `train.gpu_ids: [0]` while requesting multiple
GPUs can make distributed startup inconsistent and can surface as NCCL
collective timeouts instead of an immediate validation error.

## AutoML / HPO Notes

AutoML runs training — all requirements from **Training Requirements** above apply. The agent must read that section first.

For no-input local DINO AutoML smoke runs, use `DINO_AUTOML_PROFILE` from
**Training Requirements**. Do not inspect previous AutoML runs to infer dataset
URIs, `num_classes`, recommendation count, or interval settings.

**Recommended AutoML metric:** for quick operational checks, use explicit
`metric="mAP50"` with `direction="maximize"` and pass a custom
`metric_extractor` that reads `Validation mAP50`. For COCO or paper-style
benchmark comparisons, use `metric="val_mAP"` with `direction="maximize"` so
the reported number matches the standard mAP column rather than AP50. Do not
rely on `metric="kpi"` for generated DINO runners unless you have verified the
local resolver maps it to the intended detection metric; loose fallback parsing
can otherwise optimize `val_loss`.

Use a `metric_extractor` that reads the last `Validation mAP50` value from the
logs, then run AutoML with `automl_settings={"metric": "mAP50",
"direction": "maximize", ...}`.

When a benchmark run remains below target but the per-epoch `val_mAP` curve is
still climbing at the final epoch, extend the best full-budget configuration
before declaring the search plateaued. For dense datasets such as aerial or
driving-scene detection, also preserve high-resolution input overrides and
structural settings (`model.backbone`, `model.num_queries`,
`model.num_select`, class metadata) when evaluating or resuming the checkpoint.

**Recommended hyperparameters:**

Suggested knobs: `train.optim.lr`, `train.optim.weight_decay`,
`model.backbone`, `model.num_queries`, and `model.dropout_ratio`. Constrain
`model.backbone` to supported names such as `resnet_50` and `resnet_34`; the
LLM brain may otherwise propose legacy or invalid DINO backbone names.

`train.optim.weight_decay` is not in the default DINO spec schema — the runner accepts it with a warning. It still works; the DINO training code picks it up from the config.

All model-specific metadata is documented in the Training Requirements table and
`references/skill_info.yaml`. DINO data-source arrays are not auto-resolved from
TAO Core metadata; provide dataset paths explicitly in the spec overrides.
