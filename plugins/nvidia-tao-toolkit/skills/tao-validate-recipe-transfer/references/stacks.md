# Stack-specific notes

Read only the section for the stack in play. Porting effort differs by roughly an order of
magnitude across these, and the "raw research repo" case is by far the most work.

## Raw research repo (the common case for paper transfer)

Expect: a fork of an older framework, pinned to torch/CUDA from publication time, with the
dataset path structure hardcoded.

- The config is often Python, not YAML, with values overridden in several places. Find the
  *effective* config by printing it at the start of training rather than reading files.
- Look for a `tools/train.py` + `configs/` layout (mm-family lineage) or a single
  `main.py` with argparse (DETR lineage). Each has known idioms.
- Dataset registration is usually one class plus one registry entry plus a CLASSES constant
  in a third file. Missing the third is the classic S2 bug.
- Multi-GPU launch is often `torch.distributed.launch` (deprecated) — the port to `torchrun`
  changes how args are parsed and how local rank is read.
- Budget real time for custom op compilation. Containerize.

## MMDetection / MMSegmentation / Detectron2

- Config inheritance (`_base_`) means the effective config is assembled from several files.
  Always dump the merged config and read that: `cfg.dump()` / `print(cfg.pretty_text)`.
- MM-family `num_classes` excludes background in recent versions and included it in older
  ones. Verify against the version in use, do not assume.
- Custom datasets need `CLASSES` on the dataset class *and* matching `metainfo` in the config
  in newer versions. Mismatch produces silently wrong evaluation.
- `mmcv-full` version compatibility is a frequent build blocker; the version matrix is strict.
- Detectron2's `MetadataCatalog` needs `thing_classes` registered before the config is built.

## Ultralytics (YOLO family)

- Very fast to get running, which makes it tempting to skip Phase A. Still run the checkpoint
  gate against a COCO subset.
- **License**: AGPL-3.0. This is a hard gate for customer deployments — resolve before use.
- Hyperparameters live in a single YAML plus a large set of defaults; dump the resolved args
  from the run directory rather than reading the YAML.
- Mosaic is on by default and is usually disabled for the final epochs (`close_mosaic`).
  For small objects or small datasets, consider disabling entirely.
- Auto-anchor recalculation happens by default and is genuinely useful for transfer — but
  confirm it ran and log the resulting anchors, since it silently changes the recipe.
- The letterbox implementation is specific; the export path must match it exactly.

## timm / torchvision (classification)

- Cleanest transfer case. `timm` train script arguments map closely to published recipes.
- Match the pretrained model's expected preprocessing exactly — `timm.data.resolve_data_config`
  gives the correct mean/std/crop-pct for each checkpoint. Using generic ImageNet constants
  with a model trained differently costs real accuracy.
- Head replacement: `model.reset_classifier(num_classes)`, and consider a higher LR multiplier
  on the fresh head.

## NVIDIA TAO

- Recipe fields are exposed through spec files rather than code, so field classification maps
  cleanly onto the spec.
- The tradeoff is reduced visibility: fewer places to insert the verification ladder. Compensate
  by leaning harder on the data-side checks (leakage, gap report, annotation QC) and on
  evaluating exported engines directly.
- Export and INT8 calibration are first-class here — build the calibration set from real
  production images, not from training data, and verify with `preprocess_parity.py`.
- Pretrained models from NGC carry their own license terms; check per-model.

## Export and serving (TensorRT / ONNX / Triton / DeepStream)

Regardless of training stack:

- Verify numerical parity between the PyTorch model and the exported engine on the same
  images before trusting any deployment number.
- Preprocessing must be a single shared artifact across train, eval, and export. Most
  production failures live here.
- Test across the real range of input aspect ratios, not one sample image.
- INT8 calibration data must be representative of production; an unrepresentative calibration
  set is a common cause of "fine in FP16, bad in INT8".
