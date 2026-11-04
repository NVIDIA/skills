# Fine-tuning input contract

Read this reference when validating a user dataset or adapting an existing
AnomalyGenNext recipe.

## Dataset layout

```text
DATASET_ROOT/
  defect_spec.jsonl
  TEXTURE/
    clean_image/*
    anomaly_image/DEFECT/*
    mask/DEFECT/<anomaly-stem>_mask.png
    cad_mask/DEFECT/*              # only for spatial_dependency=cad
VALIDATION_ROOT/testcase.jsonl
```

The defect specification may be supplied separately with `--defect-spec`.
Each row identifies `defect_type` as `TEXTURE+DEFECT`, selects
`spatial_dependency` (`free`, `text`, or `cad`), and provides the
placement fields required by that mode. Text-routed defects require a nonempty
`roi_prompt_defect_location`.

Each validation row must contain `image_filename`, `mask_filename`, and
`anomaly_type`. Freeze at least three rows per trained type. The builder
resolves supported relative paths, verifies every file, and emits a normalized
validation JSONL with absolute paths.

All dataset and validation images and masks must use `.jpg`, `.jpeg`, or
`.png`, matching the AnomalyGenNext 1.1 runtime loader. The builder rejects
`.bmp`, `.tif`, `.tiff`, and `.webp` instead of allowing preparation to pass
and failing later in `ValidationKPI`.

Masks must contain exactly the binary pixel values `0` and `255`.
Empty/all-zero masks, full/all-255 masks, values such as `1`, and soft
grayscale edges are invalid.

## Type and recipe identity

The recipe's `anomaly_types` order defines model class IDs. Resolve the order
from an explicit list first, then the user template, otherwise lexical dataset
discovery. Every type must exist in the dataset and defect specification.

A user recipe is a template, not an unchecked pass-through. Preserve supported
training knobs, but replace dataset identity, paths, validation testcase,
base/VAE checkpoints, anomaly-type order, and iteration-zero validation with
the verified values. Preserve the emitted canonical recipe beside the selected
adapter; generation must use that pair unchanged.

The dataset root, defect specification, validation testcase, optional template,
Cosmos3-Nano checkpoint, VAE, and checkpoint root may live at separate absolute
paths. The Cosmos3-Nano checkpoint must be a DCP directory with
`checkpoint.json`, `model/.metadata`, and one or more `model/*.distcp` shards.
If the official checkpoint is staged in Hugging Face format, use the pinned
image's `cosmos_framework.scripts.convert_model_to_dcp` command documented in
`container-runtime.md`, then pass its persistent output directory as
`--base-checkpoint`.
The selected platform must expose all of them to the container. The
checkpoint root must be mounted over the image's complete `checkpoints/` tree
and contain the required Qwen tokenizer model assets under `hf/` plus a
Transformers-compatible
`facebook/dinov2-large/config.json` and model weights.
