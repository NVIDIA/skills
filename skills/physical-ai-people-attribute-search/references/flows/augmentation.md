# Flow: Augmentation Only

**OSMO YAML**: `assets/configs/osmo/augmentation.yaml`

## Group Sequence

```
setup_group -> augmentation_group
```

## Purpose

Perform image-edit augmentation on person crops without the auto-labeling
captioning stage. Use when you only need augmented images and verification
metadata but not text-domain captions.

## Stage Details

### setup_group
Same as the e2e flow setup — copies worker scripts and cookbook configs.

### augmentation_group
- **Task**: `augmentation_worker_0`
- **Image**: `nvcr.io/nvidia/paidf-augmentation:1.0.0`
- **Purpose**: Full image-edit augmentation pipeline (preprocess, config gen,
  augment with verification, post-process).
- **Endpoints**: Image Edit, VLM, LLM

## Submit

See **Submit (all flows)** in `SKILL.md`, substituting `augmentation.yaml` for
`<flow>.yaml`. This flow uses the Image Edit, VLM, and LLM endpoints; endpoints
default to the in-cluster NIMs, so append `image_edit_url=`, `vlm_url=`, and/or
`llm_url=` to the same `--set-string` list only for external endpoints.

## Output Structure

```
<dataset>-outputs/<run_id>/
├── setup_b0/
└── outputs/augmented/
    ├── <person_id>/aug_<n>/
    │   ├── output.jpg
    │   ├── output.txt
    │   └── output_metadata.json
    └── dataset/
        ├── augmented_data.json
        └── augmented_imgs/
```
