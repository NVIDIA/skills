# Flow: End-to-End (e2e)

**OSMO YAML**: `assets/configs/osmo/e2e.yaml`

## Group Sequence

```
setup_group -> augmentation_group -> auto_labeling_group
```

## Stage Details

### setup_group
- **Task**: `setup`
- **Image**: `nvcr.io/nvidia/base/ubuntu:22.04_20240212`
- **Purpose**: Copies worker scripts and cookbook configs to shared output URL;
  writes `.env` with endpoint wait flags.
- **Inputs**: Dataset URL (person crops)
- **Outputs**: Setup artifacts at `<storage_url>/datasets/<dataset>-outputs/<run_id>/setup_b0`

### augmentation_group
- **Task**: `augmentation_worker_0`
- **Image**: `nvcr.io/nvidia/paidf-augmentation:1.0.0`
- **Purpose**: Full image-edit augmentation pipeline:
  1. Preprocess: combine multi-view crops into pane images (`combine_panes.py`)
  2. Config generation: sample attributes from distribution config
  3. Image-edit augmentation with MCQ verification and retry
  4. Post-process: split panes back to per-view crops, build dataset JSON
- **Inputs**: Setup artifacts + dataset
- **Outputs**: Augmented images at `<storage_url>/datasets/<dataset>-outputs/<run_id>/outputs/augmented`
- **Endpoints**: Image Edit, VLM, LLM

### auto_labeling_group
- **Task**: `auto_labeling_worker_0`
- **Image**: `nvcr.io/nvidia/paidf-auto-labeling:1.0.0`
- **Purpose**: Generate person-attribute captions and synonymous search queries
  using the shipped `person_attributes` question bank.
- **Inputs**: Setup artifacts + augmentation output
- **Outputs**: Captions at `<storage_url>/datasets/<dataset>-outputs/<run_id>/outputs/auto_labeled`
- **Endpoints**: VLM, LLM

## Submit

See **Submit (all flows)** in `SKILL.md`, substituting `e2e.yaml` for
`<flow>.yaml`. This flow uses the Image Edit, VLM, and LLM endpoints; endpoints
default to the in-cluster NIMs, so append `image_edit_url=`, `vlm_url=`, and/or
`llm_url=` to the same `--set-string` list only for external endpoints.

## Output Structure

```
<dataset>-outputs/<run_id>/
├── setup_b0/
│   ├── augmentation_worker.sh
│   ├── auto_labeling_worker.sh
│   ├── endpoint_common.sh
│   ├── .env
│   └── configs/
│       ├── distribution_config.yaml
│       ├── augmentation_config.yaml
│       └── README.md
├── outputs/
│   ├── augmented/
│   │   ├── <person_id>/aug_<n>/
│   │   │   ├── output.jpg
│   │   │   ├── output.txt
│   │   │   └── output_metadata.json
│   │   └── dataset/
│   │       ├── augmented_data.json
│   │       └── augmented_imgs/
│   └── auto_labeled/
│       └── caption_<id>/
│           └── task/open_qa.json
```
