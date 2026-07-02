# Flow: Auto-Labeling Only

**OSMO YAML**: `assets/configs/osmo/auto_labeling.yaml`

## Group Sequence

```
setup_group -> auto_labeling_group
```

## Purpose

Generate person-attribute captions and synonymous search queries on
pre-augmented person crop images. Use when augmented images are already
available and you only need the text-domain labels.

## Stage Details

### setup_group
Copies worker scripts and writes `.env` with endpoint wait flags.
Does not copy augmentation configs since augmentation is not run.

### auto_labeling_group
- **Task**: `auto_labeling_worker_0`
- **Image**: `nvcr.io/nvidia/paidf-auto-labeling:1.0.0`
- **Purpose**: Loop over input images and generate captions using the shipped
  `person_attributes` question bank with question-driven VLM-LLM mode.
- **Endpoints**: VLM, LLM (no Image Edit needed)

## Submit

See **Submit (all flows)** in `SKILL.md`, substituting `auto_labeling.yaml` for
`<flow>.yaml`. This flow uses the VLM and LLM endpoints (no Image Edit); endpoints
default to the in-cluster NIMs, so append `vlm_url=` and/or `llm_url=` to the same
`--set-string` list only for external endpoints.

## Output Structure

```
<dataset>-outputs/<run_id>/
├── setup_b0/
└── outputs/auto_labeled/
    └── caption_<id>/
        ├── config.yaml
        ├── logs/
        ├── prompts/
        ├── sidecars/
        └── task/open_qa.json
```

## Output Format

`task/open_qa.json` contains captions grouped by the shipped person-attribute
question bank entries. Example:

```json
{
  "id": "person_clothing_full_description",
  "question": "Describe the person's visible clothing, colors, and footwear in one concise sentence.",
  "answer": "A person wearing a red hoodie, black leggings, and white sneakers."
}
```
