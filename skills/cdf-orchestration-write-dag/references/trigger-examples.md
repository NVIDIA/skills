# Trigger Examples

Sample prompts that should trigger this skill, mapped to modes documented in CDF's
`cdf-image-attribute-augmentation/launchable.md`. The same shape of request applies to
other pipelines (e.g. an EVG-style detection/captioning/visual-QA subset) — these three are the
IAA-scoped instances that exist in this repo today.

- **`augmentation` mode** — "Create a new K8s DAG that runs only the IAA Cosmos image-edit stage,
  no auto-labeling/person-attribute-search afterward." → `CosmosTaskGroup` (+ conditional
  `CosmosPostProcessingTaskGroup`), no `ImageAttributeAugmentationTaskGroup`, no final dataset-merge
  step.
- **`auto_labeling` mode** — "Create a new K8s DAG that runs person-attribute captioning/query
  generation over already-augmented crops, with no image-edit generation step." →
  `ImageAttributeAugmentationTaskGroup` only, no `CosmosTaskGroup`; input preparation lists
  pre-augmented crops directly instead of combining raw person-ID panes.
- **Biased/custom generation** — "Generate a payload for the augmentation-only IAA DAG biased
  toward red tops and sneakers, 5 variants per person." → not a new DAG; a payload built against an
  already-generated DAG's `cosmos.variable_distribution`, sourcing valid attribute values from the
  `augmentation` component skill's `cosmos_config.yaml` `verification_options`, not invented.

`e2e` mode is not a `cdf-orchestration-write-dag` case — `image_attribute_augmentation_dag_k8s` already covers it.
