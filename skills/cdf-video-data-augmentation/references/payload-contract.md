# Video Data Augmentation payload contract

Use `scripts/payload.py` as the installed-skill renderer and preflight validator. The checked-in
Pydantic model at
`airflow/dags/workflows/video_data_augmentation_dag/models/payload.py` remains authoritative when
the repository is available.

This reference describes the VDA trigger payload and DAG integration. For the configuration
consumed inside each container, follow the [component routes](../SKILL.md#component-details):
use `cdf-augmentation` for augmentation YAML and `cdf-auto-labeling` for stage configuration.
Map their options through the workflow's supported fields; container YAML is not a DagRun payload.

## Top-level fields

| Field | Rule |
|---|---|
| `input_path` | One video or a directory prefix; supports `s3://`, HTTP, and HTTPS input |
| `max_videos` | Positive integer for a bounded run; `null`, zero, or a negative integer for all matching videos |
| `output_directory` | Writable `s3://` base; artifacts are written below `/<dag_run_id>/` |
| `external_services` | `false` for managed Kubernetes services; `true` for supplied endpoints |
| `service_lifecycle` | VLM, LLM, and Cosmos Transfer enabled only in internal mode |
| `cosmos` | Video-to-video generation, distribution, endpoints, and evaluator policy |
| `captioning`, `visual_qa`, `reasoning` | Auto-labeling endpoint and stage options |
| `super_resolution`, `detection_and_tracking` | Per-stage processing options |
| `training_export` | Export formats and optional task filters |

The runtime model ignores unknown fields in several sections. Successful parsing therefore does
not prove that a misspelled option took effect.

Validation normalizes `max_videos=0` and negative integers to `null`, the canonical unlimited
value, before directory listing and input slicing.

## Service modes

Internal mode is Kubernetes-only. All lifecycle entries must be enabled. The checked-in manifest
uses one shared Qwen endpoint for both VLM and LLM roles and a two-GPU Cosmos3-Nano Transfer
endpoint.

External mode requires real values for:

- `cosmos.vlm_service_url`
- `cosmos.llm_service_url`
- `cosmos.cosmos_transfer_service_url`
- `captioning.vlm_service_url`
- `captioning.llm_service_url` when `enable_llm_summary` is true
- `visual_qa.vlm_service_url`
- `visual_qa.llm_service_url`
- `reasoning.llm_service_url`

OSMO supports external mode only. It also requires
`cosmos.cosmos_evaluator_service_url` whenever any evaluator check remains enabled.

Nested `output_directory` and `external_services` values may be omitted because the runtime model
copies them from the top level. If supplied, they must match exactly.

## Cosmos augmentation

- `num_augmentation` is at least one and counts variants per input video.
- Include an explicit non-empty `variable_distribution` whenever a `cosmos` object is present.
  Otherwise the shared `CosmosTaskConfig` can fall back to an unrelated clothing distribution.
- The bundled config fixes the transfer model to `nvidia/Cosmos3-Nano`.
  `cosmos.cosmos_transfer_model` does not replace it.
- `base_config_path` may select an MSC-readable custom Cosmos YAML. Omit it for the bundled VDA
  prompt, endpoint, generation, and evaluator contract.

For distribution syntax, sampling rules, and custom Cosmos YAML, read the pinned
`cdf-augmentation` skill's `references/configuration-schema.md` and
`references/config-decision-tree.md` through the component routes above.
The bundled [variable-distribution.json](../assets/variable-distribution.json) is a runnable VDA
example. The standalone renderer supports its direct-weight form; advanced payloads still need
validation against the repository's workflow model.

## Evaluator policy

For checker configuration and internal retry/gating semantics, use the pinned augmentation skill's
`references/evaluator-setup-guide.md`. The rules below are this DAG's evaluator integration.

The bundled config defines these gates:

- `metropolis.hallucination`
- `metropolis.attribute_verification`
- `metropolis.physical_plausibility`

An omitted override leaves a check enabled. `false` removes that check for the run; `true` retains
it. Kubernetes can deploy managed evaluator components when checks remain enabled and no external
URL is supplied. OSMO cannot deploy evaluator components and needs an external Arbitrator
root URL reachable from its tasks, unless all checks are explicitly disabled.

Rejected candidates are removed from the generated-video auto-labeling branch. A run may proceed
with remaining passing candidates but fails if none remain.

## Auto-labeling details

For stage settings, prompts, question banks, and export schemas, use the pinned auto-labeling
skill's stage references and `references/prompt-authoring.md`. The following are VDA-specific
composition rules and defaults, not a replacement for the component contract.

- The DAG runs the same ordered chain on source and generated videos: super resolution,
  detection/tracking, captioning, visual QA, reasoning, and training export.
- The DAG replaces `visual_qa.question_bank_file` with its bundled VQA question bank. A payload
  value does not change VDA questions.
- Default training export format is `tao-vl-reason-v1.0`.
- The two export branches remain separate; the workflow does not combine them into one dataset
  manifest.

## Render and validate

Internal Kubernetes example:

```bash
python3 scripts/payload.py render \
  --platform k8s \
  --service-mode internal \
  --input-path s3://bucket/vda/input/ \
  --output-directory s3://bucket/vda/output \
  --max-videos 1 \
  --num-augmentation 2 \
  --variable-distribution assets/variable-distribution.json \
  --output /tmp/vda-payload.json
```

External example:

```bash
python3 scripts/payload.py render \
  --platform k8s \
  --service-mode external \
  --input-path s3://bucket/vda/input/ \
  --output-directory s3://bucket/vda/output \
  --vlm-url https://vlm.example/v1 \
  --llm-url https://llm.example/v1 \
  --cosmos-transfer-url https://transfer.example/v1 \
  --cosmos-evaluator-url https://evaluator.example \
  --output /tmp/vda-payload.json
```

Validate an existing payload:

```bash
python3 scripts/payload.py validate \
  --platform k8s \
  --payload /tmp/vda-payload.json \
  --output /tmp/vda-normalized.json
```
