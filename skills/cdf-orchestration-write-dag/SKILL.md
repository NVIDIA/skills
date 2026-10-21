---
name: cdf-orchestration-write-dag
description: Use when a user describes a custom CDF Orchestration pipeline — a specific ordered combination of stages such as augmentation only, auto-labeling only, detection+captioning only, or image attribute augmentation without full auto-labeling — that no existing DAG in airflow/dags/workflows/ covers, and asks for a new Kubernetes DAG. Also use to check that a generated or existing DAG's model/container/prompt choices match an external spec document (e.g. a CDF `launchable.md`). Not for requests that an existing DAG already covers, including payload-only changes such as different inputs, endpoints, models, or variant counts — route those to the matching workflow skill, unless the user still wants a separate DAG after being pointed to it.
version: "1.0.1"
license: CC-BY-4.0 AND Apache-2.0
metadata:
  owner: NVIDIA
  service: cosmos-data-factory
  version: 1.0.1
  reviewed: '2026-10-08'
  author: NVIDIA
  tags:
    - physical-ai
    - cdf-orchestration
    - airflow
    - dag
---

# Write DAG

## What This Skill Does

Composes a new Kubernetes-only Airflow DAG from the **existing task groups and stages** that the
shipped DAGs are built from: the shared task groups in `airflow/dags/shared/task_groups/`, plus
the workflow-local stages the shipped DAGs define for themselves (for example DIG's simulation and
AnomalyGen tasks, AV's curation and retrieval tasks). It does not invent new task-group *logic*.
Output is:

1. `airflow/dags/workflows/<dag_name>_dag/configs/<dag_name>_k8s_manifest.yaml` — compute + component manifest
2. `airflow/dags/workflows/<dag_name>_dag/<dag_name>_dag.py` — the DAG builder
3. `airflow/dags/workflows/<dag_name>_dag/models/payload.py` — the payload pydantic model
4. `airflow/dags/workflows/<dag_name>_dag/callables/` — **required workflow-local glue**, not
   optional. Every `*TaskGroup` used here takes one or more `Callable` constructor args with **no
   default** (e.g. `InputPreparationTaskGroup(prepare_input_callable=...)`,
   `CosmosTaskGroup(config_generation_callable=..., output_validation_callable=...)`,
   `ValidatedOutputTaskGroup(validation_callable=...)`). These callables are genuinely new,
   per-DAG code — they are what turns generic container-arg building into *this* pipeline's actual
   input listing / config generation / output validation. Do not skip writing them, and do not
   treat "don't invent task logic" as license to invent them freely either: for each task group
   you use, find the closest existing DAG that uses the *same* task group, read its equivalent
   callable, and adapt it — trim the parts that don't apply (e.g. dropping an
   `output_group_id`/downstream-XCom concern when there is no next stage) rather than writing from
   a blank page.

**Kubernetes, and OSMO for workflows that support it.** This repository checks in K8s manifests for
every workflow, and additionally checks in OSMO manifests for Event Video Generation and Video
Data Augmentation (`event_video_generation_dag_osmo`, `video_data_augmentation_dag_osmo`) — see the
`cdf-event-video-generation` skill for the pattern. OSMO DAGs are **external-services-only**:
their manifests define OSMO batch tasks but no in-cluster VLM/LLM/image-edit/evaluator service
components, so an OSMO variant must reject `external_services: false` payloads and cannot use
`service_lifecycle` to deploy anything. A user asking for an OSMO variant of a new custom DAG is
in scope — replicate the external-services-only contract from an existing OSMO manifest/payload
rather than refusing. NVCF is still unsupported — do not offer it unless the user explicitly says
their environment supports it, and if so, treat it as new scope requiring its own investigation
rather than a flag on this skill.

**Not scoped to one workflow.** Task groups are shared across the shipped DAGs; a custom DAG can
mix stages from any of them (e.g. `CosmosTaskGroup` + `CosmosEvaluatorTaskGroup` with no
auto-labeling stages, or `ImageAttributeAugmentationTaskGroup` with no `CosmosTaskGroup` for a
labeling-only pass over already-augmented crops).

**These DAGs are not checked in as permanent artifacts.** Unlike the five workflow DAGs that ship
with this repository (listed in [Step 0](#step-0-confirm-no-existing-dag-already-covers-this)), a
DAG this skill generates is a one-off, user-requested pipeline — it is not added to any workflow
skill's DAG-ID table. Because nothing else will ever guide the user through deploying
or running it, **this skill owns the full lifecycle, not just authoring** — Step 8 below is not
optional follow-through, it's the only place readiness-check/deploy/trigger/monitor for this DAG
will ever happen.

---

## Making Use of Component Skills

The shared task groups are thin Airflow wrappers around CDF components. Each component owns its
config schema, model defaults, prompts, and errors in its own skill; the DAG's `container_args`
and generated config files are the contract that skill documents. This skill does not reimplement
that contract.

**Open the matching component skill before writing manifest task entries, config templates,
prompts, or question banks** — never invent config schema, model defaults, or prompt wording from
memory. [`references/component-skills-excerpt.md`](references/component-skills-excerpt.md) links
every component skill and keeps an offline summary of each. If the component repo is unreachable,
use that summary; if what you need isn't in it, say so and ask rather than extrapolating.

Use this table to pick the component and to see which shipped DAG already runs that stage — the
closest shipped DAG is also the one to copy callables and manifest entries from:

| Task group(s) or stage | Component | Component skill | Shipped DAGs that already use it |
|---|---|---|---|
| `CosmosTaskGroup` | CDF Augmentation | `cdf-augmentation` | IAA (image edit), DIG (image edit), EVG (image-to-video), VDA (Cosmos Transfer); AV subclasses it as workflow-local `AVCosmosTaskGroup` (Cosmos3-Nano with RGB or WSM controls) |
| `CosmosEvaluatorTaskGroup` | CDF Augmentation (Cosmos Evaluator) | `cdf-augmentation` | IAA, EVG, VDA; AV subclasses it as workflow-local `AVEvaluatorTaskGroup` (physical plausibility) |
| `DetectionAndTrackingTaskGroup`, `CaptioningTaskGroup`, `VisualQATaskGroup` | CDF Auto-Labeling | `cdf-auto-labeling` | EVG, VDA |
| `ImageAttributeAugmentationTaskGroup`, `EventAndPersonAttributeSearchTaskGroup` | CDF Auto-Labeling | `cdf-auto-labeling` | IAA, EVG |
| `SuperResolutionTaskGroup`, `ReasoningTaskGroup`, `TrainingExportTaskGroup` | CDF Auto-Labeling | `cdf-auto-labeling` | VDA |
| Simulation render (manifest task `image_generation`) | CDF Simulation | `cdf-simulation` | DIG |
| AnomalyGen defect generation (manifest task `anomaly_generation`) | CDF AnomalyGen | `cdf-anomalygen` | DIG |
| Curation and retrieval (manifest tasks `curator_model_download`, `curator`, `media_probe`, `cds_ingest`; endpoints `milvus_service`, `cds_service`) | CDF Curation and Retrieval | `cdf-curation-and-retrieval` | AV |

For `CosmosTaskGroup`, the input → output shape decides the model and adapter, not the task-group
name: Cosmos Transfer for video scene changes, Cosmos Predict for new or continued video, image
edit for still images, and image-to-video to animate a first frame.

Simulation, AnomalyGen, and curation and retrieval have no class in `shared/task_groups/`. Each is
a manifest task (or a few) wired directly in the DIG or AV DAG with `self.builder.build_task` /
`partial_task`, plus workflow-local callables that prepare its inputs and validate its outputs.
Reuse them the same way as any workflow-local stage (see **Workflow-local task groups** below):
copy the manifest task entries, the wiring, and the callables from that DAG, and adapt the
callables to your payload model.

- **Simulation and AnomalyGen** — `defect_image_generation_smt_dag.py` with its manifest
  `configs/smt/defect_image_generation_k8s_manifest.yaml`. The `generate_configs` task
  (`generate_smt_configs`) builds one simulation config per `golden_generation` variant from
  `configs/smt/image_generation_config.yaml`, and one AnomalyGen config per `defect_generation`
  variant from `configs/smt/anomaly_gen_config.yaml`. Simulation (`image_generation`) is mapped over the
  `image_generation_environments` list, selected by `select_generated_config_list`. AnomalyGen
  (`anomaly_generation`) is mapped over the `anomaly_package` tasks (`package_anomalygen_inputs`),
  and the `validate_anomalygen_outputs` task checks its results. Callables live in
  `callables/smt/`, and the payload model is `SmtDefectImageGenerationPayload` in
  `callables/smt/models.py`. DIG's image-edit stage between them is a `CosmosTaskGroup` that, in
  internal mode, also needs the `image_edit_checkpoint_download` task.
- **Curation and retrieval** — `autonomous_vehicle_dataset_generation_dag.py` with
  `configs/autonomous_vehicle_dataset_generation_k8s_manifest.yaml` and
  `configs/cosmos_curator_config.yaml`. The chain is `generate_configs` (`create_avdg_configs`) and
  `input_preparation` (`prepare_avdg_input`) → `build_curator_args` → `curator_model_download`
  (task `download_curator_model`) → `curator` (task `run_curator`) → `prepare_curator_probe` →
  `media_probe` (task `probe_curator_outputs`) → `reconcile_curator_outputs` → `cds_ingest` (task
  `ingest_cds`, after the `milvus_service` and `cds_service` endpoints are ready) →
  `collect_results` → `validate_retrieval`. AV reuses the same `media_probe` task to check WSM
  controls and generated videos. The pods run scripts from the DAG's `callables/` that
  `config_generation.py` base64-encodes into the generated configs (`cds_ingest.py`,
  `msc_credentials.py`, `media_probe.py`, `control_video_probe.py`), and the DAG file injects
  `cds_vector_pipeline.py` and `cds_vector_components.py` into the `cds_service` deployment. Copy
  all of them. These scripts use boto3, so inputs and outputs must be on S3 or an S3-compatible
  store.

DIG and AV also run subsets of their chains through payload switches, and those subsets count as
payload differences in Step 0:

- DIG `pipeline_mode`: `simulation` (simulation only), `sim2real` (simulation, then image edit),
  `full` (adds AnomalyGen, the default), `defect` (AnomalyGen only).
- AV `cosmos.enabled`: `false` runs curation and retrieval only; `true` adds Cosmos3-Nano
  generation and physical-plausibility evaluation.

If the user's request doesn't map onto a documented config/prompt pattern in the component skill,
say so and ask rather than inventing prompt text or config keys.

---

## Available Task Groups

Import each directly from its submodule (matches every existing DAG file — do not import from the
`task_groups` package `__init__`, which is an incomplete re-export and omits
`EventAndPersonAttributeSearchTaskGroup`).

| Task Group | Module | What It Does | Manifest component(s) needed |
|---|---|---|---|
| `ValidatePayloadTaskGroup` | `validate_payload` | Validates the trigger payload against the DAG's pydantic model. Always first. | none |
| `InputPreparationTaskGroup` | `input_preparation` | Lists/normalizes input media from the payload's `input_path`. Required whenever the pipeline reads input media; every shipped DAG uses it except DIG, which generates its inputs by simulation. | none |
| `ServiceLifecycleTaskGroup` | `service_lifecycle` | Deploys/tears down internal VLM/LLM/image-edit/image2video pods and waits for readiness; skipped per-service when `external_services: true`. | `components.endpoints` entries matching the `ServiceLifecycleSpec`s passed in |
| `CosmosTaskGroup` | `cosmos` | Runs the `augmentation` component (image-edit or Cosmos Transfer/Predict/image2video) via a manifest task, dynamically mapped over inputs. | `components.tasks.augmentation` (+ VLM/LLM/image-edit endpoints) |
| `ImageAttributeAugmentationTaskGroup` | `image_attribute_augmentation` | Runs person-attribute search/captioning over cosmos-augmented crops (IAA's auto-labeling stage). Reads a cosmos-output XCom. | `components.tasks.event_and_person_attribute_search` |
| `EventAndPersonAttributeSearchTaskGroup` | `event_and_person_attribute_search` | EVG's equivalent — person/event attribute search over auto-labeled video output. Reads an auto-labeling XCom. | `components.tasks.event_and_person_attribute_search` |
| `DetectionAndTrackingTaskGroup` | `detection_and_tracking` | RFDETR detection + BoxMOT tracking on video/cosmos output. | `components.tasks.detection_and_tracking` |
| `CaptioningTaskGroup` | `captioning` | VLM captioning on detection/tracking output. | `components.tasks.captioning` |
| `VisualQATaskGroup` | `visual_qa` | VLM visual QA; instantiate once per QA purpose with a distinct `group_id`/`component_name` (EVG uses it twice: anomaly QA and person-attribute QA). | `components.tasks.visual_qa` |
| `CosmosEvaluatorTaskGroup` | `cosmos_evaluator` | Deploys and tears down the Cosmos Evaluator (Postgres, database, arbitrator, and the configured checkers: hallucination, attribute verification, physical plausibility) around Cosmos generation. | `components.endpoints.cosmos_evaluator_*` entries for the core services and each selected checker |
| `SuperResolutionTaskGroup` | `super_resolution` | Super-resolves videos with a manifest-defined container task (VDA). | `components.tasks.super_resolution` |
| `ReasoningTaskGroup` | `reasoning` | Produces reasoning labels with a manifest-defined container task (VDA). | `components.tasks.reasoning` |
| `TrainingExportTaskGroup` | `training_export` | Aggregates completed scenes into one training-export container run (VDA). | `components.tasks.training_export` |
| `ValidatedOutputTaskGroup` | `validated_output` | Runs a workflow-specific output validation callable. Last processing step before reporting in every shipped DAG except DIG, which validates in its own `validate_anomalygen_outputs` task. | none |
| `PerformanceReportingTaskGroup` | `reporting` | Optional YAML+HTML performance report, gated by `enable_performance_reporting` in the payload. Hangs off `validated_output` as a terminal branch — never upstream of `fail_pipeline`/`pipeline_success`. | none |

**Before using any task group**, read its `__init__` in the source file — constructor kwargs
(`input_xcom_task_id`, `group_id`, `component_name`, `prepare_args_callable`,
`prepare_args_op_kwargs`) vary per group and determine both manifest wiring and XCom data flow.
Copy the calling pattern from a shipped DAG that already uses that group (see the "Shipped DAGs
that already use it" column above) — do not guess kwargs, and do not assume
every group exposes an `input_xcom_task_id` param: `ValidatedOutputTaskGroup`, for one, doesn't —
it takes a plain `op_kwargs` dict (default `{"payload": ..., "run_id": ...}`), so pulling an
upstream XCom into it means passing a custom `op_kwargs`/callable, not a task-id string.

**Workflow-local task groups exist too, outside `shared/task_groups/`.** IAA's
`CosmosPostProcessingTaskGroup` lives under
`airflow/dags/workflows/image_attribute_augmentation_dag/tasks/` and is wired in *conditionally* —
`image_attribute_augmentation_dag.py` only includes it when
`"cosmos_post_processing" in self.builder.manifest.deployment.components.tasks`. Check each
reference DAG's own `tasks/` and `callables/` directories, not just `shared/task_groups/`, for
stage-specific groups like this — AV's `AVCosmosTaskGroup` (`tasks/cosmos.py`) and
`AVEvaluatorTaskGroup` (`callables/evaluator_lifecycle.py`), and the DIG and AV stages that are
plain manifest tasks wired in the DAG file (see the component table above). Where a stage is
optional, decide inclusion the same way: gate on manifest task presence, don't hardcode it in.

**Constructor kwarg defaults are not guaranteed to match your pipeline.** e.g.
`CosmosTaskGroup`'s `external_augmentation_pool` defaults to `external_image_edit_service_pool`,
which is right for image-edit but wrong for image2video or Cosmos Transfer — EVG passes
`external_image2video_service_pool` and VDA passes `cosmos_transfer_3_service_pool` explicitly.
Internal-mode augmentation always runs on `default_pool`. Always diff every pool/profile-name-shaped
default against `deploy/values.yaml`'s `airflowPools.pools` and the manifest you wrote in Step 3.

### Data flow

`input_preparation` → `service_lifecycle` (parallel with input_preparation, gated by `services_ready`)
→ first processing task group (reads `input_preparation`'s XCom) → ... → last processing task group
→ `validated_output` → `[fail_pipeline, pipeline_success]` and `performance_reporting` (parallel
branches) → `service_lifecycle` shutdown (`ALL_DONE`).

Each processing task group after the first must read its *predecessor's* XCom (`input_xcom_task_id`
pointed at the previous group's group-qualified task id, where that param exists — see above), not
always `input_preparation` — e.g. in IAA, `event_and_person_attribute_search` reads
`cosmos_augmentation.validate_outputs`, not `input_preparation.prepare_input`.

DIG does not follow this shape: it starts from `generate_configs` instead of
`input_preparation`, routes stages with `pipeline_mode` branch and short-circuit tasks, and, in
`full` and `defect` modes, ends at `validate_anomalygen_outputs` instead of `validated_output`. A chain that starts with
simulation should copy DIG's wiring for that part.

---

## Step 0: Confirm No Existing DAG Already Covers This

Run this before Step 1, every time — including when this skill was invoked directly, since nothing
upstream of it is guaranteed to have made this check correctly. **Do not skip it because the
request already says "custom" or "new"; the user's framing is not evidence.**

The test is about the *task-group chain*, not the payload. These DAGs ship with this repository;
read the DAG file for the exact task-group order:

| Shipped DAG | What it runs | Workflow skill |
|---|---|---|
| `image_attribute_augmentation_dag_k8s` | `input_preparation → cosmos_augmentation` (image edit, with attribute-verification evaluation) `→ cosmos_post_processing` (when in the manifest) `→ event_and_person_attribute_search → generate_augmented_dataset → validated_output` | `cdf-image-attribute-augmentation` |
| `event_video_generation_dag_k8s` / `_osmo` | `input_preparation → cosmos_augmentation` (image-to-video, with Cosmos evaluation) `→ auto_labeling` (`detection_and_tracking → captioning → anomaly_visual_qa → person_attribute_visual_qa → person_attribute_search`) `→ generate_anomaly_dataset → validated_output` | `cdf-event-video-generation` |
| `video_data_augmentation_dag_k8s` / `_osmo` | `input_preparation`, then two branches into `validated_output`: auto-labeling of the source videos, and `cosmos_augmentation` (Cosmos Transfer, with Cosmos evaluation) followed by auto-labeling of the generated videos. Each auto-labeling branch runs `super_resolution → detection_and_tracking → captioning → visual_qa → reasoning → training_export` | `cdf-video-data-augmentation` |
| `autonomous_vehicle_dataset_generation_dag_k8s` | Curation and retrieval (`curator → media_probe → cds_ingest → collect_results → validate_retrieval`), then, when `cosmos.enabled`, Cosmos3-Nano generation with physical-plausibility evaluation, then `validated_output` | `cdf-autonomous-vehicle-dataset-generation` |
| `defect_image_generation_smt_dag_k8s` | `generate_configs → image_generation` (simulation) `→ augmentation` (image edit) `→ anomaly_package → anomaly_generation → validate_anomalygen_outputs`, with `pipeline_mode` selecting a subset | `cdf-defect-image-generation` |

If the requested pipeline's stage sequence — after mapping it to task groups and stages per the
Available Task Groups and component tables — **matches one of those chains exactly** (or the chain
of any other DAG under `airflow/dags/workflows/`), do not write a new DAG unless the user still
wants one after the recommendation below:

- Different input or output paths, service URLs, model overrides, prompts, variant counts, or
  distributions are **payload differences**. The shipped DAGs already parameterize these through
  their payload schemas — none of it justifies a new DAG.
- A subset that a shipped DAG already runs through a payload switch (DIG `pipeline_mode`, AV
  `cosmos.enabled`; see the component section) is also a payload difference.
- Only a **structural** difference — a task group added, removed, or reordered relative to the
  existing chain — is a genuine case for this skill.

If the check matches a shipped DAG: do not write any files yet. Recommend that DAG's workflow skill,
name the matching DAG, and show how the payload covers the differences they described. If it's a
genuine structural difference, proceed to Step 1.

If the user still wants a separate DAG after that recommendation, proceed to Step 1. Record their
reason (for example a feature they plan to add, or a DAG they need to own and change
independently) and include it in the Step 7 summary. Re-ask only if they haven't seen the
recommendation; their original "custom" or "new" framing alone is not that confirmation.

---

## Step 1: Gather Requirements

Ask all of the following together, not one at a time:

- **Pipeline** — the ordered sequence of stages, described in the user's own words. Map each
  stage to a task group from the table above. If ambiguous (e.g. "augmentation and labeling" —
  does that include a verification/QA pass?), ask before writing anything.
- **DAG name** — snake_case identifier.
- **Input path, output directory, max items to process.**
- **Service mode** — `external_services: true` (user supplies endpoint URLs) or `false` (DAG
  deploys VLM/LLM/image-edit pods in-cluster). If external, gather every service URL the selected
  task groups need.
- **Container images / models** — for each processing task group, confirm the container image and
  served model against the **component skill's current default** (per the component table
  above), not a hardcoded value. Ask the user to confirm or override.
- **Prompts / question banks / MCQ variables** — if the pipeline needs domain-specific prompts
  (edit prompt template, verification questions, person-attribute question bank), author them with
  the matching component skill's prompt-authoring guidance, not from scratch.

## Step 2: Read the Base Manifest and Component Skill References

Read the K8s manifest of every shipped DAG that supplies a stage in your chain, in full, before
writing anything (all paths under `airflow/dags/workflows/`):

| Stages in the chain | Manifest |
|---|---|
| Image-edit augmentation, image attribute augmentation | `image_attribute_augmentation_dag/configs/image_attribute_augmentation_k8s_manifest.yaml` |
| Image-to-video, detection and tracking, captioning, visual QA, event and person attribute search | `event_video_generation_dag/configs/event_video_generation_k8s_manifest.yaml` |
| Cosmos Transfer, super resolution, reasoning, training export | `video_data_augmentation_dag/configs/video_data_augmentation_k8s_manifest.yaml` |
| Simulation, AnomalyGen | `defect_image_generation_dag/configs/smt/defect_image_generation_k8s_manifest.yaml` |
| Curation and retrieval, Cosmos3-Nano, physical-plausibility evaluation | `autonomous_vehicle_dataset_generation_dag/configs/autonomous_vehicle_dataset_generation_k8s_manifest.yaml` |

Pick the one supplying most of the chain as the base, and copy the other stages' entries from
their own manifests. These are the source of truth for `deployment.profiles`, pool names,
`container_args`, secrets, and GPU configuration — copy fields verbatim, do not invent new
profile shapes. Copy any config templates those entries reference too (for example DIG's
`configs/smt/*.yaml`, AV's `cosmos_curator_config.yaml`).

In parallel, invoke the component skill(s) selected in Step 1 and read the referenced files. This
is where model defaults, endpoint `adapter` values (`nim`, `openai.chat.completions`,
`openai.images.edits`), and prompt/config schema come from — the manifest's `container_args` and
the DAG's generated config file must agree with what that skill documents.

## Step 3: Write the K8s Manifest

**Path:** `airflow/dags/workflows/<dag_name>_dag/configs/<dag_name>_k8s_manifest.yaml`

Copy the base manifest chosen in Step 2 verbatim, then:

1. **Trim `components.tasks`** to only the entries the selected task groups need (see the table's
   "Manifest component(s) needed" column).
2. **Trim `components.endpoints`** to only what the remaining tasks need.
3. **Apply container image / model overrides** confirmed in Step 1 — cross-check the resulting
   `container_args` (served model name, required flags such as `--omni` for `vllm-omni serve
   Qwen/Qwen-Image-Edit-2511`, endpoint adapter matching the exposed API path) against the
   component skill reference and against any external spec the user cited (see **Parity Check**
   below).
4. Do not change profiles, pools, secrets, or connection settings.

## Step 4: Write the DAG Python File

**Path:** `airflow/dags/workflows/<dag_name>_dag/<dag_name>_dag.py`

Follow the structure of the shipped DAG that shares the most stages with the pipeline (the same
base you picked in Step 2): `image_attribute_augmentation_dag.py`, `event_video_generation_dag.py`,
`video_data_augmentation_dag.py`, `defect_image_generation_smt_dag.py`, or
`autonomous_vehicle_dataset_generation_dag.py`. Wire stages taken from other DAGs the way their
own DAG wires them. Either way, it's the current, working pattern — do not resurrect an older
webserver/NVCF-era template:

- A `<DagName>DAGBuilder` class wrapping a `ComponentBuilder(manifest_path=...)`.
- `build_dag()` instantiates `ValidatePayloadTaskGroup`, `InputPreparationTaskGroup`,
  `ServiceLifecycleTaskGroup` (with a `ServiceLifecycleSpec` tuple matching only the endpoints this
  DAG uses), then the chosen processing task groups in order, then `ValidatedOutputTaskGroup` and
  optionally `PerformanceReportingTaskGroup`. A chain that starts with simulation follows DIG
  instead (see **Data flow** above).
- Wire `fail_pipeline`/`pipeline_success`/shutdown exactly as in the reference DAG — reporting is a
  terminal branch off `validated_output`, never upstream of the outcome tasks.
- Module-level registration uses the manifest-existence guard:

```python
_DAG_NAME_K8S_MANIFEST = os.environ.get(
    "<DAG_NAME_UPPER>_K8S_MANIFEST_PATH",
    <DAG_NAME_CONFIG_DIR> / "<dag_name>_k8s_manifest.yaml",
)
<dag_name>_dag_k8s = _create_<dag_name>_dag_if_manifest_exists(
    manifest_path=_DAG_NAME_K8S_MANIFEST,
    platform="k8s",
    description="...",
    tags=["sdg", "kubernetes", "<dag_name>"],
)
```

A missing manifest means the DAG is silently absent from Airflow, not broken — this is the only
"registration" mechanism in the current codebase. There is no webserver database and no
DAG-seeding step. If the user also wants an OSMO variant (see "Kubernetes, and OSMO for workflows
that support it" above), add a second `_create_<dag_name>_dag_if_manifest_exists` block mirroring
this one with `platform="osmo"` and its own manifest path — model it on
`event_video_generation_dag.py`'s existing k8s/osmo pair rather than inventing the pattern. There
is still no NVCF branch to add; NVCF remains unsupported.

**Path:** `airflow/dags/workflows/<dag_name>_dag/callables/`

For every task group in the chosen chain, write the required callable(s) by adapting the closest
existing DAG's equivalent (see **Workflow-local task groups** above for the sibling `tasks/`
directory used for non-shared groups like `CosmosPostProcessingTaskGroup`). This is real, new code
— it is expected to differ per DAG — but its *shape* (signature, XCom pull/push pattern, error
handling via `AirflowFailException`) should mirror the reference implementation, not be designed
from scratch.

**Watch for payload-schema coupling when reusing a `tasks/` callable across DAGs.** IAA's
`generate_image_attribute_augmentation_performance_report` re-validates the templated payload
against `ImageAttributeAugmentationDagPayloadConfig` by name inside its own `_parse_payload` — it is
not schema-agnostic despite living next to schema-agnostic reporting primitives. Reusing it
unmodified from a different DAG's payload model re-validates against the *wrong* schema; it may not
even fail loudly, since a before-validator on the foreign model can silently backfill fields your
payload never had. Any `tasks/` callable that imports a specific `*DagPayloadConfig` by name needs
the same treatment as the other callables above: adapt your own copy pointed at your own payload
model, don't import the original across DAGs.

**Path:** `airflow/dags/workflows/<dag_name>_dag/models/payload.py`

Follow the payload model pattern of the reference DAG you picked in Step 4 above (its
`models/payload.py`; DIG's is `callables/smt/models.py`), and copy the nested config fields for
stages taken from other DAGs from their payload models: top-level
`input_path`, `output_directory`, `external_services`, `service_lifecycle:
ServiceLifecycleTaskConfig`, one nested `*TaskConfig` field per processing task group,
`enable_performance_reporting: bool`, plus a `model_validator(mode="before")` that propagates
`output_directory`/`external_services` into nested configs and a `model_validator(mode="after")`
that fails fast when `external_services: true` but a required service URL is missing. DIG's
payload has no `input_path`: simulation reads the USD scene from `input_scene` and its variants
from `golden_generation`, and AnomalyGen reads its variants from `defect_generation`. Copy the
fields for the stages you take.

## Step 5: Register for Deployment

There is no DB-seeding step in the current deployment flow, and no S3 DAG sync either — DAGs and
plugins are baked into the Airflow image at build time (`docker/Dockerfile.airflow` `COPY`s
`airflow/dags` and `airflow/plugins` in). The deploy commands are `make build sdg-controller` then
`make install sdg-controller` (run from the repo root). **DO NOT RUN THEM YET** — Step 8 gates this
with explicit user confirmation after the readiness check:

```bash
make build sdg-controller     # rebuilds and pushes the image with the new DAG file baked in
make install sdg-controller   # redeploys the controller with that image
```

Airflow only picks up the new DAG file and manifest once the redeployed controller is running the
rebuilt image — a `make install sdg-controller` against a stale (not-yet-rebuilt) image tag will
not register the new DAG even though the Helm release itself reports success.

## Step 6: Verify

1. **YAML syntax:** `python -c "import yaml; yaml.safe_load(open('<manifest-path>'))"`
2. **DAG import and build:**
   `cd airflow && PYTHONPATH=plugins python -c "import dags.workflows.<dag_name>_dag.<dag_name>_dag as m; assert m.<dag_name>_dag_k8s is not None"`.
   DAG modules import plugin packages such as `slot_holds`, which Airflow puts on the path inside
   the controller image, so `PYTHONPATH=plugins` is required locally. The `assert` matters because
   the manifest-existence guard logs a warning and returns `None` when the DAG fails to build, so
   a bare import succeeds even for a broken DAG.
3. **Task chain is not empty** and every processing task group's `input_xcom_task_id` points at its
   actual upstream (not always `input_preparation`).
4. **Manifest/code agreement:** every `components.tasks.*` / `components.endpoints.*` key the code
   references exists in the manifest, and vice versa — no orphaned manifest entries.

### Parity Check (against an external spec, e.g. `launchable.md`)

If the user cited an external doc describing the same pipeline (model names, required serving
flags, endpoint contract, workflow modes), diff the generated manifest and config against it
explicitly and report any mismatch instead of silently reconciling:

- Served model names and versions match.
- Required serving flags match (e.g. `--omni` for `vllm-omni serve Qwen/Qwen-Image-Edit-2511`;
  confirm from the component skill, don't assume the flag name).
- The endpoint's exposed API path matches the manifest's `adapter` choice (e.g. a doc requiring
  `/v1/chat/completions` needs `adapter: openai.chat.completions`, not `openai.images.edits`).
- Any named "modes" in the doc (e.g. end-to-end vs. augmentation-only vs. labeling-only) map to
  either separate generated DAGs or a documented payload switch — state explicitly which, and if
  the current task groups can't express a mode the doc describes (see Constraints), say so rather
  than approximating it.

Fix any failures before reporting success.

---

## Step 7: Print Summary

```
Generated files
├── airflow/dags/workflows/<dag_name>_dag/configs/<dag_name>_k8s_manifest.yaml
│   Endpoints: <...>
│   Tasks:     <...>
├── airflow/dags/workflows/<dag_name>_dag/models/payload.py
├── airflow/dags/workflows/<dag_name>_dag/callables/  (adapted from: <source DAG per callable>)
└── airflow/dags/workflows/<dag_name>_dag/<dag_name>_dag.py
    Task chain: input_preparation → <...> → validated_output
    DAG ID: <dag_name>_dag_k8s

Parity check against <cited doc>: <pass / mismatches listed>

Shipped DAG that already covers this: <none, or DAG ID + the user's reason for a separate DAG>

GPU footprint — external mode: <N> GPU(s) — <task>: 1 GPU (k8s_gpu_task), ... (0 if every task pod
  in the chain is CPU-profiled)
GPU footprint — internal mode: <N> GPU(s) = external-mode total + <endpoint>: <n> GPU(s) each, ...

To register: make build sdg-controller && make install sdg-controller
```

**Compute both GPU footprint lines from the manifest itself — never state or imply external mode is
0 without checking.** See [`references/gpu-footprint.md`](references/gpu-footprint.md) for the two
sources (GPU-profiled task pods in either mode, plus internally deployed endpoints) and how they add.

Print a complete example payload covering every field in the new payload model, including nested
task configs and any prompt/config file paths the pipeline needs.

---

## Step 8: Readiness Check, Deploy, Trigger, and Monitor

Because this DAG is never added to a run-skill's checked-in DAG-ID table (see "These DAGs are not
checked in as permanent artifacts" above), nothing else will ever walk the user through running it.

**Default: item 1 below (readiness check) runs immediately and automatically after Step 7 — proceed
straight into it without stopping.** It is read-only (`kubectl get`, `curl` GETs) and touches no
shared state, so it needs no confirmation.

**Item 2 (`make build sdg-controller` + `make install sdg-controller`) is different: it rebuilds
the Airflow image with the new DAG file baked in, pushes it to the registry, and redeploys the
shared Airflow cluster to run it — a state-changing action on shared infrastructure.**
Show the user the Step 7 summary plus the readiness-check results, then ask for explicit
confirmation before running it. Invoking this skill authorizes generating the files; it does not
by itself authorize deploying them to a shared cluster, so do not treat the initial invocation as
standing approval for this step — confirm every time. The same confirm-before-execute rule applies,
for the same reason, to triggering an actual run (steps 3-4 below), which additionally consumes
real GPU/compute and produces billable, resource-consuming workflow executions.

1. **Readiness check**, mirroring the checklist in `cdf-image-attribute-augmentation`'s SKILL.md
   `## Scope` section and
   [its `airflow-direct-api.md`](../cdf-image-attribute-augmentation/references/airflow-direct-api.md)
   — same cluster, same controller, only the target `dag_id` differs:
   - Establish the cluster connection: ask the user for the credential file path; never assume one,
     never fall back to a default location, never log the file contents.
   - Controller pods Running: `kubectl get pods -n sdg-workflow -l "release=sdg-workflow-controller"`.
   - Airflow API reachable via the ClusterIP. **The ClusterIP is not guaranteed routable from
     wherever this skill runs** — it depends on the host's network path to the cluster, which
     varies by environment. Test it first (`curl --max-time 8 "$AIRFLOW_URL/api/v2/version"`); if
     it times out, fall back to `make port-forward` as a background job (never foreground — it
     blocks) and use `http://localhost:8080` for every subsequent API call in this step. Don't
     treat a ClusterIP timeout as a readiness-check failure that routes to `cdf-orchestration-setup` —
     it's a routing difference, not a cluster problem, as long as the port-forward fallback works.
   - Once deployed (step 2 below), confirm this DAG's specific `dag_id` is loaded and
     `is_paused: False`. A DAG that doesn't appear after `make build sdg-controller` + `make
     install sdg-controller` most often means the image was deployed before the rebuild finished,
     or the rebuild step was skipped — re-check the deployed image tag rather than re-running Step
     4-7.
   - Pools with open slots: the cluster's GPU-task pool (e.g. `k8s_rtx6000_1` — read the actual
     name from `deploy/values.yaml`'s `airflowPools.pools` rather than assuming one), `default_pool`,
     the manifest's own task pools, and the external augmentation pool passed to `CosmosTaskGroup`
     (e.g. `external_image_edit_service_pool`) — read the actual names from the manifest you wrote
     in Step 3 and the DAG wiring, don't assume they match the reference DAG's.
   - Compute-cluster GPU capacity (free vs. total, not just allocatable — the cluster is shared)
     against the GPU footprint computed in Step 7, for the service mode the payload will actually
     use — not the other mode's number.
   - Stale failed pods in the namespace (report, don't clean up without confirming ownership).
   - If any check fails, route to the `cdf-orchestration-setup` skill rather than diagnosing further.
2. **Deploy**: after the user confirms (see the documented default above), run `make build
   sdg-controller` then `make install sdg-controller` (Step 5), then re-run the Airflow-API
   DAG-loaded check.
3. **Build and confirm the payload** with the user — same non-negotiable rule as
   `cdf-image-attribute-augmentation`: every required field (`input_path`,
   `output_directory`, service mode, and any external service URLs the chosen task groups need)
   comes from the user and is never invented. Values the user supplied earlier, including for a
   previous run, may be reused. Then run
   `python3 skills/cdf-orchestration-preflight/scripts/storage_check.py --payload <payload.json>`
   from the repository root; continue only if it prints `Storage mapping: ready`. Using an
   unmapped bucket requires changing the controller's storage configuration and redeploying it;
   with the user's authorization, route that change to
   [orchestration setup](../cdf-orchestration-setup/SKILL.md), then rerun the check.
4. **Trigger**: POST the confirmed payload to this DAG's `dagRuns` endpoint (or the CLI
   equivalent) — the auth + request shape is in the same `airflow-direct-api.md` linked in item 1.
   Capture the run id.
5. **Monitor**: poll run status until terminal (`success`/`failed`), reporting progress
   periodically rather than going silent for the full run.
6. **On completion**: fetch and summarize outputs per the payload model's `output_directory`
   layout (and, if `enable_performance_reporting` was set, the generated report) — same shape as
   `cdf-image-attribute-augmentation`'s result retrieval, applied to this DAG's own output
   paths.

---

## Examples

Sample prompts that should and should not trigger this skill, mapped to the IAA `augmentation`,
`auto_labeling`, and `e2e` modes, are in
[`references/trigger-examples.md`](references/trigger-examples.md).

---

## Constraints

- **K8s always; OSMO only when explicitly requested and external-services-only.** Add an OSMO
  manifest/registration path only if the user asks for one, and enforce the external-services-only
  contract (see "Kubernetes, and OSMO for workflows that support it" above) — never let an OSMO
  variant deploy in-cluster services. Do not add NVCF manifests or registration paths at all.
- **Do not modify existing DAG files, task groups, or component-skill source.** Only write the new
  DAG's own files (manifest, payload model, callables, DAG builder, plus any new
  prompt/question-bank/config files the component skill's authoring reference calls for).
- **Do not reimplement or fork task-group class logic.** Every processing step is an instance of an
  existing `*TaskGroup` class (shared or, where applicable, workflow-local) or an existing
  manifest task from a shipped DAG (DIG's simulation and AnomalyGen, AV's curation and
  retrieval), called the way that DAG already calls it. The callables passed *into* those task groups are expected new
  code — adapted from the closest existing DAG's equivalent, not invented from a blank page (see
  Step 4).
- **Do not invent prompts, config keys, or model defaults.** Get them from the matching component
  skill (per the component table) or from the base manifest.
- If the user's pipeline needs a task group that doesn't exist yet (e.g. a third
  `VisualQATaskGroup` purpose beyond what an existing `group_id` naming convention supports, or two
  separate `CosmosTaskGroup` rounds — it takes a `group_id` (default `cosmos_augmentation`; DIG
  passes `augmentation`), but every instance builds from the same manifest `augmentation` task and
  its `max_concurrency`, and no shipped DAG runs two rounds), explain the gap and ask for
  clarification rather than writing workaround code.
