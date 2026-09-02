---
name: paidf-orchestration-write-dag
description: Use when a user describes a custom PAIDF Orchestration pipeline — a specific ordered combination of stages such as augmentation only, auto-labeling only, detection+captioning only, or image attribute augmentation without full auto-labeling — that no existing DAG in airflow/dags/workflows/ covers, and asks for a new Kubernetes DAG. Also use to check that a generated or existing DAG's model/container/prompt choices match an external spec document (e.g. a PAIDF `launchable.md`).
version: "1.0.0"
license: CC-BY-4.0 AND Apache-2.0
metadata:
  owner: NVIDIA
  service: physical-ai-data-factory
  version: 1.0.0
  reviewed: '2026-09-02'
  author: NVIDIA
  tags:
    - physical-ai
    - paidf-orchestration
    - airflow
    - dag
---

# Write DAG

## What This Skill Does

Composes a new Kubernetes-only Airflow DAG from the **existing, shared task groups** in
`airflow/dags/shared/task_groups/` — the same building blocks `image_attribute_augmentation_dag.py`
and `event_video_generation_dag.py` are built from. It does not invent new task-group *logic*.
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

**Kubernetes only.** This repository's `main`/current branches check in K8s manifests only (see
`image-attribute-augmentation-workflow` and `event-video-generation-workflow` skills) — no NVCF, no
OSMO. Do not offer those backends unless the user explicitly says their environment supports them,
and if so, treat it as new scope requiring its own investigation rather than a flag on this skill.

**Not scoped to one workflow.** Task groups are shared across IAA and EVG today; a custom DAG can
mix stages from either lineage (e.g. `CosmosTaskGroup` + `AutoLabelingTaskGroup` with no
`DetectionAndTrackingTaskGroup`, or `ImageAttributeAugmentationTaskGroup` with no `CosmosTaskGroup`
for a labeling-only pass over already-augmented crops).

**These DAGs are not checked in as permanent artifacts.** Unlike `image_attribute_augmentation_dag`
or `event_video_generation_dag`, a DAG this skill generates is a one-off, user-requested pipeline —
it is not added to `image-attribute-augmentation-workflow`, `event-video-generation-workflow`, or
any other run-skill's DAG-ID table. Because nothing else will ever guide the user through deploying
or running it, **this skill owns the full lifecycle, not just authoring** — Step 8 below is not
optional follow-through, it's the only place readiness-check/deploy/trigger/monitor for this DAG
will ever happen.

---

## Making Use of Component Skills

The task groups here are thin Airflow wrappers around two component pipelines that have their own
skills, external to this repo — **`augmentation`** (image-edit / Cosmos Transfer / Cosmos Predict /
image2video container) and **`auto-labeling`** (detection, captioning, visual QA, person attribute
search container). The DAG's `container_args` and generated config files are literally the CLI
contract those skills document — this skill does not reimplement that contract, it points at it.

**Those two skills are only reachable when this session's workspace happens to include their
repos** — there's no submodule, plugin registration, or other durable link from `sdg-workflow` to
them. Try `Skill(skill="augmentation")` / `Skill(skill="auto-labeling")` first; if the skill isn't
found, fall back to
[`references/component-skills-excerpt.md`](references/component-skills-excerpt.md) — a small,
frozen excerpt of the schema/prompt/troubleshooting content this skill has actually needed so far.
Treat it as a stale snapshot, not a substitute for the real skill: if what you need isn't in it, say
so and ask rather than extrapolating.

**Always invoke the matching component skill before writing manifest task entries, config
templates, prompts, or question banks** — never invent config schema, model defaults, or prompt
wording from memory:

| Writing... | Invoke skill | Read |
|---|---|---|
| `CosmosTaskGroup` config — image-edit (IAA-style) | `augmentation` | `references/configuration-schema.md` (endpoints list, adapters), `references/image-attribute-augmentation.md` (image-edit prompts, MCQ `exclude_variables`, distribution configs) |
| `CosmosTaskGroup` config — image2video / Cosmos Transfer/Predict (EVG-style) | `augmentation` | `references/event-video-gen.md`, `references/config-decision-tree.md` (which model/adapter for the input→output shape you actually have) |
| `ImageAttributeAugmentationTaskGroup` / `EventAndPersonAttributeSearchTaskGroup` prompts, question banks | `auto-labeling` | `references/prompt-authoring.md`, `references/event-and-person-attribute-search.md`, `references/stages/person-attribute-search.md` |
| `DetectionAndTrackingTaskGroup` / `CaptioningTaskGroup` / `VisualQATaskGroup` / `AutoLabelingTaskGroup` config | `auto-labeling` | the matching file under `references/stages/` (`detection-and-tracking.md`, `captioning.md`, `visual-qa.md`) |
| Config validation or runtime errors from either container | matching skill | its `references/troubleshooting.md` |

`config-decision-tree.md` is the one to open first whenever the task group is doing something other than
a straight image-edit (IAA's case) — it's what tells you Cosmos Transfer vs. Predict vs. image2video,
not the task-group name.

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
| `InputPreparationTaskGroup` | `input_preparation` | Lists/normalizes input media from the payload's `input_path`. Always required. | none |
| `ServiceLifecycleTaskGroup` | `service_lifecycle` | Deploys/tears down internal VLM/LLM/image-edit/image2video pods and waits for readiness; skipped per-service when `external_services: true`. | `components.endpoints` entries matching the `ServiceLifecycleSpec`s passed in |
| `CosmosTaskGroup` | `cosmos` | Runs the `augmentation` component (image-edit or Cosmos Transfer/Predict/image2video) via a manifest task, dynamically mapped over inputs. | `components.tasks.augmentation` (+ VLM/LLM/image-edit endpoints) |
| `ImageAttributeAugmentationTaskGroup` | `image_attribute_augmentation` | Runs person-attribute search/captioning over cosmos-augmented crops (IAA's auto-labeling stage). Reads a cosmos-output XCom. | `components.tasks.event_and_person_attribute_search` |
| `EventAndPersonAttributeSearchTaskGroup` | `event_and_person_attribute_search` | EVG's equivalent — person/event attribute search over auto-labeled video output. Reads an auto-labeling XCom. | `components.tasks.event_and_person_attribute_search` |
| `DetectionAndTrackingTaskGroup` | `detection_and_tracking` | RFDETR detection + BoxMOT tracking on video/cosmos output. | `components.tasks.auto_labeling` (or the manifest's detection task key — check the base manifest) |
| `CaptioningTaskGroup` | `captioning` | VLM captioning on detection/tracking output. | `components.tasks.auto_labeling` |
| `VisualQATaskGroup` | `visual_qa` | VLM visual QA; instantiate once per QA purpose with a distinct `group_id`/`component_name` (EVG uses it twice: anomaly QA and person-attribute QA). | `components.tasks.visual_qa` |
| `AutoLabelingTaskGroup` | `auto_labeling` | Generic mapped auto-labeling container execution (detection+tracking as one component) — used when a workflow doesn't need the split `DetectionAndTrackingTaskGroup`/`CaptioningTaskGroup` stages. | `components.tasks.auto_labeling` |
| `ValidatedOutputTaskGroup` | `validated_output` | Runs a workflow-specific output validation callable. Always last processing step before reporting. | none |
| `PerformanceReportingTaskGroup` | `reporting` | Optional YAML+HTML performance report, gated by `enable_performance_reporting` in the payload. Hangs off `validated_output` as a terminal branch — never upstream of `fail_pipeline`/`pipeline_success`. | none |

**Before using any task group**, read its `__init__` in the source file — constructor kwargs
(`input_xcom_task_id`, `group_id`, `component_name`, `prepare_args_callable`,
`prepare_args_op_kwargs`) vary per group and determine both manifest wiring and XCom data flow.
Copy the calling pattern from whichever of `image_attribute_augmentation_dag.py` or
`event_video_generation_dag.py` already uses that group — do not guess kwargs, and do not assume
every group exposes an `input_xcom_task_id` param: `ValidatedOutputTaskGroup`, for one, doesn't —
it takes a plain `op_kwargs` dict (default `{"payload": ..., "run_id": ...}`), so pulling an
upstream XCom into it means passing a custom `op_kwargs`/callable, not a task-id string.

**Workflow-local task groups exist too, outside `shared/task_groups/`.** IAA's
`CosmosPostProcessingTaskGroup` lives under
`airflow/dags/workflows/image_attribute_augmentation_dag/tasks/` and is wired in *conditionally* —
`image_attribute_augmentation_dag.py` only includes it when
`"cosmos_post_processing" in self.builder.manifest.deployment.components.tasks`. Check each
reference DAG's own `tasks/` directory, not just `shared/task_groups/`, for stage-specific groups
like this, and decide inclusion the same way: gate on manifest task presence, don't hardcode it in.

**Constructor kwarg defaults are not guaranteed to match your manifest.** e.g.
`CosmosTaskGroup`'s `internal_augmentation_pool` default is `internal_image_edit_service_pool`,
but IAA's manifest profile is actually named `iaa_internal_image_edit_service_pool` — the reference
DAG overrides the kwarg explicitly. Always diff every pool/profile-name-shaped default against the
manifest you wrote in Step 3, not just against the task group's source.

### Data flow

`input_preparation` → `service_lifecycle` (parallel with input_preparation, gated by `services_ready`)
→ first processing task group (reads `input_preparation`'s XCom) → ... → last processing task group
→ `validated_output` → `[fail_pipeline, pipeline_success]` and `performance_reporting` (parallel
branches) → `service_lifecycle` shutdown (`ALL_DONE`).

Each processing task group after the first must read its *predecessor's* XCom (`input_xcom_task_id`
pointed at the previous group's group-qualified task id, where that param exists — see above), not
always `input_preparation` — e.g. in IAA, `event_and_person_attribute_search` reads
`cosmos_augmentation.validate_outputs`, not `input_preparation.prepare_input`.

---

## Step 0: Confirm No Existing DAG Already Covers This

Run this before Step 1, every time — including when this skill was invoked directly, since nothing
upstream of it is guaranteed to have made this check correctly. **Do not skip it because the
request already says "custom" or "new"; the user's framing is not evidence.**

The test is about the *task-group chain*, not the payload: `image_attribute_augmentation_dag_k8s`
already runs `input_preparation → cosmos_augmentation → cosmos_post_processing →
event_and_person_attribute_search → generate_augmented_dataset → validated_output`, and
`event_video_generation_dag_k8s` already runs its own full chain (see that DAG's file for the exact
task groups). If the requested pipeline's stage sequence — after mapping it to task groups per the
table below — **matches one of those chains exactly**, this is not a `write-dag` case, no matter
what triggered the invocation:

- Different `input_path`/`output_directory`/service URLs/model overrides/`num_augmentation`/
  `variable_distribution`/`max_imgs` are **payload differences**. Both checked-in DAGs already
  parameterize all of that through their existing payload schema — none of it justifies a new DAG.
- Only a **structural** difference — a task group added, removed, or reordered relative to the
  existing chain — is a genuine `write-dag` case.

If the check matches an existing DAG: stop, do not write any files, and tell the user to use
`image-attribute-augmentation-workflow` or `event-video-generation-workflow` (whichever chain
matched) instead — with the payload differences they described, since that skill's payload already
covers them. If it's a genuine structural difference, proceed to Step 1.

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
  served model against the **component skill's current default** (`augmentation` or
  `auto-labeling`, per the table above), not a hardcoded value. Ask the user to confirm or override.
- **Prompts / question banks / MCQ variables** — if the pipeline needs domain-specific prompts
  (edit prompt template, verification questions, person-attribute question bank), author them with
  the matching component skill's `prompt-authoring.md`/`image-attribute-augmentation.md` guidance,
  not from scratch.

## Step 2: Read the Base Manifest and Component Skill References

Read the closest existing K8s manifest in full before writing anything —
`airflow/dags/workflows/image_attribute_augmentation_dag/configs/image_attribute_augmentation_k8s_manifest.yaml`
if the pipeline includes `CosmosTaskGroup`/`ImageAttributeAugmentationTaskGroup`, or
`airflow/dags/workflows/event_video_generation_dag/configs/event_video_generation_k8s_manifest.yaml`
if it includes detection/tracking/captioning/visual-QA. These are the source of truth for
`deployment.profiles`, pool names, `container_args`, secrets, and GPU configuration — copy fields
verbatim, do not invent new profile shapes.

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

Follow the structure of whichever reference DAG matches the pipeline's task groups —
`image_attribute_augmentation_dag.py` for an image-edit/IAA-style chain,
`event_video_generation_dag.py` for a detection/captioning/visual-QA/EVG-style chain, or the closer
match by shared task groups when the pipeline mixes both lineages. Either way, it's the current,
working pattern — do not resurrect an older webserver/NVCF-era template:

- A `<DagName>DAGBuilder` class wrapping a `ComponentBuilder(manifest_path=...)`.
- `build_dag()` instantiates `ValidatePayloadTaskGroup`, `InputPreparationTaskGroup`,
  `ServiceLifecycleTaskGroup` (with a `ServiceLifecycleSpec` tuple matching only the endpoints this
  DAG uses), then the chosen processing task groups in order, then `ValidatedOutputTaskGroup` and
  optionally `PerformanceReportingTaskGroup`.
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
"registration" mechanism in the current codebase. There is no webserver database, no DAG-seeding
step, and no NVCF/OSMO branch to add.

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

Follow the payload model pattern of whichever reference DAG you picked in Step 4 above
(`ImageAttributeAugmentationDagPayloadConfig` or `EventVideoGenerationDagPayloadConfig`): top-level
`input_path`, `output_directory`, `external_services`, `service_lifecycle:
ServiceLifecycleTaskConfig`, one nested `*TaskConfig` field per processing task group,
`enable_performance_reporting: bool`, plus a `model_validator(mode="before")` that propagates
`output_directory`/`external_services` into nested configs and a `model_validator(mode="after")`
that fails fast when `external_services: true` but a required service URL is missing.

## Step 5: Register for Deployment

There is no DB-seeding step in the current deployment flow. The deploy command is `make sync-dag`
(run from the repo root). **DO NOT RUN IT YET** — Step 8 gates this with explicit user confirmation
after the readiness check:

```bash
make sync-dag   # packages and uploads DAGs, plugins, and configs to S3
```

Airflow picks up the new DAG file and manifest on its normal DAG-folder refresh — no additional
registration call.

## Step 6: Verify

1. **YAML syntax:** `python -c "import yaml; yaml.safe_load(open('<manifest-path>'))"`
2. **DAG import:** `cd airflow && python -c "import dags.workflows.<dag_name>_dag.<dag_name>_dag"`
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

GPU footprint — external mode: <N> GPU(s) — <task>: 1 GPU (k8s_gpu_task), ... (0 if every task pod
  in the chain is CPU-profiled)
GPU footprint — internal mode: <N> GPU(s) = external-mode total + <endpoint>: <n> GPU(s) each, ...

To register: make sync-dag
```

**Compute both GPU footprint lines from the manifest itself — never state or imply external mode is
0 without checking.** Two independent sources, only one of which is mode-dependent:

1. **Task pods that do their own local model inference, in either mode.** For every entry in
   `components.tasks`, resolve its `deployment_profile` against `deployment.profiles`: a
   `k8s_gpu_task`-profiled (or otherwise `gpu:`-bearing) task claims 1 GPU per running pod
   *regardless of `external_services`*, because it's doing in-pod inference, not calling an
   endpoint — a `k8s_cpu_task`/`augmentation_task`-profiled one claims none. This is exactly why
   IAA's external mode is genuinely 0 GPU (its tasks — `augmentation`, `cosmos_post_processing`,
   `event_and_person_attribute_search` — are all CPU-profiled) while EVG's external mode is not
   (`detection_and_tracking`/`captioning`/`visual_qa` all run on `k8s_gpu_task`, 3 GPUs minimum
   before any endpoint is even considered).
2. **Endpoints deployed internally**, i.e. `external_services: false` for that service. Per
   replica: VLM / LLM / image-edit / Cosmos Transfer or Predict = 1 GPU; image2video = **2** GPUs
   (`gpu_count: 2`, `host_ipc: true`). `external_services: true` contributes 0 here.

Internal-mode total = source 1 + source 2 — the task-pod cost doesn't disappear in internal mode,
it's additive. Report both totals in the summary; don't make the user re-derive them from the
manifest, and don't reuse a number from a different pipeline's shape (IAA's "external = 0" does not
generalize to a pipeline that includes any `k8s_gpu_task`-profiled stage).

Print a complete example payload covering every field in the new payload model, including nested
task configs and any prompt/config file paths the pipeline needs.

---

## Step 8: Readiness Check, Deploy, Trigger, and Monitor

Because this DAG is never added to a run-skill's checked-in DAG-ID table (see "These DAGs are not
checked in as permanent artifacts" above), nothing else will ever walk the user through running it.

**Default: item 1 below (readiness check) runs immediately and automatically after Step 7 — proceed
straight into it without stopping.** It is read-only (`kubectl get`, `curl` GETs) and touches no
shared state, so it needs no confirmation.

**Item 2 (`make sync-dag`) is different: it uploads DAG files, plugins, and configs to S3 and
deploys them to the shared Airflow cluster — a state-changing action on shared infrastructure.**
Show the user the Step 7 summary plus the readiness-check results, then ask for explicit
confirmation before running it. Invoking `write-dag` authorizes generating the files; it does not
by itself authorize deploying them to a shared cluster, so do not treat the initial invocation as
standing approval for this step — confirm every time. The same confirm-before-execute rule applies,
for the same reason, to triggering an actual run (steps 3-4 below), which additionally consumes
real GPU/compute and produces billable, resource-consuming workflow executions.

1. **Readiness check**, mirroring the checklist in `image-attribute-augmentation-workflow`'s SKILL.md
   `## Scope` section and `references/airflow-direct-api.md` — same cluster, same controller, only
   the target `dag_id` differs:
   - Establish the cluster connection: ask the user for the credential file path; never assume one,
     never fall back to a default location, never log the file contents.
   - Controller pods Running: `kubectl get pods -n sdg-workflow -l "release=sdg-workflow-controller"`.
   - Airflow API reachable via the ClusterIP. **The ClusterIP is not guaranteed routable from
     wherever this skill runs** — it depends on the host's network path to the cluster, which
     varies by environment. Test it first (`curl --max-time 8 "$AIRFLOW_URL/api/v2/version"`); if
     it times out, fall back to `make port-forward` as a background job (never foreground — it
     blocks) and use `http://localhost:8080` for every subsequent API call in this step. Don't
     treat a ClusterIP timeout as a readiness-check failure that routes to `orchestration-setup` —
     it's a routing difference, not a cluster problem, as long as the port-forward fallback works.
   - Once deployed (step 2 below), confirm this DAG's specific `dag_id` is loaded and
     `is_paused: False`. A DAG that doesn't appear yet after `make sync-dag` means the sync/refresh
     hasn't landed, not that generation failed; re-check rather than re-running Step 4-7.
   - Pools with open slots: `k8s_gpu_1`, `default_pool`, and the manifest's own task pools (e.g.
     `iaa_internal_image_edit_service_pool` — read the actual pool names out of the manifest you
     wrote in Step 3, don't assume they match the reference DAG's).
   - Compute-cluster GPU capacity (free vs. total, not just allocatable — the cluster is shared)
     against the GPU footprint computed in Step 7, for the service mode the payload will actually
     use — not the other mode's number.
   - Stale failed pods in the namespace (report, don't clean up without confirming ownership).
   - If any check fails, route to the `orchestration-setup` skill rather than diagnosing further.
2. **Deploy**: after the user confirms (see the documented default above), run `make sync-dag`
   (Step 5), then re-run the Airflow-API DAG-loaded check.
3. **Build and confirm the payload** with the user — same non-negotiable rule as
   `image-attribute-augmentation-workflow`: every required field (`input_path`,
   `output_directory`, service mode, and any external service URLs the chosen task groups need)
   comes from the user, never invented, reused from a previous run, or filled from a checked-in
   file.
4. **Trigger**: POST the confirmed payload to this DAG's `dagRuns` endpoint (or the CLI
   equivalent) — see `references/airflow-direct-api.md` in the same skill for the auth + request
   shape. Capture the run id.
5. **Monitor**: poll run status until terminal (`success`/`failed`), reporting progress
   periodically rather than going silent for the full run.
6. **On completion**: fetch and summarize outputs per the payload model's `output_directory`
   layout (and, if `enable_performance_reporting` was set, the generated report) — same shape as
   `image-attribute-augmentation-workflow`'s result retrieval, applied to this DAG's own output
   paths.

---

## Examples

Sample prompts that should trigger this skill, mapped to modes documented in PAIDF's
`physical-ai-image-attribute-augmentation/launchable.md`. The same shape of request applies to
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

`e2e` mode is not a `write-dag` case — `image_attribute_augmentation_dag_k8s` already covers it.

---

## Constraints

- **K8s only** — do not add NVCF/OSMO manifests or registration paths.
- **Do not modify existing DAG files, task groups, or component-skill source.** Only write the new
  DAG's own files (manifest, payload model, callables, DAG builder, plus any new
  prompt/question-bank/config files the component skill's authoring reference calls for).
- **Do not reimplement or fork task-group class logic.** Every processing step is an instance of an
  existing `*TaskGroup` class (shared or, where applicable, workflow-local), called the way an
  existing DAG already calls it. The callables passed *into* those task groups are expected new
  code — adapted from the closest existing DAG's equivalent, not invented from a blank page (see
  Step 4).
- **Do not invent prompts, config keys, or model defaults.** Get them from the matching component
  skill (`augmentation` or `auto-labeling`) or from the base manifest.
- If the user's pipeline needs a task group that doesn't exist yet (e.g. a third
  `VisualQATaskGroup` purpose beyond what an existing `group_id` naming convention supports, or two
  separate `CosmosTaskGroup` rounds — it has no `group_id` parameter, only one augmentation step
  per DAG is currently possible), explain the gap and ask for clarification rather than writing
  workaround code.
