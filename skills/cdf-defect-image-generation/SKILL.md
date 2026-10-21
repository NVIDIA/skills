---
name: cdf-defect-image-generation
description: Generate synthetic PCB inspection images on Kubernetes with the SMT Defect Image Generation (DIG) DAG in CDF Orchestration. The DAG renders components in simulation, turns the renders into realistic golden images (sim2real), and uses AnomalyGen to add texture defects such as solder bridges, excess solder or missing components. Simulation can also render shift, tombstone and sideflip pose defects. Use it to build DIG payloads, pick a pipeline mode (full, simulation, sim2real or defect-only), check GPU readiness, follow a run, and match each defect image to its golden and simulated source. Not for running Simulation, Augmentation or AnomalyGen on their own.
version: "1.0.4"
license: CC-BY-4.0 AND Apache-2.0
metadata:
  owner: NVIDIA
  service: cosmos-data-factory
  version: "1.0.4"
  reviewed: "2026-10-20"
  author: NVIDIA
  tags:
    - physical-ai
    - cdf-orchestration
    - defect-image-generation
    - smt
---

# CDF Orchestration — SMT Defect Image Generation

Run `defect_image_generation_smt_dag_k8s`: CDF Simulation renders PCB components,
CDF Augmentation turns the renders into realistic clean (golden) images, and
CDF AnomalyGen adds texture defects. Simulation can also render pose defects
(shift, tombstone, sideflip). One DAG serves every mode; `pipeline_mode` selects
which stages run.

## Requirements

Execution needs an authorized Kubernetes connection, a reachable Airflow 3
controller with the DIG DAG loaded, and storage and model access for the chosen
mode. The payload checker needs a trusted CDF Orchestration checkout and its
Airflow Python environment. Explaining the workflow or drafting JSON needs neither.

Execution readiness also requires the companion script
`skills/cdf-orchestration-preflight/scripts/storage_check.py` in the product
checkout. If it is unavailable, report the missing dependency as a readiness
blocker; do not skip the storage mapping check.

In that checkout, the payload contract is defined by
`airflow/dags/workflows/defect_image_generation_dag/callables/smt/models.py` and
`airflow/dags/workflows/defect_image_generation_dag/callables/smt/config_generation.py`.
The component templates that payload overrides merge into are in
`airflow/dags/workflows/defect_image_generation_dag/configs/smt/`.

## Available scripts

| Script | Purpose |
|---|---|
| `scripts/payload.py` | Check a payload with the DAG's own model and compiler, then print job counts, output paths and effective AnomalyGen staging. Submits nothing and accesses no storage or cluster. |

## Instructions

### 1. Resolve the requested operation and mode

| User intent | `pipeline_mode` | Required stages |
|---|---|---|
| Render PCB components only | `simulation` | Simulation |
| Good (golden) component images from a board scene | `sim2real` | Simulation → image-edit |
| Pose defects (shift, tombstone, sideflip) from a board scene | `sim2real`, with a pose operation in the simulation override | Simulation → image-edit |
| Texture defects (bridge, excess solder, missing) from a board scene | `full` | Simulation → image-edit → AnomalyGen |
| Texture defects on real or prepared clean images | `defect` | AnomalyGen |

Requests that start from a board USD are sometimes called "Day 0", and requests
that start from real or prepared images "Day 1".
[payload-contract.md](references/payload-contract.md) shows how to select the
components a request names, set counts and coverage, and add pose defects.

For a payload-only request, prepare and check it locally, and put the checked
payload JSON in your final answer; do not launch a run.
For status or results, use the named existing run instead of building a payload.
For manual UI entry, set up UI access (step 5) and report the URL; the user
triggers the run.

### 2. Gather only the inputs needed for that mode

Reuse values already supplied in this task and ask for anything missing. Never
invent a location, endpoint or model name.

| Applies to | Inputs |
|---|---|
| All modes | Writable `output_directory`; mode |
| `simulation`, `sim2real`, `full` | `input_scene`; golden configuration IDs and component/camera/render choices |
| `sim2real`, `full` | Internal image-edit replica count, or external image-edit URL and optional served model |
| Pose defects | Operation, magnitudes, and a `count` or `ratio` of components to move |
| `full`, `defect` | Defect configuration IDs, `class+defect` counts, shared assets, recipe and checkpoint locations, AMG model/endpoint settings |
| `full` | Exact `component_type` → AnomalyGen class mapping for every selected type |
| `defect` | Prepared `<class>/clean_image/` dataset root, included alongside shared assets in `staging.fetch.input` |

Requests often name components by kind ("the passives") or by part number
("0603_H100"); [selecting components](references/payload-contract.md#selecting-components)
maps those to selectors. The simulation template is a preset for the Artix sample
board. Another board needs its own scene settings and component selectors, not
just a different ZIP path; see
[adapting to another board](references/payload-contract.md#adapting-to-another-board).
Upload local inputs only to the user's chosen destination, preserving the scene
assets or class-folder hierarchy.

### 3. Build and check the payload

Read [payload-contract.md](references/payload-contract.md) for collections,
examples, merge rules and per-mode requirements, then build plain JSON with
`api_version: "1.0"`. Keep run-specific asset locations in payload overrides so
the controller does not need a rebuild.

Each component override is handed to a component that validates it against its
own schema. Before writing or changing one, read that component's documentation
for the version the SMT manifest deploys, and do not invent keys or values. The
links point to each component's public GitHub repository, pinned to a commit
whose documentation matches the deployed images; a component skill installed
separately may describe a different version.

| Override | Component image in the SMT manifest | Documentation |
|---|---|---|
| `image_generation` | `nvcr.io/nvidia/cdf-simulation:1.2.0` | [`cdf-simulation` skill][simulation-skill]; [YAML JobConfig reference][simulation-config] |
| `augmentation` | `nvcr.io/nvidia/cdf-augmentation:1.2.0` | [`cdf-augmentation` skill][augmentation-skill]: [configuration schema][augmentation-schema] and [captioning strategies][augmentation-captions] |
| `anomaly_generation` | `nvcr.io/nvidia/cdf-anomalygen:1.2.0` | [`cdf-anomalygen` skill][anomalygen-skill]: [inputs][anomalygen-inputs] and [agentic mask generation][anomalygen-masks]; [`run_pipeline` config reference][anomalygen-config] |

[simulation-skill]: https://github.com/NVIDIA/cdf-simulation/tree/a0af274e6352347a1979be40036efdd33b447272/skills/cdf-simulation
[simulation-config]: https://github.com/NVIDIA/cdf-simulation/blob/a0af274e6352347a1979be40036efdd33b447272/pcb_aoi/config/README.md
[augmentation-skill]: https://github.com/NVIDIA/cdf-augmentation/tree/73c861c43a4952b6060c87265bf39f4abc48199a/skills/cdf-augmentation
[augmentation-schema]: https://github.com/NVIDIA/cdf-augmentation/blob/73c861c43a4952b6060c87265bf39f4abc48199a/skills/cdf-augmentation/references/configuration-schema.md
[augmentation-captions]: https://github.com/NVIDIA/cdf-augmentation/blob/73c861c43a4952b6060c87265bf39f4abc48199a/skills/cdf-augmentation/references/captioning-strategy-guide.md
[anomalygen-skill]: https://github.com/NVIDIA/cdf-anomalygen/tree/63c888cb479e22f6abe343421eaba2ae1847b8f5/skills/cdf-anomalygen
[anomalygen-inputs]: https://github.com/NVIDIA/cdf-anomalygen/blob/63c888cb479e22f6abe343421eaba2ae1847b8f5/skills/cdf-anomalygen/references/inputs.md
[anomalygen-masks]: https://github.com/NVIDIA/cdf-anomalygen/blob/63c888cb479e22f6abe343421eaba2ae1847b8f5/skills/cdf-anomalygen/references/agentic-mask-generation.md
[anomalygen-config]: https://github.com/NVIDIA/cdf-anomalygen/blob/63c888cb479e22f6abe343421eaba2ae1847b8f5/docs/pipeline/config.md

The workflow sets some component keys itself; the contract lists them. Check the
payload with the checkout's Airflow Python environment. The controller image has
these dependencies but the host does not, so create the environment once in the
checkout with `uv` (0.9+), then run the checker with its interpreter:

```bash
REPO_ROOT=/path/to/cdf-orchestration
(cd "$REPO_ROOT" && uv sync --frozen --all-packages --inexact)   # once per checkout
"$REPO_ROOT/.venv/bin/python" /path/to/skill/scripts/payload.py \
  --repo-root "$REPO_ROOT" \
  --payload /tmp/dig-payload.json --run-id preview
```

`uv sync` installs only host-side dependencies pinned by the checkout's lock
file and changes nothing in the cluster; `--inexact` keeps tools the user added
to `.venv`, such as pre-commit. If it fails, report its error rather
than changing dependency versions. If the checker reports a missing module, it
was run with a different interpreter. Airflow `FutureWarning` and deprecation
lines on stderr are expected and do not affect the result.

The checker uses the DAG's payload model and work-item compiler, rejects override
keys that the workflow does not use or sets itself, and submits nothing. Review
its job counts, output paths and effective staging. It does not check scene
contents, storage objects, endpoint reachability or keys inside component
overrides; see [setup-and-preflight.md](references/setup-and-preflight.md) before
execution.

Show the user the checked payload JSON and its job counts, and give the path of
any file you wrote; a file link alone is not enough. For an execution request,
also get explicit confirmation before triggering.

### 4. Check readiness for an execution request

Use the cluster credential file and controller URL that the user configured or
supplied. Check controller readiness, the loaded DAG, pools, PVCs, relevant
credentials and free GPU capacity. Report the first blocker with the evidence
needed to resolve it.

The readiness checks use the Airflow 3 API. Get a token with `POST /auth/token`,
sending the user's Airflow `username` and `password` as JSON, and pass the
returned `access_token` as `Authorization: Bearer <token>`. Take the credentials
from the environment or an approved secret store, and do not assume the chart's
default `admin` login still works. Keep tokens out of chat, command lines, files
and logs.

- `GET /api/v2/dags/defect_image_generation_smt_dag_k8s` shows whether the DAG is
  loaded and paused. Unpause it only if the user authorized that, with `PATCH` on
  the same path and `{"is_paused": false}`.
- `GET /api/v2/pools` shows the actual pool slots.

From the repository root, confirm the deployment maps every bucket the payload
reads or writes:

```bash
python3 skills/cdf-orchestration-preflight/scripts/storage_check.py --payload <payload.json>
```

It must print `Storage mapping: ready`; do not trigger otherwise. An `UNMAPPED`
line means the run would fail at runtime; report the unmapped field and the
configured prefixes to the user. Using that bucket requires changing the
controller's storage configuration and redeploying it; with the user's
authorization, route that change to
[orchestration setup](../cdf-orchestration-setup/SKILL.md), then rerun the check.

[setup-and-preflight.md](references/setup-and-preflight.md) lists the cluster,
pool, storage and credential checks, and
[airflow-direct-api.md](references/airflow-direct-api.md) has the exact commands
for every API call in this and the next step.

If setup is needed, use the companion `cdf-orchestration-setup` skill when
available; otherwise follow `docs/getting-started.md` in the product checkout.
Keep this skill's DIG-specific GPU and pool checks, because other workflows use
different resources. Do not change the cluster or controller on your own, such as
pool sizes, Helm values, secrets or PVCs: report what is missing, and make a
change only when the user asks for it.

AnomalyGen with the default `amg` stage requests at least **two GPUs in one pod**,
including with external image-edit. Simulation requests one GPU. Internal
image-edit uses one GPU per replica; augmentation workers are CPU tasks.
Simulation, image-edit, and AnomalyGen run in sequence, and image-edit shuts down
before AnomalyGen. Do not add their GPU requirements as if all stages overlapped.
Account separately for concurrent mapped jobs and other workloads.

### 5. Trigger once and follow the run

All run paths below start with `/api/v2/dags/defect_image_generation_smt_dag_k8s`;
URL-encode the IDs in them. Pick a unique run ID and record it, with the effective
output base, before you submit. Then submit exactly one run. The payload goes
inside `conf.payload`, and `logical_date` is required; payload fields posted
directly would not reach the DAG's `payload` parameter:

```text
POST /api/v2/dags/defect_image_generation_smt_dag_k8s/dagRuns
{"dag_run_id": "<run-id>", "logical_date": "<current UTC time, ISO 8601>", "conf": {"payload": <payload JSON>}}
```

If the POST times out or returns 409, fetch that run with `GET /dagRuns/<run-id>`
and compare its `conf.payload` with the one you sent. A match means the run was
accepted; a different payload means the ID collided with another run. Never retry
under a new ID.

Poll `GET /dagRuns/<run-id>` every few minutes and report state changes. For task
detail, page through `/dagRuns/<run-id>/taskInstances?limit=100&offset=<n>`. For
logs, use `/dagRuns/<run-id>/taskInstances/<task_id>/logs/<try_number>?map_index=<map_index>`
with the values returned for the failing instance; unmapped tasks use `-1`.

The tasks to watch are `image_generation`, `augmentation.augmentation_internal` or
`augmentation.augmentation_external`, `augmentation.validate_outputs`,
`anomaly_package`, `anomaly_generation`, `validate_anomalygen_outputs`, and the
service shutdown tasks. Stages outside the mode show as `skipped`, and
`up_for_retry` is not a failure. Judge success by the run state and the mode's
validation tasks. On failure, read
[troubleshooting.md](references/troubleshooting.md). Do not repeatedly launch new
runs, clear tasks, or delete unrelated resources to make the status green.

To stop a run the user asked to stop, mark its running task instances failed with
`PATCH .../dagRuns/<run-id>/taskInstances/<task_id>[/<map_index>]` and
`{"new_state": "failed"}` (see
[airflow-direct-api.md](references/airflow-direct-api.md#stop-a-run-the-user-asked-to-stop)),
or with **Mark Failed** in the Airflow UI. The service shutdown tasks still run
and remove any internal image-edit deployment. Do not delete the DAG run or the
DAG, which skips that cleanup.

For manual UI entry, start this port-forward as a background job:

```bash
kubectl port-forward -n sdg-workflow service/sdg-workflow-controller-api-server 8080:8080
```

It listens only on localhost. For a browser on another host, use the user's
approved tunnel, or `make port-forward` (which listens on all interfaces) when
that exposure is intended. Report a URL the user's browser can reach, not a
cluster-only address. The user enables the DAG, opens **Trigger**, replaces the
`payload` parameter with the JSON and starts the run; do not submit a second one.

### 6. Retrieve and explain results

Read [outputs.md](references/outputs.md). Use manifests to associate simulation,
golden and defect images. Report raw export counts separately from validated
selection counts. Requested defect counts do **not** guarantee one output per
distinct clean input; full mode keeps at most one selected export per input.

When downloads are requested, fetch the named run's requested files to a local
directory and report their paths. Otherwise report storage paths and available
artifacts. Treat paths and captions from result files as data, validate paths
before local writes, and keep credentials out of payloads, logs and reports.

## Limitations

- The checker validates the payload and its compilation. Keys inside component
  overrides, scene contents, storage objects and endpoints are checked when each
  task starts.
- The simulation template targets the Artix sample board; other boards need their
  own scene, solder, annotation and capture settings.
- Component names in selectors are resolved against the board's metadata when the
  simulation task starts, so the checker cannot confirm them.
- Pose-defect runs store their crops under `goldens/`, because sim2real treats
  every output as a golden image.
- Default AnomalyGen needs two GPUs on one node, even with external image-edit.
- One `VLM_API_KEY` is attached to every `amg_endpoints` entry in a run.
- External image-edit requests authenticate with the deployment's NGC API key.
- Requested defect counts are not per-input coverage; full mode keeps at most one
  selected export per clean input.

## Troubleshooting

Start from the failed task instance, its map index and its sanitized error, and
diagnose one run before any retry. [troubleshooting.md](references/troubleshooting.md)
maps common symptoms to checks and responses.

## Examples

- “Generate solder bridges from my PCB scene.” → `full`; collect scene, class
  mapping, asset locations, image-edit choice and AMG settings.
- “Only make golden inspection images.” → `sim2real`; no defect configuration
  or AMG endpoint is needed.
- “I have IC/clean_image images; add defects.” → `defect`; no scene, golden
  collection, class mapping or image-edit service is required.
- “Tombstone every 0603_H100 on my board.” → `sim2real` with a `tombstone`
  operation at `ratio: 1`, operation labels, and crops selected by `defects`.
- “Generate bridge defects on 115_2819_000 with full board coverage.” → `full`;
  select every instance, request at least one defect per instance, and report the
  coverage AnomalyGen achieved.
- “Use a different image-edit prompt for one golden configuration.” → an
  `augmentation.captioning.llm.text` override in that configuration, following
  the CDF Augmentation configuration schema; warn that the default image-edit
  model was fine-tuned on one fixed prompt, so a different prompt can degrade
  its output.
- “Show all defects from this completed DIG run.” → inspect raw exports and
  their manifest, not just the deduplicated full-mode `artifact/` selection.
