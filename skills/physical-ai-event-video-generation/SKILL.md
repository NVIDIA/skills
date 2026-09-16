---
name: physical-ai-event-video-generation
description: Run the PAIDF Orchestration Event Video Generation DAG on Kubernetes - image-to-video anomaly generation, auto-labeling, and anomaly dataset generation. Select for requests about event video generation, anomaly video generation, image-to-video synthesis, Cosmos3 image2video, anomaly dataset creation, safety/surveillance SDG, or generating person-falling, person-climbing, person-running, fighting, smoking/vaping, fire/smoke, or shoplifting video clips from a seed image. Runs environment setup first when controller readiness is unknown. Not for person-crop clothing/attribute augmentation (that is image-attribute-augmentation-workflow) and not for video style transfer.
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
    - event-video-generation
    - cosmos
---

# PAIDF Orchestration — Event Video Generation

Run the Event Video Generation DAG end to end: seed-image input preparation, Cosmos3
image-to-video anomaly augmentation, auto-labeling (detection and tracking, captioning,
anomaly visual QA, person-attribute visual QA, person attribute search), anomaly dataset
generation, and result retrieval.

## DAG selection

The workflow builds one DAG per compute platform from
`airflow/dags/workflows/event_video_generation_dag/`:

| Platform | DAG ID | Manifest |
|---|---|---|
| Kubernetes | `event_video_generation_dag_k8s` | `event_video_generation_k8s_manifest.yaml` |

Kubernetes is the only platform whose manifest is checked in, so
`event_video_generation_dag_k8s` is the only DAG this repository registers. A DAG is
registered only if its manifest exists; a missing manifest means the DAG is absent from Airflow
rather than broken. List the DAGs Airflow actually loaded before triggering, and never name a DAG
ID that is not in that list.

There is a **single end-to-end pipeline** — there are no generation-only or labeling-only DAG
variants. If a user asks for video generation without auto-labeling, tell them the checked-in DAG
does not offer that flow rather than inventing a DAG ID.

## Manual payload entry in the Airflow UI

If the user wants to enter their own payload directly in the Airflow UI rather than have you
construct and trigger one, your job is limited to getting them to the UI: confirm controller
readiness, ensure `make port-forward` is running (see
[airflow-direct-api.md](references/airflow-direct-api.md#port-forward-ui-access)), and report the
reachable URL. Do not render a payload, run preflight, or trigger a run yourself in this case —
the user is doing that from the UI. Resume monitoring (step 6 below) once they tell you a run has
been triggered; you can find it via the Airflow API without needing the payload they used.

## Scope

**Before building any payload, collect all of the following from the user.** Do not fall back to
repository defaults, CI payloads, or any hardcoded endpoint URL or bucket path.

| Required | Field | What to ask |
|---|---|---|
| Always | `input_path` | S3 (or HTTP/HTTPS) URL to a single seed image or a directory of seed images |
| Always | `output_directory` | Writable S3 URL where results should be written |
| Always | service mode | `external` (user provides endpoint URLs) or `internal` (DAG deploys services in-cluster) |
| External mode | `cosmos.vlm_service_url` | Full HTTPS URL for the VLM inference endpoint |
| External mode | `cosmos.llm_service_url` | Full HTTPS URL for the LLM inference endpoint |
| External mode | `cosmos.image2video_service_url` | Full HTTPS URL for the Cosmos3 image-to-video inference endpoint |
| Optional | `max_images` | Number of images to process from a directory (default: 10; 0 or negative = all) |
| Optional | `cosmos.num_augmentation` | Anomaly videos generated per image (default: 1) |
| Optional | `cosmos.variable_distribution` | anomaly_type / env_type sampling distribution (see payload-contract.md) |

If the user does not provide a required value, ask for it explicitly before proceeding. Do not
invent or reuse values from previous runs or checked-in files.

**Always run the following readiness checks before triggering a run.** The checks are
short-circuiting — stop at the first failure and route to the environment-setup skill immediately.

**Before any check**, establish the cluster connection. The cluster is reached only through
credentials the user supplies — they are never part of the repository. Check whether the cluster
credential file path is already exported in the shell environment; if not, ask the user for the
absolute path before running any cluster command. Never assume a path or fall back to any on-disk
default — see [setup-and-preflight.md](references/setup-and-preflight.md#cluster-access) for the
full procedure.

The controller (Airflow) and DAG compute tasks run on the same cluster unless a different remote
cluster connection was configured. GPU capacity is checked on this cluster.

1. **Controller pods** — check that the Airflow controller pods (not DAG task pods) are Running.
   DAG task pods in `Pending` or `Failed` state are normal and must not be mistaken for controller
   failures:

   ```bash
   kubectl get pods -n sdg-workflow -l "release=sdg-workflow-controller"
   ```

   All pods matching the `release=sdg-workflow-controller` label must be `Running`. If the
   namespace is absent, this is a first-install condition — route to the environment-setup skill,
   do not diagnose further.

2. **Airflow API** — reachable only if check 1 passes. First establish `AIRFLOW_URL` from the
   Kubernetes ClusterIP (always routable from the host, no port-forward required):

   ```bash
   AIRFLOW_URL="http://$(kubectl get svc -n sdg-workflow \
     sdg-workflow-controller-api-server \
     -o jsonpath='{.spec.clusterIP}'):8080"
   ```

   Then confirm the target DAG is loaded and `is_paused: False`. See
   [airflow-direct-api.md](references/airflow-direct-api.md) for the full auth + check sequence.
   If the API is unreachable, route to the environment-setup skill.

3. **Pools** — only if check 2 passes. Required pools with open slots: `k8s_gpu_1`,
   `default_pool`, and the image2video pool for the chosen mode
   (`external_image2video_service_pool` for external, `internal_image2video_service_pool` for
   internal).

4. **Compute-cluster GPUs** — check the cluster (using the cluster connection established above):

   ```bash
   kubectl get nodes \
     -o custom-columns='NAME:.metadata.name,GPU_ALLOC:.status.allocatable.nvidia\.com/gpu'
   # Also check pods already consuming GPUs — capacity ≠ availability on a shared cluster
   kubectl get pods -n sdg-workflow \
     --field-selector=status.phase=Running -o wide
   ```

   The compute cluster is **shared** — other users' runs may be active. Report GPUs as
   free-versus-total, not just allocatable. Two independent GPU sources, only one of which is
   mode-dependent:
   - **Task pods that run local model inference regardless of service mode**:
     `detection_and_tracking`, `captioning`, and `visual_qa` all run on the `k8s_gpu_task` profile
     (1 GPU each) — this cost applies in **both** external and internal mode, since these do in-pod
     inference rather than calling an endpoint. `event_and_person_attribute_search` and
     `augmentation` run on CPU profiles and cost nothing. **External mode therefore needs a minimum
     of three GPUs**, not zero.
   - **Internally deployed endpoints** (`external_services: false`): one GPU per VLM/LLM replica,
     plus **two** GPUs per image2video replica (`gpu_count: 2`, `host_ipc: true`) — four GPUs for
     one replica of each service.
   - **Internal mode total = both sources combined**: the three task-pod GPUs plus the four
     endpoint GPUs — **at minimum seven GPUs**, not four.

5. **Stale failed pods** — before triggering, check for accumulated failed pods in the compute
   namespace and report them. They are retained by design and do not affect run correctness, but
   they consume namespace quota and clutter log searches:

   ```bash
   kubectl get pods -n sdg-workflow \
     --field-selector=status.phase=Failed \
     -o custom-columns='NAME:.metadata.name,AGE:.metadata.creationTimestamp,DAG:.metadata.labels.dag_id'
   ```

   Clean up only pods whose `dag_id` label matches a run you own, after confirming with the user.

Document each check result explicitly.

**If any check fails**: invoke the environment-setup skill automatically — do not wait for the user
to say "set up" or ask them to name the skill.

**If the user's request implies first-time or explicit deployment** ("deploy", "install", "set up",
"reinstall", "redeploy", "full setup"): invoke the environment-setup skill even if all checks
pass, and confirm the planned commands first.

**If all checks pass** and the user only wants to run the workflow: proceed directly to payload
and trigger.

## Bundled tools

- `scripts/upload_images.py`: validate/upload a local seed image or flat directory of seed images.
- `scripts/payload.py`: render or validate a standalone
  `EventVideoGenerationDagPayloadConfig`-compatible JSON.
- `scripts/summarize_results.py`: summarize a downloaded `anomaly_dataset/dataset.json` dataset.

Run commands from this skill directory. Credentials must be inherited from the shell that launched
the agent; never ask the user to paste secret values into the prompt.

## Procedure

1. Determine the input source.
   - For local data, validate before upload:

     ```bash
     python scripts/upload_images.py --path /path/to/seed-images --validate-only
     ```

   - Then upload:

     ```bash
     python scripts/upload_images.py \
       --path /path/to/seed-images --destination-path event-video-generation/my-run
     ```

   - For an existing storage URL, use it unchanged after confirming it names either a single
     image or a flat directory of images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`,
     `.webp`). Unlike person-crop workflows, there is no subdirectory convention — every matching
     file directly under `input_path` is one input image. When `input_path` names a directory,
     the DAG sorts matching files and takes the first `max_images` of them.

2. Select service mode.
   - `external` requires explicit VLM, LLM, and image2video endpoint URLs.
   - `internal` lets the DAG's service lifecycle deploy all three services in-cluster.
   - Choose service mode independently from controller placement. A local controller may use
     external inference endpoints.
   - Keep nested service mode and output directory consistent with the top level.
   - On Kubernetes, VLM and LLM each claim one GPU from `k8s_gpu_1`, but image2video claims
     **two** GPUs per replica (`gpu_count: 2`, `host_ipc: true`) — four GPUs for the internally
     deployed endpoints alone. That's on top of, not instead of, the three GPUs
     `detection_and_tracking`/`captioning`/`visual_qa` always claim from `k8s_gpu_task` regardless
     of service mode (see the readiness-check GPU breakdown above): **external mode needs a
     minimum of three allocatable GPUs, internal mode a minimum of seven** — not zero and four.

3. Read [payload-contract.md](references/payload-contract.md), then render a payload from the
   values collected above. Do not copy checked-in dev or CI payloads — they contain deployment-
   specific endpoint URLs and bucket paths that must not be inherited by user runs.

   External:

   ```bash
   python scripts/payload.py render \
     --input-path s3://bucket/input/seed-images/ \
     --output-directory s3://bucket/output/event-video-generation/ \
     --service-mode external \
     --vlm-url https://vlm.example/v1 \
     --llm-url https://llm.example/v1 \
     --image2video-url https://image2video.example/v1 \
     --max-images 10 --num-augmentation 3 \
     --variable-distribution assets/variable-distribution.json \
     --output /tmp/evg-payload.json
   ```

   Internal:

   ```bash
   python scripts/payload.py render \
     --input-path s3://bucket/input/seed-images/ \
     --output-directory s3://bucket/output/event-video-generation/ \
     --service-mode internal \
     --max-images 10 --num-augmentation 3 \
     --output /tmp/evg-payload.json
   ```

   Show the user the rendered payload (or its validated contents) and get explicit confirmation
   before proceeding. Only continue to preflight and triggering if they confirm; if they want
   changes, re-render and re-confirm.

4. Preflight the DAG through the Airflow API. Check that the DAG is loaded, required pools have
   slots, and controller pods are healthy — see
   [airflow-direct-api.md#preflight-direct-path](references/airflow-direct-api.md#preflight-direct-path).
   Confirm presence only; never print credential values.

5. Submit exactly one DAG run. Pass the payload from step 3 as `conf.payload` — see
   [airflow-direct-api.md#trigger-a-run](references/airflow-direct-api.md#trigger-a-run) for the
   full request shape. Record and return the `dag_run_id`, input path, output directory, and
   service mode.

6. Immediately after triggering — without waiting to be asked — monitor the run until it reaches
   a terminal state (`success` or `failed`). Poll the Airflow API every 60–120 seconds:

   ```bash
   # Poll run state
   RESPONSE=$(curl -s -H "Authorization: Bearer $TOKEN" \
     "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns/$RUN_ID")
   RESPONSE="$RESPONSE" python3 -c "import json, os; print(json.loads(os.environ['RESPONSE'])['state'])"
   ```

   For a per-task breakdown when state is `running` or `failed`, see
   [airflow-direct-api.md](references/airflow-direct-api.md#per-task-breakdown-useful-for-diagnosing-failures).

   Stop polling as soon as the run state is `success` or `failed`. Use the polling loop that
   fits your runtime — a shell `while` loop, a background process, or a tool-native scheduler.
   Do not block the user waiting for each poll; report state changes as they occur.

   Tell the user they can also watch progress live in the Airflow UI. `make port-forward` runs in
   the foreground and never exits, so start it as a background job — and prefer that the user runs
   it in their own terminal, since an agent-owned forward dies with the session. Resolve the host's
   real address rather than reporting a placeholder or `localhost`, which is meaningless from
   another machine:

   ```bash
   HOST_IP=$(hostname -I | awk '{print $1}')
   echo "Airflow UI: http://$HOST_IP:8080"
   ```

   Default credentials are `admin`/`admin`, defined in `deploy/values.yaml` under
   `airflow.createUserJob.defaultUser` (not `webserver.defaultUser`). Update them before
   production use.

   For a full per-task breakdown see
   [airflow-direct-api.md](references/airflow-direct-api.md#per-task-breakdown-useful-for-diagnosing-failures).

   To stop an in-progress run: open the Airflow UI, find the active DagRun, locate the running
   task, and mark it **Failed** (task menu → Mark Failed). This triggers the DAG's shutdown path,
   cleaning up Deployments, Services, and GPU pods. Do not delete the DagRun or the DAG — that
   bypasses cleanup and leaves stale cluster resources.

7. After the run reaches `success` or `failed`, ask the user:
   **"Would you like to download and analyze the results?"**
   Do not download automatically — wait for confirmation.

   If the user confirms, use whatever AWS credentials are already available in the shell
   environment (standard `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION`,
   an AWS profile, or instance role). Never ask the user to paste credentials into the prompt.
   Run artifacts live under `<output_directory>/<run_id>/`, where `<output_directory>` is the
   payload value and `<run_id>` is the `dag_run_id` from step 5. The final dataset is in
   `anomaly_dataset/`:

   ```bash
   aws s3 sync "<output_directory>/<run_id>/anomaly_dataset/" /tmp/evg-results/
   python scripts/summarize_results.py --results-dir /tmp/evg-results
   ```

   To inspect intermediate generated videos instead, sync `<output_directory>/<run_id>/cosmos/`
   and read `metadata.json` from each `<video_key>/<augmentation_index>/` folder.

   Read [outputs.md](references/outputs.md) before interpreting files.

## Guardrails

- **Never use default endpoint URLs, bucket paths, or input paths from the codebase or
  checked-in payloads.** Always ask the user for every deployment-specific value before building
  a payload. If a required value is missing, stop and ask — do not substitute a guess.
- Preserve explicit user inputs and endpoint/model selections throughout the session.
- Do not submit if payload validation, local dataset validation, or Airflow preflight fails.
- Do not show AWS credentials, Airflow bearer tokens, or S3 signed URLs.
- Do not start multiple runs unless the user explicitly requests them.
- Ask for a dataset location if none was supplied; this workflow has no implicit demo dataset.
- Do not invent generation-only or labeling-only DAG IDs — only the DAG listed above exists.
- Only offer a platform whose manifest exists and whose DAG is loaded in Airflow.
- Do not route person-crop clothing/attribute augmentation requests here — that is
  `image-attribute-augmentation-workflow`.

## References

- Read [setup-and-preflight.md](references/setup-and-preflight.md) for environment, storage, and
  policy requirements.
- Read [payload-contract.md](references/payload-contract.md) when creating or changing a payload.
- Read [outputs.md](references/outputs.md) when retrieving or interpreting results.
- Read [troubleshooting.md](references/troubleshooting.md) after validation, API, or runtime errors.
- Read [airflow-direct-api.md](references/airflow-direct-api.md) for all Airflow interactions:
  preflight, triggering, monitoring, per-task breakdown, and log retrieval.
