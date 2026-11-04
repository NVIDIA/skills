---
name: cdf-event-video-generation
description: Run the CDF Orchestration Event Video Generation DAG on Kubernetes or OSMO - image-to-video anomaly generation, auto-labeling, and anomaly dataset generation. Select for requests about event video generation, anomaly video generation, image-to-video synthesis, Cosmos3 image2video, anomaly dataset creation, safety/surveillance SDG, or generating person-falling, person-climbing, person-running, fighting, smoking/vaping, fire/smoke, or shoplifting video clips from a seed image. Covers both event_video_generation_dag_k8s and the external-endpoints-only event_video_generation_dag_osmo. Runs environment setup first when controller readiness is unknown. Not for person-crop clothing/attribute augmentation (that is cdf-image-attribute-augmentation) and not for video style transfer.
version: "1.0.2"
license: CC-BY-4.0 AND Apache-2.0
metadata:
  owner: NVIDIA
  service: cosmos-data-factory
  version: "1.0.2"
  reviewed: "2026-10-09"
  author: NVIDIA
  tags:
    - physical-ai
    - cdf-orchestration
    - event-video-generation
    - cosmos
---

# CDF Orchestration — Event Video Generation

Run the Event Video Generation DAG end to end: seed-image input preparation, Cosmos3
image-to-video anomaly augmentation, auto-labeling (detection and tracking, captioning,
anomaly visual QA, person-attribute visual QA, person attribute search), anomaly dataset
generation, and result retrieval.

## Component details

This workflow skill and its references are the primary guide for configuring and running the
workflow. Consult a component skill only when the requested configuration or diagnosis requires
component-specific details not covered here. Routine payload authoring with documented fields,
preflight, triggering, monitoring, and result retrieval do not require component skill access.

When additional component guidance is needed, consult that component's documentation, and do not
invent keys or values. The links below are pinned to the latest `main` commit of each component
repository as of 2026-10-20; a component skill installed separately may describe a different
version.

| Component | Covers | Documentation at that commit |
|---|---|---|
| `cdf-augmentation` | Image-to-video configuration, generation prompts, model parameters, evaluators, and augmentation errors | [`cdf-augmentation` skill][augmentation-skill] |
| `cdf-auto-labeling` | Detection/tracking, captioning, visual QA, question banks, attribute-search configuration, and auto-labeling errors | [`cdf-auto-labeling` skill][auto-labeling-skill] |

[augmentation-skill]: https://github.com/NVIDIA/cdf-augmentation/tree/73c861c43a4952b6060c87265bf39f4abc48199a/skills/cdf-augmentation
[auto-labeling-skill]: https://github.com/NVIDIA/cdf-auto-labeling/tree/6c75c76c52a6c76c3866a38d30885e11123b0981/skills/cdf-auto-labeling

Use only documentation at these commits; do not substitute a moving branch or another version.

If the repository is inaccessible, continue tasks supported by this workflow's guidance. Identify
the unavailable reference and the unresolved component-specific detail only when that missing
guidance prevents completing the requested work; do not invent component settings or treat
component access as a prerequisite for ordinary workflow use.

Apply component guidance through this workflow's supported payload fields and DAG constraints.
Component configuration and the Airflow trigger payload are different contracts; component
capabilities do not automatically become supported DAG options. Keep execution through this
workflow's existing procedure. If the pinned guidance differs from the deployed component version
or DAG integration, report the mismatch before proposing the affected settings.

## DAG selection

The workflow builds one DAG per compute platform from
`airflow/dags/workflows/event_video_generation_dag/`:

| Platform | DAG ID | Manifest |
|---|---|---|
| Kubernetes | `event_video_generation_dag_k8s` | `event_video_generation_k8s_manifest.yaml` |
| OSMO | `event_video_generation_dag_osmo` | `event_video_generation_osmo_manifest.yaml` |

Both manifests are checked in, so both DAGs are registered. A DAG is registered only if its
manifest exists; a missing manifest means the DAG is absent from Airflow rather than broken. List
the DAGs Airflow actually loaded before triggering, and never name a DAG ID that is not in that
list.

**`event_video_generation_dag_osmo` supports external inference endpoints only.** Its manifest
defines OSMO batch tasks but no VLM/image-to-video/Cosmos-Evaluator service components, so an
internal-services payload cannot deploy what it needs:

- Always set top-level `external_services: true` (and `cosmos.external_services` to `true`, or
  omit so it inherits the top-level value). Any OSMO payload with `external_services: false` is
  invalid for the deployed manifest even though the shared payload schema can parse it — never
  submit one.
- Supply `cosmos.vlm_service_url`, `cosmos.llm_service_url`, and `cosmos.image2video_service_url`.
- Leave `service_lifecycle.vlm_service.enabled` / `service_lifecycle.image2video_service.enabled`
  as `false`, or omit `service_lifecycle` entirely.
- The OSMO DAG has no managed Cosmos Evaluator either — if any quality check stays enabled, supply
  an external Arbitrator URL via `cosmos.cosmos_evaluator_service_url` reachable from OSMO tasks,
  or explicitly disable both checks. Never reuse a K8s-DAG internal-mode starter payload for the OSMO DAG.
- Pools differ from the K8s DAG. Derive them from the OSMO manifest rather than assuming: in the
  currently checked-in `event_video_generation_osmo_manifest.yaml` the OSMO tasks
  (`detection_and_tracking`, `captioning`, `visual_qa`, `augmentation`) run on the `osmo_rtx6000`
  profile → pool `osmo_rtx6000_1`, while `event_and_person_attribute_search` stays on
  `k8s_cpu_task` → `default_pool`. (`osmo_h100_1` exists in `deploy/values.yaml` but this manifest
  does not reference it.) The OSMO manifest's `components.endpoints` key is absent — that is the
  structural reason internal mode cannot work here.

**Check `OSMO_TOKEN`, `OSMO_URL` and `OSMO_POOL` before triggering an OSMO run — and only
then.** They are optional at deploy time, so a perfectly healthy controller can be running with
them empty: `make setup` does not validate them and the chart renders empty values, so the DAG
registers and triggers normally and only fails once a task tries to reach OSMO. Do not raise this
during environment setup or for Kubernetes runs; it is a precondition of the OSMO path
specifically. Verify against the cluster (authoritative — local deployment configuration may have
changed since the last deploy):

```bash
(set -o pipefail
kubectl get secret -n sdg-workflow osmo-token-secret -o json | python3 -c '
import base64, json, sys
data = json.load(sys.stdin).get("data", {})
for key, name in (("osmo_token", "OSMO_TOKEN"), ("osmo_url", "OSMO_URL"), ("osmo_pool", "OSMO_POOL")):
    print(f"{name}: " + ("set" if base64.b64decode(data.get(key, ""), validate=True) else "unset"))
')
# A kubectl or decoding error is a failed check; never print the values or their lengths.
```

If any is unset, stop before triggering. Tell the user which settings must be configured and the
controller redeployed so the secret is recreated and the pods pick them up, and route that
configuration to the environment-setup skill. Do not set, write, or handle these values yourself.

There is a **single end-to-end pipeline per platform** — there are no generation-only or
labeling-only DAG variants. If a user asks for video generation without auto-labeling, tell them
the checked-in DAGs do not offer that flow rather than inventing a DAG ID.

## Manual payload entry in the Airflow UI

If the user wants to enter their own payload directly in the Airflow UI rather than have you
construct and trigger one, your job is limited to getting them to the UI: confirm controller
readiness, ensure `make port-forward` is running (see
[airflow-direct-api.md](references/airflow-direct-api.md#port-forward-ui-access)), and report the
reachable URL. Do not render a payload, run preflight, or trigger a run yourself in this case —
the user is doing that from the UI. Resume monitoring (step 6 below) once they tell you a run has
been triggered; you can find it via the Airflow API without needing the payload they used.

## Scope

**Before building any payload, collect all of the following from the user.** Never invent an
endpoint URL or bucket path — use only values the user supplied.

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

If the user does not provide a required value, ask for it explicitly before proceeding. Never
invent one. Values the user explicitly supplied earlier, including for a previous run, may be reused
when clearly applicable to the current target and request. Ask for clarification when the target
or context is ambiguous.

**Always run the following readiness checks before triggering a run.** The checks are
short-circuiting — stop at the first failure and route to the environment-setup skill immediately.

**Before any check**, establish the cluster connection. The cluster is reached only through
credentials the user supplies — they are never part of the repository. Check whether the cluster
credential file path is already exported in the shell environment; if not, ask the user for the
absolute path before running any cluster command. Never assume a path or fall back to any on-disk
default — see [setup-and-preflight.md](references/setup-and-preflight.md#cluster-access) for the
full procedure.

For Kubernetes, check GPU capacity on the configured task cluster. For OSMO, check the
compute pool selected by the OSMO manifest; Kubernetes GPU availability does not establish it.

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
   Kubernetes ClusterIP. That works from a cluster node; from a workstation, fall back to a
   port-forward as described in
   [airflow-direct-api.md](references/airflow-direct-api.md#establish-airflow_url-and-authenticate):

   ```bash
   AIRFLOW_URL="http://$(kubectl get svc -n sdg-workflow \
     sdg-workflow-controller-api-server \
     -o jsonpath='{.spec.clusterIP}'):8080"
   ```

   Then confirm the target DAG is loaded and `is_paused: False`. See
   [airflow-direct-api.md](references/airflow-direct-api.md) for the full auth + check sequence.
   If the API is unreachable, route to the environment-setup skill.

3. **Pools** — only if check 2 passes. **Derive the required pool names from the manifest for the
   platform you are running; do not assume them.** Pool names are deployment-specific: they come
   from the manifest's `deployment.profiles.*.pool` and are created by `deploy/values.yaml`'s
   `airflowPools.pools`, so a customer on different hardware or with renamed pools will not match
   whatever this repository ships today. See
   [airflow-direct-api.md](references/airflow-direct-api.md) for the derivation snippet, which
   also adds the mode-dependent image2video pool the DAG selects at runtime
   (`external_image2video_service_pool` for external mode, `default_pool` for internal — set in
   the DAG's `CosmosTaskGroup(...)` wiring, not in the manifest). On the currently checked-in K8s
   manifest this resolves to `default_pool`, `k8s_rtx6000_1`, and
   `external_image2video_service_pool`; on the OSMO
   manifest it is `osmo_rtx6000_1`, `default_pool`, and `external_image2video_service_pool`.
   Every derived pool must exist with open slots.

4. **Compute capacity** — for OSMO, check the pool described in
   [setup-and-preflight.md](references/setup-and-preflight.md#osmo-capacity), then continue to
   step 5. For Kubernetes, check the cluster using the configured connection:

   ```bash
   # Per node: allocatable GPUs minus GPUs requested by pods in every namespace.
   kubectl get nodes -o json > /tmp/evg-nodes.json
   kubectl get pods -A --field-selector=status.phase!=Succeeded,status.phase!=Failed \
     -o json > /tmp/evg-pods.json
   python3 - <<'PY'
   import json
   alloc = {n["metadata"]["name"]: int(n["status"].get("allocatable", {}).get("nvidia.com/gpu", 0))
            for n in json.load(open("/tmp/evg-nodes.json"))["items"]}
   used = dict.fromkeys(alloc, 0)
   for p in json.load(open("/tmp/evg-pods.json"))["items"]:
       node = p["spec"].get("nodeName")
       if node in used:
           for c in p["spec"].get("containers", []):
               used[node] += int(c.get("resources", {}).get("requests", {}).get("nvidia.com/gpu", 0))
   for node in sorted(alloc):
       print(f"{node}: {alloc[node] - used[node]} free / {alloc[node]} allocatable")
   print("cluster free GPUs:", sum(alloc.values()) - sum(used.values()))
   PY
   ```

   The compute cluster is **shared** — other users' runs and other namespaces may hold GPUs. Report
   GPUs as free-versus-total from this calculation, not just allocatable; pending pods hold no node
   yet and will compete for the next free GPU. Size the run as resident endpoint GPUs plus at
   least one batch GPU, because the batch stages run one at a time:
   - **Batch task pods, in both service modes**: `augmentation` (`augmentation_task` profile) and
     `detection_and_tracking`, `captioning`, and `visual_qa` (`k8s_gpu_task`) request 1 GPU per
     pod; `event_and_person_attribute_search` is CPU-only. Do not assume augmentation is
     CPU-profiled here — it is in IAA, but not in EVG. The DAG runs augmentation first, then the
     auto-labeling stages in a `chain`, so only one stage is active at a time. One free GPU lets
     the run progress; more GPUs only widen that stage's mapped-pod fan-out, bounded by pool slots.
   - **Cosmos Evaluator checkers, in both service modes by default**: the default evaluator
     configuration enables both checks, and the DAG deploys
     `cosmos_evaluator_attribute_verification` and `cosmos_evaluator_physical_plausibility` in
     cluster at 1 GPU each regardless of `external_services` — **2 GPUs**, resident until the
     validated output is written. Postgres, database, and arbitrator are CPU-only. The checkers are
     skipped only when the payload supplies `cosmos.cosmos_evaluator_service_url` (an existing
     Arbitrator) or disables the checks with `cosmos.checker_overrides`.
   - **Internally deployed endpoints** (`external_services: false`): `service_lifecycle` deploys
     two services, not three — `vlm_service` at 1 GPU per replica (the same Qwen serves the
     text-only traffic, so there is no separate `llm_service`) plus `image2video_service` at **two**
     GPUs per replica (`gpu_count: 2`, `host_ipc: true`). `image2video_service` is torn down after
     augmentation so auto-labeling can reuse its GPUs.
   - **Floors on the checked-in manifest** (peak during augmentation): external mode needs **three
     GPUs by default** (two checkers plus one batch GPU), or one with an external evaluator or the
     checks disabled. Internal mode needs **six by default** (three endpoint GPUs, two checkers,
     one batch GPU), or four with an external evaluator or the checks disabled.
   - Derive all of this from the manifest rather than trusting these numbers: resolve each
     component's `deployment_profile` against `deployment.profiles`, count profiles with
     `configuration.gpu: true`, and sum `gpu_count` where set (default 1).

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
   - `external` requires explicit VLM, LLM, and image2video endpoint URLs. The LLM and VLM URLs may
     legitimately differ here — only *internal* mode collapses them onto one endpoint.
   - `internal` lets the DAG's service lifecycle deploy the services in-cluster. It deploys **two**
     endpoints, not three: `vlm_service` and `image2video_service`, with the single Qwen VLM also
     serving the text-only LLM traffic (the payload model sets
     `cosmos.llm_service_url = cosmos.vlm_service_url` in internal mode). There is no separate
     `llm_service` to deploy.
   - Choose service mode independently from controller placement. A local controller may use
     external inference endpoints.
   - Keep nested service mode and output directory consistent with the top level.
   - Size the choice against the GPU footprint derived in readiness check 4: neither mode is
     zero-GPU, and internal mode adds the deployed endpoints on top of the task pods. See
     [setup-and-preflight.md](references/setup-and-preflight.md#gpu-requirements) for the full
     per-component breakdown, and re-derive it from the manifest rather than reusing a remembered
     total.

3. Read [payload-contract.md](references/payload-contract.md), then render a payload from the
   values collected above. Every deployment-specific value — endpoint URLs, bucket paths, input
   paths — must be one the user supplied; never invent one.

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
   Confirm presence only; never print credential values. Then, from the repository root, confirm
   the deployment maps every bucket the payload reads or writes:

   ```bash
   python3 skills/cdf-orchestration-preflight/scripts/storage_check.py --payload <payload.json>
   ```

   It must print `Storage mapping: ready`. An `UNMAPPED` line means the run would fail at
   runtime; report the unmapped field and the configured prefixes to the user. Using that
   bucket requires changing the controller's storage configuration and redeploying it; with the
   user's authorization, route that change to
   [orchestration setup](../cdf-orchestration-setup/SKILL.md), then rerun the check.

5. Submit exactly one DAG run. Pass the payload from step 3 as `conf.payload` — see
   [airflow-direct-api.md#trigger-a-run](references/airflow-direct-api.md#trigger-a-run) for the
   full request shape. Record and return the `dag_run_id`, input path, output directory, and
   service mode.

   Before monitoring, read back the run's `conf` and confirm the DAG received the submitted
   payload. A payload placed at the top level of `conf` instead of under `conf.payload` is
   silently ignored, and the run then "succeeds" on placeholder defaults:

   ```bash
   RESPONSE=$(curl -s -H "Authorization: Bearer $TOKEN" \
     "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns/$DAG_RUN_ID")
   RESPONSE="$RESPONSE" python3 -c "
   import json, os
   received = (json.loads(os.environ['RESPONSE']).get('conf') or {}).get('payload')
   submitted = json.load(open('/tmp/evg-payload.json'))
   print('payload received' if received == submitted else 'PAYLOAD MISMATCH')
   "
   ```

   On a mismatch, do not treat the run as valid: report it to the user and ask before stopping
   the run (see step 6) and re-triggering with the payload under `conf.payload`.

6. Immediately after triggering — without waiting to be asked — monitor the run until it reaches
   a terminal state (`success` or `failed`). Poll the Airflow API every 60–120 seconds, with an
   upper bound so a run that never leaves `queued` or `running` is reported instead of polled
   forever:

   ```bash
   # Poll run state for up to 12 hours (720 polls x 60 s); raise MAX_POLLS for very large runs.
   MAX_POLLS=720
   for i in $(seq 1 "$MAX_POLLS"); do
     RESPONSE=$(curl -s --max-time 30 -H "Authorization: Bearer $TOKEN" \
       "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns/$DAG_RUN_ID")
     STATE=$(RESPONSE="$RESPONSE" python3 -c "import json, os; print(json.loads(os.environ['RESPONSE']).get('state', 'unknown'))" 2>/dev/null || echo "unreachable")
     echo "$(date -u +%H:%M:%S) $STATE"
     case "$STATE" in success|failed) break ;; esac
     sleep 60
   done
   ```

   `unreachable` or `unknown` means the request failed (expired token, timeout, lost
   port-forward); fix the connection rather than counting it as progress. If the loop ends without
   a terminal state, or the run stays `queued` for more than about 30 minutes, stop polling and
   report the state with the per-task breakdown below. In internal mode, GPU endpoints waiting for
   free GPUs are the usual cause.

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

   To stop an in-progress run: open the Airflow UI, find the active DagRun, locate the running
   task, and mark it **Failed** (task menu → Mark Failed). This triggers the DAG's shutdown path,
   cleaning up Deployments, Services, and GPU pods. Do not delete the DagRun or the DAG — that
   bypasses cleanup and leaves stale cluster resources.

7. After the run reaches `success` or `failed`, download and analyze results if the user has
   already requested or approved that action, respecting any condition such as "only on success".
   Otherwise ask **"Would you like to download and analyze the results?"** and wait for confirmation.

   Once authorized, use whatever AWS credentials are already available in the shell
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

- **Never invent endpoint URLs, bucket paths, or input paths — use only values the user
  supplied.** Always ask for every deployment-specific value before building a payload. If a
  required value is missing, stop and ask — do not substitute a guess.
- Preserve explicit user inputs and endpoint/model selections throughout the session.
- Do not submit if payload validation, local dataset validation, the storage mapping check, or
  Airflow preflight fails.
- Do not show credential values or S3 signed URLs. For credential diagnostics, report presence
  only (set/unset), including `AIRFLOW_USER` and `AIRFLOW_PASS` when applicable. Do not disclose
  or offer lengths, prefixes, suffixes, hashes, or commands that echo secrets. A request solely
  to display or check credential presence does not authorize authentication or token creation;
  authentication remains part of authorized workflow operations.
- Do not start multiple runs unless the user explicitly requests them.
- Ask for a dataset location if none was supplied; this workflow has no implicit demo dataset.
- Do not invent generation-only or labeling-only DAG IDs — only the DAGs listed above exist.
- Only offer a platform whose manifest exists and whose DAG is loaded in Airflow.
- Do not route person-crop clothing/attribute augmentation requests here — that is
  `cdf-image-attribute-augmentation`.

## References

- Read [setup-and-preflight.md](references/setup-and-preflight.md) for environment, storage, and
  policy requirements.
- Read [payload-contract.md](references/payload-contract.md) when creating or changing a payload.
- Read [outputs.md](references/outputs.md) when retrieving or interpreting results.
- Read [troubleshooting.md](references/troubleshooting.md) after validation, API, or runtime errors.
- Read [airflow-direct-api.md](references/airflow-direct-api.md) for all Airflow interactions:
  preflight, triggering, monitoring, per-task breakdown, and log retrieval.
