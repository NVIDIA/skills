---
name: cdf-image-attribute-augmentation
description: Run the CDF Orchestration Image Attribute Augmentation DAG on Kubernetes - person-crop clothing augmentation, attribute search, and augmented dataset generation. Select for requests about image attribute augmentation, person attribute search, person re-identification data, clothing augmentation, attribute captions, augmentation payloads, run status, or result retrieval. Runs environment setup first when controller readiness is unknown. Not for video or defect-image generation.
version: "1.0.1"
license: CC-BY-4.0 AND Apache-2.0
metadata:
  owner: NVIDIA
  service: cosmos-data-factory
  version: "1.0.1"
  reviewed: "2026-10-09"
  author: NVIDIA
  tags:
    - physical-ai
    - cdf-orchestration
    - image-attribute-augmentation
    - cosmos
---

# CDF Orchestration — Image Attribute Augmentation

Run the Image Attribute Augmentation DAG end to end: person-crop input preparation, cosmos
image-edit augmentation, cosmos post-processing, event and person attribute search, augmented
dataset generation, and result retrieval.

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
| `cdf-augmentation` | Image-edit configuration, clothing variables and distributions, captioning, verification, and augmentation errors | [`cdf-augmentation` skill][augmentation-skill] |
| `cdf-auto-labeling` | Event and person attribute-search configuration and errors | [`cdf-auto-labeling` skill][auto-labeling-skill] |

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
`airflow/dags/workflows/image_attribute_augmentation_dag/`:

| Platform | DAG ID | Manifest |
|---|---|---|
| Kubernetes | `image_attribute_augmentation_dag_k8s` | `image_attribute_augmentation_k8s_manifest.yaml` |

Kubernetes is the only platform whose manifest is checked in, so
`image_attribute_augmentation_dag_k8s` is the only DAG this repository registers. A DAG is
registered only if its manifest exists; a missing manifest means the DAG is absent from Airflow
rather than broken. List the DAGs Airflow actually loaded before triggering, and never name a DAG
ID that is not in that list.

There is a **single end-to-end pipeline** — there are no augmentation-only or labeling-only DAG
variants. If a user asks for augmentation without attribute search, tell them the checked-in DAG
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

**Before building any payload, collect all of the following from the user.** Never invent an
endpoint URL or bucket path — use only values the user supplied.

| Required | Field | What to ask |
|---|---|---|
| Always | `input_path` | S3 (or HTTP/HTTPS) URL whose immediate subdirectories are person-ID folders |
| Always | `output_directory` | Writable S3 URL where results should be written |
| Always | service mode | `external` (user provides endpoint URLs) or `internal` (DAG deploys services in-cluster) |
| External mode | `cosmos.vlm_service_url` | Full HTTPS URL for the VLM inference endpoint |
| External mode | `cosmos.llm_service_url` | Full HTTPS URL for the LLM inference endpoint |
| External mode | `cosmos.image_edit_service_url` | Full HTTPS URL for the image-edit inference endpoint |
| Optional | `max_imgs` | Number of person-ID folders to process (default: 1; 0 or negative = all) |
| Optional | `cosmos.num_augmentation` | Clothing variants per person (default: 1) |
| Optional | `cosmos.variable_distribution` | Clothing attribute distribution file path (see payload-contract.md) |

If the user does not provide a required value, ask for it explicitly before proceeding. Never
invent one. Values the user supplied earlier in the session, including for a previous run, may be
reused; list each reused value in the payload shown for confirmation in step 3, so the user
confirms it again before anything is submitted.

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

3. **Pools** — only if check 2 passes. **Derive the required pool names from the manifest; do not
   assume them.** Pool names are deployment-specific: they come from the manifest's
   `deployment.profiles.*.pool` and are created by `deploy/values.yaml`'s `airflowPools.pools`, so
   a customer on different hardware or with renamed pools will not match whatever this repository
   ships today. See
   [airflow-direct-api.md](references/airflow-direct-api.md) for the derivation snippet, which
   also adds the mode-dependent augmentation pool the DAG selects at runtime
   (`external_image_edit_service_pool` for external, `default_pool` for internal — set in the
   DAG's `CosmosTaskGroup(...)` wiring, not in the manifest). On the currently checked-in manifest
   this resolves to `default_pool`, `k8s_rtx6000_1`, and `external_image_edit_service_pool`.
   Every derived pool must exist with open slots.

4. **Compute-cluster GPUs** — check the cluster (using the cluster connection established above):

   ```bash
   # Per node: allocatable GPUs minus GPUs requested by pods in every namespace.
   kubectl get nodes -o json > /tmp/iaa-nodes.json
   kubectl get pods -A --field-selector=status.phase!=Succeeded,status.phase!=Failed \
     -o json > /tmp/iaa-pods.json
   python3 - <<'PY'
   import json
   alloc = {n["metadata"]["name"]: int(n["status"].get("allocatable", {}).get("nvidia.com/gpu", 0))
            for n in json.load(open("/tmp/iaa-nodes.json"))["items"]}
   used = dict.fromkeys(alloc, 0)
   for p in json.load(open("/tmp/iaa-pods.json"))["items"]:
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
   GPUs as free-versus-total from this calculation, not just allocatable. Pending pods are not
   counted because they hold no node yet; mention any that are waiting for GPUs, since they will
   compete for the next free one. External mode needs no GPUs for inference — every task
   pod (`augmentation`, `cosmos_post_processing`, `event_and_person_attribute_search`) runs on a
   CPU profile, unlike EVG's `k8s_gpu_task`-profiled auto-labeling stages.

   **Internal mode deploys two GPU endpoints, not three.** `service_lifecycle` has exactly two
   services — `vlm_service` and `image_edit_service` — because one deployed Qwen serves both the
   multimodal and the text-only traffic: in internal mode the payload model assigns
   `llm_service_url = vlm_service_url` and `llm_model = vlm_model`. There is no separate
   `llm_service` endpoint to deploy. So one replica of each needs **two** allocatable GPUs, more
   only if a `replicas` value is raised. Derive this from the manifest rather than trusting this
   number — count components whose `deployment_profile` resolves to a profile with
   `configuration.gpu: true`, summing `gpu_count` where set (default 1).

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

- `scripts/upload_images.py`: validate/upload local `<person_id>/<image>.(jpg|jpeg|png)` data.
- `scripts/payload.py`: render or validate a standalone
  `ImageAttributeAugmentationDagPayloadConfig`-compatible JSON.
- `scripts/payload_schema.py`: payload field allowlists and defaults imported by `payload.py`. It is
  a module, not an entry point; do not run it directly.
- `scripts/summarize_results.py`: summarize a downloaded `augmented_data.json` dataset.
- `scripts/workflow.py`: drive the SDG webserver API — submit a run, poll its status, retrieve
  results, or cancel a single named run by ID (cancels only that run; does not touch cluster
  resources or other runs). Requires `WEBSERVER_ENDPOINT` and `NGC_API_KEY`. Prefer the Airflow
  API path below for normal operation.

Run the bundled `scripts/` commands from this skill directory; they need only Python and, for
uploads, `boto3`. The Airflow pool check reads the DAG manifest, so it needs the CDF Orchestration
checkout: set `REPO_ROOT` to it, as shown in
[airflow-direct-api.md#preflight-direct-path](references/airflow-direct-api.md#preflight-direct-path).
Credentials must be inherited from the shell that launched the agent; never ask the user to paste
secret values into the prompt.

## Procedure

1. Determine the input source.
   - For local data, validate before upload:

     ```bash
     python scripts/upload_images.py --path /path/to/crops --validate-only
     ```

   - Then upload while preserving the hierarchy:

     ```bash
     python scripts/upload_images.py \
       --path /path/to/crops --destination-path image-attribute-augmentation/my-run
     ```

   - For an existing storage URL, use it unchanged after confirming it contains person-ID
     subdirectories. Each immediate subdirectory of `input_path` is treated as one person ID, and
     its images are combined into a single horizontal strip per person.

2. Select service mode.
   - `external` requires explicit VLM, LLM, and image-edit endpoint URLs. The LLM and VLM URLs may
     legitimately differ here — only *internal* mode collapses them onto one endpoint.
   - `internal` lets the DAG's service lifecycle deploy the services in-cluster. It deploys **two**
     endpoints, not three: `vlm_service` and `image_edit_service`, with the single Qwen VLM also
     serving the text-only LLM traffic.
   - Choose service mode independently from controller placement. A local controller may use
     external inference endpoints.
   - Keep nested service mode and output directory consistent with the top level.
   - On Kubernetes each deployed endpoint claims one GPU from the manifest's GPU pool, so internal
     mode needs at least **two** allocatable GPUs (more if any `replicas` value is raised);
     external mode needs none for inference.

3. Read [payload-contract.md](references/payload-contract.md), then render a payload from the
   values collected above. Every deployment-specific value — endpoint URLs, bucket paths, input
   paths — must be one the user supplied; never invent one.

   External:

   ```bash
   python scripts/payload.py render \
     --input-path s3://bucket/input/person-crops/ \
     --output-directory s3://bucket/output/image-attribute-augmentation/ \
     --service-mode external \
     --vlm-url https://vlm.example/v1 \
     --llm-url https://llm.example/v1 \
     --image-edit-url https://image-edit.example/v1 \
     --max-imgs 10 --num-augmentation 3 \
     --variable-distribution assets/variable-distribution.json \
     --output /tmp/iaa-payload.json
   ```

   Internal:

   ```bash
   python scripts/payload.py render \
     --input-path s3://bucket/input/person-crops/ \
     --output-directory s3://bucket/output/image-attribute-augmentation/ \
     --service-mode internal \
     --max-imgs 10 --num-augmentation 3 \
     --output /tmp/iaa-payload.json
   ```

   Add `--cosmos-evaluator-url <arbitrator-root-url>` to use an existing evaluator, or
   `--checker-override metropolis.attribute_verification=false` to turn the checker off for one
   run. The renderer and `validate` reject unknown fields and any distribution that omits one of
   the six clothing attributes; fix the payload instead of removing the field that was rejected.

   Show the user the rendered payload (or its validated contents) and get explicit confirmation
   before proceeding. Only continue to preflight and triggering if they confirm; if they want
   changes, re-render and re-confirm.

4. Preflight the DAG through the Airflow API. Check that the DAG is loaded, required pools have
   slots, and controller pods are healthy — see
   [airflow-direct-api.md#preflight-direct-path](references/airflow-direct-api.md#preflight-direct-path).
   Confirm credential presence with set/unset only; never print values, lengths, prefixes,
   suffixes, hashes, or fingerprints. Then, from the repository root, confirm
   the deployment maps every bucket the payload reads or writes:

   ```bash
   python3 skills/cdf-orchestration-preflight/scripts/storage_check.py --payload /tmp/iaa-payload.json
   ```

   It must print `Storage mapping: ready`. An `UNMAPPED` line means the run would fail after its
   GPU services start; report the unmapped field and the configured prefixes to the user. Using
   that bucket requires changing the controller's storage configuration and redeploying it; with
   the user's authorization, route that change to
   [orchestration setup](../cdf-orchestration-setup/SKILL.md), then rerun the check.

5. Submit exactly one DAG run. Pass the payload from step 3 as `conf.payload` — see
   [airflow-direct-api.md#trigger-a-run](references/airflow-direct-api.md#trigger-a-run) for the
   full request shape and lost-response recovery. Record and return the `DAG_RUN_ID`, input path,
   output directory, and service mode.

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
   report the state with the per-task breakdown below: tasks waiting on pool slots or GPUs are the
   usual cause.

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

   Use the configured Airflow login. The documented default login is defined in
   `deploy/values.yaml` under `airflow.createUserJob.defaultUser` (not
   `webserver.defaultUser`); deployments may override it. Do not include resolved credentials
   in chat or generated scripts. Change the default login before production use.

   To stop an in-progress run: open the Airflow UI, find the active DagRun, locate the running
   task, and mark it **Failed** (task menu → Mark Failed). This triggers the DAG's shutdown path,
   cleaning up Deployments, Services, and GPU pods. Do not delete the DagRun or the DAG — that
   bypasses cleanup and leaves stale cluster resources.

7. After the run reaches `success` or `failed`, if the user has not already requested or
   authorized downloading and analyzing the results, ask:
   **"Would you like to download and analyze the results?"**
   Wait for confirmation unless that authorization is already present in the session.

   When authorized, use whatever AWS credentials are already available in the shell
   environment (standard `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION`,
   an AWS profile, or instance role). Never ask the user to paste credentials into the prompt.
   Run artifacts live under `<output_directory>/<run_id>/`, where `<output_directory>` is the
   payload value and `<run_id>` is the `dag_run_id` from step 5. The final dataset is in
   `augmented_dataset/`; sync its contents into one local directory and summarize that directory:

   ```bash
   aws s3 sync "<output_directory>/<run_id>/augmented_dataset/" /tmp/iaa-results/
   python scripts/summarize_results.py --results-dir /tmp/iaa-results
   ```

   The summarizer reads the canonical `augmented_data.json` at the top of the dataset and ignores
   the per-scene copies under `<scene>/sidecars/`.

   To inspect intermediate augmented images instead, sync `<output_directory>/<run_id>/cosmos/`
   and read `output_metadata.json` from each `<person_id>/<augmentation_index>/` folder.

   Read [outputs.md](references/outputs.md) before interpreting files.

8. Report execution status and output quality separately. A `success` run state means every task
   finished; it does not mean every scene is complete. Before calling a run fully successful:

   - Fetch the `generate_augmented_dataset` task log (`map_index=-1`; see
     [airflow-direct-api.md](references/airflow-direct-api.md#fetch-task-logs)) and look for
     `Missing optional output metadata` and `Ignoring ... cosmos skipped sidecar` warnings. A
     missing metadata file leaves that scene with no selected attributes and generic `person`
     queries.
   - If results were downloaded, read the `Quality warnings` section from `summarize_results.py`.
     It lists entries without selected attributes, entries missing any clothing attribute, and
     skipped augmentations.

   If either check finds a problem, report the run as **succeeded with quality warnings**, list the
   affected entries, and do not describe the dataset as complete.

## Guardrails

- **Never invent endpoint URLs, bucket paths, or input paths — use only values the user
  supplied.** Always ask for every deployment-specific value before building a payload. If a
  required value is missing, stop and ask — do not substitute a guess.
- Preserve explicit user inputs and endpoint/model selections throughout the session.
- Do not submit if payload validation, local dataset validation, the storage mapping check, or
  Airflow preflight fails.
- Do not show AWS credentials, Airflow bearer tokens, or S3 signed URLs.
- Report a file as saved only after a successful write and an existence check. Otherwise,
  describe the content as proposed and not saved.
- Do not start multiple runs unless the user explicitly requests them.
- Ask for a dataset location if none was supplied; this workflow has no implicit demo dataset.
- Do not invent augmentation-only or labeling-only DAG IDs — only the DAG listed above exists.
- Only offer a platform whose manifest exists and whose DAG is loaded in Airflow.

## References

- Read [setup-and-preflight.md](references/setup-and-preflight.md) for environment, storage, and
  policy requirements.
- Read [payload-contract.md](references/payload-contract.md) when creating or changing a payload.
- Read [outputs.md](references/outputs.md) when retrieving or interpreting results.
- Read [troubleshooting.md](references/troubleshooting.md) after validation, API, or runtime errors.
- Read [airflow-direct-api.md](references/airflow-direct-api.md) for all Airflow interactions:
  preflight, triggering, monitoring, per-task breakdown, and log retrieval.
