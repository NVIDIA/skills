# Direct Airflow API

Use Airflow's API to submit and monitor `autonomous_vehicle_dataset_generation_dag_k8s`. The agent constructs the payload; the DAG validates inputs, executes components and validates outputs. These request examples do not replace those DAG tasks.

## Connection and Readiness

Use the controller established through the shared [Getting Started Guide](../../../docs/getting-started.md) or [orchestration setup skill](../../cdf-orchestration-setup/SKILL.md). Set `AIRFLOW_URL` to an address reachable from the terminal making the requests. A Kubernetes ClusterIP is usable only where cluster routing is available; a workstation can use the configured ingress or port-forward. Do not create an AV-specific installation.

Readiness, storage, pool and GPU checks belong to the shared [orchestration preflight skill](../../cdf-orchestration-preflight/SKILL.md). Reuse a verified environment for a bounded run. Missing capacity or unhealthy services are reasons to resolve readiness before submission, not to reinstall a shared controller automatically.

The workflow also needs the operator-provisioned shared FFmpeg installation claim. Where `kubectl` access is available, confirm it before submitting:

```bash
kubectl get pvc shared-ffmpeg -n sdg-workflow
```

A missing or unbound claim leaves `run_curator` and the media probe pods `Pending`. Follow the [shared FFmpeg installation instructions](../../../docs/autonomous-vehicle-dataset-generation/prerequisites.md#shared-ffmpeg-installation) when setup is requested.

Use a supplied `AIRFLOW_TOKEN`, or authenticate with the configured `AIRFLOW_USERNAME` and `AIRFLOW_PASSWORD`. The examples use `curl` and `jq`; keep credentials in the protected environment, with shell tracing disabled. Do not print tokens or include credentials in the payload.

```bash
: "${AIRFLOW_URL:?Set the reachable controller URL}"
AIRFLOW_URL="${AIRFLOW_URL%/}"
if [ -z "${AIRFLOW_TOKEN:-}" ]; then
  : "${AIRFLOW_USERNAME:?Set the configured Airflow username}"
  : "${AIRFLOW_PASSWORD:?Set the configured Airflow password}"
  export AIRFLOW_USERNAME AIRFLOW_PASSWORD
  AIRFLOW_TOKEN=$(
    jq -n '{username: env.AIRFLOW_USERNAME, password: env.AIRFLOW_PASSWORD}' |
      curl --fail --silent --show-error --connect-timeout 10 --max-time 60 \
        -H 'Content-Type: application/json' --data-binary @- \
        "$AIRFLOW_URL/auth/token" | jq -er '.access_token'
  )
fi
: "${AIRFLOW_TOKEN:?Airflow authentication failed}"
DAG_ID='autonomous_vehicle_dataset_generation_dag_k8s'
```

Read the loaded DAG and its published Run Parameter schema:

```bash
curl --fail --silent --show-error --max-time 30 \
  -H "Authorization: Bearer $AIRFLOW_TOKEN" \
  "$AIRFLOW_URL/api/v2/dags/$DAG_ID/details"
```

Confirm the DAG exists, is unpaused, and exposes the requested fields in `params.payload.schema` (including referenced schema definitions). The absence of `wsm_manifest_uri` or evaluator toggles is expected. The [Payload Guide](../../../docs/autonomous-vehicle-dataset-generation/payload-guide.md) explains the values to construct; Airflow performs authoritative validation on submission and in the DAG.

## Submit Once

For a requested run, set `PAYLOAD_FILE` to the completed local JSON, `RUN_ID` to the chosen unique run identity, and `REQUEST_FILE` and `RUN_FILE` to writable absolute filenames outside the repository. A payload-only request stops at the JSON deliverable.

First query the chosen ID:

```bash
: "${RUN_ID:?Choose and record a unique run ID}"
: "${RUN_FILE:?Choose an absolute local filename for the run response}"
RUN_ID_PATH=$(jq -rn --arg id "$RUN_ID" '$id | @uri')
RUN_HTTP_STATUS=$(curl --silent --show-error --max-time 30 \
  -H "Authorization: Bearer $AIRFLOW_TOKEN" \
  -o "$RUN_FILE" -w '%{http_code}' \
  "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns/$RUN_ID_PATH")
printf 'Run lookup HTTP status: %s\n' "$RUN_HTTP_STATUS"
```

- HTTP 200: the run already exists. Compare its `conf.payload` with the intended payload and monitor that run. Different input is a conflict; do not overwrite it.
- HTTP 404: the chosen ID is absent. For the initial submission, continue below.
- Other status or a network error: resolve it before submission. An uncertain lookup is not evidence that no run exists.

Only for an absent ID and an authorized execution, construct the standard request and POST once:

```bash
: "${PAYLOAD_FILE:?Set the absolute payload filename}"
: "${REQUEST_FILE:?Choose an absolute local filename for the request}"
umask 077
LOGICAL_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
jq -n --arg id "$RUN_ID" --arg date "$LOGICAL_DATE" \
  --slurpfile payload "$PAYLOAD_FILE" \
  '{dag_run_id: $id, logical_date: $date, conf: {payload: $payload[0]}}' \
  > "$REQUEST_FILE"
curl --fail --silent --show-error --connect-timeout 10 --max-time 60 \
  -H "Authorization: Bearer $AIRFLOW_TOKEN" \
  -H 'Content-Type: application/json' --data-binary "@$REQUEST_FILE" \
  "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns"
```

Record the returned run ID. If POST times out or returns a duplicate-ID conflict, GET the original ID and inspect that run. Do not blindly repeat POST or choose a replacement ID. If status remains unknown, report the uncertainty; any confirmed retry uses the same ID and payload. A rejected payload should be corrected from the reported validation error, not sent to a local substitute validator.

## Monitor and Report

Read the same run through the API or watch it in Airflow's Grid/Graph view:

```bash
curl --fail --silent --show-error --max-time 30 \
  -H "Authorization: Bearer $AIRFLOW_TOKEN" \
  "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns/$RUN_ID_PATH"
curl --fail --silent --show-error --max-time 30 \
  -H "Authorization: Bearer $AIRFLOW_TOKEN" \
  "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns/$RUN_ID_PATH/taskInstances?limit=100"
```

Use paginated task results if more than 100 exist. Continue monitoring within the user's requested window until terminal success or failure; if observation ends first, report the run as still active. Do not submit another run to work around a queued or failed one.

If a managed service pod restarts with `Orchestrator did not become ready within <N>s`, follow [model startup timeouts](../../cdf-orchestration-setup/references/model-startup-timeouts.md). Propose a timeout increase to the user.

The DAG's payload, retrieval, WSM/media and output-validation tasks determine
success. Generation validates and resamples RGB inputs by default, or probes
prepared WSM controls in `wsm` mode, before running Cosmos. Accepted candidates
undergo media validation afterward. Plausibility rejections are filtered out;
a successful run may publish no accepted videos. `finalize_augmentation_outputs` publishes the result manifest.
Report terminal state, output paths, media checks and required physical-plausibility
results. Read `search_result.json`, `cosmos3-nano/selection.json` and
`cosmos3-nano/result_manifest.json`.

For cancellation, follow the [workflow UI instructions](../../../docs/autonomous-vehicle-dataset-generation/getting-started.md#run-the-workflow) so shutdown tasks can run. Do not delete a DagRun or shared services as a recovery shortcut.
