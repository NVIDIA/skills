# Direct Airflow API

Use this path to preflight, trigger, and monitor Video Data Augmentation runs in the existing
Airflow installation. Never reinstall Airflow merely because a DAG is absent; inspect import
errors and report the installation mismatch first.

## Select the DAG

```bash
# Kubernetes (internal or external inference services)
DAG_ID="video_data_augmentation_dag_k8s"

# OSMO (external endpoints only)
# DAG_ID="video_data_augmentation_dag_osmo"
```

Confirm the selected ID is loaded. A DAG is registered only when its platform manifest exists.

## Establish the API and authenticate

For the repository's Kubernetes controller, derive the API address from the live service:

```bash
AIRFLOW_URL="http://$(kubectl get svc -n sdg-workflow \
  sdg-workflow-controller-api-server \
  -o jsonpath='{.spec.clusterIP}'):8080"
curl -sS --fail "$AIRFLOW_URL/api/v2/version"
```

Airflow 3 uses a JWT. Use the credentials configured for the running installation; do not print
the password or token in chat and do not write either to committed files.

```bash
RESPONSE=$(curl -sS --fail -X POST "$AIRFLOW_URL/auth/token" \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"$AIRFLOW_USER\",\"password\":\"$AIRFLOW_PASS\"}")
TOKEN=$(RESPONSE="$RESPONSE" python3 -c \
  "import json,os; print(json.loads(os.environ['RESPONSE'])['access_token'])")
test -n "$TOKEN"
```

## Preflight

Check the DAG, import errors, pools, controller pods, and PVCs before submission:

```bash
RESPONSE=$(curl -sS --fail -H "Authorization: Bearer $TOKEN" \
  "$AIRFLOW_URL/api/v2/dags/$DAG_ID")
RESPONSE="$RESPONSE" python3 -c \
  "import json,os; d=json.loads(os.environ['RESPONSE']); print('paused:',d.get('is_paused'),'file:',bool(d.get('file_token')))"

curl -sS --fail -H "Authorization: Bearer $TOKEN" \
  "$AIRFLOW_URL/api/v2/importErrors"

RESPONSE=$(curl -sS --fail -H "Authorization: Bearer $TOKEN" \
  "$AIRFLOW_URL/api/v2/pools")
RESPONSE="$RESPONSE" DAG_ID="$DAG_ID" python3 -c '
import json, os
watched = {"default_pool", "cosmos_transfer_3_service_pool"}
watched.add("osmo_rtx6000_1" if os.environ["DAG_ID"].endswith("_osmo") else "k8s_rtx6000_1")
for pool in json.loads(os.environ["RESPONSE"]).get("pools", []):
    if pool["name"] in watched:
        print(pool["name"], "open:", pool["open_slots"], "total:", pool["slots"])
'

kubectl get pods -n sdg-workflow -l "release=sdg-workflow-controller"
kubectl get pvc -n sdg-workflow
```

For Kubernetes runs, verify the `AIRFLOW_CONN_KUBERNETES_REMOTE` environment variable from the
scheduler rather than querying `/api/v2/connections`. Environment-backed Airflow connections are
not stored in the metadata database, so a REST 404 is not evidence that the connection is absent.

```bash
RESPONSE=$(kubectl exec -n sdg-workflow \
  deploy/sdg-workflow-controller-scheduler -c scheduler -- \
  printenv AIRFLOW_CONN_KUBERNETES_REMOTE)
RESPONSE="$RESPONSE" python3 -c \
  "import json,os; print(json.loads(os.environ['RESPONSE']).get('conn_type'))"
```

For OSMO, also confirm that the `osmo_token`, `osmo_url` and `osmo_pool` keys of
`osmo-token-secret` are non-empty (check presence only, never print values; route missing
settings to the environment-setup skill) and `osmo_rtx6000_1` capacity. OSMO cannot deploy
the VLM, LLM, Cosmos Transfer, or Cosmos Evaluator services.

## Trigger one run

The body must wrap the validated workflow payload under `conf.payload`. `logical_date` is required
by this Airflow API version.

```bash
PAYLOAD=$(cat /tmp/vda-payload.json)
LOGICAL_DATE=$(date -u +"%Y-%m-%dT%H:%M:%S.000Z")
curl -sS --fail -X POST "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"logical_date\":\"$LOGICAL_DATE\",\"conf\":{\"payload\":$PAYLOAD}}"
```

Record the returned `dag_run_id`. If the response is uncertain, search for that run before
retrying; do not create a duplicate.

## Monitor and diagnose

```bash
RESPONSE=$(curl -sS --fail -H "Authorization: Bearer $TOKEN" \
  "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns/$DAG_RUN_ID")
RESPONSE="$RESPONSE" python3 -c \
  "import json,os; print(json.loads(os.environ['RESPONSE'])['state'])"

RESPONSE=$(curl -sS --fail -H "Authorization: Bearer $TOKEN" \
  "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns/$DAG_RUN_ID/taskInstances?limit=500")
RESPONSE="$RESPONSE" python3 -c '
import json, os
for task in json.loads(os.environ["RESPONSE"]).get("task_instances", []):
    print((task.get("state") or "not scheduled").ljust(15), task.get("task_id", ""))
'
```

Fetch logs for a specific task and map index:

```bash
TASK_ID="cosmos_augmentation.augmentation_external"
TRY=1
MAP_INDEX=0
curl -sS --fail -H "Authorization: Bearer $TOKEN" \
  "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns/$DAG_RUN_ID/taskInstances/$TASK_ID/logs/$TRY?map_index=$MAP_INDEX"
```

Poll every 60–120 seconds until `success` or `failed`. Successful shutdown tasks do not turn a
failed processing run into a success.

## UI access

`make port-forward` is a foreground command. Run it in a background-capable terminal and verify
the forwarded endpoint before reporting it. For cancellation behavior, use the canonical
[Stopping a run](troubleshooting.md#stopping-a-run) guidance.
