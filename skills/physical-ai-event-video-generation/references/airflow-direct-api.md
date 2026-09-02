# Direct Airflow API

This is the primary path for triggering and monitoring Event Video Generation runs.

## Preflight (direct path)

Run these checks before submitting a run:

```bash
# 0. Establish AIRFLOW_URL (ClusterIP — always routable from host)
AIRFLOW_URL="http://$(kubectl get svc -n sdg-workflow \
  sdg-workflow-controller-api-server \
  -o jsonpath='{.spec.clusterIP}'):8080"

# 1. Airflow API is reachable
RESPONSE=$(curl -s "$AIRFLOW_URL/api/v2/version")
RESPONSE="$RESPONSE" python3 -c "import json, os; print(json.loads(os.environ['RESPONSE']))"

# 2. DAG is loaded
TOKEN=<token from auth step below>
DAG_ID="event_video_generation_dag_k8s"
RESPONSE=$(curl -s -H "Authorization: Bearer $TOKEN" "$AIRFLOW_URL/api/v2/dags/$DAG_ID")
RESPONSE="$RESPONSE" python3 -c "
import json, os
d = json.loads(os.environ['RESPONSE'])
print('is_paused:', d.get('is_paused'), '| file_token:', bool(d.get('file_token')))
"

# 2b. No DAG import errors (a DAG can be absent because it failed to parse)
RESPONSE=$(curl -s -H "Authorization: Bearer $TOKEN" "$AIRFLOW_URL/api/v2/importErrors")
RESPONSE="$RESPONSE" python3 -c "
import json, os
d = json.loads(os.environ['RESPONSE']); print('import errors:', d.get('total_entries'))
for e in d.get('import_errors',[]): print(' ', e['filename'], '::', e['stack_trace'][:200])
"

# 3. Required pools have slots
RESPONSE=$(curl -s -H "Authorization: Bearer $TOKEN" "$AIRFLOW_URL/api/v2/pools")
RESPONSE="$RESPONSE" python3 -c "
import json, os
watched = ('k8s_gpu_1','default_pool','internal_image2video_service_pool','external_image2video_service_pool')
for p in json.loads(os.environ['RESPONSE']).get('pools',[]):
    if p['name'] in watched:
        print(p['name'], '- open_slots:', p['open_slots'], '/ slots:', p['slots'])
"

# 4. Controller pods are healthy
kubectl get pods -n sdg-workflow --field-selector=status.phase!=Running 2>/dev/null | grep -v "^NAME" || echo "All pods running"

# 5. All PVCs are Bound (a Pending PVC does not show up in pod status)
kubectl get pvc -n sdg-workflow
```

All `kubectl` commands above assume the cluster connection is established from the path the user
supplied (see [setup-and-preflight.md](setup-and-preflight.md#cluster-access)). Ask for it when
unset rather than guessing.

If any check fails, resolve it before submitting a run.

## Verifying the `kubernetes_remote` connection

Every Event Video Generation task manifest sets `kubernetes_conn_id: kubernetes_remote`, so this
connection must resolve or all tasks fail at runtime. **Do not verify it through the REST API.**
This deployment injects connections as `AIRFLOW_CONN_*` environment variables sourced from
Kubernetes secrets (see `deploy/values.yaml`), and env-var connections are never written to the
Airflow metadata database. `GET /api/v2/connections` therefore returns `total_entries: 0` on a
perfectly healthy cluster, and `GET /api/v2/connections/kubernetes_remote` returns 404. Neither
means anything is wrong.

Verify the env var instead:

```bash
RESPONSE=$(kubectl exec -n sdg-workflow deploy/sdg-workflow-controller-scheduler -c scheduler -- \
  printenv AIRFLOW_CONN_KUBERNETES_REMOTE)
RESPONSE="$RESPONSE" python3 -c "import json, os; d=json.loads(os.environ['RESPONSE']); print('conn_type:', d.get('conn_type'))"
# expect: conn_type: kubernetes
```

Backing secrets should also exist — `kubernetes-remote-connection-secret`,
`nvcf-connection-secret`, `airflow-api-connection-secret`, and
`multistorageclient-configuration-secret`:

```bash
kubectl get secret -n sdg-workflow | grep -E "connection|multistorage"
```

## Establish AIRFLOW_URL and authenticate

The Kubernetes ClusterIP is always routable from the host machine and is the most reliable
address for agent API calls. Set `AIRFLOW_URL` from it before any API command:

```bash
AIRFLOW_URL="http://$(kubectl get svc -n sdg-workflow \
  sdg-workflow-controller-api-server \
  -o jsonpath='{.spec.clusterIP}'):8080"
echo "AIRFLOW_URL=$AIRFLOW_URL"
```

Note: the ClusterIP (internal Kubernetes network) is **not** the same as the host's LAN IP.
To expose the UI in a browser from another machine, run `make port-forward` separately — it binds
`0.0.0.0:8080`, making the UI reachable at `http://<host-LAN-IP>:8080`.

Airflow 3 uses JWT. Credentials live in `deploy/values.yaml` under
`airflow.createUserJob.defaultUser`, which defaults to `admin`/`admin`. Note the YAML location:
it is **not** `airflow.webserver.defaultUser` — that path does not exist in this chart and raises
`KeyError: 'defaultUser'`.

```bash
AIRFLOW_USER="admin"
AIRFLOW_PASS="admin"

RESPONSE=$(curl -s -X POST "$AIRFLOW_URL/auth/token" \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"$AIRFLOW_USER\",\"password\":\"$AIRFLOW_PASS\"}")
TOKEN=$(RESPONSE="$RESPONSE" python3 -c "import json, os; print(json.loads(os.environ['RESPONSE'])['access_token'])")

test -n "$TOKEN" && echo "auth OK" || echo "auth FAILED"
```

If auth fails, the defaults were changed at install time. Read what was actually applied to the
running release rather than guessing:

```bash
VALUES=$(helm get values "$RELEASE_NAME" -n "$NAMESPACE" -a -o json)
VALUES="$VALUES" python3 -c "import json, os; print(json.loads(os.environ['VALUES'])['airflow']['createUserJob']['defaultUser']['username'])"
```

Do not print the password in chat or write a resolved credential into a committed file.

## Port-forward (UI access)

`make port-forward` runs `kubectl port-forward` in the **foreground** and blocks until interrupted.
Start it as a background job, never as a blocking call, or the session stalls:

```bash
make port-forward   # binds 0.0.0.0:8080; run as a background job
```

Do not wrap it in `nohup` or `setsid` — these are blocked by policy in many agent environments. Use
the background-execution facility of your shell tool instead.

Confirm it is actually serving before reporting success, and resolve the host's real LAN address so
the user gets a URL they can click:

```bash
curl -s -o /dev/null -w "http=%{http_code}\n" --max-time 8 http://localhost:8080/api/v2/version
HOST_IP=$(hostname -I | awk '{print $1}')
echo "Airflow UI: http://$HOST_IP:8080"
```

A healthy forward logs `Forwarding from 0.0.0.0:8080 -> 8080`, then `Handling connection for 8080`
per request. Always report the resolved `http://<host-LAN-IP>:8080` URL and the credentials to the
user — never a bare `localhost:8080`, which is meaningless from their browser if they are on a
different machine. Tell them they can watch task progress, inspect logs, and stop runs via Mark
Failed.

## Trigger a run

This calls the Airflow REST API on the same cluster you already authenticated against above — it
is not a third-party or remote endpoint, and no response body is ever executed as code.

`logical_date` is a **required** field on this Airflow API version — `null` or an omitted field
is rejected with `{"detail":[{"type":"missing","loc":["body","logical_date"],"msg":"Field
required"...}]}`. Pass an explicit ISO 8601 UTC timestamp (the current time is fine; Airflow does
not require it to be meaningful for a manual trigger). Pass `conf.payload` to override the DAG's
default `Param` value. Without it the DAG runs with `EventVideoGenerationDagPayloadConfig()`
defaults, which point at non-working placeholder paths and endpoints — always pass an explicit
payload for user runs.

```bash
# The only DAG this repository registers; confirm it is loaded before triggering
DAG_ID="event_video_generation_dag_k8s"

# With a custom payload
PAYLOAD=$(cat /tmp/evg-payload.json)
LOGICAL_DATE=$(date -u +"%Y-%m-%dT%H:%M:%S.000Z")
curl -s -X POST "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"logical_date\": \"$LOGICAL_DATE\", \"conf\": {\"payload\": $PAYLOAD}}"

# Returns dag_run_id — record it for status checks
```

## Check run state

```bash
DAG_RUN_ID="manual__2026-07-12T12:23:13.076612+00:00"
RESPONSE=$(curl -s -H "Authorization: Bearer $TOKEN" \
  "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns/$DAG_RUN_ID")
RESPONSE="$RESPONSE" python3 -c "import json, os; print(json.loads(os.environ['RESPONSE'])['state'])"
```

## Per-task breakdown (useful for diagnosing failures)

```bash
RESPONSE=$(curl -s -H "Authorization: Bearer $TOKEN" \
  "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns/$DAG_RUN_ID/taskInstances?limit=50")
RESPONSE="$RESPONSE" python3 -c "
import json, os
for t in json.loads(os.environ['RESPONSE']).get('task_instances',[]):
    print(f\"{t['state']:15} {t['task_id']}\")
"
```

## Fetch task logs

```bash
# try_number starts at 1; map_index=0 for non-mapped tasks
TASK_ID="cosmos_augmentation.augmentation_external"
TRY=1
RESPONSE=$(curl -s -H "Authorization: Bearer $TOKEN" \
  "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns/$DAG_RUN_ID/taskInstances/$TASK_ID/logs/$TRY?map_index=0")
RESPONSE="$RESPONSE" python3 -c "import json, os; print(json.loads(os.environ['RESPONSE']).get('content',''))"
```

## State vocabulary

| State | Meaning |
|---|---|
| `queued` | Scheduled, not yet running |
| `running` | Actively executing |
| `success` | All tasks completed |
| `failed` | At least one task failed — check task instances for root cause |
| `up_for_retry` | Task failed but retries remain; wait before diagnosing |
| `skipped` | Expected for bypassed branches (e.g. service startup in external mode) |
