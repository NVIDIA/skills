# Direct Airflow API

This is the primary path for triggering and monitoring Event Video Generation runs.

## Preflight (direct path)

Run these checks before submitting a run. Step 3 reads the DAG manifest from the CDF Orchestration
checkout, so set `REPO_ROOT` to that checkout; the rest of the checks need only `kubectl`, `curl`,
and `python3`.

```bash
REPO_ROOT=/path/to/cdf-orchestration
PLATFORM=k8s  # set to osmo for OSMO; keep this choice through submission

# 0. Establish AIRFLOW_URL (see "Establish AIRFLOW_URL and authenticate" for the port-forward
#    fallback when the ClusterIP is not routable from this host)
AIRFLOW_URL="http://$(kubectl get svc -n sdg-workflow \
  sdg-workflow-controller-api-server \
  -o jsonpath='{.spec.clusterIP}'):8080"

# 1. Airflow API is reachable; a timeout or empty reply is a failure, not a pass
RESPONSE=$(curl -s --max-time 8 "$AIRFLOW_URL/api/v2/version") || echo "Airflow API UNREACHABLE"
RESPONSE="$RESPONSE" python3 -c "import json, os; print(json.loads(os.environ['RESPONSE']))"

# 2. DAG is loaded
# Use TOKEN from the authentication step below.
DAG_ID="event_video_generation_dag_${PLATFORM}"
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

# 3. Required pools have slots.
# Derive the pool names from the manifest for the platform you are running rather than hardcoding
# them — pool names are deployment-specific (they come from the manifest's
# deployment.profiles.*.pool and are created by deploy/values.yaml's airflowPools.pools). A
# customer on different hardware, or one who renamed them, will not have the names this
# repository ships with today. Use the _osmo_ manifest when triggering the OSMO DAG.
# Needs PyYAML. The checkout's .venv (`uv sync --frozen --all-packages --inexact`) has it; otherwise use any
# python3 where `import yaml` succeeds.
MANIFEST="$REPO_ROOT/airflow/dags/workflows/event_video_generation_dag/configs/event_video_generation_${PLATFORM}_manifest.yaml"
REQUIRED_POOLS=$(python3 - "$MANIFEST" <<'PY'
import sys, yaml
m = yaml.safe_load(open(sys.argv[1]))
pools = {p["pool"] for p in m["deployment"]["profiles"].values() if p.get("pool")}
# The Cosmos task group picks the augmentation pool at runtime by service mode: external mode uses
# the service pool below, internal mode uses default_pool. Neither is named in the manifest.
pools.update({"default_pool", "external_image2video_service_pool"})
print(" ".join(sorted(pools)))
PY
)
echo "manifest-derived pools: $REQUIRED_POOLS"
RESPONSE=$(curl -s -H "Authorization: Bearer $TOKEN" "$AIRFLOW_URL/api/v2/pools?limit=100")
RESPONSE="$RESPONSE" REQUIRED_POOLS="$REQUIRED_POOLS" python3 -c "
import json, os
watched = set(os.environ['REQUIRED_POOLS'].split())
found = {p['name']: p for p in json.loads(os.environ['RESPONSE']).get('pools',[])}
for n in sorted(watched):
    p = found.get(n)
    print(n, '- open_slots:', p['open_slots'], '/ slots:', p['slots']) if p else print(n, '- MISSING')
"

# 4. Controller is ready. Checks only the controller's own Deployments and StatefulSets, so
#    completed setup Jobs and retained task pods do not affect the result. A kubectl error
#    stops with a failure instead of being reported as healthy.
kubectl get deploy,statefulset -n sdg-workflow -l release=sdg-workflow-controller \
  -o jsonpath='{range .items[*]}{.kind}/{.metadata.name} {.status.readyReplicas}/{.spec.replicas}{"\n"}{end}' \
  > /tmp/evg-controller.txt || { echo "controller check FAILED: kubectl error"; }
test -s /tmp/evg-controller.txt || echo "controller check FAILED: no controller workloads found"
awk '{split($2, r, "/"); s = (r[1] == r[2] && r[2] != "") ? "ready" : "NOT READY"; print s, $1, $2}' /tmp/evg-controller.txt

# 5. All PVCs are Bound (a Pending PVC does not show up in pod status)
kubectl get pvc -n sdg-workflow
```

Report check 4 as healthy only when the command succeeded, listed at least one workload, and every
line reads `ready`. Failed or completed task pods from earlier runs are not controller failures.

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

Which address reaches the API depends on where the agent runs:

| Agent location | `AIRFLOW_URL` |
|---|---|
| A cluster node, or a host that routes the cluster service network | The API server's ClusterIP, port 8080 |
| A workstation or any other host | `http://localhost:8080` through a port-forward |
| Behind an ingress the operator configured | The ingress URL the user supplies |

Try the ClusterIP first and fall back to a port-forward when it times out:

```bash
AIRFLOW_URL="http://$(kubectl get svc -n sdg-workflow \
  sdg-workflow-controller-api-server \
  -o jsonpath='{.spec.clusterIP}'):8080"
curl -s --max-time 8 -o /dev/null "$AIRFLOW_URL/api/v2/version" \
  || echo "ClusterIP not routable from this host; start a port-forward and use http://localhost:8080"
echo "AIRFLOW_URL=$AIRFLOW_URL"
```

Start the fallback with `make port-forward` as a background job (see
[Port-forward](#port-forward-ui-access)), then set `AIRFLOW_URL=http://localhost:8080`. The
ClusterIP is **not** the host's LAN IP; to open the UI from another machine, use the port-forward,
which binds `0.0.0.0:8080`, and `http://<host-LAN-IP>:8080`.

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
VALUES="$VALUES" python3 -c "import json, os; u = json.loads(os.environ['VALUES'])['airflow']['createUserJob']['defaultUser'].get('username'); print('Airflow username: set' if u else 'Airflow username: unset')"
```

For credential diagnostics, report presence only (set/unset), including `AIRFLOW_USER`,
`AIRFLOW_PASS`, and the bearer token when applicable. Never print credential values, lengths,
fragments, or hashes, or suggest commands that echo secrets. Do not authenticate or create a
token solely to answer a request to display credentials or check their presence. The authentication
sequence above is for authorized workflow operations. Never write resolved credentials into
committed files.

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
per request. Always report the resolved `http://<host-LAN-IP>:8080` URL to the
user, with instructions to use their configured Airflow credentials — never a bare
`localhost:8080`, which is meaningless from their browser if they are on a different machine. Tell them they can watch task progress, inspect logs, and stop runs via Mark
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

Submit exactly one run. Choose the run ID before submitting so a lost response can be checked
instead of resubmitted:

```bash
# Keep the DAG_ID selected and checked during preflight.
DAG_RUN_ID="evg__$(date -u +%Y%m%dT%H%M%SZ)"
PAYLOAD_FILE=/tmp/evg-payload.json   # the payload the user confirmed

TRIGGER_BODY=$(python3 - "$PAYLOAD_FILE" "$DAG_RUN_ID" <<'PY'
import datetime, json, sys
print(json.dumps({
    "dag_run_id": sys.argv[2],
    "logical_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "conf": {"payload": json.load(open(sys.argv[1]))},
}))
PY
)
RESPONSE=$(curl -s --max-time 30 -X POST "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "$TRIGGER_BODY")
RESPONSE="$RESPONSE" python3 -c "import json, os; d = json.loads(os.environ['RESPONSE'] or '{}'); print(d.get('state'), d.get('dag_run_id') or d)"
echo "DAG_RUN_ID=$DAG_RUN_ID"   # every later command uses this value
```

If the request times out, returns nothing, or returns `409`, **do not resubmit under a new ID**.
Fetch the run and compare its payload with the one you sent:

```bash
RESPONSE=$(curl -s -H "Authorization: Bearer $TOKEN" \
  "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns/$DAG_RUN_ID")
RESPONSE="$RESPONSE" python3 - "$PAYLOAD_FILE" <<'PY'
import json, os, sys
run = json.loads(os.environ["RESPONSE"] or "{}")
if "conf" not in run:
    print("run not found; it is safe to submit once more with the same DAG_RUN_ID")
elif run["conf"].get("payload") == json.load(open(sys.argv[1])):
    print("the run was accepted; monitor it")
else:
    print("DAG_RUN_ID collided with a different run; choose a new ID")
PY
```

## Check run state

`DAG_RUN_ID` is the value recorded when the run was triggered. For a run the user started from the
UI, list recent runs and take its `dag_run_id`:
`curl -s -H "Authorization: Bearer $TOKEN" "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns?order_by=-start_date&limit=5"`.

```bash
HTTP_STATUS=$(curl -s --max-time 30 -o /tmp/evg-run.json -w '%{http_code}' \
  -H "Authorization: Bearer $TOKEN" \
  "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns/$DAG_RUN_ID") || HTTP_STATUS=timeout
if [ "$HTTP_STATUS" = 200 ]; then
  python3 -c "import json; print(json.load(open('/tmp/evg-run.json'))['state'])"
else
  echo "state check failed: HTTP $HTTP_STATUS"   # 401/403: refresh the token; timeout: retry
fi
```

## Per-task breakdown (useful for diagnosing failures)

A run can have more task instances than one page returns, so the listing pages until it has all of
them. Tasks that have not been scheduled yet have no state:

```bash
OFFSET=0; : > /tmp/evg-tasks.jsonl
while :; do
  RESPONSE=$(curl -s --max-time 30 -H "Authorization: Bearer $TOKEN" \
    "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns/$DAG_RUN_ID/taskInstances?limit=100&offset=$OFFSET")
  COUNT=$(RESPONSE="$RESPONSE" python3 -c "
import json, os
d = json.loads(os.environ['RESPONSE'])
with open('/tmp/evg-tasks.jsonl', 'a') as out:
    for t in d['task_instances']:
        out.write(json.dumps(t) + '\n')
print(len(d['task_instances']), d['total_entries'])
") || { echo "task listing failed"; break; }
  set -- $COUNT; OFFSET=$((OFFSET + $1))
  [ "$1" -eq 0 ] || [ "$OFFSET" -ge "$2" ] && break
done
python3 -c "
import json
for line in open('/tmp/evg-tasks.jsonl'):
    t = json.loads(line)
    print(str(t['state'] or 'not scheduled').ljust(15), t['task_id'], t['map_index'])
"
```

## Fetch task logs

```bash
# try_number starts at 1. Mapped tasks take their map index (0, 1, ...); unmapped tasks use -1.
# Take both from the per-task breakdown above.
TASK_ID="cosmos_augmentation.augmentation_external"
TRY=1
MAP_INDEX=0
RESPONSE=$(curl -s -H "Authorization: Bearer $TOKEN" \
  "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns/$DAG_RUN_ID/taskInstances/$TASK_ID/logs/$TRY?map_index=$MAP_INDEX")
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
