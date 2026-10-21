# Direct Airflow API

Copy-pasteable commands for the readiness, trigger, monitor and stop steps in `SKILL.md`. Every
command targets the controller the user configured; nothing here deploys or changes the cluster
except the trigger and stop calls, which need the user's confirmation first. Responses are
captured into a variable and parsed locally; no response body is ever executed.

## Connect and authenticate

```bash
AIRFLOW_URL="http://$(kubectl get svc -n sdg-workflow sdg-workflow-controller-api-server \
  -o jsonpath='{.spec.clusterIP}'):8080"
curl -s --max-time 8 "$AIRFLOW_URL/api/v2/version" || echo "ClusterIP not routable; use a port-forward"
```

If the ClusterIP times out, start
`kubectl port-forward -n sdg-workflow service/sdg-workflow-controller-api-server 8080:8080` as a
background job and set `AIRFLOW_URL=http://localhost:8080`.

Read the Airflow credentials from the environment or an approved secret store. The login body and
the bearer header are written to owner-only temporary files, so neither the password nor the token
appears on a command line, in the process list, or in chat:

```bash
umask 077
AUTH_BODY=$(mktemp); AUTH_HEADER=$(mktemp)
python3 -c 'import json, os; print(json.dumps({"username": os.environ["AIRFLOW_USER"], "password": os.environ["AIRFLOW_PASSWORD"]}))' > "$AUTH_BODY"
RESPONSE=$(curl -s -X POST "$AIRFLOW_URL/auth/token" -H "Content-Type: application/json" -d @"$AUTH_BODY")
rm -f "$AUTH_BODY"
RESPONSE="$RESPONSE" python3 -c 'import json, os; print("Authorization: Bearer " + json.loads(os.environ["RESPONSE"])["access_token"])' > "$AUTH_HEADER"
unset RESPONSE
api() { curl -s -H @"$AUTH_HEADER" "$@"; }
DAG_ID=defect_image_generation_smt_dag_k8s
```

A `401` or `403` later means the token expired; repeat the block above. Remove `"$AUTH_HEADER"`
when the session ends.

## Readiness

```bash
RESPONSE=$(api "$AIRFLOW_URL/api/v2/dags/$DAG_ID")
RESPONSE="$RESPONSE" python3 -c 'import json, os; d = json.loads(os.environ["RESPONSE"]); print("is_paused:", d.get("is_paused", d))'

RESPONSE=$(api "$AIRFLOW_URL/api/v2/importErrors")
RESPONSE="$RESPONSE" python3 -c 'import json, os; print("import errors:", json.loads(os.environ["RESPONSE"])["total_entries"])'

RESPONSE=$(api "$AIRFLOW_URL/api/v2/pools?limit=100")
RESPONSE="$RESPONSE" python3 -c '
import json, os
watched = {"default_pool", "k8s_rtx6000_1", "external_image_edit_service_pool"}
pools = {p["name"]: p for p in json.loads(os.environ["RESPONSE"])["pools"]}
for name in sorted(watched):
    p = pools.get(name)
    print(name, "open", p["open_slots"], "/", p["slots"]) if p else print(name, "MISSING")
'
```

`external_image_edit_service_pool` matters only for external image-edit; internal augmentation
runs on `default_pool`. Unpause the DAG only when the user authorized it:
`api -X PATCH -H "Content-Type: application/json" "$AIRFLOW_URL/api/v2/dags/$DAG_ID" -d '{"is_paused": false}'`.

## Trigger exactly one run

Choose and record a unique run ID before submitting. The payload goes inside `conf.payload`, and
`logical_date` must be present:

```bash
DAG_RUN_ID="dig__$(date -u +%Y%m%dT%H%M%SZ)"
PAYLOAD_FILE=/path/to/confirmed_payload.json
TRIGGER_BODY=$(mktemp)

python3 - "$PAYLOAD_FILE" "$DAG_RUN_ID" > "$TRIGGER_BODY" <<'PY'
import datetime, json, sys
print(json.dumps({
    "dag_run_id": sys.argv[2],
    "logical_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "conf": {"payload": json.load(open(sys.argv[1]))},
}))
PY

RESPONSE=$(api -X POST -H "Content-Type: application/json" \
  "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns" -d @"$TRIGGER_BODY")
RESPONSE="$RESPONSE" python3 -c 'import json, os; d = json.loads(os.environ["RESPONSE"]); print(d.get("state"), d.get("dag_run_id") or d)'
```

On a timeout or `409`, do not resubmit under a new ID. Fetch the run and compare payloads:

```bash
RESPONSE=$(api "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns/$DAG_RUN_ID")
RESPONSE="$RESPONSE" python3 - "$PAYLOAD_FILE" <<'PY'
import json, os, sys
remote = json.loads(os.environ["RESPONSE"]).get("conf", {}).get("payload")
local = json.load(open(sys.argv[1]))
print("same run was accepted" if remote == local else "run ID collided with a different payload")
PY
```

## Monitor

```bash
RESPONSE=$(api "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns/$DAG_RUN_ID")
RESPONSE="$RESPONSE" python3 -c 'import json, os; print(json.loads(os.environ["RESPONSE"])["state"])'

RESPONSE=$(api "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns/$DAG_RUN_ID/taskInstances?limit=100&offset=0")
RESPONSE="$RESPONSE" python3 -c '
import json, os
for t in json.loads(os.environ["RESPONSE"])["task_instances"]:
    print(str(t["state"]).ljust(15), t["task_id"], t["map_index"], "try", t["try_number"])
'
```

Fetch the log of a failing instance with the `task_id`, `try_number` and `map_index` the listing
returned (`-1` for unmapped tasks):

```bash
TASK_ID=anomaly_generation; TRY=1; MAP_INDEX=-1
RESPONSE=$(api "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns/$DAG_RUN_ID/taskInstances/$TASK_ID/logs/$TRY?map_index=$MAP_INDEX")
RESPONSE="$RESPONSE" python3 -c '
import json, os
content = json.loads(os.environ["RESPONSE"]).get("content", "")
for entry in content if isinstance(content, list) else [content]:
    print(entry.get("event", entry) if isinstance(entry, dict) else entry)
'
```

## Stop a run the user asked to stop

Mark the **running task instances** failed, not the DAG run, so the service shutdown tasks still
remove any internal image-edit deployment. Never delete the DAG run or the DAG.

```bash
RESPONSE=$(api "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns/$DAG_RUN_ID/taskInstances?state=running&limit=200")
RUNNING=$(RESPONSE="$RESPONSE" python3 -c '
import json, os
for t in json.loads(os.environ["RESPONSE"])["task_instances"]:
    print(t["task_id"] + ("" if t["map_index"] < 0 else "/" + str(t["map_index"])))
')
for TI in $RUNNING; do
  api -X PATCH -H "Content-Type: application/json" \
    "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns/$DAG_RUN_ID/taskInstances/$TI" \
    -d '{"new_state": "failed"}' >/dev/null && echo "marked failed: $TI"
done
```

The run's failure callback queues `k8s_resource_cleanup_dag` for this run. If pods or Services
labeled with this run are still present afterward, trigger that DAG with
`{"logical_date": "<now>", "conf": {"target_run_id": "<DAG_RUN_ID>"}}`. Never leave
`target_run_id` empty: that deletes every k8s-plugin resource in the namespace.
