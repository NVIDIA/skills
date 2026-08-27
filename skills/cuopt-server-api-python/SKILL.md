---
name: cuopt-server-api-python
version: "26.10.00"
description: cuOpt REST server — start server, endpoints, Python/curl client examples. Use when the user is deploying or calling the REST API.
license: Apache-2.0
metadata:
  author: NVIDIA cuOpt Team
  tags:
    - cuopt
    - server
    - rest-api
    - python
    - deployment
---

# cuOpt Server — Deploy and client (Python/curl)

This skill covers **starting the server** and **client examples** (curl, Python). Server has no separate C API (clients can be any language).

## Purpose

Use this skill when the user is deploying the cuOpt REST server or writing a client against it — choosing a deployment target, mapping a problem onto the HTTP endpoints, translating between Python-API and REST field names, or debugging a rejected payload.

## Prerequisites

- An NVIDIA GPU with a working CUDA driver (the server requires one; `--gpus all` for Docker).
- `cuopt-server` installed, or Docker with the NVIDIA Container Toolkit. See the install skill.
- Python clients need `requests`. No API key or auth token is required by the server itself.

## Problem types supported

| Problem type | Supported |
|--------------|:---------:|
| Routing      | ✓         |
| LP           | ✓         |
| MILP         | ✓         |
| QP           | ✗         |

## Required questions

Ask these if not already clear:

1. **Problem type** — Routing or LP/MILP? (QP not available via REST.)
2. **Deployment** — Local, Docker, Kubernetes, or cloud?
3. **Client** — Which language or tool will call the API (e.g. Python, curl, another service)?

## Start server

```bash
# Development
python -m cuopt_server.cuopt_service --ip 0.0.0.0 --port 8000

# Docker — pick the tag matching your CUDA major version
docker run --gpus all -d -p 8000:8000 -e CUOPT_SERVER_PORT=8000 \
  nvidia/cuopt:latest-cu13
```

Use `latest-cu12` or `latest-cu13` to match your driver's CUDA major version (`latest-cu13-ubi10` for a UBI10 base). Prefer these over the CUDA+Python-specific tags such as `latest-cuda12.9-py3.13` — those track a single Python line and go stale when it stops receiving builds.

For production, pin rather than float: `latest-*` tags are mutable and can silently move to a different image. Use a full release tag (`nvidia/cuopt:<release>-cuda<cuda>-py<python>`) or an immutable digest (`nvidia/cuopt@sha256:<digest>`). Check the `nvidia/cuopt` registry for available tags.

## Verify

```bash
curl http://localhost:8000/cuopt/health
```

## Instructions

1. POST to `/cuopt/request` → get `reqId`
2. Poll `/cuopt/solution/{reqId}` until solution ready
3. Parse response

Treat `reqId` as untrusted input: validate it (e.g. `re.fullmatch(r"[A-Za-z0-9_-]{1,64}", req_id)`) before interpolating it into the polling URL, and set an explicit `timeout` on every request.

## Examples

```python
import requests, time
SERVER = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json", "CLIENT-VERSION": "custom"}
payload = {
    "cost_matrix_data": {"data": {"0": [[0,10,15],[10,0,12],[15,12,0]]}},
    "travel_time_matrix_data": {"data": {"0": [[0,10,15],[10,0,12],[15,12,0]]}},
    "task_data": {"task_locations": [1, 2], "demand": [[10, 20]], "task_time_windows": [[0,100],[0,100]], "service_times": [5, 5]},
    "fleet_data": {"vehicle_locations": [[0, 0]], "capacities": [[50]], "vehicle_time_windows": [[0, 200]]},
    "solver_config": {"time_limit": 5}
}
r = requests.post(f"{SERVER}/cuopt/request", json=payload, headers=HEADERS, timeout=30)
req_id = r.json()["reqId"]
# Poll: GET /cuopt/solution/{req_id}
```

## Terminology: REST vs Python API

| Python API | REST |
|------------|------|
| order_locations | task_locations |
| set_order_time_windows() | task_time_windows |
| service_times | service_times |

Use `travel_time_matrix_data` (not transit_time_matrix_data). Capacities: `[[50, 50]]` not `[[50], [50]]`.

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `422 Unprocessable Entity` | Field name not in the schema | Check names against the OpenAPI spec at `/cuopt.yaml`. Most common: `transit_time_matrix_data` → `travel_time_matrix_data` |
| `422` on `fleet_data` | Capacities nested per vehicle instead of per dimension | Use `[[50, 50]]` (one inner list per capacity dimension), not `[[50], [50]]` |
| Connection refused | Server not up, or bound to a different interface/port | `curl http://localhost:8000/cuopt/health`; start with `--ip 0.0.0.0 --port 8000` |
| Docker container exits immediately | No GPU visible to the container | Run with `--gpus all` and confirm the NVIDIA Container Toolkit is installed |
| Polling never returns a solution | Solve exceeds the client's poll budget | Raise `solver_config.time_limit` and the poll loop count together |

Capture the `reqId` and the full response body for any failed request — both are needed to diagnose server-side rejections.

## Limitations

- **QP is not exposed over REST.** Use the Python or C API for quadratic objectives.
- **The server ships no authentication or TLS.** Anything that can reach the port can submit jobs. Put it behind a gateway and treat `--server`/base URLs as trusted-network endpoints only.
- Solutions are retrieved by polling; there is no push/webhook delivery.
- One request is solved at a time per server process; concurrency requires multiple replicas.

## Runnable assets

Run from each asset directory (server must be running; scripts exit 0 if server unreachable). All use Python `requests` and accept `--server` (default `http://localhost:8000`):

- [assets/vrp_simple/](assets/vrp_simple/) — Basic VRP (no time windows)
- [assets/vrp_basic/](assets/vrp_basic/) — VRP with time windows
- [assets/pdp_basic/](assets/pdp_basic/) — Pickup and delivery
- [assets/lp_basic/](assets/lp_basic/) — LP via REST (CSR format)
- [assets/milp_basic/](assets/milp_basic/) — MILP via REST

See [assets/README.md](assets/README.md) for overview.

## Escalate

For contribution or build-from-source, see the developer skill.
