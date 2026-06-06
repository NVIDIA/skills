# Deploy Readiness Gate

`docker compose up -d` returns when containers are *created*, not when
the processes inside have finished initialising. Cold deploys
(first-time NIM image pulls, model warmup, vLLM CUDA-graph capture)
can legitimately take 10–20 min. Use this gate before declaring a
deploy "done".

## Step 1 — wait for the compose project to settle

Every container must be either `running` or cleanly `exited 0`. One-shot init
jobs (e.g. `vss-kibana-init`) legitimately exit 0 and stay exited, which is
fine. Anything `restarting`, `unhealthy`, or `exited <N≠0>` is a deploy
failure even though `up -d` returned 0.

```bash
# docker compose 2.21+ emits NDJSON (one bare object per line) from
# `ps --format json`, not a JSON array — so no `.[]` here; jq's default
# input loop already iterates each line. The filter accepts only
# `running` and `exited 0`; everything else (restarting, unhealthy,
# exited with non-zero code) is a failure.
docker compose -f resolved.yml ps --format json \
  | jq -r 'select((.State == "running" or (.State == "exited" and .ExitCode == 0)) | not)
           | "\(.Name)\t\(.State)\texit=\(.ExitCode // "?")\t\(.Status)"' \
  | { mapfile -t bad; if [ "${#bad[@]}" -gt 0 ]; then
        printf 'FAIL: %s\n' "${bad[@]}" >&2; exit 1;
      fi; }
```

## Step 2 — probe the profile's documented readiness endpoints

Container state alone isn't enough — the processes inside may still be
importing modules, loading models, and binding ports. Each profile reference
(`base.md`, `lvs-profile.md`, `alerts.md`, `warehouse.md`, …) lists the
endpoints that must be reachable for that profile (agent REST API, UI,
inference NIMs, etc., on the ports the profile actually opens). Run those
`curl` checks with a generous deadline (15 min is reasonable for cold NIM
warmup) and only declare the deploy done once every documented endpoint
returns the expected success exit code.

## Step 3 — triage slow containers

If any probe times out, dump `docker compose ps` and
`docker compose logs --tail 100 <slow-service>` and report the slow
container. Never claim success on a half-warm stack.
