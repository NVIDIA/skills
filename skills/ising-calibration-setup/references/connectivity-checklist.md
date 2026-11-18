# Connectivity checklist

Run this after setup to confirm the connection actually works, and to tell an
auth problem apart from a network/model-not-ready problem.

## Step 1: confirm the key is set

Hosted API -- every request carries the key, so check it first:

```bash
test -n "$NVIDIA_API_KEY" && echo "NVIDIA_API_KEY is set" || echo "NVIDIA_API_KEY is EMPTY"
```

If empty, go back to `hosted-api-quickstart.md` and generate the key -- do not
proceed to step 2.

Self-hosted NIM -- there is no request-time key. `NGC_API_KEY` is used for the
image pull and is passed into the container at startup; the container does not
authenticate requests. An empty `NGC_API_KEY` in your current shell therefore
does not break a container that is already running. Check that the container
is up instead:

```bash
docker ps
```

## Step 2: send one lightweight request

Hosted API:

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia/ising-calibration-1.5-31b","messages":[{"role":"user","content":"ping"}],"max_tokens":8}'
```

Self-hosted NIM -- check readiness first, which distinguishes "still loading
weights" from "broken" in one call:

```bash
curl -s http://localhost:8000/v1/health/ready
# {"object":"health.response","message":"Service is ready","status":"ok"}
```

Then send a real request (adjust host/port -- `localhost:8000` only applies if you are
running the container on the same machine as this check; for a cloud/remote
GPU instance, use its address, e.g. its public IP or an SSH tunnel, and
confirm the port is actually open/reachable from where curl runs):

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"ping"}],"max_tokens":8}'
```

The `model` field is omitted here because a NIM serving a single model resolves
it for you. Note that the self-hosted model ID
(`nvidia/NVIDIA-Ising-Calibration-1.5-31B`) differs from the hosted one, so
carrying the hosted string over into a self-hosted request fails.

## Step 3: read the status code

| Code | Meaning | Hosted API fix | Self-hosted NIM fix |
|------|---------|----------------------|----------------------------|
| `200` | Working. | -- | -- |
| `401` / `403` | Hosted API: auth failure. Self-hosted NIM: something in front of the container. | Regenerate `NVIDIA_API_KEY` at `build.nvidia.com/settings/api-keys`; make sure you did not accidentally export the NGC key instead. | A running NIM does not authenticate requests, so a 401 cannot come from the container or your NGC key. Check whatever sits between you and the container -- a reverse proxy, API gateway, or cloud load balancer -- and its credentials. To confirm, curl the container directly from the machine running Docker at `localhost:<host-port>` (left side of the `-p` mapping in `docker ps`; 8000 if unchanged); a `200` there isolates the problem to the layer in front. Do not regenerate the NGC key for this. |
| `000` / connection refused / timeout | Network or service-not-ready failure, not auth. | Check general internet connectivity and that `integrate.api.nvidia.com` is reachable. | Container is not up yet or the port mapping is wrong. Run `docker ps` to confirm the container is running and that the `-p` mapping's host port matches the one you are curling (the quickstart uses `-p 8000:8000`; the container side is always 8000, the host side is whatever you chose), then `docker logs <container>` to check startup progress. On a cloud/remote instance, also check that the port is actually exposed to you (firewall / security group rules), not just bound inside the instance. |
| `429` | Rate limited. | Wait and retry; check the trial terms for limits. | N/A for self-hosted. |
| `5xx` | Server-side error. | Retry; if persistent, the hosted service may be degraded. | Check `docker logs <container>` for a crash or OOM. |

Do not assume a connection-refused error against a self-hosted NIM means the
key is wrong -- confirm the container is actually running and ready first.
