# Brev Environment Reference

How to deploy VSS on a Brev GPU instance so the UI and API are reachable
from a browser via Brev **secure links** (a Cloudflare-fronted reverse proxy).

This reference derives from `scripts/deploy_vss_launchable.ipynb`, which is the
interactive reference implementation.

## When this applies

A Brev-managed instance sets `BREV_ENV_ID=<instance-id>` in `/etc/environment`.
If that file doesn't contain `BREV_ENV_ID`, you're not on a Brev-provisioned
instance and this reference doesn't apply — use the normal host IP + port
access pattern from [`base.md`](base.md).

## Architecture

```
Browser  ──https──>  77770-<BREV_ENV_ID>.brevlab.com  (Cloudflare Access)
                             │
                             ▼
                   Brev network tunnel
                             │
                             ▼
              vss-proxy (nginx) :7777 on the instance
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
        UI :3000      Agent API :8000     VST :30888
```

**Why one port.** Each Brev secure link terminates a separate Cloudflare
Access session. If you gave each VSS service its own secure link, the UI's
AJAX calls to the Agent API would cross Cloudflare Access sessions and
trigger CORS rejections. Consolidating behind nginx on port 7777 keeps
everything in one origin.

## Secure-link URL format

```
https://<link-prefix>-<BREV_ENV_ID>.brevlab.com
```

- `<BREV_ENV_ID>` is the instance's ID from `/etc/environment`.
- `<link-prefix>` depends on how the Brev secure link is configured:
  - **Launchable-created links (default):** `${PROXY_PORT}0` — e.g. port 7777 → prefix `77770`.
  - **Manually-created links:** `${PROXY_PORT}` — e.g. port 7777 → prefix `7777`.
- Override with `BREV_LINK_PREFIX=<prefix>` if your setup differs.

## Per-profile secure link requirements

| Profile | Required links | Optional |
|---|---|---|
| `base` | **7777** (nginx proxy — UI + Agent + VST) | 6006 (Phoenix tracing) |
| `lvs` | **7777**, **5601** (Kibana) | 6006 |
| `search` | **7777**, **5601**, **31000** (nvstreamer) | 6006 |
| `alerts` | **7777**, **5601**, **31000** (nvstreamer) | 6006 |

Ports that should NOT get their own secure link (they're behind the nginx proxy):
3000 (UI), 8000 (Agent), 30888 (VST).

## Setup flow

Source the helper script **before** `docker compose up`:

```bash
source skills/deploy/scripts/brev_setup.sh
```

Or equivalently:

```bash
source "$(claude-config-dir)/skills/deploy/scripts/brev_setup.sh"
```

This exports:

| Var | Value | Used by |
|---|---|---|
| `BREV_ENV_ID` | Instance ID from `/etc/environment` | `docker-compose.yml` → nginx config |
| `PROXY_PORT` | `7777` (default, overridable) | `docker-compose.yml` → nginx container's published port |
| `BREV_LINK_PREFIX` | `${PROXY_PORT}0` (launchable default) | Report / log URL rewriting in the agent |

The compose stack reads those via `${VAR:-default}` so missing vars fall back
to internal IPs — you can skip the source step on non-Brev hosts without
breaking anything.

## Verifying the deploy is reachable externally

After `docker compose up -d`:

```bash
# 1. Nginx proxy is up and routing
curl -sf http://localhost:${PROXY_PORT:-7777}/health >/dev/null && echo "proxy OK"

# 2. UI reachable through the proxy (internally)
curl -sfI http://localhost:${PROXY_PORT:-7777}/ | head -1

# 3. Print the browser URL the user should open
echo "https://${BREV_LINK_PREFIX}-${BREV_ENV_ID}.brevlab.com"
```

If step 1 fails, the nginx container (`vss-proxy`) hasn't come up — check
`docker logs vss-proxy`. Common reason: `PROXY_PORT` collision with something
else on the host, or missing `BREV_LINK_PREFIX` var when nginx does URL rewrites.

## Brev launchable quirk — the `0` suffix

Brev launchables always create secure links with a trailing `0` appended to
the port number. A launchable opened for port 7777 ends up reachable at
`77770-<id>.brevlab.com`, **not** `7777-<id>.brevlab.com`.

If the user manually created a secure link via the Brev dashboard, that `0`
suffix may or may not be there — in which case set `BREV_LINK_PREFIX=7777`
(without the `0`) to match.

## Troubleshooting

| Symptom | Cause |
|---|---|
| UI loads but AJAX calls to `/api/*` CORS-fail | A second secure link was created for port 8000 → browser treats it as a different origin. Delete the extra link; the UI should use the proxy only. |
| `curl https://77770-...brevlab.com` → 502 | nginx container (`vss-proxy`) is down — `docker logs vss-proxy` |
| `curl https://77770-...brevlab.com` → Cloudflare Access login page forever | User hasn't been granted access in the Brev org; not a deploy issue |
| Agent-generated report URLs don't open | `BREV_LINK_PREFIX` wasn't exported before compose → reports hard-code internal IPs. Source `brev_setup.sh` and redeploy |


## Brev secure-link env vars (extracted from SKILL.md Step 1c)

### Step 1c — If deploying on Brev, set up secure-link env vars

On a Brev-managed instance, VSS is accessed from the browser via a
Cloudflare-fronted secure link that tunnels to an nginx proxy on port 7777.
The proxy consolidates UI + Agent API + VST behind one origin (CORS-safe).

Source the helper **before** `docker compose up`:

```bash
source skills/deploy/scripts/brev_setup.sh
```

It detects `/etc/environment`'s `BREV_ENV_ID` and exports `PROXY_PORT=7777`
and `BREV_LINK_PREFIX=77770` (launchable default; override with
`BREV_LINK_PREFIX=7777` if the secure link was created manually without
the `0` suffix). On non-Brev instances the script is a no-op.
