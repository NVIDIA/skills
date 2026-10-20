# Brev Environment Reference

How to deploy VSS on a Brev GPU instance so the UI and API are reachable
from a browser via Brev **secure links** (a Cloudflare-fronted reverse proxy).

This reference derives from `deploy/docker/scripts/deploy_vss_launchable.ipynb`, which is the
interactive reference implementation.

## When this applies

A Brev-managed instance publishes an **environment context file** —
`$BREV_ENVIRONMENT_CONTEXT_PATH`, conventionally
`/etc/brev/environment-context.json` — listing one entry per exposed port with
its `destination_port`, `public_port`, and `fqdn`. It also sets
`BREV_ENV_ID=<instance-id>` in `/etc/environment`. If neither is present you are
not on a Brev-provisioned instance and this reference doesn't apply — use the
normal host IP + port access pattern from the selected file under `profiles/`.

**Resolve every hostname from the context file; never build one from a
pattern.** The domain is not always `brevlab.com` and the prefix convention
varies. Read the `fqdn` for the port you want. A port with no entry is not
exposed and has no URL. Resolve it with the helpers under
[Resolving a secure link](#resolving-a-secure-link).

## Architecture

```
Browser  ──https──>  <fqdn published for port 7777>  (Cloudflare Access)
                             │
                             ▼
                   Brev network tunnel
                             │
                             ▼
              vss-haproxy-ingress :7777 on the instance
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
        UI :3000      Agent API :8000     VST :30888
```

## Resolving a secure link

Look the port up in the context file. These helpers are used throughout this
reference; define them once per shell:

```bash
BREV_CTX="${BREV_ENVIRONMENT_CONTEXT_PATH:-/etc/brev/environment-context.json}"
# FQDN published for a destination port; empty when the port is not exposed.
brev_fqdn() { jq -er --argjson p "$1" '[.ports[]? | select(.destination_port == $p) | .fqdn][0] // empty' "$BREV_CTX" 2>/dev/null; }
# Public port for that link (443 in current Brev; read it, don't assume it).
brev_public_port() { jq -er --argjson p "$1" '[.ports[]? | select(.destination_port == $p) | .public_port][0] // empty' "$BREV_CTX" 2>/dev/null; }
# Browse origin for that port. Use this for any URL you hand the user or curl:
# an origin built from the FQDN alone is wrong wherever public_port is not 443.
brev_origin() {
  local host port
  host="$(brev_fqdn "$1")" && port="$(brev_public_port "$1")" || return 1
  if [ "$port" = 443 ]; then echo "https://${host}"; else echo "https://${host}:${port}"; fi
}

brev_fqdn 7777        # hostname only — for EXTERNAL_IP / VSS_PUBLIC_HOST
brev_public_port 7777
brev_origin 7777      # the haproxy ingress link — the VSS browse origin
```

`7777` is the destination port on the instance — the HAProxy ingress VSS exposes.
An empty result means no link is published for it: expose one in Brev, or deploy
behind a port that has one. Do not fall back to constructing a hostname.

### When the helpers return nothing
<a id="confined-read"></a>

`/etc/brev` is `0700 root:root`. An agent process holding no capabilities is
denied it even as uid 0 ([`prerequisites.md`](prerequisites.md#confinement)).
`read_brev_environment_context` retries that denial as `sudo -n cat` before
turning it into `{}` — which recovers the file for a host shell holding
passwordless sudo, and not for an agent whose `sudo` is refused by policy. When
the retry fails too, the helpers above return nothing instead of failing,
logging `context ... is unreadable` at `WARNING` and carrying on. **An empty
result is not "no link is published".** Distinguish the two before reporting
either; the warning line tells you which one you have.

`dockerd` holds the capabilities the agent lacks, so a read-only bind mount
resolves the entry. It is a privileged read of host config: **ask the user to
approve it**, never take it unilaterally, and request the one port's entry rather
than the whole file.

```bash
docker run --rm -v /etc/brev:/brev:ro ubuntu:22.04 cat /brev/environment-context.json \
  | jq -r '.ports[]|select(.destination_port==7777)|"public_port=\(.public_port) fqdn=\(.fqdn)"'
```

Any image already on the host serves — nothing but `cat` is used — and a small
one is worth pulling on a host with none. When the approval is declined, ask the
user for the origin they browse this instance on. Never construct one either
way: the domain is `gobrev.dev` on some instances and `brevlab.com` on others,
and a wrong host `404`s every browser request while `curl localhost:7777` still
succeeds.

## Per-profile secure link requirements

| Profile | Required links | Optional |
|---|---|---|
| `base` | **7777** (HAProxy ingress — UI + Agent + VST) | 6006 (Phoenix tracing) |
| `lvs` | **7777**, **5601** (Kibana) | 6006 |
| `search` | **7777**, **5601**, **31000** (nvstreamer) | 6006 |
| `alerts` | **7777**, **5601**, **31000** (nvstreamer) | 6006 |

On a NemoClaw build, add the harness dashboard **relay** port
(`NEMOCLAW_DASHBOARD_RELAY_PORT`, default **18790**) to whichever row applies:
it is the Agent UI's own origin and is not behind HAProxy, so it needs its own
link. Resolve it the same way as 7777 — the answer decides whether the summary
reports a secure-link URL or a loopback one behind an SSH tunnel.

Publish it **before** the notebook's section 3.1. That is the step that reads the
link and exports `CHAT_UI_URL`, which onboard bakes into the image's
`gateway.controlUi` to turn device auth off for the remote UI (any origin is
allowed). A sandbox built without it cannot sign in over the link and needs
recreating once the link exists, since `gateway.*` cannot be edited afterwards.

`NEMOCLAW_DASHBOARD_PORT` (default 18789) gets no link of its own. NemoClaw
keeps that forward on loopback and the relay in front of it is the published
path.

Ports that should NOT get their own secure link (they're behind HAProxy):
3000 (UI), 8000 (Agent), 30888 (VST).

### Search Harbor eval origin selection
<a id="host-local-search-eval"></a>

The host-side search Harbor eval accepts the deployment's host origin when
published links are missing or unreachable. This allows retrieval, ingestion,
and cleanup to run on eval workers without public links. Report missing ports
or unreadable context explicitly. The browser-facing deployment flow below
still requires the links in the profile table; a host-local eval result does
not establish public browser access.

Select this branch **during preflight**, only when the caller explicitly
authorizes a host-side Search Harbor eval. When the 7777 link is not published
or the context is unreadable, skip the browser-only secure-link overrides
in *Setup flow* and the external browser verification below. Keep the
deployment's local `HOST_IP` / `EXTERNAL_IP` settings and their dependent-value
closure from [`composition.md`](composition.md); do not replace them with an
unavailable FQDN. Continue through [`deployment.md`](deployment.md) and the local
Search checks in [`readiness.md`](readiness.md). Missing public links do not
fail this branch. The completed workflow supplies the host origin to the
selector; operate skills do not construct it. That host origin is the build's
fronting origin `http://$HOST_IP:$HAPROXY_HOST_PORT` from the read path in
[`deployment.md`](deployment.md).

An unreadable context in this branch is reported as unreadable and treated as
"no published link": the privileged `docker run` read above needs interactive
user approval, so a non-interactive eval does not take it. Resolve links from the
context file only; never query Brev API endpoints for them.

After local deployment readiness, **always** run the selector, passing the
published HTTPS origin for 7777 (or an empty string when none is available — the
selector handles that case, so do not skip the call) and the **host origin
supplied by the deployment workflow** to
[`select_brev_origin.sh`](../../operations/vss-search-archive/scripts/select_brev_origin.sh).
Never guess a Brev hostname, construct a replacement endpoint, or modify ports.
The selector validates each public candidate as an HTTPS origin before probing;
a non-HTTPS or malformed candidate selects host-local scope without a request.
For a valid HTTPS candidate, it makes one bounded request to
`/vst/api/v1/sensor/version`, without following redirects. HTTP 200 with JSON
`type: "vst"` and a nonempty string `version` selects the public origin with
`media_scope: "public"`. Missing candidates, transport errors, redirects,
authentication pages, and invalid JSON select the supplied host origin with
`media_scope: "host-local"`. A missing host origin or invalid invocation fails
with empty stdout. Check the exit status, configure `vss` with the returned
`origin`, and report `media_scope`. Host-local media URLs are usable from the
eval host and must not be advertised as public links. Do not retry the selector
or repair routing during ingestion or search.

## Setup flow

This recipe configures a browser-facing deployment. For an explicitly authorized host-local search eval,
select [the eval branch](#host-local-search-eval) before executing this recipe;
its missing-link stop does not apply to that branch.

Before resolving Compose, set the Brev secure-link values in the build's
`_builds/<name>/override.env`. **`EXTERNAL_IP` alone is not enough** — the Brev secure
link is served over **HTTPS on the link's own `public_port`** (443 in current
Brev — read it, per above), but the profile `.env` ships
`VSS_PUBLIC_HTTP_PROTOCOL=http`, `VSS_PUBLIC_WS_PROTOCOL=ws`, and
`VSS_PUBLIC_PORT=${HAPROXY_HOST_PORT}` (7777). Leaving those at the defaults makes the
agent emit `http://…:7777` UI/API/WS URLs from an `https://` page → the browser
blocks them as mixed content. Set the host, protocol, and port together:

```bash
BUILD_OVERRIDE="_builds/<name>/override.env"
host="$(brev_fqdn 7777)"                      # helpers from "Resolving a secure link"
port="$(brev_public_port 7777)"
[ -n "$host" ] && [ -n "$port" ] || { echo "No Brev link published for port 7777" >&2; exit 1; }
sed -i \
  -e "s|^EXTERNAL_IP=.*|EXTERNAL_IP=${host}|" \
  -e "s|^VSS_PUBLIC_HOST=.*|VSS_PUBLIC_HOST=${host}|" \
  -e "s|^VSS_PUBLIC_HTTP_PROTOCOL=.*|VSS_PUBLIC_HTTP_PROTOCOL=https|" \
  -e "s|^VSS_PUBLIC_WS_PROTOCOL=.*|VSS_PUBLIC_WS_PROTOCOL=wss|" \
  -e "s|^VSS_PUBLIC_PORT=.*|VSS_PUBLIC_PORT=${port}|" \
  "$BUILD_OVERRIDE"
```

Stop on an empty `host`; never substitute a constructed one. These values also
fill HAProxy's `known_host` allowlist, so a wrong hostname `404`s every browser
request while `curl localhost:7777` still succeeds.

On a NemoClaw build other than `alerts`, drop the `EXTERNAL_IP` line above and
leave that variable to the sandbox ([`agent-harness.md`](agent-harness.md)). The
rest is unchanged: `VSS_PUBLIC_HOST` already carries the secure link, and the
`known_host` ACL admits it in its own right.

## Verifying the deploy is reachable externally

After `docker compose up -d`:

```bash
# 1. Proxy is up and routing. Probe /, not /health: the ingress is HAProxy and its
# route table has no /health, so that path 404s on a perfectly healthy proxy.
curl -sf -o /dev/null http://localhost:7777/ && echo "proxy OK"

# 2. UI reachable through the proxy (internally)
curl -sfI http://localhost:7777/ | head -1

# 3. Print the browser URL the user should open — resolved, never constructed
brev_origin 7777
```

If step 1 fails, the haproxy container (`vss-haproxy-ingress`) hasn't come up — check
`docker logs vss-haproxy-ingress`. Common reason: another service on the host is
already bound to port 7777, or `EXTERNAL_IP` in the build override doesn't
match the secure-link domain (haproxy's `known_host` ACL rejects the
request as 404 from the browser even though `curl localhost:7777` works).

## Troubleshooting

| Symptom | Cause |
|---|---|
| User says the Brev link won't load at all | Compare their URL with `brev_origin 7777`. A hostname absent from the context file is not exposed. If the deployment sits behind a different exposed port, resolve that port and redeploy. |
| UI loads but AJAX calls to `/api/*` CORS-fail | A second secure link was created for port 8000 → browser treats it as a different origin. Delete the extra link; the UI should use the proxy only. |
| `curl "$(brev_origin 7777)"` → 502 | HAProxy container (`vss-haproxy-ingress`) is down — `docker logs vss-haproxy-ingress` |
| `curl "$(brev_origin 7777)"` → Cloudflare Access login page forever | User hasn't been granted access in the Brev org; not a deploy issue |
| Agent-generated report URLs don't open | `EXTERNAL_IP` in the build override is still the internal `${HOST_IP}` default → reports hard-code internal IPs. Set it from `brev_fqdn 7777` in `_builds/<name>/override.env` (see [Setup flow](#setup-flow)), regenerate `resolved.yml`, and redeploy. |
