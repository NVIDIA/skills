# Agent Harness

- [Model](#model)
- [Connecting the Web UI to NemoClaw](#connecting-the-web-ui-to-nemoclaw)
- [`nemoclaw` is never a service key](#nemoclaw-is-never-a-service-key)
- [What removing the agent implies](#what-removing-the-agent-implies)
- [Ordering](#ordering)
- [Prerequisites](#prerequisites)
- [Default provider](#default-provider)
- [Bring-up](#bring-up)
- [Verification](#verification)
- [Teardown](#teardown)
- [Sources](#sources)

## Model

A **harness** is what a person or another agent talks to in order to drive a
build. Two exist, they are **mutually exclusive**, and **at most one** is
deployed. `vss-agent` is removed unless the request names it, so a build carries
the NemoClaw sandbox, the in-stack agent, or no harness at all — never two.

**Q3 is binary — these are its only two outcomes:**

| Q3 answer | Outcome | Where it runs | Reached by |
|---|---|---|---|
| **yes** *(default)* | `nemoclaw` | a sandbox on the host, outside the Compose project | its chat UI, with the VSS skills installed into it |
| **no** | no harness | — | the `vss` CLI from the host |

**Separate explicit-name path — not a Q3 option:** `vss-agent` runs inside the
Compose project and is reached through the agent REST API (`/generate`) and Web
UI. Select it through the Agent owner ([`services/agent.md`](services/agent.md))
only when the request names it; skip Q3 entirely.

`vss-agent` is in-stack: it is a container, it is reached through the build's own
origin, and forward closure retains it whenever agentic orchestration is
requested or another owner declares it as a peer — unless NemoClaw is selected,
which removes it. NemoClaw is host-side: an OpenShell sandbox running an agent
harness (OpenClaw or Hermes) with the repository's skills installed, driving the
deployment from outside over the same public routes an operator would use.

**NemoClaw is the default harness.** When a build has an interactive surface and
the request names no harness, Q3 asks one yes/no question — deploy the NemoClaw
sandbox as the harness? — and **yes** is the default, including for an
unanswered Q3. A **no** means no harness: the build is driven by the `vss` CLI
from the host. Never turn that into a menu of harnesses; the only question is
whether NemoClaw is deployed.

**`vss-agent` is removed on either answer.** The in-stack agent is deployed only
when the request names it — the chat agent, the Web UI, the agent REST API — and
such a request skips Q3 the way any named harness does. Honour it; do not steer
it to NemoClaw. Everything below about the removal therefore applies to a `no`
as much as to a `yes`; only the [Prerequisites](#prerequisites),
[ingress](#ingress-is-still-required), and bring-up sections are NemoClaw's
alone.

A build that reaches no interactive surface still has **no harness at all** — the
correct outcome for one that only ingests, indexes, or serves an API. Defaulting
to NemoClaw never means adding a harness to a headless build.

## Connecting the Web UI to NemoClaw

When a NemoClaw build uses the default OpenClaw runtime and includes `vss-ui`,
connect its chat sidebar and Chat tab through the embedded adapter. This
requires a relay-aware `deploy_nemoclaw.ipynb`: NemoClaw keeps its dashboard
forward on `127.0.0.1` (its `connect`/`recover`/`start` recovery re-creates it
there and retires a `0.0.0.0` forward as stale), so section 3.5 starts a
**dashboard relay** on a second host port, bound to Docker's default-bridge
gateway — what `host.docker.internal` resolves to inside the container. If the
checked-out notebook lacks section 3.5's relay, stop and report the checkout as
incompatible; a loopback-only forward cannot serve the containerized UI.

Before Step 7 writes the build artifacts, select the dashboard port (default
`18789`) and the relay port (default `18790`, must differ). Put these values in
`_builds/<name>/override.env` so the first deploy starts the UI with the
adapter selected. Resolve `<relay-port>` to the selected relay port, not the
loopback dashboard forward:

| Variable | Value |
|---|---|
| `VSS_AGENT_ADAPTER_ENABLED` | `true` |
| `VSS_AGENT_BACKEND_PROTOCOL` | `openclaw-ws` |
| `VSS_AGENT_BACKEND_URL` | `ws://host.docker.internal:<relay-port>` |

Leave `VSS_AGENT_BACKEND_TOKEN` unset. The gateway token does not exist until
onboarding, and the Web UI accepts it at runtime. Leave
`VSS_AGENT_BACKEND_PATH` unset; `/` is the `openclaw-ws` default.

Export the selected ports and adapter flag before onboarding. The notebook
derives `AGENT_DASHBOARD_PORT` and `AGENT_DASHBOARD_RELAY_PORT` from these values:

```bash
export NEMOCLAW_DASHBOARD_PORT="${NEMOCLAW_DASHBOARD_PORT:-18789}"
export NEMOCLAW_DASHBOARD_RELAY_PORT="${NEMOCLAW_DASHBOARD_RELAY_PORT:-18790}"
export VSS_AGENT_ADAPTER_ENABLED=true
```

After onboarding, the user runs `nemoclaw <sandbox> gateway-token --quiet`
with the name section 3.5 printed (`vss-harness-sandbox` by default) on the
deployment host and enters the result in the Web UI's **Connect NemoClaw
chat** panel. The UI checks the gateway and enables both chat surfaces when it
accepts the token. If the relay is unavailable, the panel offers a retry; if
the token is rejected, the user can enter a current one. The token stays in
the browser tab session and is sent only to the UI's same-origin agent API.
No Compose regeneration or UI container recreation is needed.

This connects the chat sidebar and Chat tab. It does not restore the Search tab
or the ingress `/api`, `/chat`, `/websocket` routes, which address the in-stack
agent directly. `openclaw-ws` is specific to OpenClaw; do not apply this block
to an explicit Hermes build.

## `nemoclaw` is never a service key

`nemoclaw` is **not** a Compose service and **must not** enter
`COMPOSE_PROFILES`, appear in `compose.yml`, or receive a file under
`patches/`. It has no image in the root Compose graph and resolution would fail
on the invented key. Treat it exactly as `<name>` is treated in the artifact
contract: a label outside the Compose model.

Selecting NemoClaw therefore never *adds* a key. All it adds is a host-side step
after deployment (below).

## What removing the agent implies

Applies to **both** Q3 answers — a `yes` and a `no` alike — and not to a build
whose request named the in-stack agent.

**Remove `vss-agent` from the Foundation's `COMPOSE_PROFILES` together with the
two peers only it uses: the `llm_*` key and `phoenix`.** They go as one unit.
The `llm_*` peer is the in-stack agent's LLM: of the `base` services only
`vss-agent` calls the LLM (`services/agent/compose.yml`; the only other consumer
is `lvs-server`, outside `base` - `alert-bridge` never calls it: its compose
passes `LLM_MODE`, which nothing in the service reads, and its URL rewriting
keys on `VLM_MODE`), and the harness brings its own model provider - the NIM is
the build's largest GPU claim and image pull.
`phoenix` collects the agent's traces and has no other client; `haproxy` only
routes to it, and tolerates an absent backend exactly as it does the absent
agent. Keep `llm_*` only when an enabled service still consumes it (an
`lvs` or combined build keeps it for `lvs-server`) or
the harness LLM is route (a) *against the build's own LLM NIM*, which needs the
NIM resident - name the key in `REQUESTED_PROFILES` so Step 5's validator keeps
it. This is the agent-owned removal Step 5's harness-only invariant applies,
and `scripts/resolve_service_graph.py` enforces. `vss-ui` stays: pruning it is
a capability decision, not a harness one.

`vss-ui`'s dependency on the agent ships as `required: false` so the filtered
project still resolves, and `scripts/normalize_resolved_yml.py` drops the
dangling entry. Never re-add a hard `depends_on` in a build override — Compose
rejects a project whose enabled service hard-depends on a filtered one, and
Step 8 fails with no `resolved.yml`.

`vss-ui` is worth keeping with no agent: its Alerts, Dashboard, and Video
Management tabs address Alert Bridge, Kibana, and VST directly. An explicit
"headless" request drops it as well — honour that, and report the loss of those
three tabs.

### What the removal costs, and what it does not

Report every one of these that the build has, whenever the agent is removed:

| Surface | Effect |
|---|---|
| Web UI chat sidebar, Chat tab, Search tab | the sidebar and Chat tab answer through NemoClaw when the adapter is wired ([Connecting the Web UI to NemoClaw](#connecting-the-web-ui-to-nemoclaw)); with no harness they stop. The Search tab stays dead either way because it addresses `/api/v1/search`, which the adapter does not replace. The Alerts, Dashboard, and Video Management tabs keep working, because they address Alert Bridge, Kibana, and VST directly — including Video Management's upload and delete, which never went through the agent |
| Alerts tab, *Generate Report* | goes with the sidebar it drives. The incident list and rule CRUD stay, on `video-analytics-api` and Alert Bridge |
| Web UI summarization on `lvs` | with no harness, gone: the UI ships no LVS client, so the capability is `vss summarize` from the host and the UI is a dashboard |
| Ingress `/api`, `/chat`, `/websocket` | `503`. HAProxy still starts — `bk_vss_agent` is declared `init-addr none` — and the origin's root still serves the UI |
| Search **ingestion and deletion** | use `vss vios add` / `delete` on Agent-backed and headless deployments; configured notification receivers perform fan-out and cleanup |
| `vss-generate-video-report-rag` | unavailable: it drives the agent's `/v1/chat` and `/executions`. Route reports through `vss-generate-video-report`, which never calls the agent |

Nothing else in the operate set needs it. No `vss` command group declares the
agent a requirement: `configure` probes each route independently and simply omits
`/api`, while `summarize`, `search` (query), `vlm`, `vios`, and `memory` address
LVS, Elasticsearch, RT-Embed, RT-VLM, and VIOS directly. `vss-ui` holds the only
`depends_on` in the whole graph, and it is optional; no other service in any
profile calls the agent, so alerting, analytics, ingest, and summarization are
unaffected.

### Source provisioning on every deployment

On Agent-backed and headless deployments, source provisioning follows
`vss-manage-video-io-storage`
[`provision-vios-source.md`](../../operations/vss-manage-video-io-storage/references/provision-vios-source.md):
register one source with `vss vios add`, which the mounted notification config
fans out to enabled receivers. There is no Agent-presence gate or `/complete`
prerequisite. Check the receivers for the requested capability before claiming
indexing readiness. Alert rules stay with `vss-manage-alerts`, which addresses
Alert Bridge.

### Ingress is still required

The sandbox reaches the build only over host-published HTTP, and the host-CLI
read path has no ingress-less form (see
[`deployment_resolution.md`](deployment_resolution.md) and
[`services/ingress.md`](services/ingress.md)). `vss-haproxy-ingress` must be in
the effective service set, carrying the operate route-set. A request that pairs
NemoClaw with "no ingress" is a **capability contradiction** — take it to the
clarification gate; do not resolve it by dropping either side.

Shipping the ingress is necessary but not sufficient. HAProxy 404s any `Host`
header outside its `known_host` allowlist, which admits only `VSS_PUBLIC_HOST`,
`EXTERNAL_IP`, `HOST_IP`, and localhost ([`services/ingress.md`](services/ingress.md)).
The sandbox reaches the host as `host.openshell.internal`, so every Compose
NemoClaw build 404s on its own origin until that name holds one of those slots.
Set `EXTERNAL_IP=host.openshell.internal` in `_builds/<name>/override.env` before
resolving, and comment the line. `EXTERNAL_IP` is the slot to spend: `HOST_IP`
must stay bridge-reachable, and the URLs users follow resolve from
`VSS_PUBLIC_HOST`. Do not ask the user — it is a mechanical consequence of
choosing NemoClaw.

Not on `alerts`: `alert-bridge` rewrites clip URLs from `INTERNAL_IP` to
`EXTERNAL_IP`, so repointing it makes alert evidence unopenable. There the
curated `haproxy.cfg` is **required** — admit `host.openshell.internal` per
[`services/ingress.md`](services/ingress.md) and leave `EXTERNAL_IP` alone.
Do not settle for the origin 404 on the grounds that alerts operate reaches
Alert Bridge and VA-MCP on their host ports: it costs the sandbox the ingress
origin every other skill resolves against.

### Either answer makes it a delta build

Removing `vss-agent` is a capability delta, so a named profile that reaches Q3 is
a **delta build** on a `no` as much as a `yes` — create `_builds/<name>/` and
follow Delta mode from Step 2. Only a request that names the in-stack agent, and
so never reaches Q3, can stay a stock deploy.

### Cost to report, not to optimize away

Two things the user should hear up front rather than discover:

- **GPU and memory budget.** `vss-agent` reserves no GPU, so its removal frees
  memory rather than a device; pruning its `llm_*` peer (above) is what frees
  GPU memory, and a build that keeps the NIM for `lvs-server` or
  for route (a) must say so. Budget the build against [`sizing.md`](sizing.md)
  plus the harness's own model provider —
  and note that a NemoClaw-managed local model claims every visible GPU unless
  pinned (see [Prerequisites](#prerequisites)).
- **One harness, two entry points.** On a default OpenClaw `yes`, the build UI
  chat and Agent UI both reach NemoClaw. Report **both as markdown links** and
  name NemoClaw as the driver. On a `no`, report the browse origin alone and
  name the `vss` CLI as the driver. The build's
  origin is `VSS_PUBLIC_HOST`; on Brev that is the FQDN the context file
  publishes for the ingress port, resolved rather than constructed
  ([`brev.md`](brev.md)). Never `EXTERNAL_IP`, which on a NemoClaw build holds
  `host.openshell.internal` and resolves only inside the sandbox.

## Ordering

The harness needs the build's origin, and that origin does not exist until the
deployment answers. Run the steps in this order and no other:

| # | Step | Why here |
|---|---|---|
| 1 | Deploy `resolved.yml` | nothing to point a harness at yet |
| 2 | Readiness gate ([`readiness.md`](readiness.md)) | a harness pointed at a half-warm stack reports failures that are not its own |
| 3 | Resolve the origin | `http://$HOST_IP:$HAPROXY_HOST_PORT` from the deployed build |
| 4 | Bring up the harness | consumes that origin |

Skipping the readiness gate is the common failure: onboarding succeeds, the
first call from the sandbox fails, and the cause looks like the harness.

## Prerequisites

Beyond everything in [`prerequisites.md`](prerequisites.md) and
[`credentials.md`](credentials.md), the harness step needs the following.

**Check them at Q3, not at bring-up.** NemoClaw is the default, so a build can
reach these requirements without anyone having asked for them. Any one missing is
a **blocker at harness selection**: name it, ask whether to supply it, proceed
with no harness, or name the in-stack agent instead, and deploy nothing until
that is answered. Discovering it after the readiness gate means a deployed build
with no way to drive it.

- **`uv`, `docker`, `python3`, and `curl` on `PATH`**, and outbound reach to the
  installer. The host's own Python version is not a requirement here — the
  bring-up below pins its interpreter with `uv run --python 3.12`. **Do not
  require the NemoClaw CLI here**: section 3.1 installs it at the pinned
  `NEMOCLAW_INSTALL_REF` whenever that ref is not already present, so a fresh
  host is a supported starting point and preflighting the post-install CLI would
  reject one. The notebook's own preflight (section 2) reports on `docker`,
  `python3`, and `curl` without failing the run, so it neither informs the
  harness choice nor stops a host that is short one of them.
- **An agent model provider**, and only the credential that provider needs. This
  is the harness's *own* LLM, unrelated to the build's `LLM_*` and `VLM_*` knobs.
  The notebook offers three — (a) an OpenAI-compatible endpoint, (b) a
  NemoClaw-managed local model, (c) a build.nvidia.com hosted model — and **this
  skill defaults to (a) serving Claude Opus 5** (see [Default
  provider](#default-provider) below). The provider is a default; the endpoint,
  model id, and bearer token it needs are **collected from the user** at Q3a,
  which is where a missing credential surfaces. Section 1.2 of the notebook
  remains the authority on which variables each provider needs; do not infer
  them, and do not preflight a variable a different provider would have used.
- **A GPU budget that accounts for the harness.** The default remote provider
  costs no GPU. This applies only when the user overrides to the local
  provider: (b)
  `install-vllm` takes every visible GPU unless `NEMOCLAW_VLLM_GPU_DEVICE` pins
  it, which will strand the build's own models. Reconcile that against
  [`sizing.md`](sizing.md) before choosing it, not after.
- **The checkout's own assets**: `assets/vss_nemoclaw_policy.yaml`, `skills/`,
  and `.openclaw/` (the sandbox `Dockerfile`, the VSS OpenClaw
  plugin and the `workspace/` docs). The notebook resolves them from
  `VSS_REPO_DIR`; for OpenClaw, onboard builds the sandbox image from that
  Dockerfile (`--from`), so the skills and docs arrive baked rather than
  installed. Do not build a sandbox image of your own for this.
- **`NEMOCLAW_DASHBOARD_PORT` (`18789`) and `NEMOCLAW_DASHBOARD_RELAY_PORT`
  (`18790`) free, or held by this build's own sandbox.** See [Ports the harness
  claims](#ports-the-harness-claims) — the one prerequisite whose remedy is the
  user's to run, because clearing it destroys someone else's sandbox.

Preflight the selected provider's row below and no other — a credential check
that fires for every build rejects the supported paths that need no key:

| Provider | Required at Q3 | Not required |
|---|---|---|
| (a) public OpenAI-compatible endpoint — the skill default | `NEMOCLAW_ENDPOINT_URL`, `NEMOCLAW_MODEL`, `COMPATIBLE_API_KEY` | `NVIDIA_API_KEY` |
| (a) self-hosted endpoint, or one on a private address — including the build's own LLM NIM | `NEMOCLAW_ENDPOINT_URL`, `NEMOCLAW_MODEL`, `NEMOCLAW_INFERENCE_PROXY=0`; use the supplied `COMPATIBLE_API_KEY`, or `EMPTY` only when the endpoint has no authentication | `NVIDIA_API_KEY` |
| (b) NemoClaw-managed local model | `NEMOCLAW_PROVIDER` (`install-vllm`, `ollama`, `nim-local`, …) | any API key; `HF_TOKEN` only for a gated `install-vllm` model |
| (c) build.nvidia.com hosted model | `NVIDIA_API_KEY` | `COMPATIBLE_API_KEY`, `NEMOCLAW_ENDPOINT_URL` |

`NEMOCLAW_INFERENCE_PROXY` follows the upstream's transport rather than the row
above. Leave the default `1` when the endpoint is HTTPS on 443: it is inert for a
public address, and it is what rescues a public name that corp or DGX DNS
resolves to a private one — the case the proxy was built for. Set `0` when the
upstream is plain HTTP or on another port, which the proxy cannot represent.

Egress from the sandbox to the build is already allowed: the shipped policy's
`vss-backend` entries cover the HAProxy origin (`7777`) along with each
backend's own host port. **A build that moves `HAPROXY_HOST_PORT` off `7777`
has no matching policy entry**, so every call from the sandbox returns
`CONNECT tunnel failed, response 403`. Report that as a blocker naming the port;
the policy is a checked-in asset, not something to rewrite per build.

The LLM NIM's published port is in the same position. `vss-backend` allowlists
literal `host` + `port` pairs and interpolates nothing, and neither NIM port is
among them — `30081`, which `LLM_PORT` resolves to, nor `30082` for a VLM NIM.
A harness pointed at the build's NIM therefore relies on NemoClaw authorizing
the inference endpoint it was onboarded with. A `403 CONNECT tunnel failed` on
the harness's first turn is that missing entry rather than a deployment fault —
report it naming the port, and do not fall back to a remote provider to get a
working chat.

### Ports the harness claims

The gateway forward binds `NEMOCLAW_DASHBOARD_PORT` (`18789`) and the section
3.5 relay binds `NEMOCLAW_DASHBOARD_RELAY_PORT` (`18790`). Neither is ever
taken from its holder — a held `18789` [stops onboard or moves the sandbox
elsewhere](#troubleshooting-port-18789-is-not-available), and the relay cell
stops on a foreign listener on `18790` — and every sandbox defaults to the
same two ports, so any sandbox still running on this host holds them.

Probe both on a **yes** to Q3, before accepting it and with the rest of the
prerequisites — a build with no harness claims neither port. Probe the values
this build will bind: an override from the environment or the request, which
Step 7 records, or the defaults when there is none. The bind test decides
free or held and needs only `python3`; `lsof` or `ss` only names the holder,
and a host may lack both:

```bash
# Absent until notebook cell 3.1 installs it, so do not let it fail the probe.
command -v openshell >/dev/null \
  && openshell sandbox list                         # compare against NEMOCLAW_SANDBOX_NAME
# A name is not a claim. Print which build claimed which name: a sibling build
# holding the default name is as foreign here as a stranger's sandbox.
grep -H . "$(git rev-parse --show-toplevel)"/_builds/*/sandbox 2>/dev/null
for port in "${NEMOCLAW_DASHBOARD_PORT:-18789}" "${NEMOCLAW_DASHBOARD_RELAY_PORT:-18790}"; do
  python3 -c 'import socket,sys; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(("",int(sys.argv[1])))' "$port" 2>/dev/null \
    && { echo "$port free"; continue; }
  # The holder's own command line is what names its sandbox: the forward is an
  # `openshell ... forward service <name>` process, so no --sandbox pattern
  # finds it, and the PID alone says nothing about ownership.
  echo "$port held by:"
  for pid in $(lsof -tnP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null \
    || ss -Hltnp "sport = :$port" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2); do
    printf '  %s: ' "$pid"; tr '\0' ' ' 2>/dev/null <"/proc/$pid/cmdline"; echo
  done
done
# stderr first: a process that exits mid-scan makes the shell's own open fail.
# `--sandbox` is required, not decoration: the echo below carries the relay path
# in this block's own command line, which `[d]` does not hide from the scan.
for f in /proc/[0-9]*/cmdline; do tr '\0' ' ' 2>/dev/null <"$f"; echo; done \
  | grep -E '[d]ashboard-(relay|forward-watchdog)\.py .*--sandbox '  # --sandbox names the owner
# A relay must carry this path to count as ours, not just the sandbox name. Set
# it from the ref decision: VSS_REPO_DIR is not exported until bring-up, so
# reading it here silently names the wrong checkout.
HARNESS_SRC="$(git rev-parse --show-toplevel)"      # ref build: <that>/_builds/<name>/harness-src
echo "this build's relay: $HARNESS_SRC/deploy/docker/scripts/nemoclaw/dashboard-relay.py"
```

**No `openshell` on the host is a pass, not a failed probe.** Cell 3.1 installs
it, so the fresh host Q3 supports has none — and with none, no sandbox is
running to hold either port. Both ports free and no relay or watchdog in the
process list is the whole preflight satisfied: accept the yes and continue. A
held port still blocks even when nothing can name its holder.

**Read ownership off the holder's command line, never off the port.** The
forward prints `forward service <name>`, the relay and watchdog print
`--sandbox <name>`, and the name to match is
`${NEMOCLAW_SANDBOX_NAME:-vss-harness-sandbox}`. With several sandboxes on the
host, the listing and the PID settle nothing on their own.

**A matching name is not ownership; this build's own `_builds/<name>/sandbox`
record is.** Written before bring-up runs, it is the only thing that says
which sandbox this build may replace. The default name proves nothing on its
own: `deploy_nemoclaw.ipynb` run by hand names its sandbox
`vss-harness-sandbox` too, and [Teardown](#teardown) leaves exactly that one
standing as unowned. **A sibling build's record is not this build's claim
either** — several builds in one checkout take the same default, and the
notebook recreates by name without reading any record, so accepting another
`_builds/*/sandbox` here is how a build discards a sandbox and sessions
someone is still using. Read every record to learn who holds the name; own
only the one written under the build being deployed. A build on its first
deploy has no record and so owns nothing, which is right — it has onboarded
nothing yet. A re-onboard of that same build is what the record makes owned.

**The relay must match the bring-up's own script path as well as the name.**
Section 3.5 keeps a relay only when its command line carries the resolved
`deploy/docker/scripts/nemoclaw/dashboard-relay.py` under `VSS_REPO_DIR`, so a
same-name relay from a second clone is foreign however familiar its name looks.
Apply that test here or the build deploys and then fails at 3.5 with `Port
<relay-port> is held by pid …, which is not '<sandbox>'s dashboard relay` — the
conflict Q3 exists to catch, surfacing after the cost.

**Take that path from the ref decision, never from `VSS_REPO_DIR`.** The export
does not exist until bring-up, and a [harness source ref](#harness-source-ref)
points it at the `_builds/<name>/harness-src` worktree rather than the
checkout. So on a ref build the checkout's own relay is foreign — reading the
unset variable at Q3 would adopt it as this build's and approve its port.
A re-onboard of the same build is the one case a worktree relay is owned,
because that worktree survives for exactly that purpose.

The unset case is not the checkout either: the notebook defaults
`VSS_REPO_DIR` to `$HOME/video-search-and-summarization`, so a checkout
anywhere else must export it — the [Bring-up](#bring-up) block does — or
bring-up reads its assets and its relay from a path the probe never looked at.

A holder this build owns is not a conflict: `NEMOCLAW_RECREATE_SANDBOX=1`
replaces the sandbox named `NEMOCLAW_SANDBOX_NAME`, and the relay cell replaces
that name's relay from this checkout. Do not assume that case — ownership is
this build's own record naming the holder, and for a relay the resolved script
path as well, so a deployment that named itself leaves a sandbox and a relay
foreign to the next run, as do a sibling build's sandbox and a same-name relay
from another checkout.

**A holder this build's own record does not name is the one case to ask about
rather than hand over.** The notebook takes the name regardless: section 3.1
adds `--recreate-sandbox` for it and discards that sandbox's agent sessions.
So put the choice before the deploy, not in the final summary — say which name
is held, which build recorded it or that none did, and whose sessions a yes
discards. Take either a yes to recreate it, or another `NEMOCLAW_SANDBOX_NAME`
with a free port pair to go with it, which leaves that sandbox running and
onboards beside it. Two ordinary situations land here: a fresh checkout on a
host that has built before — an eval box — and a second build beside one whose
harness is still up. Both are the question doing its job, not a false positive
to wave through.

**Anything else is a hard blocker**, including a held port nothing could name —
report it as held by an unidentified listener. Report what holds which port, hand the
block below over, and **do not proceed until no foreign holder remains** —
destroy nothing and kill nothing on the user's behalf. A port the sandbox in
this build's own record still holds is not what that waits on, nor is one the
user has just agreed to recreate; `NEMOCLAW_RECREATE_SANDBOX=1` replaces
either. Stopping here costs nothing: Q3 precedes every build artifact.

```bash
# Watchdog first: it answers a dying forward with `nemoclaw recover`. Relay
# last: `destroy` releases the forward, never the relay. Drop the `destroy`
# when `openshell sandbox list` no longer shows <other>.
OTHER='<other>'
# pkill -f compiles an extended regex, so escape every metacharacter in the
# name: an unescaped one matches other sandboxes' processes too, and these
# scripts are shared by every sandbox on the host.
OTHER_RE="$(printf '%s' "$OTHER" | sed 's/[][(){}.*+?^$|\\]/\\&/g')"
pkill -f -- "dashboard-forward-watchdog\.py --sandbox ${OTHER_RE}( |$)"
nemoclaw "$OTHER" destroy --yes --cleanup-gateway
pkill -f -- "dashboard-relay\.py --sandbox ${OTHER_RE}( |$)"
```

Re-probe both ports and resume once each is free or back to a holder this
build owns.

## Default provider

**Default to notebook option (a) — the remote OpenAI-compatible endpoint —
serving Claude Opus 5 through the NVIDIA Inference Hub.** First ask the user
which model the sandbox runs on, per [Harness model —
Q3a](../SKILL.md#harness-model--q3a). Another model on this same endpoint
replaces `NEMOCLAW_MODEL` alone and keeps every other value below. Only a
request for a different provider shows the notebook's three provider options
(a), (b), and (c), whereupon collect only the selected provider's settings.

| Variable | Default | Note |
|---|---|---|
| `NEMOCLAW_PROVIDER` | `custom` — not asked | option (a); the only provider that consumes `NEMOCLAW_ENDPOINT_URL` |
| `NEMOCLAW_MODEL` | `aws/anthropic/bedrock-claude-opus-5` | Claude Opus 5 as the Inference Hub routes it. The id is **the endpoint's, not the model's**: `azure/anthropic/claude-opus-5` is the same model over another route, a provider's public API serves the bare `claude-opus-5`, and a model router takes a **route id**. Replace it whenever the endpoint changes |
| `NEMOCLAW_ENDPOINT_URL` | `https://inference-api.nvidia.com/v1` | the NVIDIA Inference Hub gateway, reachable from NVIDIA infrastructure — the same upstream the bundled `inference-api-proxy.py` defaults to. Any OpenAI-compatible base URL replaces it: another gateway, a model router, a self-hosted server, or a provider's public API |
| `COMPATIBLE_API_KEY` | **no default** | a real bearer token is required for a public endpoint. Take it from the environment or the platform secret store — never a literal in a command, a file, or skill output |

### Where the user gets each value

What to tell a user who selects option (a) but cannot use its defaults or does
not have the key.

| Value | Where it comes from |
|---|---|
| A key for the default endpoint | Create one at <https://inference.nvidia.com/key-management?action=new-key>; the model ids the Hub serves are listed at <https://inference.nvidia.com/?new=0>. Its gateway answers from NVIDIA infrastructure, so a host outside that network takes one of the rows below — settle that before onboard, not after the first agent turn fails |
| Another gateway or router the user runs | Whoever operates it, plus its own model list for the matching Opus id. Usually the endpoint their other tooling already points at, so ask about that first when the default does not apply. A router's route ids come from [`deploy_vss_switchyard.ipynb`](../../../deploy/docker/scripts/deploy_vss_switchyard.ipynb), and it forwards the caller's credential upstream, so the key stays the **upstream** provider's |
| A provider's public API key | That provider's own documentation, for both the key and the model ids it serves. Point the user there rather than describing a key format or a console layout this skill does not own |
| No key at all | The build's own LLM NIM, per *(a) against the build's own LLM NIM* below (`NIM_SERVED_MODEL_NAME` and `COMPATIBLE_API_KEY=EMPTY`) on a build that resolved `LLM_MODE=local` or `local_shared`; or option (c) with an `nvapi-…` key from <https://build.nvidia.com>, which replaces this route and serves Nemotron rather than Opus |

Two cases need no question at all. **A provider explicitly configured in the
environment with all of its required values is the answer**: values exported
by the caller, by CI, or by a platform secret store win over anything Q3a
collects, so read them first and confirm what was found instead of re-asking.
And in **autonomous mode**
([`SKILL.md`](../SKILL.md#exception--autonomous-mode)) the caller's instruction
plus that environment answer Q3a; a token missing from both is still a blocker
to report, never grounds to substitute a provider.

Requesting a local model, or any model this endpoint does not serve, moves
the build to the provider-selection question. "Use a local model", "air-gapped",
"use Nemotron", or a named endpoint of their own already answers that question,
so carry it through rather than re-asking. A local request has **two**
destinations, not one:

| Request | Route | Where the model runs |
|---|---|---|
| "reuse the deployment's LLM", "one model for both", "point it at the LLM NIM" | (a) `custom` against the build's own NIM | the build's `llm_local*` container — no additional GPU |
| "use a local model", "air-gapped", with no server named | (b) a NemoClaw-managed server (`install-vllm`, `ollama`, `nim-local`) | a second server NemoClaw starts and owns, on its own GPU |
| a named endpoint of their own | (a) `custom` against that endpoint | wherever they run it |

**(a) against the build's own LLM NIM** is the cheapest local harness and the
least obvious route, so it is spelled out here. It applies only to a build whose
`LLM_MODE` resolved to `local` or `local_shared`; one resolved to a remote LLM
has no local NIM to point at.

| Variable | Value |
|---|---|
| `NEMOCLAW_PROVIDER` | `custom` |
| `NEMOCLAW_ENDPOINT_URL` | `http://host.openshell.internal:<LLM_PORT>/v1`, with `LLM_PORT` read from the build's `resolved.yml` (`30081` on every current profile). Use that hostname rather than `HOST_IP` — the egress policy's entries are keyed on it |
| `NEMOCLAW_MODEL` | the NIM's `NIM_SERVED_MODEL_NAME` from `resolved.yml`, e.g. `nvidia/nemotron-3.5-lightning-30b-a3b`; never assume the slug |
| `COMPATIBLE_API_KEY` | `EMPTY` — set it explicitly, per the self-hosted failure mode below |
| `NEMOCLAW_INFERENCE_PROXY` | `0` — required, per that same failure mode |

Say what the route costs **before** taking it, so the user can pick (b) or the
remote default instead: the harness adds no GPU but shares one NIM instance with
the build's own LLM work, at `NIM_KVCACHE_PERCENT=0.3` and
`NIM_MAX_MODEL_LEN=65536` on the shared-GPU profiles. On a build whose
capabilities drive that NIM — `lvs` summarization, `search` critique — agent
turns and the build's own requests contend for KV cache.

[Ordering](#ordering) already covers the timing, and it matters more here: the
NIM is the slowest service in the build on a cold cache, and it has to answer on
that port before the notebook runs. Onboard is the only step that applies the
endpoint, so changing it afterwards needs the sandbox recreated.

Two failure modes to handle rather than paper over:

- **No API key available for a provider that needs one.** Report it as a blocker
  and stop. Never silently substitute (c) build.nvidia.com or a local model: the
  harness would come up on a different LLM than the one reported, and nothing
  downstream could tell. Switching providers is the user's call to make on that
  report, not a fallback to take for them.
- **A private or self-hosted endpoint.** Section 3.1 picks the transport from the
  endpoint's address, not from the provider name: a host resolving to a private
  address gets the bundled proxy, and a blank key is sent as `EMPTY`. That proxy
  reaches its upstream as `https://<host>` on 443 and drops the port, so it is
  **wrong for an endpoint served over plain HTTP or on another port** — a local
  NIM, vLLM, or Ollama. Export `NEMOCLAW_INFERENCE_PROXY=0` for those, and pass
  `COMPATIBLE_API_KEY=EMPTY` yourself, since the placeholder is applied only on
  the branch that export turns off. Disabling the proxy leaves NemoClaw's own
  SSRF guard to accept the endpoint; an onboard that rejects it is the case the
  proxy exists for — report it rather than re-aiming the harness at another
  provider. A loopback-only server must be rebound to `0.0.0.0`, because the
  sandbox reaches this host as `host.openshell.internal`.

These four variables are exactly the parameter contract
`run_setup_notebook.py` already carries for this notebook, so exporting them is
sufficient — the injection re-reads them after the section 1.2 cells have run,
which is what lets the export win over whichever provider cell executed last.
`NEMOCLAW_INFERENCE_PROXY` is not in that table and must not be added to it:
the injection assigns the raw environment string and section 3.1 accepts only a
real bool. Section 1.3 reads it from the environment itself, so exporting `0`
works the same way.

## Bring-up

Do not reimplement any of this. `deploy_nemoclaw.ipynb` is the single source of
host-side harness logic — sandbox onboarding, policy, skill install, workspace
docs, webhooks, UI link — and
`deploy/docker/scripts/run_setup_notebook.py` executes it non-interactively.

Its section 2.1 also runs `deploy/docker/scripts/pin_docker_version.sh`, which
the Docker gate in [`prerequisites.md`](prerequisites.md#docker-pin) already ran
at Step 3. That is deliberate: the script is idempotent and only re-applies the
`apt-mark hold`s on a host already pinned. **A host that skipped Step 3 gets its
Docker downgrade here, with the build running** — dockerd restarts under it. The
pin belongs at Step 3 for that reason; reaching this step unpinned is a
prerequisite that was missed, not a step the harness owns.

Set the environment, then run the notebook:

| Variable | Value for a build | Why |
|---|---|---|
| `VSS_REPO_DIR` | the checkout root | resolves the policy, skills, and workspace docs |
| `VSS_PUBLIC_URL` | **leave unset** for a Compose build | the deployment origin `vss configure` records; empty means this host's Compose deployment and 3.2 fills it in — see [`VSS_PUBLIC_URL` is the deployment origin](#vss_public_url-is-the-deployment-origin---leave-it-empty-on-compose) below |
| `NEMOCLAW_SANDBOX_NAME` | `vss-harness-sandbox` unless already set | a new build replaces the sandbox of the same name, which is why Q3 settles first whether that name is this build's to take |
| `NEMOCLAW_RECREATE_SANDBOX` | `1` | onboard is the only step that applies the provider, endpoint, model and key, so a reused sandbox would run on whatever it was onboarded with. Section 3.1 adds `--recreate-sandbox` when a sandbox of that name exists, discarding it and its agent sessions |
| `AGENT_RUNTIME` | `openclaw` (default) or `hermes` | selects the harness profile; a change needs a fresh onboard |
| `NEMOCLAW_DASHBOARD_PORT` | selected port; default `18789` | NemoClaw's own forward, loopback only |
| `NEMOCLAW_DASHBOARD_RELAY_PORT` | selected port; default `18790` | the section 3.5 relay the UI adapter backend URL must use (`ws://host.docker.internal:<relay-port>`); the Brev secure link and `CHAT_UI_URL` publish this port |
| `VSS_AGENT_ADAPTER_ENABLED` | `true` when connecting `vss-ui` to OpenClaw | the relay is what makes the gateway reachable from the container; this flag turns the UI's adapter on |
| `NEMOCLAW_DASHBOARD_WATCHDOG` | leave unset (on) | section 3.5 starts a host watchdog that repairs a dashboard forward holding its port without carrying HTTP; the gateway token is unchanged by a repair |
| `NEMOCLAW_PROVIDER`, model settings, and the selected provider's credential | the Q3a answers, per [Default provider](#default-provider) | remote Claude Opus 5 when the user accepts the default; otherwise the exact notebook provider and settings selected in Q3a. The block below spells out the default remote route alone; every other route **replaces** these values rather than defaulting through them |
| `NEMOCLAW_INFERENCE_PROXY` | unset, or `0` against a local endpoint | `0` is required when (a) points at the build's own LLM NIM, or at any plain-HTTP server: the default rewrites such an endpoint to an `https` upstream on 443 |
| `ORCHESTRATOR_ENABLE_HTTPS` | `false` | leave at the default; the HTTPS MCP path is a separate opt-in |

```bash
set -o pipefail   # report the notebook's status, not `tee`'s
umask 077         # the setup log echoes the notebook's own settings dump

REPO="$(git rev-parse --show-toplevel)"
BUILD_NAME="<name>"                                # the build's _builds/<name>/ directory

export VSS_REPO_DIR="$REPO"
export NEMOCLAW_SANDBOX_NAME="${NEMOCLAW_SANDBOX_NAME:-vss-harness-sandbox}"
export NEMOCLAW_RECREATE_SANDBOX=1
export NEMOCLAW_DASHBOARD_PORT="${NEMOCLAW_DASHBOARD_PORT:-18789}"
export NEMOCLAW_DASHBOARD_RELAY_PORT="${NEMOCLAW_DASHBOARD_RELAY_PORT:-18790}"
export VSS_AGENT_ADAPTER_ENABLED=true

# Harness LLM: notebook option (a), Claude Opus 5 through the NVIDIA
# Inference Hub. Substitute the Q3a answers whenever the user replaced a
# default; the `:-` form lets an already-exported value win — right for a
# caller-supplied endpoint or CI, and exactly why these lines must not be
# copied as-is for the NIM route, where a stale endpoint and model would
# survive.
export NEMOCLAW_PROVIDER=custom
export NEMOCLAW_MODEL="${NEMOCLAW_MODEL:-aws/anthropic/bedrock-claude-opus-5}"
export NEMOCLAW_ENDPOINT_URL="${NEMOCLAW_ENDPOINT_URL:-https://inference-api.nvidia.com/v1}"
# From the environment or the secret store; never a literal here. A key the user
# dropped in a file is read in first: set -a; . "$REPO/nemoclaw.env"; set +a
: "${COMPATIBLE_API_KEY:?bearer token for NEMOCLAW_ENDPOINT_URL is required}"
export COMPATIBLE_API_KEY

# Against the build's own unauthenticated LLM NIM, replace the endpoint, model,
# and key outright. A different endpoint that requires authentication (such
# as the skill eval's local NIM proxy) must use that endpoint's supplied key:
#   export NEMOCLAW_ENDPOINT_URL="http://host.openshell.internal:<LLM_PORT>/v1"
#   export NEMOCLAW_MODEL="<NIM_SERVED_MODEL_NAME from resolved.yml>"
#   export COMPATIBLE_API_KEY=EMPTY
#   export NEMOCLAW_INFERENCE_PROXY=0

# Record the name BEFORE the run, not on success: 3.1 onboards and 3.2-3.5 keep
# configuring, so a failure in between leaves a live sandbox that teardown
# reaches only through this file. It is also this build's claim on the name at
# the next Q3 probe, which owns no other build's record, whatever it is called.
printf '%s\n' "$NEMOCLAW_SANDBOX_NAME" \
  > "$REPO/_builds/${BUILD_NAME}/sandbox"

uv run --isolated --no-project --python 3.12 \
  --with nbformat --with nbclient --with ipykernel -- \
  python "$REPO/deploy/docker/scripts/run_setup_notebook.py" \
    --notebook "$REPO/deploy/docker/scripts/deploy_nemoclaw.ipynb" \
    --require-output "Sandbox '${NEMOCLAW_SANDBOX_NAME}' ready." \
    --echo-output \
  2>&1 | tee "$REPO/_builds/${BUILD_NAME}/nemoclaw-setup.log"
```

**Take the status from the notebook, not from `tee`.** Keep `pipefail` set, or
read `${PIPESTATUS[0]}` on the line right after the pipeline. Non-zero is a
blocker: report it with the log path and stop, rather than going on to the UI
link. Say that the build may hold a live sandbox and name it — the claim is on
disk, so [Teardown](#teardown) removes it like any other build's.

**`--echo-output` is what puts the notebook's output in the log.** Without it the
runner keeps every output in memory, prints one summary line, and discards the
rest — so section 3.5's `Sandbox:` and `Agent UI:` lines, and the `WARNING:`
that cell prints when the dashboard forward does not come up, never reach the
`tee`d log. `--require-output` matches the in-memory copy either way, so a run
that omits the flag exits 0 with a log that proves nothing. Keep the whole run
for the same reason: `| tail` or `| head` drops those lines back out.

The flag redacts the UI link's `#token=` fragment, so the log gives you the
origin, the sandbox name and that warning — never a live token. Read the log for
which origin the notebook *chose*; it is the only record of that decision, and
it can differ from what the context file publishes when the notebook's own read
of `/etc/brev` was denied.

Do not reconstruct the URL from the notebook source. Its origin branches on
whether the Brev context file publishes a secure link for the relay port.
Outside the notebook, resolve that FQDN from the context file
([`brev.md`](brev.md) → *Resolving a secure link*) rather than assembling a
hostname.

### `VSS_PUBLIC_URL` is the deployment origin - leave it empty on Compose

Compose and Kubernetes follow one contract in the sandbox: `vss configure
--base-url <origin>` records the path routes behind a single origin, and every
operation skill uses the recording. `VSS_PUBLIC_URL` is that origin. **Empty
means the Compose deployment on this host**: section 3.2 fills in the haproxy
origin `http://host.openshell.internal:<HAPROXY_PORT>` (already allowlisted in
the `vss-backend-readwrite` egress entry), uploads it in `ENV.md`, runs
`vss configure` and `vss configure check` inside the sandbox and prints the
availability table. Set it only for a Kubernetes deployment, to that cluster's
Ingress origin; 3.2 then also names the host in the `vss-k8s-ingress` egress
entry. Do not set it to the Compose origin by hand - the notebook derives it,
and it is `host.openshell.internal` as the sandbox sees it, not `HOST_IP`.

Run **only** `deploy_nemoclaw.ipynb`. Its companion,
`deploy_vss_orchestrator.ipynb`, exists so the sandbox can deploy and manage VSS
itself — work this skill has already done by the time the harness comes up.
Add it as a second `--notebook` only when the user explicitly wants the agent to
own the deployment lifecycle too.

### Harness source ref

Only when SKILL.md's [Harness source ref](../SKILL.md#harness-source-ref)
selected one. The sandbox Dockerfiles fetch `develop` unless their build
context holds a staged snapshot (`.vss-src/`), and the notebook takes every
harness asset — the Dockerfile, the plugin, the policy, the skills, the
workspace docs — from `VSS_REPO_DIR`. So build the whole harness from one
revision: a worktree of the ref, staged with that ref's own script, as
`VSS_REPO_DIR`. Run this **before** the notebook, in place of the plain
`export VSS_REPO_DIR="$REPO"` above:

```bash
REF="<ref>"                                        # e.g. nightly-20260928, v3.3.0, a sha
SRC="$REPO/_builds/${BUILD_NAME}/harness-src"

# Resolve REF to what origin has *now*, never to a stale local copy: a tag is
# force-fetched (`+`, so a moved tag updates the local one), a branch or full sha
# is read from the FETCH_HEAD of that same fetch, and only a sha the remote will
# not serve by name (an abbreviated one) falls back to the local object store.
COMMIT=""
if git -C "$REPO" fetch -q origin "+refs/tags/$REF:refs/tags/$REF" 2>/dev/null; then
  COMMIT="$(git -C "$REPO" rev-parse --verify -q "refs/tags/$REF^{commit}" || true)"
elif git -C "$REPO" fetch -q origin "$REF" 2>/dev/null; then
  COMMIT="$(git -C "$REPO" rev-parse --verify -q "FETCH_HEAD^{commit}" || true)"
elif [[ "$REF" =~ ^[0-9a-f]{7,40}$ ]]; then
  COMMIT="$(git -C "$REPO" rev-parse --verify -q "$REF^{commit}" || true)"
fi
[ -n "${COMMIT:-}" ] || { echo "harness ref $REF does not resolve" >&2; exit 1; }
git -C "$REPO" show "$COMMIT:.openclaw/Dockerfile" | grep -q 'vss-sr\[c\]' \
  || { echo "$REF predates staged harness builds" >&2; exit 1; }

git -C "$REPO" worktree remove --force "$SRC" 2>/dev/null || true
GIT_LFS_SKIP_SMUDGE=1 git -C "$REPO" worktree add -q --detach "$SRC" "$COMMIT"
python3 "$SRC/skills/vss-build-vision-ai/scripts/stage_vss_src.py" \
  --repo-root "$SRC" .openclaw .hermes

export VSS_REPO_DIR="$SRC"                         # every harness asset from the ref
```

The staging line prints the snapshot's sha and version (for example
`sha 1942cc2fe0 ... version 3.3.0rc0.post1.dev21+g1942cc2fe0`); quote both in
the summary — that version is what `vss --version` will report inside the
sandbox. Keep `--notebook` on `$REPO/deploy/docker/scripts/deploy_nemoclaw.ipynb`
(the runner is the checkout's; the assets are the ref's). `GIT_LFS_SKIP_SMUDGE`
keeps the worktree to source files; the harness needs no LFS object.

The worktree is the build's, not the checkout's: it stays under
`_builds/<name>/` for a later re-onboard, and [Teardown](#teardown) removes it.

The OpenClaw sandbox image ships only the **operation** skills — the ones whose
`SKILL.md` declares `vss-requires` — and activates those the recorded deployment
can serve (`vss-openclaw-sync` — or `vss-hermes-sync` on Hermes — inside the sandbox re-selects after
`vss configure`). It does **not** receive this skill: the sandbox operates the
build it was given and is not expected to compose further builds or manage the
`_builds/` tree this run produced. A Hermes sandbox (`.hermes/`) ships the same operation skills and
does not receive it either.

## Verification

The notebook runs with errors fatal and asserts each step itself, so a clean
exit already means onboarding, policy, skills, and workspace docs all landed.
Confirm the two things that exit code cannot cover:

1. **The harness is reachable.** Section 3.5 prints `Sandbox: <name>` and
   `Agent UI: <url>`. The OpenClaw URL carries the gateway token in `#token=`;
   treat the complete URL as a secret. In the final summary, remove the
   fragment and report only the token-free origin as a markdown link. Never
   copy the token or the secret-bearing URL into the response, and do not send
   the user to the setup log for it — `--echo-output` redacts the fragment, so
   the log does not hold one. Hand over the recipe rather than the value: on
   the deployment host `nemoclaw <name> gateway-token --quiet` prints a fresh
   token, and the UI URL is the reported origin plus `/#token=<token>`.
   Carry all three — origin, token command, and that `/#token=` form — into the
   summary; a token-free origin on its own lands the user on an unauthenticated
   page.

   **On Brev the host is the secure-link FQDN for the relay port, and a
   `127.0.0.1` origin is a claim to prove rather than a fallback to take.**
   Resolve `brev_origin <relay-port>` yourself and read what it returned
   before reporting anything; an unread lookup is not an empty one.

   A denied read of `/etc/brev` is a third answer, distinct from both. A
   confined agent is refused the directory even as uid 0 — under a user
   namespace it appears owned by `nobody`, so the denial can look like an
   ordinary missing file. Escalate it, never resolve it as "no link": re-run
   the lookup unconfined if the environment allows, or use the approved
   `docker run -v /etc/brev:/brev:ro` read in [`brev.md`](brev.md) →
   *When the helpers return nothing*. Only a lookup that ran with access and
   came back empty justifies the loopback origin, and then it is paired with
   the SSH tunnel the same section prints. Reporting a tunnel the user does not
   need, against an origin that does not resolve for them, is the failure this
   paragraph exists to prevent.

   Confirm the two listeners behind it:

   ```bash
   openshell forward list         # BIND column for the dashboard port
   pgrep -af dashboard-relay.py   # --listen addresses for the relay port
   ```

   The forward must read `127.0.0.1` on every host, Brev included — NemoClaw's
   recovery re-creates it there and retires a wider bind as stale, so a
   loopback forward is the healthy state rather than a fault to repair.
   Off-loopback clients are the relay's job: `0.0.0.0` on Brev, and Docker's
   bridge gateway on an adapter-enabled run without a secure link. If the relay
   is missing or bound elsewhere, re-run section 3.5 rather than re-binding the
   forward, which the next lifecycle command undoes.
2. **The sandbox can reach the build.** From the sandbox, one call against the
   origin recorded in `ENV.md`. A `403 CONNECT tunnel failed` is the egress
   policy (see Prerequisites), not a deployment fault — the distinction matters
   because the two have opposite fixes.

Section 3.6 is an optional deeper pass over the live sandbox, active policy
metadata, webhooks, and the installed workspace docs. Run it when onboarding
behaved unexpectedly.

### Name the sandbox in the summary

Report the sandbox name next to the Agent UI link — the `NEMOCLAW_SANDBOX_NAME`
the bring-up was given, read back from section 3.5's `Sandbox: <name>` line
rather than assumed. It is the handle every later command takes:
`nemoclaw <name> status`, `openshell sandbox exec -n <name>`, and the
[Teardown](#teardown) destroy. The sandbox lives outside the Compose project, so
nothing that lists the build reveals it, and a second build under the same
name replaces it.

Say with it whether the bring-up **rebuilt** an existing sandbox of that name.
`NEMOCLAW_RECREATE_SANDBOX=1` discards the previous sandbox and its agent
sessions, and nothing else in the run tells the user that happened — Q3 asks
first about every sandbox except the one this build's own record names.

### Troubleshooting: "Port 18789 is not available."

Onboard's port preflight ends the run that way when the dashboard port is
explicit — the bring-up exports `NEMOCLAW_DASHBOARD_PORT`, so it always is —
and a listener NemoClaw cannot attribute to its own healthy runtime holds it.
The report names the blocking process and offers
`NEMOCLAW_DASHBOARD_PORT=<port> nemoclaw onboard`. When NemoClaw does
recognize the holder as its own, the run continues and the new sandbox takes
the next free port in `18789`-`18799` instead — quieter, and worse, because
the notebook's later cells still address the port it was given. Neither
outcome takes the port from its holder, which is why Q3 probes it first.

A deploy recreates the sandbox, so its state is expendable: destroy the
sandboxes this deploy owns - the failed one and any left by earlier runs -
confirm `18789` is free, and rerun the notebook. A sandbox someone else owns
holding `18789` is not yours to destroy; it means this host cannot run the
deploy until they release it.

```bash
openshell sandbox list
nemoclaw <name> destroy             # each sandbox of this deploy
lsof -nP -iTCP:18789 -sTCP:LISTEN   # must print nothing
```

### Troubleshooting: "OpenClaw onboarding for '<name>' is incomplete"

`nemoclaw onboard` stops after `[8/8] Policy presets` with:

> OpenClaw onboarding for '<name>' is incomplete because <cause>. Resume or
> rerun onboarding.

Only that prefix is stable, so search on it and report the whole line. The
cause is one of NemoClaw's pairing and CLI scope-settlement strings — "its
canonical CLI device pairing did not appear", "its canonical CLI scope upgrade
remained pending", "the sandbox scope-upgrade approval watcher was not
running". A NemoClaw older than the pinned `NEMOCLAW_INSTALL_REF` ended the
line "its canonical CLI device did not receive the required baseline scopes",
which the pin no longer prints at all. The VSS stack and the inference route
have nothing to do with any of them.

The sandbox is expendable here as well: destroy it and rerun, as above.

Anything beyond that — repairing a sandbox in place, NemoClaw's recreate
guards — is NemoClaw's domain: see the
[NemoClaw documentation](https://docs.nvidia.com/nemoclaw/latest/index.html).

## Teardown

The harness and the build are independent lifecycles: nothing in Compose
reaches the sandbox. Tearing down a build therefore starts with
[`teardown.md`](teardown.md) →
[NemoClaw harness](teardown.md#nemoclaw-harness--before-compose), which
stops the dashboard-forward watchdog, destroys the sandbox, stops the dashboard
relay the destroy leaves behind, and removes the `harness-src` worktree when the
build has one.

Destroy the sandbox **before** the Compose project when doing both, so the
harness is not left pointed at an origin that has stopped answering. Removing
the build's containers alone leaves a healthy sandbox with a dead origin, which
reports as a skill failure rather than a missing deployment.

## Sources

- `deploy/docker/scripts/deploy_nemoclaw.ipynb`
- `deploy/docker/scripts/run_setup_notebook.py`
- `deploy/docker/scripts/nemoclaw/README.md`
- `deploy/docker/services/ui/compose.yml`
- `services/ui/DOCKER-README.md`
- `assets/vss_nemoclaw_policy.yaml`
- `.openclaw/` — `Dockerfile`, `plugin/`, `workspace/` (and its `_nemoclaw` overlay)
