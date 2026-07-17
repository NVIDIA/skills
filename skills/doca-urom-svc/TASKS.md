# DOCA UROM Service — Tasks

**Where to start:** The order is `configure → build → modify →
run → test → debug`. The `## test` verb is an iterative loop,
not a one-shot pass — see the eval-loop overlay in `## test`
below. For the DOCA UROM Service, `build` and `modify` are
about *deployment configuration* (container image selection,
the daemon's CLI flags / env, host-library version pairing), not
about compiling source.

These verbs cover the in-scope service operational workflows
for an external operator deploying this service on BlueField.
Every step assumes the operator has consulted the live public
DOCA UROM Service Guide (reachable through
[doca-public-knowledge-map ## DOCA services](../../doca-public-knowledge-map/SKILL.md#doca-services))
and is using it as the authoritative reference; this file
prescribes the *order* and *what to look up where*, not a
copy-paste runbook.

## configure

Preparing the BlueField, picking the configuration axes,
confirming the paired host-library version, and wiring the
substrate before the container starts.

1. **Confirm this service is actually the right answer.** Per
   the path-selection rule in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy):
    - Does the host fleet use the `doca-urom` library AND have
      an MPI / UCX collective pattern where host CPU is the
      bottleneck on communication?
    - Does the BlueField generation in use support the UCX
      components / collectives the host workload wants to
      offload?
    - If either answer is *no*, stop here and tell the user
      honestly: keep the host's existing communication path,
      *do not* deploy this service speculatively.
2. **Confirm DOCA is installed on the BlueField.** This skill
   expects DOCA already on the BlueField Arm. If that has not
   been verified, run
   [`doca-setup ## test`](../../doca-setup/TASKS.md#test)
   first. If there is no install yet, route to
   [`doca-setup ## no-install`](../../doca-setup/TASKS.md#no-install)
   for the public NGC DOCA container path.
3. **Confirm the paired host-library version.** Per the
   version-contract rule in
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility),
   the host-side `doca-urom` library version and this service's
   container tag are a paired contract. Walk the host fleet:
   capture `pkg-config --modversion doca-urom` on the hosts
   that will offload to this BlueField; cross-check that the
   service container tag the operator intends to deploy is
   one the DOCA Compatibility Policy lists as compatible with
   that host library version. If the user does not know which
   tag to deploy, the right answer is to look it up in the
   live public DOCA UROM Service Guide whose version matches
   the host library — NOT to memorize a tag. Route any
   version-mismatch ambiguity to
   [`doca-version`](../../doca-version/SKILL.md).
4. **Confirm the underlying RDMA transport substrate.** This
   service consumes the BlueField's `doca-rdma` substrate to
   actually move bytes. Walk the substrate bring-up per
   [`doca-rdma ## configure`](../../libs/doca-rdma/TASKS.md#configure)
   — at minimum, confirm the BlueField port the service will
   use is up and the inter-node fabric routes between the
   BlueFields the host fleet will exchange UROM operations
   between. A service deployed over a broken RDMA fabric will
   surface as stalled completions at the host (per
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   layer 3) and the user will incorrectly blame the service.
5. **Decide the three service-configuration axes.** Per the
   configuration-axes table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   commit before starting the container to:
    - **UCX-component / collective surface** — which
      components and collective primitives this service
      instance exposes for host offload. Cap-bound to the
      BlueField generation; cross-check against what the
      host's `doca_urom_cap_*` query (per
      [`doca-urom CAPABILITIES.md ## Capabilities and modes`](../../libs/doca-urom/CAPABILITIES.md#capabilities-and-modes))
      believes is supported.
    - **Enqueue queue depth** — sized to the cluster's
      intended in-flight depth per host. Undersized queues
      generate `DOCA_ERROR_AGAIN` at the host; oversized
      queues hide service-side handler stalls.
    - **DOCA Comch endpoint pairing** — how the host's
      `doca-urom` library reaches the service (over a DOCA
      Comch endpoint pair). Access is governed by that Comch
      pairing plus the underlying RDMA permissions; there is
      NO service-side authorization list to author — a
      host-visible `DOCA_ERROR_NOT_PERMITTED` is a Comch /
      RDMA / `doca_dev` signal, not a UROM-service authz
      decision.
6. **Set the daemon's flags and env.** From the public
   DOCA UROM Service Guide, derive the daemon's CLI flags
   (passed via the container's `SERVICE_ARGS` — e.g.
   `--max-msg-size` / `-m`, log level `-l` / `--sdk-log-level`)
   and the `UROM_PLUGIN_PATH` env for the chosen
   UCX-component / collective surface and queue depth. Quote
   flags from the live guide / the daemon's `--help`, do NOT
   infer them from generic UCX or container-runtime knowledge.
   `doca_urom.yaml` mounts only the `plugins/` directory and
   the log directory — there is no service config file to
   author or place on the filesystem.
7. **Sanity check before container start.** Confirm with the
   user: which BlueField the service will run on; which
   container tag the operator intends to pull; which host
   library version the deployment is paired against; which
   UCX components / collectives the service will expose; what
   the queue depth is; how the paired hosts reach the service
   over DOCA Comch. If any of those are unclear, stop and
   ask — do not start the container against an
   underspecified configuration; a deploy that races past
   step 7 is the canonical source of the layer-4 (paired-version
   mismatch) debug failures in
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).

## build

The DOCA UROM Service is shipped as a container, not a
library. There is no service *application* artifact for the
operator to build — the container ships from NGC and is
configured through the daemon's CLI flags (`SERVICE_ARGS`) and
the `UROM_PLUGIN_PATH` env at container start, not through a
mounted config file.

If the user is asking how to build a **host-side UROM
application** (an MPI / UCX consumer that enqueues operations
through `doca-urom`), that is not a service question:

- For host-side `doca-urom` applications, the build is the
  host library's build — route to
  [`doca-urom ## build`](../../libs/doca-urom/TASKS.md#build).
  The service's contribution to that conversation is
  *that the service's container tag must be one the DOCA
  Compatibility Policy lists as paired with the host's
  `doca-urom` version* — the build itself is owned by the
  host-library skill.
- For applications that build directly against `doca-rdma`
  (skipping UROM), the build is the RDMA library's build —
  route to [`doca-rdma ## build`](../../libs/doca-rdma/TASKS.md#build).

If the user is instead asking how to build the **service
container itself** from source, that is *not* an
external-operator workflow — the container ships pre-built
from NGC and rebuilding it is out of scope for this skill.
Route to the public DOCA UROM Service Guide via
[doca-public-knowledge-map ## DOCA services](../../doca-public-knowledge-map/SKILL.md#doca-services).

## modify

The DOCA UROM Service does not have a "modify a sample"
workflow analogous to DOCA libraries; there is no service
sample program a user starts from. The service analog of
"modify" is **adapt the documented container config recipe to
the user's environment**, with the paired-contract version
pinning re-checked on every modification.

1. **Start from the documented recipe.** Identify the public
   DOCA UROM Service Guide's recipe that matches the user's
   target BlueField generation and intended offload surface.
   Quote it; do not author a new one from scratch.
2. **Diff against the user's environment.** Note the specific
   substitutions the user must make: container image tag
   (always derived from the paired host library version per
   the DOCA Compatibility Policy), BlueField port for the
   substrate, the `plugins/` mount and `UROM_PLUGIN_PATH`,
   UCX-component / collective list, queue depth. Each
   substitution is a point at which the operator can break
   one of the three configuration axes or the version-pairing
   contract.
3. **Apply minimum-change.** Change only what the user's
   environment forces. Every additional deviation from the
   documented recipe widens the surface for an unintended
   service / host pairing mismatch the operator will have to
   debug later.
4. **Re-validate against the three configuration axes.** Each
   substitution is a chance to accidentally break one of the
   three configuration axes (component / collective surface,
   queue depth, DOCA Comch endpoint pairing). Walk
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   one row at a time after every substitution.
5. **Re-validate against the version-contract rule.** Per the
   version-contract rule in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   any change that affects the container tag, OR that adds /
   removes a host endpoint paired against a different host
   library version, must also re-check the host-library +
   service-version pair against the DOCA Compatibility
   Policy. A modify that lands a new container tag without
   re-running the version-pair check is the canonical source
   of layer-4 subtle failures.
6. **Re-validate against the host fleet.** A service config
   change that the operator validates in isolation may break
   one of the paired hosts' workloads. Coordinate with the
   host-side skill (see
   [`doca-urom ## test`](../../libs/doca-urom/TASKS.md#test))
   before scaling a service config change to a fleet that's
   actively offloading.

The agent's anti-pattern alert: a *"clean rewrite of the
service config from scratch"* is almost always slower than
adapting the public DOCA UROM Service Guide's recipe, because
the service's config schema is documented per the public guide
and is not 1:1 with generic UCX-server configuration.

## run

Bringing up the service container, confirming the paired host
library can reach it, and walking a single-operation smoke
BEFORE layering any HPC workload on top.

1. **Pull the service container image from NGC** at the tag
   the public DOCA UROM Service Guide names for the
   operator's DOCA release AND host library version pair.
   Quote the tag from the live guide; do NOT memorize or
   invent the tag. A wrong tag is the canonical source of
   layer-1 (container runtime) and layer-4 (paired-version
   mismatch) errors simultaneously.
2. **Start the container per the public Container Deployment
   Guide pattern.** Set the daemon's `SERVICE_ARGS` /
   `UROM_PLUGIN_PATH` and mount the `plugins/` directory as
   the public DOCA UROM Service Guide names (there is no
   service config file to mount). The runtime
   command shape (e.g. `docker run` / `crictl` / BlueField
   container manager) is documented in the Container
   Deployment Guide reachable through
   [`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).
3. **Confirm the container is running, not restart-looping.**
   A restart loop is a layer-1 symptom per
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   (container runtime / image tag / plugins mount); diagnose
   it before touching service config.
4. **Watch the service container's logs for startup.** The
   container's stdout is the primary observability surface.
   The service should report that it is ready to receive host
   enqueues, name the UCX-component / collective surface it
   exposed, and report its queue sizing. The agent should NOT
   invent log line formats; quote what the live container is
   emitting.
5. **Confirm at least one paired host can `doca_ctx_start`
   against this BlueField.** On a single paired host, walk
   [`doca-urom ## configure`](../../libs/doca-urom/TASKS.md#configure)
   through `doca_ctx_start`; success here means the service
   accepted the host's connection and the host's library can
   begin enqueueing. Failure here surfaces layer-1 (container
   not reachable, or the DOCA Comch pairing not established) or
   layer-4 (host-library / service version mismatch) — walk
   the layers in [`## debug`](#debug).
6. **Single-operation smoke (next: `## test` step 1).**
   Before driving any HPC workload, walk `## test` step 1
   once: one host enqueue → observe the service receive it
   in logs → observe one completion at the host. Only then
   layer the workload on top.

For the runtime version + container-tag cross-checks that
underlie *"my service behaves differently from what the host
library expects"*, see
[`doca-version TASKS.md ## run`](../../doca-version/TASKS.md#run)
and apply the host-library + service-version paired-version
overlay from
[`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility).

## test

The DOCA UROM Service has no "compile and unit-test" workflow
— testing is operational and end-to-end through the paired
host library.

**`## test` is an iterative loop, not a one-shot pass.** Every
mutation (UCX-component / collective surface change, queue
depth change, Comch-pairing change, container tag change,
substrate change) re-opens the smoke sweep. Skipping the
re-run after a mutation is the failure mode this loop
replaces.

The eval-loop overlay (rows apply to every service deployment,
not just one BlueField generation):

| Step | Why this is a loop, not a step | Where the substance lives |
| --- | --- | --- |
| 1 → 3 → 1 | Step 3 (cluster-pattern check) often reveals an as-deployed gap in queue sizing or in the exposed UCX-component / collective surface; loop back to step 1 | [`## test`](#test) step 3 |
| 1 → ## debug | When the single-host smoke (step 1) does NOT see the service receive the enqueue, the deployment is non-functional — escalate to `## debug` immediately, do not run later steps | [`## debug`](#debug) |
| 2 → ## configure → 2 | When the cap-query disagreement smoke (step 2) shows the host's `doca_urom_cap_*` and the service's exposed surface don't match, the service config is wrong — loop back to `## configure` step 5 and re-run | [`## configure`](#configure) |
| 1..4 → ## run | Each loop iteration ends with a smoke; if all four pass, hand off to live `## run` traffic | [`## run`](#run) |

The agent's rule: every mutation re-opens the sweep. A
configuration change followed by *"it probably still works"*
is exactly the failure mode the iterative loop is here to
prevent.

1. **Single-host single-operation smoke.** From ONE paired
   host, after the host's `doca_ctx_start` has succeeded
   against this BlueField, enqueue ONE put (a small payload
   into a peer's exported memory) and observe: (a) the
   service-side logs show the enqueue was received; (b) the
   host-side progress engine fires the completion. This is
   the cheapest place to identify layer-1 / layer-2 / layer-3
   / layer-4 gaps per
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   before any HPC stack effort is wasted.
2. **Cap-query agreement smoke.** From a paired host, run the
   matching `doca_urom_cap_*` queries against the active
   `doca_devinfo` (per
   [`doca-urom ## configure`](../../libs/doca-urom/TASKS.md#configure)
   step 4) and confirm the operation families / collective
   variants the host says are supported are the same set the
   service exposed in its config. A disagreement here means
   either the service is exposing more than the host library
   knows how to use, or the service is exposing less than the
   host library expects — either way, layer-2 or layer-4
   debugging starts here.
3. **Cluster-pattern smoke.** With 2 paired hosts on the
   service, exercise a small batch (e.g. a few hundred puts
   across two nodes) AND a small collective (e.g. a small
   all-reduce across the same two nodes). Confirms the
   service's queue depth supports the in-flight pattern AND
   that the service's exposed collective surface actually
   executes end-to-end. A service that passes step 1 but
   fails step 3 most often has a queue-depth (layer 2) or
   collective-variant (layer 4) gap.
4. **Capability snapshot.** Save the *as-deployed* answer to:
   which service container tag is running, which UCX
   components / collectives are exposed, which queue depth is
   configured, how paired hosts reach the service over DOCA
   Comch, which
   host-library versions paired hosts are on. This snapshot
   is the artifact that lets future debug sessions skip
   rediscovery — and is the diff target when the user later
   asks *"why does this deployment behave differently from
   the last one"*.

## debug

Layered diagnosis. Walk the layers in this order; do not skip
down without clearing the layer above. The first four layers match
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).
The host-visible `DOCA_ERROR_*` overlay is documented on the
host side in
[`doca-urom CAPABILITIES.md ## Error taxonomy`](../../libs/doca-urom/CAPABILITIES.md#error-taxonomy);
this ladder maps the host symptom back to the service-side
layer the operator must touch.

1. **Container runtime layer.** Is the service container
   actually running and not restart-looping? Symptoms:
   container exits immediately, image pull fails, restart
   count climbing; host's `doca_urom` create or
   `doca_ctx_start` fails because there is no reachable
   service. Resolution: confirm the image tag matches what
   the public guide names for the operator's host-library
   pair; confirm the `plugins/` mount and `UROM_PLUGIN_PATH`
   match what the public guide names; confirm the BlueField
   has the runtime configured per the public Container
   Deployment Guide.
   This layer is owned by the container runtime, not by UROM.
   Note: a host-visible `DOCA_ERROR_NOT_PERMITTED` here is a
   DOCA Comch-pairing / RDMA-permission signal (or a host-side
   `doca_dev` access problem fixed per
   [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug)) —
   NOT a service-side authorization rejection; there is no
   such authz layer in the shipped binary.
2. **Service-side resource layer.** Host enqueues succeed;
   completions never fire; OR after N successful enqueues
   every further enqueue returns `DOCA_ERROR_AGAIN`
   indefinitely. Resolution: walk the service logs for
   queue-saturation or handler-stuck lines; if the queue is
   sized too small for the host's in-flight depth, raise the
   queue depth per [`## configure`](#configure) step 5; if a
   service-side handler is stuck on a single in-flight
   operation, the stuck-handler line names the operation and
   the cluster team can isolate it. **Do NOT silently raise
   the queue depth without investigating** — a queue that is
   constantly saturating may be hiding a layer-3 substrate
   problem.
3. **Underlying RDMA substrate layer.** Host sees
   `DOCA_ERROR_IO_FAILED`; or enqueue succeeds but
   completions never fire and the substrate counters show
   errors. Resolution: confirm the BlueField port the service
   is using is up and the inter-node fabric routes correctly;
   route to
   [`doca-rdma TASKS.md ## debug`](../../libs/doca-rdma/TASKS.md#debug)
   layers 5-7 for substrate-layer diagnosis. **Do NOT try to
   fix this from service config; it is a substrate property,
   not a service knob.**
4. **Paired-version mismatch layer.** Host's `doca_urom_cap_*`
   claims an operation family / collective is supported, but
   runtime enqueue returns `DOCA_ERROR_NOT_SUPPORTED` (despite
   a healthy container). Resolution:
   cross-check `pkg-config --modversion doca-urom` on the
   host AND this service's container tag against the DOCA
   Compatibility Policy and the four-way-match rule in
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   layer 2. The fix is either upgrading the service
   container or downgrading the host library; **do NOT paper
   over with a retry, and do NOT change the service's exposed
   surface to "fix" a version-pair gap.**
5. **Performance-not-helping layer.** Container running,
   queue not saturated, substrate
   healthy, versions paired correctly — but the offload's
   end-to-end performance is worse than the host-CPU
   baseline. Resolution: this is most often not a bug —
   either the workload's pattern doesn't actually benefit
   from DPU offload (small ops, low collective depth, host
   CPU not the bottleneck) per the path-selection rule in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   OR the service is configured to process operations one at
   a time when batching is documented for that operation
   family. Re-read the public guide for batching knobs on the
   specific collective in use; if the workload genuinely
   doesn't benefit, route back to path selection in
   [`## configure`](#configure) step 1.
6. **Cross-cutting layer.** For env-side and program-side
   debug that is not service-specific (BlueField install,
   BlueField kernel, DOCA library errors the service may
   surface), drop to
   [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).

## Command appendix

DOCA UROM Service-specific commands the verbs above reach for,
grouped by purpose so the agent picks the right family without
searching prose. Every row is a class — the agent must not
invent flags beyond what the row names; flag and command
discovery is `--help` on the installed tool or the public
guide, not prose recall.

**Infra-aware preamble (every row below).** Per the bundle's
detect → prefer → fall back → report contract documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent should:

1. Probe for the matching structured helper FIRST (`doca-env --json`
   for version + devices + libraries + drivers + hugepages in
   one shot on the BlueField; the BlueField container manager's
   structured status output when available).
2. If the probe succeeds, the structured tool's output is the
   authoritative answer and the agent SHOULD NOT also run the
   manual command in the row below. Report *"using structured
   `<tool>`"*.
3. If the probe fails, fall back to the manual command in the
   row. Report *"falling back to manual chain"*.
4. The schemas the structured tools emit are defined in
   [`doca-structured-tools-contract ## Schemas`](../../doca-structured-tools-contract/SKILL.md#schemas);
   the version-handling semantics (four-way match, NGC,
   headers-win) are owned by
   [`doca-version`](../../doca-version/SKILL.md).

| Purpose | Command (class shape) | Owning step | Reads as healthy when … |
| --- | --- | --- | --- |
| Container lifecycle | The BlueField container manager's start / stop / status command for the service container, per the public Container Deployment Guide | [`## run`](#run) | Container `running`, restart count stable. |
| Container logs | The BlueField container manager's log-stream command for the service container | [`## debug`](#debug) layers 1-2 + 4 | Service-ready line on startup; no queue-saturation / handler-stuck lines repeating; no paired-version warnings. |
| Container tag in use | The BlueField container manager's image-inspect command for the running service container | [`## run`](#run) step 1; [`## debug`](#debug) layer 4 | Tag matches what the public DOCA UROM Service Guide names for the paired host library version. |
| Host-library version (on paired host) | `pkg-config --modversion doca-urom` on the host | [`## configure`](#configure) step 3; [`## debug`](#debug) layer 4 | A semver string the DOCA Compatibility Policy lists as compatible with this service's container tag. |
| BlueField DOCA version | `doca_caps --version` on the BlueField | [`## configure`](#configure) step 2; [`## debug`](#debug) layer 4 | A semver string matching the BlueField's `pkg-config --modversion doca-common`; disagreement = partial install per [`doca-version`](../../doca-version/SKILL.md). |
| Host-side cap-query agreement | A `doca_urom_cap_*` query from a paired host against the active `doca_devinfo`, then compare to the service's exposed surface | [`## test`](#test) step 2; [`## debug`](#debug) layer 4 | The host's cap-query result is a subset of (or equal to) the service's exposed UCX-component / collective surface. |
| Substrate port check | `ibv_devinfo` (sudo) for the BlueField port the service is using; OR the substrate-side checks from [`doca-rdma TASKS.md ## debug`](../../libs/doca-rdma/TASKS.md#debug) | [`## configure`](#configure) step 4; [`## debug`](#debug) layer 3 | `PORT_ACTIVE` with a sane MTU; substrate counters not error-spiking. |
| Single-operation smoke | A single put or get from one paired host through `doca-urom` to a peer, with the service's logs observed in parallel | [`## run`](#run) step 6; [`## test`](#test) step 1 | Service logs show the enqueue received; host's progress engine fires the completion. |

Three cross-cutting rules for this appendix:

- **Never invent a daemon flag, container tag, or config
  key.** The public DOCA UROM Service Guide
  is the contract; the daemon's `--help` and the BlueField
  container manager's `--help` output are the secondary sources
  for runtime mechanics. Prose-derived daemon flags are the
  most common hallucination failure for this skill.
- **Container before service before paired host.** When
  triaging, confirm the container layer (running, not
  restart-looping, image tag correct) before reading any
  service log, and confirm the   service-side state before
  blaming the paired host. A non-running container makes
  every service-layer command meaningless; a broken DOCA Comch
  pairing / RDMA-permission problem makes every host-side debug
  session a wild-goose chase.
- **Cross-link instead of duplicate.** Cross-cutting env
  commands (port-state, `devlink`, `ip link`, `ethtool`) live
  in
  [`doca-setup TASKS.md ## Command appendix`](../../doca-setup/TASKS.md#command-appendix);
  this appendix names only the DOCA-UROM-Service-specific
  ones.

## Deferred task verbs

- **Installing DOCA on the BlueField** — out of scope here.
  Route to
  [`doca-setup ## configure`](../../doca-setup/TASKS.md#configure)
  for env preparation and
  [`doca-setup ## test`](../../doca-setup/TASKS.md#test) for
  install health verification, or
  [`doca-setup ## no-install`](../../doca-setup/TASKS.md#no-install)
  for the public NGC DOCA container path.
- **Writing a host-side `doca-urom` application** — not a
  service question. Route to
  [`doca-urom ## configure`](../../libs/doca-urom/TASKS.md#configure)
  for the host-side bring-up,
  [`doca-urom ## build`](../../libs/doca-urom/TASKS.md#build)
  for the build pattern, and
  [`doca-urom ## test`](../../libs/doca-urom/TASKS.md#test)
  for the host-side smoke. The host library and the service
  are a paired contract, NOT a single artifact.
- **Bringing up the RDMA / RoCE / IB substrate between
  BlueFields** — owned by
  [`doca-rdma ## configure`](../../libs/doca-rdma/TASKS.md#configure).
  This service *uses* the substrate; it does not stand it
  up.
- **Designing the HPC stack (MPI collective algorithms, UCX
  transport wiring)** — out of scope here. The service
  executes UROM-shaped offloads; the stack decides which
  operations to enqueue. Route to upstream MPI / UCX
  documentation.
- **Other DOCA services** (DMS / DTS / BlueMan / Firefly /
  Flow-Inspector / HBN / Argus / …) — not this service.
  Route to
  [doca-public-knowledge-map ## DOCA services](../../doca-public-knowledge-map/SKILL.md#doca-services)
  for the routing table and the matching `services/<service>`
  skill when it exists (e.g.
  [`doca-firefly ## configure`](../doca-firefly/TASKS.md#configure)
  for time synchronization,
  [`doca-dms ## configure`](../doca-dms/TASKS.md#configure)
  for device management). The container-shaped deployment
  pattern is shared; the per-service domain is different.

## Cross-cutting

- The public DOCA UROM Service Guide is the single source of
  truth. Any daemon flag, UCX-component / collective name,
  container tag, or observability
  output the agent quotes must come from there, not from
  generic UCX or container-runtime knowledge.
- The host library and this service are a paired contract.
  Every deployment decision (container tag pulled, daemon
  flag / env set, how the host pairs over DOCA Comch) is also
  a host-side decision; the agent must name BOTH sides
  whenever a change on one side affects the other.
- Path-selection is mandatory up front. This service is the
  wrong answer when the host fleet does not use `doca-urom`,
  when the HPC stack is neither MPI nor UCX, or when the
  BlueField generation cannot support the intended offload.
- Smoke before scale. Every workload (full collectives,
  dense MPI patterns) goes after the single-operation smoke
  and the cap-query agreement smoke pass, never before.
- For URL routing to the DOCA UROM Service Guide and other
  public DOCA documentation, see
  [doca-public-knowledge-map ## DOCA services](../../doca-public-knowledge-map/SKILL.md#doca-services).
