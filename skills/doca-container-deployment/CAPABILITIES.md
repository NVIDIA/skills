# DOCA container deployment — Capabilities

**Where to start:** The pattern overview below names the recurring
container-deployment patterns the agent walks for every DOCA service
on the BlueField. Pick the pattern first, then drill into the H2
that owns the substance. For the *how* of executing each pattern,
jump to [TASKS.md](TASKS.md). For per-service overlays of the four
in-bundle services (`doca-argus`, `doca-dms`, `doca-firefly`,
`doca-urom-svc`), follow
the matching per-service skill that layers on top of this one.

This file enumerates the cross-cutting DOCA container-deployment
runtime contract as described in the public **DOCA Container
Deployment Guide** (reachable through
[`doca-public-knowledge-map ## DOCA services`](../doca-public-knowledge-map/SKILL.md#doca-services))
and as overlaid on the BlueField OS image that ships kubelet in
standalone mode. Treat this file as a *map of what is documented*,
not a substitute for reading the live Container Deployment Guide
plus the matching per-service guide when standing up a real
deployment.

## Pattern overview

> **Non-goals (externally-productized NVIDIA services not in the
> DOCA monorepo):** BlueMan, HBN, SNAP, Virtio-net, DOCA Telemetry
> Service (DTS, as productized), and any future external NVIDIA
> networking software not under `doca/services/`. A user asking
> *"how do I deploy BlueMan?"* (or any of the others) is routed to
> the public NVIDIA docs at `docs.nvidia.com/doca/sdk/` for that
> specific product, NOT silently extrapolated from this skill's
> contract. The strict-to-DOCA invariant is documented at
> `AGENTS.md ## Non-goals` row 7.

Every container-deployment question this skill teaches resolves into
one of SIX patterns. The patterns are CLASSES — they apply across
every DOCA service on the BlueField, not just one service.

| Container-deployment pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Recognise the runtime — kubelet standalone on BlueField Arm watching a static-pod manifests directory | No full Kubernetes control plane; no `kubectl` against a cluster API; the operator's unit of input is a YAML pod-spec file dropped into the documented manifests directory; kubelet schedules the pod and runs the container locally on the BlueField | [`## Capabilities and modes`](#capabilities-and-modes) runtime-shape table |
| 2. Confirm the BlueField preconditions BEFORE dropping the pod spec | DOCA installed on the Arm; container runtime + kubelet present per the BlueField OS image; BFB version compatible with the service's container tag; per-service firmware-slot enabled when the service emulates a device; image-pull reachability from the BlueField to NGC | [`## Capabilities and modes`](#capabilities-and-modes) precondition table |
| 3. Author the pod-spec YAML from the documented recipe — never from generic Kubernetes intuition | Pod-spec field names, the static-pod manifests directory path, the volume-mount shape, and the image string all come from the public DOCA Container Deployment Guide plus the per-service guide; the agent does NOT infer them | [`## Capabilities and modes`](#capabilities-and-modes) pod-spec-shape rule + [`## Safety policy`](#safety-policy) "do not invent" rule |
| 4. Observe the deployment at three layers — kubelet status, container runtime logs, service-side liveness | Each layer has its own owner and its own command; healthy means all three agree | [`## Observability`](#observability) three-layer table |
| 5. Map a failure back to its layer | Pod-spec syntax → pod scheduling → image pull → runtime → volume mount → network policy → version → cross-cutting host; eight layers, each with its own owner | [`## Error taxonomy`](#error-taxonomy) layered split |
| 6. Smoke before bulk — pod Running, ENTRYPOINT clean, trivial liveness probe answered, only then layer real workload | One-shot deploy → service-side liveness probe → multi-pod smoke when the service has paired publishers / followers / clients | [`## Safety policy`](#safety-policy) smoke-before-bulk rule |

Two cross-cutting rules that apply to *every* pattern above:

- **Operate the documented path; do not invent one.** Pod-spec
  field names, the static-pod manifests directory path, kubelet
  command-line flags, image strings, and volume-mount shapes all
  come from the public DOCA Container Deployment Guide and the
  matching per-service guide reached through
  [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md).
  Inferring them from generic Kubernetes / upstream `kubelet`
  prose is the most common hallucination failure for this skill.
- **The runtime is uniform across services; the per-service
  overlay is the variable.** The six in-bundle DOCA services
  (Argus, DMS, Firefly, Flow-Inspector, OS-Inspector, UROM
  service) all use this same kubelet-standalone + pod-spec-drop
  pattern. What changes per service is the config-file mount, the
  image string, the per-service preconditions (PTP-aware path
  for Firefly, paired host↔DPU workload for UROM service, mirror
  source configuration for Flow-Inspector, gNMI endpoint exposure
  for DMS, host-OS introspection target for OS-Inspector,
  policy-set pinning for Argus), the "healthy" definition (PHC
  offset + ports state for Firefly, mirrored-flow ingest for
  Flow-Inspector, gNMI session up for DMS, per-target sample for
  OS-Inspector, event-stream output for Argus, per-operation
  counter for UROM service), and the paired-workload contract.
  The agent's job is to walk the shared runtime here, then route
  to the per-service skill for the overlay.

## Capabilities and modes

### Runtime shape — kubelet standalone on BlueField Arm

DOCA service containers run on the **BlueField Arm cores** and are
scheduled by a **kubelet running in standalone mode** that the
BlueField OS image ships pre-configured. There is no full
Kubernetes control plane on the BlueField in this deployment shape;
no `kubectl` against a cluster API server; no scheduler / controller
manager / etcd. The kubelet on the BlueField watches the documented
**static-pod manifests directory** that the public DOCA Container
Deployment Guide names; the operator's unit of input is a YAML
pod-spec file dropped into that directory.

| Property | What it means for the operator |
| --- | --- |
| Where it runs | On the BlueField Arm cores. Not on the host. The host's relationship to a DOCA service container is whatever the per-service overlay defines (control-plane peer / data-plane peer / time follower / etc.). |
| Who watches what | A kubelet running in standalone mode on the BlueField watches the documented static-pod manifests directory and reconciles every YAML pod spec it finds there into a running pod. The operator interacts with kubelet via files in that directory, not via a cluster API. |
| Unit of operator input | A YAML pod-spec file dropped into the documented manifests directory. Adding the file = telling kubelet to bring the pod up. Removing the file = telling kubelet to tear it down. Editing the file in place is documented to trigger a reconcile; the exact semantics live in the public guide. |
| Container runtime | The container runtime kubelet talks to (CRI-compatible) ships with the BlueField OS image per the public DOCA Container Deployment Guide. The operator does NOT install a separate `docker` / `containerd` from upstream. |
| What the operator does NOT do | Operate a Kubernetes cluster, run `kubectl` against an API server, define `Deployment` / `Service` / `Ingress` objects, attach to a CNI overlay across BlueFields, set up an etcd, or run a scheduler — none of that is in scope for the BlueField's kubelet-standalone shape. |

### BlueField preconditions

Every DOCA service container shares the same BlueField precondition
matrix. The matrix below names the *classes* of precondition; the
exact commands / flags live in the public DOCA Container Deployment
Guide and (for firmware slots) in the public BlueField firmware-
configuration documentation reached through
[`doca-setup`](../doca-setup/SKILL.md).

| Precondition class | What it covers | Where to confirm |
| --- | --- | --- |
| DOCA install on the BlueField Arm | DOCA installed and healthy on the BlueField Arm side. Without this, kubelet's documented runtime hooks and the DOCA Container Deployment Guide's documented mount paths do not exist | [`doca-setup ## test`](../doca-setup/TASKS.md#test) (install verification) and the *no-install* path at [`doca-setup ## no-install`](../doca-setup/TASKS.md#no-install) when there is no install yet |
| Container runtime + kubelet standalone | Both ship with the BlueField OS image per the public DOCA Container Deployment Guide. The operator confirms `kubelet` is running and that the static-pod manifests directory exists; the operator does NOT install upstream Kubernetes from scratch | Public DOCA Container Deployment Guide |
| BFB version compatible with the service's container tag | The BlueField OS image (the BFB) ships a kubelet + container runtime version pair that the public Container Deployment Guide for the operator's DOCA release certifies. A BFB on a different release line can fail to run a service container even when the host DOCA install looks healthy | Public DOCA Container Deployment Guide + the per-service guide for the operator's DOCA release |
| Per-service firmware slot (when the service emulates a host-facing PCIe device) | Some externally-productized NVIDIA services (notably SNAP and Virtio-net, both out of scope for this bundle per the strict-to-DOCA invariant) stand up emulated PCIe devices via the BlueField firmware emulation slots; the matching slot must be enabled BEFORE the pod is scheduled, and a BlueField reset is typically required after the slot is flipped. **None of the six DOCA services in this bundle (Argus, DMS, Firefly, Flow-Inspector, OS-Inspector, UROM service) emulate a host-facing PCIe device**, so this precondition does NOT apply to any in-bundle service. The precondition is documented here only because an operator who reads this skill while deploying an external NVIDIA service needs to know the pattern exists; the operator's source of truth in that case is the public NVIDIA docs at `docs.nvidia.com/doca/sdk/` | The public NVIDIA docs at `docs.nvidia.com/doca/sdk/` for the external service plus [`doca-setup ## configure`](../doca-setup/TASKS.md#configure) for the firmware-tool workflow itself |
| Image-pull reachability to NGC | The BlueField needs network reach to the NGC catalog (`nvcr.io`) so the container runtime can pull the service's image at the documented tag. A BlueField that can ping the world but cannot reach `nvcr.io` (corporate proxy, firewall, air-gap) will pass every other precondition and still fail to deploy | The container runtime's image-pull log surface + the operator's network team |
| Host-OS permissions on the BlueField | The operator needs the permissions the public Container Deployment Guide names to write into the static-pod manifests directory, read kubelet's log surface, and inspect the container runtime. The permission boundary lives at the BlueField OS, not inside any DOCA service | The BlueField OS plus the public Container Deployment Guide |

### Pod-spec YAML — what the operator drops, where, and shaped by what

The unit of operator input is a single YAML pod-spec file dropped
into the documented static-pod manifests directory on the BlueField.
The public DOCA Container Deployment Guide is the source of truth
for:

- The exact directory path kubelet watches.
- The pod-spec YAML field names and structure (the `apiVersion` and
  `kind` the guide names; the container / image / command /
  resources / volumes / volumeMounts / securityContext fields the
  guide actually quotes; the mount-path convention for the per-
  service config file; the host-network / host-PID / privileged
  posture the per-service guide expects).
- The image string and tag for the service the operator is
  deploying (per the matching per-service guide).

The agent quotes pod-spec field names from the live public guide;
the agent does NOT infer them from generic Kubernetes pod-spec
intuition. Field names, defaulting behavior, and validation rules
vary across kubelet versions and the BlueField OS image's kubelet
build is the contract — not upstream Kubernetes prose.

### Cross-service generalization

The deployment runtime is uniform across every DOCA service on the
BlueField:

| Service | Service-specific overlay (lives in the per-service skill) |
| --- | --- |
| [`doca-argus`](../services/doca-argus/SKILL.md) | Runtime-security / monitoring policy config; event-stream output as the "healthy" signal |
| [`doca-dms`](../services/doca-dms/SKILL.md) | gNMI / gNOI endpoint config; gNMI session up + per-RPC counter as the "healthy" signal; host-side gNMI client as the paired-workload contract |
| [`doca-firefly`](../services/doca-firefly/SKILL.md) | Four-axis PTP config (role / profile / domain / interface); PHC offset + ports state as the "healthy" signal; host-side time follower as a paired requirement |
| [`doca-urom-svc`](../services/doca-urom-svc/SKILL.md) | UROM operations endpoint; per-operation counter as the "healthy" signal; paired host↔DPU workload contract |

The agent's rule: walk the shared runtime here, then route to the
matching per-service skill for the per-service overlay. Re-stating
the shared runtime inside the per-service skill is the failure mode
this skill exists to prevent.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way match
rule, NGC container semantics, and the headers-win-over-docs rule,
see [`doca-version`](../doca-version/SKILL.md). The body lives
there; this skill does not duplicate it.

**The container-deployment-specific overlay** is:

- **Three version anchors must agree, not two.** Every DOCA
  service container deployment has *three* runtime version anchors,
  not two: the host DOCA install version on the BlueField Arm
  (per [`doca-version TASKS.md ## configure`](../doca-version/TASKS.md#configure)),
  the **BFB version** on the BlueField (the BlueField OS image
  version, which determines which kubelet + container-runtime
  build is in play), and the **per-service container tag** pulled
  from NGC. A mismatch between any two of those three is the
  canonical *"the docs say this should work but it does not"*
  failure mode. Capture all three before debugging, not in the
  middle of debugging.
- **The container tag is the per-service runtime anchor, NOT
  `pkg-config --modversion` on the host.** Same overlay every
  per-service skill in the bundle (Argus / DMS / Firefly /
  Flow-Inspector / OS-Inspector / UROM service) carries. When the
  operator reports a service-container behavior, the relevant
  version anchor is the container tag pulled, not
  `pkg-config --modversion doca-common` on the host. Quote both
  versions when triaging; if they diverge, route to
  [`doca-version TASKS.md ## debug`](../doca-version/TASKS.md#debug)
  layer 2 before diagnosing the service behavior itself.
- **The public DOCA Container Deployment Guide version must match
  the BFB / container-runtime pair.** The Container Deployment
  Guide is versioned; the on-page version must match the BlueField
  OS image's kubelet + runtime build. A mismatch between the docs
  version and the live runtime is the canonical *"my pod spec
  matches the guide but kubelet rejects it"* failure mode — the
  guide on the screen may not be for the BFB on the BlueField.
- **The per-service guide version must match the container tag.**
  Within a service, the per-service guide's version must match the
  container tag pulled — config-file field names and "healthy"
  output can change between releases.

## Error taxonomy

Container-deployment errors fall into EIGHT layers, each with its
own owner. The agent walks the layers in this order; conflating
them wastes debug time and blames the wrong layer.

1. **Pod-spec syntax layer.** The YAML pod-spec file is in the
   static-pod manifests directory but kubelet rejects it as
   malformed. Symptoms: kubelet log surface shows a parse / schema
   error; the pod never appears in kubelet's status output.
   Causes: invalid YAML (indentation, tab vs space); an
   `apiVersion` / `kind` the kubelet build does not accept; a
   pod-spec field name the kubelet build does not know; a value
   type mismatch (string where number is expected). Resolution:
   re-quote the pod-spec shape from the public DOCA Container
   Deployment Guide for the BFB the operator is on; do NOT infer
   field names from generic Kubernetes intuition. Owner: this
   skill + the public Container Deployment Guide.
2. **Pod scheduling layer.** The pod-spec parses, but kubelet
   does not schedule the pod (the pod never reaches `Pending` →
   `Running`). Symptoms: kubelet reports a scheduling failure;
   the pod is stuck in `Pending` indefinitely; node resources
   or admission rules reject the pod. Causes: requested CPU /
   memory exceeds what the BlueField Arm can grant; a `nodeName`
   / `nodeSelector` mismatch; a security-context the kubelet
   build rejects; a `hostPath` the kubelet build refuses; the
   per-service precondition (firmware slot, etc.) not yet
   satisfied so the kubelet refuses to schedule the pod.
   Resolution: walk the per-service skill's preconditions
   first; then re-quote the resource shape from the public
   guide. Owner: this skill + the per-service skill.
3. **Image-pull layer.** The pod schedules but the container
   runtime cannot pull the image. Symptoms: the pod is stuck in
   `ContainerCreating` / `ImagePullBackOff`; the container
   runtime's image-pull log surface shows a 404 / 403 / timeout
   against `nvcr.io`. Causes: invented image string or tag (the
   load-bearing first-app failure); BlueField has no network
   reach to `nvcr.io` (corporate proxy, firewall, air-gap); the
   tag the per-service guide names does not exist for the
   operator's BFB / DOCA release. Resolution: re-quote the image
   string + tag from the public per-service guide reached through
   [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md);
   confirm BlueField reach to `nvcr.io` independently of the
   pod-spec. Owner: this skill + the per-service skill + the
   operator's network team.
4. **Container-runtime / ENTRYPOINT layer.** The image is
   present locally; the container starts; but its ENTRYPOINT
   immediately exits with a non-zero status, or the container
   is in a restart loop. Symptoms: kubelet status shows
   `CrashLoopBackOff`; the container's log stream shows a
   config-parse error, a missing dependency, or a failed
   internal precondition. Causes: the per-service config-file
   mount path is wrong (the file is mounted at a path the
   ENTRYPOINT does not expect); the per-service config file is
   malformed; the service inside the container fails a
   service-specific precondition (per-service skill responsibility);
   the container is incompatible with the BFB / runtime
   underneath. Resolution: read the container's log stream
   FIRST, route the service-specific config-parse error to the
   matching per-service skill, then re-check the BFB / version
   anchors. Owner: the per-service skill primarily; this skill
   for the cross-cutting mount-path / runtime issues.
5. **Volume-mount layer.** The pod ENTRYPOINT runs but cannot
   read or write something the pod-spec says it should. Symptoms:
   the container's log stream shows a permission / not-found
   error on a mounted path; a `hostPath` volume the pod-spec
   names does not exist on the BlueField host; the per-service
   config file mount is empty inside the container even though
   the file exists on the host. Causes: the host path named in
   the pod-spec does not exist; the host path exists but the
   container user cannot read / write it; the mount-propagation
   mode does not match what the service expects; a directory was
   bind-mounted where a file was expected (or vice versa).
   Resolution: confirm the host path on the BlueField, confirm
   the permissions, re-quote the volume-mount shape from the
   public Container Deployment Guide. Owner: this skill.
6. **Network-policy / host-firewall layer.** The container runs
   but the service inside cannot reach a peer it needs, or a
   peer cannot reach the service. Symptoms: per-service liveness
   probe times out; an external client cannot connect to the
   service's documented port; the container's outbound to a
   paired endpoint (NGC for image pull, upstream PTP master for
   Firefly, gNMI client for DMS, mirror source for Flow-Inspector)
   times out. Causes: the pod-spec `hostNetwork` / port mapping
   does not match what the per-service guide expects; a host
   firewall on the BlueField blocks the port; a network policy
   on the BlueField OS isolates the pod from the host network; a
   corporate firewall blocks the egress. Resolution: confirm the
   host network reachability independently of the pod; confirm
   the pod's networking posture matches the per-service guide.
   Owner: this skill + the operator's network team.
7. **Version layer.** The pod is healthy on the BlueField, the
   service answers, but the behavior does not match what the
   public per-service guide page describes. Symptoms: a config
   field name the guide quotes is rejected; a "healthy" output
   line the guide names does not appear; a paired workload sees
   a different schema / protocol than the guide describes.
   Causes: the per-service guide on the screen is for a different
   DOCA release than the container tag pulled; the BFB on the
   BlueField is on a different release than the Container
   Deployment Guide describes; partial install on the BlueField
   Arm. Resolution: walk
   [`doca-version TASKS.md ## debug`](../doca-version/TASKS.md#debug)
   layer 2 and re-confirm the three version anchors in
   [`## Version compatibility`](#version-compatibility). Owner:
   [`doca-version`](../doca-version/SKILL.md).
8. **Cross-cutting host layer.** The deployment is healthy from
   kubelet's view and the service's view, but the BlueField OS
   or the host OS introduces a cross-cutting failure (kernel
   version, driver loaded / not loaded, hugepage allocation,
   PCIe link state, BFB-side log surface). Symptoms: rare and
   hard to attribute; surface as everything-looks-fine-but-it-
   does-not. Resolution: drop to the cross-cutting debug ladder
   at [`doca-debug`](../doca-debug/SKILL.md). Owner:
   [`doca-debug`](../doca-debug/SKILL.md) + the operator's
   BlueField OS team.

Service containers do not return `DOCA_ERROR_*` to the operator's
shell — the outward surface is kubelet's pod-status output, the
container runtime's logs, and the service-specific liveness signal
each per-service skill names. The DOCA error taxonomy at
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../doca-programming-guide/CAPABILITIES.md#error-taxonomy)
becomes relevant only when a service container internally calls
into a DOCA library that surfaces `DOCA_ERROR_*` in its log
stream; that is owned by the per-service skill plus
[`doca-debug`](../doca-debug/SKILL.md), not by this one.

## Observability

Documented observability surfaces, in the order the agent reaches
for them. Three layers, each with its own owner. Healthy means all
three agree.

- **Kubelet pod-status layer (FIRST).** The first place to look.
  Kubelet's documented status surface answers *"is the pod-spec
  YAML parsed, is the pod scheduled, is the container running, is
  it restart-looping"*. The agent reaches this via the BlueField
  container manager's structured-status output when present, or
  via the documented kubelet status command the public Container
  Deployment Guide names. The agent does NOT invent a `kubectl
  get pods` invocation against a cluster API — there is no
  cluster API in this deployment shape.
- **Container-runtime log layer (SECOND).** Once kubelet reports
  the container is running (or restart-looping), the container
  runtime's log stream for that container is the agent's window
  into what the service inside is doing — config-parse success
  or failure, the documented bring-up lines, any per-service
  error / warning lines. The agent quotes the runtime command
  the public Container Deployment Guide names (or routes to it
  via [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md));
  the agent does NOT invent runtime CLI flags.
- **Service-side liveness layer (THIRD, load-bearing for
  "healthy").** Each per-service skill names the documented
  liveness signal that proves the service inside the container
  is actually serving its purpose — Argus's event-stream output,
  DMS's gNMI session-up + per-RPC counter, Firefly's PHC offset
  + ports state, Flow-Inspector's mirrored-flow ingest,
  OS-Inspector's per-target sample, UROM service's per-operation
  counter, and so on. This layer is the one the agent uses as
  the end-to-end "healthy" oracle; "pod is `Running`" alone is
  NOT the same as "the service inside is ready". Route to the
  per-service skill for the exact signal.

A note on what the agent does NOT do: this deployment shape has no
cluster-wide observability (no Prometheus scrape against a cluster
API, no metrics-server, no `kubectl logs` against a cluster).
Per-pod observability is via the BlueField's local kubelet + local
container runtime + per-service signal. Cross-cutting host-side
observability (`dmesg`, `ip link`, `devlink`, `ethtool` on the
BlueField) lives in
[`doca-setup TASKS.md ## Command appendix`](../doca-setup/TASKS.md#command-appendix)
and
[`doca-debug TASKS.md ## debug`](../doca-debug/TASKS.md#debug).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

The cross-cutting safety surface for any DOCA service container
deployment. Per-service skills add overlays on top (e.g. Firefly's
PTP-aware-path-required rule, DMS's gNMI-credential rotation rule,
UROM service's paired-workload-restart-order rule); the
cross-cutting rules below apply across every service.

- **Smoke before bulk (load-bearing).** Before putting the
  BlueField under any real workload, the agent walks the smoke
  sequence in [`TASKS.md ## test`](TASKS.md#test): (a) the pod
  reaches `Running` per kubelet's documented status surface; (b)
  the container's ENTRYPOINT log shows the per-service config
  parsed cleanly and the documented bring-up lines complete;
  (c) the per-service liveness signal (per the matching per-
  service skill) is healthy. Only then is the BlueField ready
  for the per-service workload (PTP-locked workload for Firefly,
  gNMI traffic for DMS, mirror-source traffic for Flow-Inspector,
  introspection targets for OS-Inspector, paired UROM workload
  for UROM service, policy-driven monitoring for Argus).
  Skipping the smoke and going straight to bulk workload is the
  most common reason "kubelet says the pod is up but my workload
  still fails".
- **Failed pod restart is HIGH-STAKES — clear the root cause
  BEFORE letting kubelet restart-loop the pod.** Kubelet's
  documented default is to restart a failed container; that is
  the right default for a transient failure but the WRONG
  default for a recurring failure. A pod in `CrashLoopBackOff`
  burns BlueField CPU / memory, fills the container runtime's
  log surface, and obscures the underlying error. The agent's
  rule: a pod that has crashed more than the documented
  threshold (or twice in a row with the same exit signature) is
  no longer evidence the deployment can self-heal; STOP the
  retry loop, read the container's last full ENTRYPOINT log,
  walk the error taxonomy from layer 1, and only re-enable
  scheduling once the root cause is identified. Letting kubelet
  loop a known-broken pod is not resilience; it is delayed
  diagnosis.
- **Do not invent image strings, tags, kubelet flag names,
  pod-spec field names, or the static-pod manifests directory
  path.** Quote each from the public DOCA Container Deployment
  Guide plus the matching per-service guide reached through
  [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md).
  Prose-derived strings here are the load-bearing first-app
  hallucination failure for this skill; the operator pulls the
  wrong image, names the wrong field, drops the YAML in the
  wrong directory, and the deployment never starts.
- **Confirm the BlueField preconditions BEFORE dropping the pod
  spec.** Every BlueField precondition class (DOCA install,
  container runtime + kubelet, BFB version, per-service firmware
  slot, image-pull reachability to NGC, host-OS permissions) is
  cheaper to confirm before the deploy than to debug after. The
  agent's rule: do not walk the deploy until every precondition
  in [`## Capabilities and modes`](#capabilities-and-modes)
  precondition table that the per-service skill flags as
  required is closed.
- **Edit-in-place is documented; treat it as a deploy event.**
  When the operator edits a pod-spec file in place (rather than
  removing-and-re-adding), kubelet's documented behavior is to
  reconcile. Treat every edit as a deploy event for safety:
  re-walk the smoke after every edit. A config change followed
  by "it probably still works" is exactly the failure mode the
  smoke replaces.
- **The per-service skill owns the per-service safety overlay.**
  Firefly's PTP-aware-path-required rule, DMS's
  gNMI-credential rotation rule, UROM service's paired-
  workload-restart-order rule, Argus's policy-rollback rule,
  Flow-Inspector's mirror-source-quiesce rule, OS-Inspector's
  host-OS-permission-boundary rule, and similar per-service
  safety rules are owned by the matching per-service skill, not
  by this one. The agent reads the per-service skill in parallel
  and applies both layers; this skill names the cross-cutting
  baseline.

## Public-source pointer

The canonical public source for the cross-cutting DOCA container-
deployment runtime is the **DOCA Container Deployment Guide** on
`docs.nvidia.com`, reachable through
[`doca-public-knowledge-map ## DOCA services`](../doca-public-knowledge-map/SKILL.md#doca-services).
For each per-service overlay (image string, config schema, paired-
workload contract, "healthy" definition), the canonical public
source is the matching per-service guide listed in the same
section. Verify that the version of each guide matches the
BlueField OS image (BFB), the host DOCA install, and the container
tag pulled — pod-spec field names, the static-pod manifests
directory path, and per-service config field names are documented
to evolve, so anything quoted from memory is suspect.
