# DOCA container deployment — Tasks

**Where to start:** The order is `configure → build → modify → run →
test → debug`. The `## test` verb is an iterative loop, not a
one-shot pass — see the eval-loop overlay in `## test` below. For
container deployment, `build` and `modify` are about *deployment
configuration* (per-service image selection, pod-spec YAML, mount
contracts, per-service config-file mount), not about compiling
source.

These verbs cover the in-scope cross-cutting deployment workflows
for an external operator deploying any of the six in-bundle DOCA
service containers on the BlueField — Argus, DMS, Firefly,
Flow-Inspector, OS-Inspector, UROM service. Every step assumes the
operator has consulted the live public DOCA Container Deployment
Guide and the matching per-service guide (both reachable through
[doca-public-knowledge-map ## DOCA services](../doca-public-knowledge-map/SKILL.md#doca-services))
and is using them as the authoritative reference; this file
prescribes the *order* and *what to look up where*, not a
copy-paste runbook.

> **Non-goals (externally-productized NVIDIA services not in the
> DOCA monorepo):** BlueMan, HBN, SNAP, Virtio-net, DOCA Telemetry
> Service (DTS, as productized), and any future external NVIDIA
> networking software not under `doca/services/`. A user asking
> *"how do I deploy BlueMan?"* (or any of the others) is routed to
> the public NVIDIA docs at `docs.nvidia.com/doca/sdk/` for that
> specific product, NOT silently extrapolated from this skill's
> contract. The strict-to-DOCA invariant is documented at
> `AGENTS.md ## Non-goals` row 7.

## configure

Preparing the BlueField target, confirming every precondition the
deployment will rely on, and picking the per-service skill that
owns the overlay BEFORE any pod-spec is written. This is also the
verb where the smoke-before-bulk posture is established up front —
every later verb assumes the operator has read it here.

1. **Confirm the env is healthy first.** This skill expects DOCA
   installed on the BlueField Arm and the BlueField OS image
   shipping kubelet standalone + the documented container runtime.
   If install health is unverified, run
   [`doca-setup ## test`](../doca-setup/TASKS.md#test) first.
   If the user has no install yet, route to
   [`doca-setup ## no-install`](../doca-setup/TASKS.md#no-install)
   for the public NGC DOCA container path. Confirm the BFB
   version on the BlueField is one the public DOCA Container
   Deployment Guide certifies for the operator's DOCA release.
2. **Pick the per-service skill and load it in parallel.**
   Container deployment is a *cross-cutting* runtime; the
   per-service overlay (config schema, paired workload, "healthy"
   definition) lives in the matching per-service skill. Load both
   skills together:
    - Argus → [`doca-argus`](../services/doca-argus/SKILL.md)
    - DMS → [`doca-dms`](../services/doca-dms/SKILL.md)
    - Firefly → [`doca-firefly`](../services/doca-firefly/SKILL.md)
    - UROM service → [`doca-urom-svc`](../services/doca-urom-svc/SKILL.md)

   Quote the per-service config schema and "healthy" signal from
   the per-service skill; quote the runtime contract (kubelet
   standalone, static-pod manifests directory, pod-spec shape,
   mount contract) from this skill.
3. **Confirm the BlueField preconditions.** Per the precondition
   table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   close every precondition the per-service skill flags as
   required BEFORE writing a pod spec:
    - DOCA installed on the BlueField Arm and healthy (per
      [`doca-setup ## test`](../doca-setup/TASKS.md#test)).
    - Container runtime + kubelet standalone present and running
      on the BlueField, as documented in the public Container
      Deployment Guide.
    - BFB version compatible with the per-service container tag.
    - **Per-service firmware slot** — does NOT apply to any of the
      six in-bundle DOCA services (none of them emulate a
      host-facing PCIe device); see the firmware-slot disclaimer
      in [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
      If the operator is deploying an externally-productized
      NVIDIA service that does emulate a PCIe device, that is out
      of scope here per the Non-goals callout above and the
      operator's source of truth is the public NVIDIA docs at
      `docs.nvidia.com/doca/sdk/`.
    - Image-pull reachability from the BlueField to `nvcr.io`
      (corporate proxy / firewall / air-gap concerns close here
      with the operator's network team).
    - Host-OS permissions to write into the documented static-pod
      manifests directory and to read kubelet's log surface and
      the container runtime's log surface.
4. **Identify the per-service image source.** Per the per-service
   public guide, the container image is published to the NGC
   catalog. The agent MUST route through
   [`doca-public-knowledge-map ## DOCA services`](../doca-public-knowledge-map/SKILL.md#doca-services)
   for the current image string + tag for the operator's DOCA
   release; do NOT invent an image name from memory. Wrong image
   string is the most common first-app failure for any DOCA service
   deployment, and is the load-bearing first-app failure for this
   shared runtime.
5. **Plan the per-service config-file mount.** Every DOCA service
   container is configured by a documented config file mounted
   into the container at the path the per-service guide names.
   Plan, in this order: (a) where the config file will live on
   the BlueField filesystem (a host path the kubelet runs as can
   read); (b) what mount path the per-service guide expects
   inside the container; (c) the per-service config field names
   the per-service guide quotes — quote from there, do NOT infer
   from generic Kubernetes / config intuition.
6. **Plan the rollback path.** For any in-bundle service the
   per-service skill flags as HIGH-STAKES (Firefly on a production
   time-sensitive plant; DMS replacing the BlueField's gNMI control
   surface; UROM service driving an in-flight paired host↔DPU
   workload), every deploy on a live BlueField must have: (a) the
   pre-deploy BlueField state captured (PHC state before Firefly;
   gNMI session inventory before DMS; paired-workload checkpoint
   before UROM service); (b) the previous-known-good pod spec (or
   a no-service baseline) ready to re-apply; (c) an out-of-band
   way to reach the BlueField if the deploy disrupts host
   connectivity (BlueField console, redundant management path,
   IPMI to the host); (d) a maintenance window agreed with whoever
   uses the host. For services without a HIGH-STAKES per-service
   overlay (Argus, Flow-Inspector, OS-Inspector under introspection
   workloads), the rollback bar is lower but the "be able to
   revert" rule still applies.
7. **Identify the three version anchors.** Per
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility),
   record BOTH the host DOCA install version on the BlueField
   Arm (via
   [`doca-version TASKS.md ## configure`](../doca-version/TASKS.md#configure)),
   the BFB version on the BlueField, AND the per-service
   container tag pulled. A mismatch among the three is the silent
   *"the docs say this should work"* trap; align them explicitly.

## build

Container deployment is a service runtime, not a library. There is
no *application* artifact for the operator to build — the per-
service container image ships from the public NGC catalog and the
operator pulls it; clients of any per-service surface (gNMI clients
for DMS, PTP followers for Firefly, mirror sources for
Flow-Inspector, introspection consumers for OS-Inspector) are tools
the operator already has.

If the user is asking how to build a **per-service config bundle**
(e.g. a Firefly PTP config, a DMS gNMI endpoint bundle, a
Flow-Inspector mirror-source config, an Argus policy bundle), that
is the per-service skill's `## modify` workflow, not a build. Route
to the matching per-service skill.

If the user is asking how to build a **DOCA application** that
will be deployed by the same kubelet-standalone + pod-spec-drop
pattern — that is, a user-authored container image rather than an
NVIDIA-shipped one — the build is the application's normal DOCA
build (per
[`doca-programming-guide ## build`](../doca-programming-guide/TASKS.md#build))
plus a container build the user owns. The shared runtime contract
in [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
still applies to the *deploy* step; the build step is library-
shaped, not service-shaped.

If the user is asking how to build the **per-service container
itself** from source, that is *not* an external-operator workflow
— the container ships pre-built from NGC and rebuilding it is out
of scope. Route to the per-service public guide via
[doca-public-knowledge-map ## DOCA services](../doca-public-knowledge-map/SKILL.md#doca-services).

## modify

Container deployment does not have a "modify a sample program"
workflow analogous to DOCA libraries; there is no shared sample
the user starts from. The deployment-side analog of "modify" is
**adapt the documented pod-spec recipe from the public DOCA
Container Deployment Guide plus the matching per-service guide to
the operator's environment**:

1. **Start from the documented recipe.** Identify, in the public
   DOCA Container Deployment Guide *for the BFB the operator is
   on*, the pod-spec recipe that matches the operator's service.
   If the per-service guide ships its own pod-spec recipe (e.g.
   a Firefly YAML, a DMS YAML, a Flow-Inspector YAML), use that
   as the per-service overlay on top. Quote both; do not author a
   new pod-spec shape from scratch.
2. **Diff against the operator's environment.** Note the
   substitutions the operator MUST make: the per-service
   container image tag pulled from NGC, the host path that
   holds the per-service config file, the volume-mount points
   the per-service guide names, any per-service envvars /
   security-context the per-service guide quotes, and the
   per-service paired-workload coordinates (PHC device node for
   Firefly, gNMI client endpoint for DMS, mirror source binding
   for Flow-Inspector, paired host↔DPU socket for UROM service).
   Capture each substitution explicitly; the substitution list
   is the operator's edit log for the deploy.
3. **Apply minimum-change.** Change only what the operator's
   environment forces. Every additional deviation from the
   documented recipe widens the surface for an unintended
   mount-path / network-policy / image-pull / version failure
   the operator will have to debug later.
4. **Re-validate against the precondition table.** Each
   substitution is a chance to accidentally break a BlueField
   precondition (firmware slot, BFB version, image-pull
   reachability, host-path permission). Walk
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   precondition table after every substitution.
5. **Re-validate against the per-service safety overlay.** Per-
   service skills add their own safety overlays (Firefly's
   PTP-aware-path-required rule, DMS's gNMI-credential rotation
   rule, UROM service's paired-workload-restart-order rule,
   Argus's policy-rollback rule, Flow-Inspector's
   mirror-source-quiesce rule, OS-Inspector's host-OS-
   permission-boundary rule). The modify step is the right place
   to surface the overlay, not debug time.
6. **Drop the pod spec into the documented static-pod manifests
   directory.** The path is the one the public DOCA Container
   Deployment Guide names for the BFB the operator is on; do NOT
   infer it from generic Kubernetes prose. Adding the file =
   telling kubelet to bring the pod up; removing the file =
   telling kubelet to tear it down. Editing the file in place is
   documented to trigger a reconcile — treat every edit as a
   deploy event for safety (re-walk the smoke after every edit
   per [`## test`](#test)).

The agent's anti-pattern alert: a *"start from a generic
Kubernetes pod-spec template and adapt"* is almost always slower
than starting from the public Container Deployment Guide plus
per-service guide, because the documented BlueField pod-spec
fields and the static-pod manifests directory are not 1:1 with
upstream Kubernetes intuition.

## run

Bringing up the per-service pod, confirming the runtime layer
reaches a healthy state, and confirming the per-service liveness
signal before layering any real workload on top. Every step here
assumes the prerequisites in [`## configure`](#configure) are done.

1. **Confirm the per-service image is pullable from NGC.** Per
   [`## configure`](#configure) step 4, the image string +
   tag comes from the per-service public guide reached through
   [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md);
   the BlueField has reach to `nvcr.io`; the documented image-list
   command on the BlueField container runtime can pull a fresh
   copy of the image and report the digest. A pull failure here
   is a layer-3 symptom in
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   and must be resolved before the pod-spec is dropped.
2. **Drop the documented pod spec into the static-pod manifests
   directory.** Per [`## modify`](#modify) step 6, the pod-spec
   YAML is authored from the public Container Deployment Guide
   plus the per-service guide; the operator writes the file at
   the documented path the public guide names. Adding the file
   triggers kubelet's documented reconcile.
3. **Verify kubelet has scheduled the pod.** Use the BlueField
   container manager's documented status command (or its
   structured-status surface per
   [`doca-structured-tools-contract`](../doca-structured-tools-contract/SKILL.md))
   to confirm the pod is in `Running` with no recent restarts.
   A pod stuck in `Pending` is layer 2 of
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy);
   a pod in `ImagePullBackOff` is layer 3; a pod in
   `CrashLoopBackOff` is layer 4. Drop to
   [`## debug`](#debug) BEFORE proceeding if the pod is not
   `Running`.
4. **Read the container's ENTRYPOINT log.** Use the BlueField
   container manager's documented log-stream command for the
   running container. Confirm the per-service guide's documented
   bring-up lines appear and that no documented error / warning
   lines repeat. A "container Running but ENTRYPOINT log shows a
   config-parse error" is the canonical layer-4 (runtime /
   ENTRYPOINT) trap — the pod is "running" in kubelet's view but
   the service inside is not actually serving its purpose.
5. **Verify the per-service liveness signal.** Each per-service
   skill names the documented liveness signal that proves the
   service inside the container is actually ready (event-stream
   output for Argus, gNMI session up + per-RPC counter for DMS,
   ports state + PHC offset for Firefly, mirrored-flow ingest
   for Flow-Inspector, per-target sample for OS-Inspector,
   per-operation counter for UROM service). Read it now, per the
   matching per-service skill. If the per-service signal is NOT
   healthy, drop to
   [`## debug`](#debug) layer 4 (runtime / ENTRYPOINT) or to
   the per-service skill's debug ladder — not to "restart the
   pod and hope".
6. **Smoke before bulk (next: `## test` step 1).** Before
   driving any per-service workload (real PTP-locked workload for
   Firefly, gNMI traffic for DMS, mirror-source traffic for
   Flow-Inspector, introspection targets for OS-Inspector,
   paired UROM workload for UROM service, policy-driven
   monitoring for Argus), walk `## test` step 1 once to confirm
   end-to-end readiness; only then layer the workload on top.

For the runtime version + container-tag cross-checks that
underlie *"my service behaves differently from what the docs
say"*, see
[`doca-version TASKS.md ## run`](../doca-version/TASKS.md#run)
and apply the three-version-anchor overlay from
[`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility).

## test

Container deployment has no "compile and unit-test" workflow —
testing is operational and end-to-end.

**`## test` is an iterative loop, not a one-shot pass.** Every
mutation (pod-spec field edit, image-tag bump, mount path change,
config-file edit, per-service config knob, BFB upgrade) re-opens
the smoke sweep. Skipping the re-run after a mutation is the
failure mode this loop replaces.

The eval-loop overlay (rows apply to every DOCA service
deployment, not just one service):

| Step | Why this is a loop, not a step | Where the substance lives |
| --- | --- | --- |
| 1 → 4 → 1 | Step 4 (per-service liveness probe) often reveals an as-deployed gap in the per-service config that masquerades as a pod problem; loop back to step 1 | [`## test`](#test) step 4 |
| 2 → ## debug | When the pod does NOT reach `Running`, the deployment is non-functional — escalate to [`## debug`](#debug) immediately, do not run later steps | [`## debug`](#debug) |
| 3 → ## configure → 3 | When the ENTRYPOINT log shows a precondition was not closed (firmware slot, BFB version, image-pull, mount path), loop back to [`## configure`](#configure) step 3 and re-walk the preconditions | [`## configure`](#configure) |
| 1..5 → ## run | Each loop iteration ends with a documented smoke; if all five pass, hand off to live `## run` traffic | [`## run`](#run) |

The agent's rule: every mutation re-opens the sweep. A pod-spec
edit followed by *"it probably still works"* is exactly the
failure mode the iterative loop is here to prevent — and on a
HIGH-STAKES in-bundle service (Firefly on a production plant, DMS
replacing a live gNMI control surface, UROM service driving an
in-flight paired workload) the cost of that failure mode is host
disruption, not just *"weird traffic"*.

1. **End-to-end smoke (the recommended deployment smoke).** With
   the pod scheduled, confirm in this order:
    1. Pod is in `Running` per kubelet's documented status
       surface; restart count stable (NOT in `CrashLoopBackOff`,
       NOT in `ImagePullBackOff`).
    2. Container ENTRYPOINT log shows the per-service config
       parsed cleanly and the documented bring-up lines complete.
    3. The per-service liveness signal (per the matching per-
       service skill) is healthy. This is the load-bearing
       end-to-end proof; "pod Running" alone is NOT the same as
       "service ready".
    4. A trivial liveness probe against the service's documented
       endpoint succeeds (per-service skill names what "trivial"
       means — one event for Argus, one gNMI Get for DMS, one PTP
       frame for Firefly, one mirrored packet for Flow-Inspector,
       one introspection sample for OS-Inspector, one UROM
       operation for UROM service). Only after all four pieces
       pass is the deployment ready for bulk per-service workload.
2. **Pod-spec smoke.** Confirm the negative case: temporarily
   edit a non-critical pod-spec field that the public Container
   Deployment Guide flags as kubelet-validated (e.g. the
   resource request, the documented label set, the documented
   pod-spec metadata field) and confirm kubelet's status surface
   reports the expected reconcile. Restore the correct value
   afterwards. This is the operator's evidence that the deploy
   contract is in fact reconcile-driven on their specific
   BlueField.
3. **Precondition smoke.** Independently of the pod, confirm each
   precondition the per-service skill flags is still satisfied:
   `nvcr.io` is reachable from the BlueField; the per-service
   firmware slot (if any) is still enabled; the BFB version
   matches the Container Deployment Guide; the host path holding
   the per-service config file is still in place. A divergence
   between *"precondition healthy independently"* and *"pod is
   not healthy"* is a layer-4 / layer-5 / layer-6 symptom in
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy);
   convergence is a per-service-config symptom routed to the
   per-service skill.
4. **Capability + deploy snapshot.** Save the *as-deployed*
   answer to: which per-service container tag is running, which
   BFB and host DOCA versions it pairs with, which pod-spec
   YAML was dropped, where the per-service config file lives,
   what `kubelet` reports for the pod, what the container
   ENTRYPOINT log's bring-up looks like, what the per-service
   liveness signal reads. This snapshot is the artifact that
   lets future debug sessions skip rediscovery — and on a
   HIGH-STAKES service it is the rollback baseline.
5. **Multi-pod smoke (only when the service has paired peers).**
   For in-bundle services with paired pods or paired host-side
   workloads (host-side time follower for Firefly, gNMI client
   for DMS, mirror source for Flow-Inspector, paired host↔DPU
   workload for UROM service), bring the paired peer up after
   this pod and confirm the per-service smoke the per-service
   skill names (host's `chronyc tracking` follows the PHC for
   Firefly, gNMI Get returns documented schema for DMS,
   mirrored-flow counter advances for Flow-Inspector, paired
   UROM operation completes for UROM service). One peer at a
   time; multi-peer bring-up that batches everything together
   makes ingest-failure attribution much harder.

Loop termination: stop iterating once two consecutive iterations
of the same kind don't change anything — that means the cause is
below the deployment runtime (per-service config, paired peer,
BlueField OS, host). Escalate to the per-service skill's debug
ladder plus
[`doca-debug TASKS.md ## debug`](../doca-debug/TASKS.md#debug)
with the captured layer evidence.

## debug

Layered diagnosis. Walk the layers in this order; do not skip
down without clearing the layer above. The eight layers match
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).

1. **Pod-spec syntax layer (layer 1).** Is the YAML pod-spec
   file even parsing? Symptoms: kubelet's log surface shows a
   parse / schema error; the pod never appears in kubelet's
   status output even though the YAML is in the documented
   static-pod manifests directory. Resolution: re-quote the
   pod-spec shape from the public DOCA Container Deployment
   Guide for the BFB the operator is on; do NOT infer field
   names from generic Kubernetes intuition. A common trap is a
   pod-spec field name that *exists* in upstream Kubernetes but
   is rejected by the kubelet build the BlueField OS image
   ships.
2. **Pod scheduling layer (layer 2).** Pod-spec parses, but the
   pod never reaches `Running`. Symptoms: pod stuck in
   `Pending`; kubelet reports a scheduling refusal. Resolution:
   re-walk the BlueField precondition table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   (BFB version, host-OS permissions, image-pull reachability;
   none of the six in-bundle services require a firmware
   emulation slot per the firmware-slot disclaimer in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes));
   confirm the resource requests in the pod spec are within the
   BlueField Arm's budget per the public guide.
3. **Image-pull layer (layer 3).** Pod scheduled but the
   container runtime cannot pull the image. Symptoms:
   `ImagePullBackOff`; the container runtime's image-pull log
   surface shows a 404 / 403 / timeout against `nvcr.io`.
   Resolution: re-quote the image string + tag from the
   per-service public guide reached through
   [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md);
   confirm BlueField reach to `nvcr.io` independently of the
   pod (a documented image-pull probe from the BlueField CLI is
   the cheapest check); engage the operator's network team
   when the BlueField does not have egress to `nvcr.io`. NEVER
   "fix" this by inventing an image tag from memory; that path
   only deepens the failure.
4. **Container-runtime / ENTRYPOINT layer (layer 4).** Image
   pulled, container starts, ENTRYPOINT exits non-zero or the
   pod is in `CrashLoopBackOff`. Symptoms: the container's
   ENTRYPOINT log shows a config-parse error, a missing
   dependency, or a failed internal precondition; the pod
   restart count climbs. Resolution: read the container's last
   full ENTRYPOINT log FIRST (the BlueField container manager's
   documented log-stream command); route the per-service
   config-parse error to the matching per-service skill's
   debug ladder. A pod that has crashed more than the documented
   threshold (or twice in a row with the same exit signature)
   is HIGH-STAKES per
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
   — STOP the retry loop, clear the root cause, only then
   re-enable scheduling.
5. **Volume-mount layer (layer 5).** Pod runs but the container
   cannot read / write a path the pod-spec names. Symptoms:
   ENTRYPOINT log shows permission / not-found errors on a
   mounted path; the per-service config file is mounted but
   empty / missing. Resolution: confirm the host path exists on
   the BlueField; confirm the container user can read / write
   it per the per-service guide; re-quote the volume-mount
   shape from the public Container Deployment Guide. A
   directory bind-mounted where a file was expected (or vice
   versa) is the canonical trap.
6. **Network-policy / host-firewall layer (layer 6).** Pod
   runs, the service inside is up, but a paired peer cannot
   reach it (or the service cannot reach a paired peer).
   Symptoms: liveness probe times out; an external client
   cannot connect; outbound to NGC / upstream PTP master / TOR
   / S3 endpoint hangs. Resolution: confirm host network
   reachability independently of the pod; confirm the pod's
   `hostNetwork` / port mapping matches what the per-service
   guide expects; engage the operator's network team when a
   corporate firewall is in the picture.
7. **Version layer (layer 7).** The pod is healthy, the service
   answers, but the behavior contradicts what the per-service
   guide on the screen describes. Symptoms: a documented field
   name is rejected; a documented "healthy" output line never
   appears. Resolution: walk
   [`doca-version TASKS.md ## debug`](../doca-version/TASKS.md#debug)
   layer 2 and re-confirm the three version anchors (host DOCA
   on the BlueField Arm, BFB, per-service container tag) per
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility).
   A per-service guide on a different DOCA release than the
   container tag is the canonical trap.
8. **Cross-cutting host layer (layer 8).** Deployment looks
   healthy at every layer above but a cross-cutting host issue
   (kernel version, driver loaded / not loaded, hugepage
   allocation, PCIe link state, BFB log surface) breaks the
   service's downstream behavior. Resolution: drop to
   [`doca-debug TASKS.md ## debug`](../doca-debug/TASKS.md#debug)
   for the cross-cutting debug ladder; engage the operator's
   BlueField OS team.

## Command appendix

Container-deployment commands the verbs above reach for, grouped
by purpose so the agent picks the right family without searching
prose. Every row is a class — the agent must not invent flags
beyond what the row names; flag and command discovery is `--help`
on the installed tool or the public guide, not prose recall.

**Infra-aware preamble (every row below).** Per the bundle's
detect → prefer → fall back → report contract documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent should:

1. Probe for the matching structured helper FIRST (`doca-env
   --json` for version + devices + libraries + drivers +
   hugepages in one shot; `doca-capability-snapshot` for
   per-device capability flags; `version-matrix.json` for
   *"available since"* lookups).
2. If the probe succeeds, the structured tool's output is the
   authoritative answer and the agent SHOULD NOT also run the
   manual command in the row below. Report *"using structured
   `<tool>`"*.
3. If the probe fails, fall back to the manual command in the
   row. Report *"falling back to manual chain"*.
4. The schemas the structured tools emit are defined in
   [`doca-structured-tools-contract ## Schemas`](../doca-structured-tools-contract/SKILL.md#schemas);
   the version-handling semantics (four-way match, NGC,
   headers-win) are owned by
   [`doca-version`](../doca-version/SKILL.md).

| Purpose | Command (class shape) | Owning step | Reads as healthy when … |
| --- | --- | --- | --- |
| Pod lifecycle status | The BlueField container manager's documented pod-status command (per the public DOCA Container Deployment Guide for the BFB on the BlueField) | [`## run`](#run) step 3; [`## debug`](#debug) layers 1 + 2 + 4 | The pod is in `Running` with no recent restart loop; kubelet reports no schedule / image-pull failures. |
| Container ENTRYPOINT log stream | The BlueField container manager's documented log-stream command for the running container | [`## run`](#run) step 4; [`## debug`](#debug) layers 4 + 5 + 6 | The per-service guide's documented bring-up lines appear; no documented error / warning lines repeat. |
| Image-pull verification | The BlueField container runtime's documented image-list / image-pull command after pulling the per-service image string from the public per-service guide | [`## configure`](#configure) step 4; [`## run`](#run) step 1 | The pulled image tag matches what was quoted from the per-service public guide; the image is present locally and the digest is stable. |
| Pod-spec mount inspection | The BlueField container manager's documented inspect command showing the mount list for the running pod | [`## run`](#run) step 2; [`## debug`](#debug) layer 5 | The per-service config file is mounted at the path the per-service guide names; any per-service host paths the pod-spec names are mounted in with the expected permissions. |
| Static-pod manifests directory listing | `ls` on the documented static-pod manifests directory the public DOCA Container Deployment Guide names (path varies by BFB version — quote from the live guide, do NOT memorize) | [`## configure`](#configure) step 5; [`## modify`](#modify) step 6; [`## debug`](#debug) layer 1 | The pod-spec YAML file is present at the documented path; ownership and mode are as the public guide expects; kubelet reconciles every file in the directory into a pod. |
| Per-service liveness signal | The per-service skill's documented liveness command (event-stream output for Argus, gNMI session up + per-RPC counter for DMS, `pmc` / `phc_ctl` + ports state for Firefly, mirrored-flow ingest for Flow-Inspector, per-target sample for OS-Inspector, per-operation counter for UROM service) | [`## run`](#run) step 5; [`## test`](#test) step 1.3 + 1.4; [`## debug`](#debug) layer 4 + 6 | Per the matching per-service skill's "healthy" definition. |
| BlueField precondition probe (firmware slot) | Not applicable to any of the six in-bundle DOCA services (none emulate a host-facing PCIe device) — see the firmware-slot disclaimer in [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes). Externally-productized services that do need an emulation slot are out of scope per the Non-goals callout at the top of this file. | n/a for in-bundle services | n/a |
| Image-pull reachability probe to NGC | The BlueField container runtime's documented manifest-pull / probe command against `nvcr.io` (or the documented HTTP probe from the BlueField CLI) | [`## configure`](#configure) step 3; [`## debug`](#debug) layer 3 | The BlueField can reach `nvcr.io` and pull a manifest at the documented tag; no proxy / firewall / air-gap rejects the request. |
| Version anchor — host DOCA (BlueField Arm) | `pkg-config --modversion doca-common` and `doca_caps --version` on the BlueField Arm | [`## configure`](#configure) step 7; [`## debug`](#debug) layer 7 | The two semver strings match each other and match the public DOCA Container Deployment Guide for the operator's release. |
| Version anchor — BFB version | The documented BFB-version command on the BlueField (per the BlueField OS documentation and the public DOCA Container Deployment Guide) | [`## configure`](#configure) step 1 + 7; [`## debug`](#debug) layer 7 | The BFB reports a version the public Container Deployment Guide certifies for the operator's DOCA release. |
| Version anchor — per-service container tag | The BlueField container runtime's documented image-inspect command for the running per-service container | [`## run`](#run) step 1; [`## configure`](#configure) step 7; [`## debug`](#debug) layer 7 | The tag matches what was quoted from the per-service public guide / NGC catalog for the operator's DOCA release; aligns with the host DOCA + BFB versions per the three-anchor contract. |

Three cross-cutting rules for this appendix:

- **Never invent an image string, tag, kubelet flag, pod-spec
  field name, or the static-pod manifests directory path.** The
  public DOCA Container Deployment Guide and the per-service
  guide reached through
  [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md)
  are the contract; prose-derived strings are the most common
  hallucination failure for this skill.
- **Kubelet status before container logs; container logs before
  per-service liveness.** When triaging, read kubelet's pod-
  status surface first (does kubelet think the pod is up?);
  only then read the container runtime's log stream (did the
  ENTRYPOINT parse the config and run?); only then read the
  per-service liveness signal (does the service actually serve
  its purpose?). Reading the per-service signal against a pod
  that kubelet has reported as `Pending` is meaningless.
- **Cross-link instead of duplicate.** Cross-cutting env
  commands (port-state, `devlink`, `ip link`, `ethtool` on the
  BlueField, `lspci` on the host) live in
  [`doca-setup TASKS.md ## Command appendix`](../doca-setup/TASKS.md#command-appendix)
  and
  [`doca-debug TASKS.md ## debug`](../doca-debug/TASKS.md#debug);
  this appendix names only the container-deployment-specific
  ones. Per-service liveness commands live in the matching
  per-service skill's Command appendix.

## Deferred task verbs

- **Per-service config-schema authoring** (Argus runtime-security
  policy; DMS gNMI / gNOI endpoint config; Firefly four-axis PTP
  config; Flow-Inspector inspector / sampling / mirror-source
  config; OS-Inspector host-OS introspection target config; UROM
  service operations-endpoint config) — out of scope here. Route
  to the matching per-service skill's `## configure` and
  `## modify` workflows; this skill names *that* a config file
  must be mounted, not *what* the per-service field names are.
- **Installing DOCA on the BlueField target** — out of scope
  here. Route to
  [`doca-setup ## configure`](../doca-setup/TASKS.md#configure)
  for env preparation and
  [`doca-setup ## test`](../doca-setup/TASKS.md#test) for
  install health verification, or
  [`doca-setup ## no-install`](../doca-setup/TASKS.md#no-install)
  for the public NGC DOCA container path if there is no DOCA
  install yet.
- **Flipping a BlueField firmware emulation slot** — out of scope
  here for two reasons. First, none of the six in-bundle DOCA
  services emulate a host-facing PCIe device, so none of them
  require an emulation slot (see the firmware-slot disclaimer in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)).
  Second, externally-productized NVIDIA services that *do* require
  a slot are themselves out of scope per the Non-goals callout at
  the top of this file; the firmware-tool workflow itself is owned
  by [`doca-setup ## configure`](../doca-setup/TASKS.md#configure)
  and the public BlueField firmware-configuration documentation.
- **Building a paired-workload application** (host-side time-sync
  follower behind Firefly, gNMI client behind DMS, mirror source
  behind Flow-Inspector, introspection consumer behind
  OS-Inspector, custom analyzer against Argus, paired host↔DPU
  workload behind UROM service) — not a container-deployment
  question. Route to the matching per-service skill's
  deferred-verbs block, which itself routes to the per-library
  skill ([`doca-telemetry-exporter`](../libs/doca-telemetry-exporter/SKILL.md),
  [`doca-flow`](../libs/doca-flow/SKILL.md),
  [`doca-rmax`](../libs/doca-rmax/SKILL.md), …) when the paired
  workload is a DOCA-using application.
- **Operating a full Kubernetes cluster** (cluster API,
  `kubectl`, `Deployment` / `Service` / `Ingress` objects,
  cluster-wide observability, CNI overlays) — out of scope here.
  This skill covers kubelet *standalone* mode on the BlueField,
  not a full Kubernetes control plane. For Kubernetes-scale DPU
  provisioning, see DOCA Platform Framework on GitHub via
  [`doca-public-knowledge-map ## Public source code: GitHub`](../doca-public-knowledge-map/SKILL.md#public-source-code-github).
- **Authoring or rebuilding a DOCA service container image** —
  out of scope here. The container ships pre-built from NGC at
  the tag the per-service public guide names; rebuilding it
  from source is not an external-operator workflow. Route to
  the per-service public guide via
  [`doca-public-knowledge-map ## DOCA services`](../doca-public-knowledge-map/SKILL.md#doca-services).
- **Cross-cutting host-side debug** (BlueField kernel, driver
  loaded / not loaded, hugepage allocation, PCIe link state) —
  out of scope here. Route to
  [`doca-debug TASKS.md ## debug`](../doca-debug/TASKS.md#debug).

## Cross-cutting

- The public DOCA Container Deployment Guide is the single
  source of truth for the kubelet-standalone-mode runtime, the
  static-pod manifests directory path, the pod-spec shape, the
  mount contract, the image-pull procedure, and the lifecycle
  commands. Per-service guides own the per-service overlay.
- The runtime is uniform across every DOCA service on the
  BlueField. The agent's job is to walk the shared runtime here,
  then route to the per-service skill for the overlay — not to
  re-state the runtime inside the per-service skill.
- The three version anchors (host DOCA on BlueField Arm, BFB
  version, per-service container tag) must all align. A
  mismatch among any two is the canonical *"the docs say this
  should work but it does not"* failure mode.
- Smoke before bulk. The documented smoke (pod reaches
  `Running`; ENTRYPOINT log shows config parsed and bring-up
  lines complete; per-service liveness signal healthy; trivial
  liveness probe answered) goes BEFORE any per-service workload
  on top, never after.
- Failed pod restart is HIGH-STAKES. A pod that recurringly
  crashes is no longer evidence the deployment can self-heal;
  stop the retry loop, clear the root cause, only then re-enable
  scheduling. Letting kubelet loop a known-broken pod is
  delayed diagnosis, not resilience.
- The agent does NOT invent pod-spec field names, kubelet
  flags, image tags, or the static-pod manifests directory
  path. Quote each from the public DOCA Container Deployment
  Guide plus the per-service guide reached through
  [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md).
- For URL routing to the Container Deployment Guide, the
  per-service guides, and the NGC catalog, see
  [`doca-public-knowledge-map ## DOCA services`](../doca-public-knowledge-map/SKILL.md#doca-services).
