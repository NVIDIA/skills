# doca-container-deployment — reference detail

Moved out of `SKILL.md` to keep the loader under the per-file size budget. This is supporting detail, not routing logic.

## Example questions this skill answers well

The CLASSES of container-deployment questions this skill is built to
answer, each with one worked example. The class is the load-bearing
piece; the worked example is one instance.

- **"How is a DOCA service container actually run on the BlueField —
  what is the runtime, who watches what, how does a pod-spec get
  picked up?"** — worked example: *"I have the DOCA Management
  Service (DMS) container image on the BlueField — what do I do with
  it?"*. Answered by the kubelet-standalone pattern in
  [`CAPABILITIES.md ## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes)
  + the pod-spec-drop walkthrough in
  [`TASKS.md ## run`](../TASKS.md#run).
- **"Where do I put the YAML pod spec so kubelet picks it up, and
  what shape does the spec have to be?"** — worked example: *"I have
  a YAML manifest for the Firefly container — where on the BlueField
  filesystem does it go?"*. Answered by the static-pod-manifests
  directory rule in
  [`CAPABILITIES.md ## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes)
  + the documented-recipe rule in
  [`TASKS.md ## modify`](../TASKS.md#modify) (the agent quotes the
  pod-spec shape from the public DOCA Container Deployment Guide via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md);
  it does NOT invent YAML field names).
- **"My pod spec is in the directory but the pod never starts — how
  do I diagnose it?"** — worked example: *"I dropped the OS-Inspector
  pod-spec YAML into the documented manifests directory and nothing
  happens"*. Answered by the layered error taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](../CAPABILITIES.md#error-taxonomy)
  (pod-spec syntax → pod scheduling → image pull → runtime → volume
  mount → network policy → version → cross-cutting host) + the
  layered debug ladder in [`TASKS.md ## debug`](../TASKS.md#debug).
- **"How do I find the logs of a DOCA service container, and what
  does 'healthy' look like before I put real workload on the
  BlueField?"** — worked example: *"the Flow-Inspector pod is
  `Running`, but I do not yet know whether the mirrored-flow capture
  inside is actually ready"*. Answered by the smoke-before-bulk loop in
  [`CAPABILITIES.md ## Safety policy`](../CAPABILITIES.md#safety-policy)
  + the eval-loop overlay in [`TASKS.md ## test`](../TASKS.md#test).
- **"My pod was Running and crashed; should I just have kubelet
  restart it, or is that exactly the wrong thing?"** — worked
  example: *"the DMS pod went into a restart loop after a config
  edit; should I leave it looping or step in?"*. Answered by the
  failed-pod-restart-is-high-stakes rule in
  [`CAPABILITIES.md ## Safety policy`](../CAPABILITIES.md#safety-policy)
  + the *"clear the root cause first"* layer in
  [`TASKS.md ## debug`](../TASKS.md#debug).
- **"Does this same pattern carry over to every other DOCA service,
  or is each service deployed in a different way?"** — worked
  example: *"I have DMS deployed; what changes for Firefly,
  Flow-Inspector, OS-Inspector, UROM service, and Argus?"*.
  Answered by the cross-service generalization in
  [`CAPABILITIES.md ## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes)
  + the per-service overlay routing in
  [`TASKS.md ## Deferred task verbs`](../TASKS.md#deferred-task-verbs)
  (the runtime pattern is uniform; the per-service config schema,
  precondition, observability surface, and "healthy" definition
  layer on top via the matching per-service skill).

## What this skill deliberately does not ship

This skill is **agent guidance**, not a templates or sample-pod-spec
bundle. To keep the boundary clean, it deliberately does not contain
— and pull requests should not add:

- **Pre-baked pod-spec YAML files** (full pod specs, ready-to-drop
  mount / image / command bundles for Argus / DMS / Firefly / etc.)
  intended to be copy-pasted into production. Pod specs are
  deployment-specific (per-service image tag, mount paths, host
  paths the operator picks) and the safe answer for an external
  operator is to derive them from the public DOCA Container
  Deployment Guide plus the matching per-service guide against
  their own target. The agent's job is to prescribe the *procedure*
  and quote the documented field names, not to ship a YAML the user
  might run unmodified.
- **Container image names and tags.** The canonical image source
  for any DOCA service container is the public DOCA Container
  Deployment Guide and the NGC catalog; the agent routes through
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  for the current image string and tag rather than quoting one
  from memory. Inventing an image name (e.g. a fictional
  `nvcr.io/nvidia/doca/<service>:latest`) is the load-bearing
  first-app failure for this skill.
- **Static-pod-manifests-directory path strings, kubelet flag
  names, and pod-spec field names invented from generic Kubernetes
  knowledge.** Kubelet standalone mode picks pod specs up from the
  documented directory the BlueField OS / DOCA Container Deployment
  Guide names. The agent quotes the documented path string, flag
  name, or field name from the guide; the agent does NOT infer it
  from upstream Kubernetes prose. When in doubt, route to
  [`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services)
  for the public Container Deployment Guide URL.
- **A `samples/`, `templates/`, or `reference/` subtree** of any
  kind. A mock or incomplete artifact in this skill's tree, even
  one labeled "reference", is misleading: operators will read it as
  production-ready.

## Related skills

- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  — the routing table to the public DOCA Container Deployment Guide
  (cross-service deployment pattern), the per-service public guides
  for the in-bundle services (Argus, DMS, Firefly, Flow-Inspector,
  OS-Inspector, UROM service), and the NGC catalog. This skill does
  not duplicate URLs; it points at them and adds the
  deployment-runtime overlay. See in particular
  [`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services)
  for the per-service URL set and the cross-service Container
  Deployment Guide row at the bottom of that section.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation and
  install verification on the BlueField target where the DOCA
  service container will run, including the *I have no install yet*
  path via the public NGC DOCA container and the BFB version check.
  (`doca-setup` also documents the firmware-slot enable workflow
  for externally-productized NVIDIA services that emulate a
  host-facing PCIe device; none of the six in-bundle services need
  that workflow — see the firmware-slot disclaimer in
  [`CAPABILITIES.md ## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes).)
  This skill assumes its preconditions are satisfied at the
  BlueField target.
- [`doca-version`](../../doca-version/SKILL.md) — canonical DOCA
  version-handling rules. This skill's `## Version compatibility`
  cross-links the four-way match rule plus the container-tag-vs-
  host-package overlay; the body of those rules lives in
  `doca-version`.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md)
  — the bundle's structured-tools precedence rule (detect / prefer
  / fall back / report). The Command appendix in
  [TASKS.md](../TASKS.md) honors this contract — the agent probes the
  BlueField container manager's structured status output first and
  falls back to the documented manual commands when the probe
  fails.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). Container-deployment-specific debug (pod never
  scheduled, image pull failed, ENTRYPOINT exited, volume mount
  missing, network policy blocking, restart loop) layers on top of
  the cross-cutting ladder.
- [`doca-bare-metal-deployment`](../../doca-bare-metal-deployment/SKILL.md)
  — the SIBLING deployment path: bare-metal hardware (host x86 OR
  BlueField Arm direct launch — systemd / tmux / direct invocation,
  hardware-resource binding, per-tenant isolation, restart
  discipline). This skill (container deployment) owns the
  kubelet-standalone + YAML pod-spec path; that skill owns the
  bare-metal binary-launch path. Both are routed to from
  [`doca-setup ## recognize`](../../doca-setup/TASKS.md#recognize).
- Per-service skills layered on top of this one (the four DOCA
  services in the bundle, 1:1 with `doca/services/`):
  [`doca-argus`](../../services/doca-argus/SKILL.md) (runtime
  security / monitoring),
  [`doca-dms`](../../services/doca-dms/SKILL.md) (DOCA Management
  Service — gNMI / gNOI),
  [`doca-firefly`](../../services/doca-firefly/SKILL.md) (PTP time
  sync),
  [`doca-urom-svc`](../../services/doca-urom-svc/SKILL.md) (Unified
  Communication Remote Memory Operations). Each of those skills
  shares this skill's deployment runtime and adds its own config
  schema, paired-workload contract, and "healthy" definition. The
  cross-service generalization rule is *"every DOCA service in the
  bundle uses this deployment runtime — the per-service skill is
  the overlay, not a re-statement of the runtime"*.
- **Non-goals (externally-productized NVIDIA services not in the
  DOCA monorepo and therefore not in this bundle):** BlueMan, HBN,
  SNAP, Virtio-net, DOCA Telemetry Service (as productized), and
  any future external NVIDIA networking software not under
  `doca/services/`. A user asking *"how do I deploy BlueMan?"* is
  routed to the public NVIDIA docs at `docs.nvidia.com/doca/sdk/`
  for that specific product, NOT silently extrapolated from this
  skill's contract. The strict-to-DOCA invariant is documented at
  `AGENTS.md ## Non-goals` row 7.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md)
  — general DOCA patterns. Container-deployment is service-shaped
  not library-shaped, so the build / modify / first-app pattern
  there does not apply directly, but the cross-library debug
  discipline (frontend-before-backend, env-before-program) remains
  useful when a service container surfaces an error that originated
  in a DOCA library it called.

