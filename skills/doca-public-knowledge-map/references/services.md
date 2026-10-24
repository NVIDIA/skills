## DOCA services

DOCA ships a set of *services* — long-running daemons / containers
documented separately from the libraries. Per-service skills (where
they exist) live under `skills/services/<svc>`; the URLs below are
the public guides in `docs.nvidia.com/doca/sdk/`.

> **First — the umbrella.** When the user's service is not listed below, the
> canonical first stop is the
> [**DOCA Services** umbrella page](https://docs.nvidia.com/doca/sdk/DOCA-Services/index.html),
> which lists every public DOCA service with its purpose and links to its
> service guide.

| Service | Guide | What it does |
| --- | --- | --- |
| **DOCA Services** (umbrella index) | <https://docs.nvidia.com/doca/sdk/DOCA-Services/index.html> | Canonical list of every public DOCA service with its purpose and guide link. Always check here first when a service is not in the table below. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| DOCA Management Service (DMS) | <https://docs.nvidia.com/doca/sdk/DOCA-Management-Service-Guide/index.html> | Centralized configuration / operation of BlueField and ConnectX devices via gRPC (gNMI for config, gNOI for system ops). Covered by the `doca-dms/SKILL.md` skill. |
| DOCA Firefly Service | <https://docs.nvidia.com/doca/sdk/DOCA-Firefly-Service-Guide/index.html> | PTP / time synchronization service. Covered by the `doca-firefly/SKILL.md` skill. |
| DOCA Flow Inspector Service | <https://docs.nvidia.com/doca/sdk/DOCA-Flow-Inspector-Service-Guide/index.html> | Mirrored-flow inspection service. **Not covered by this bundle** — policy-excluded from this public release (see AGENTS.md `## Non-goals` (bundle `AGENTS.md#non-goals-questions-the-agent-should-recognize-and-refuse-politely`) item 7). Route the user to the public docs above. |
| DOCA UROM Service | <https://docs.nvidia.com/doca/sdk/DOCA-UROM-Service-Guide/index.html> | Unified Communication Remote Memory Operations service. Covered by the `doca-urom-svc/SKILL.md` skill. |
| DOCA Telemetry Service (as deployed) | <https://docs.nvidia.com/doca/sdk/DOCA-Telemetry-Service-Guide/index.html> | DTS deployment and configuration by shape — BlueField built-in static pod, host docker container, DPF `DPUService` / `doca-telemetry` helm chart — plus `dts_config.ini` providers, exporters and level labels. Covered by the `doca-dts/SKILL.md` skill (the telemetry *reader* and *exporter* library APIs stay with `doca-telemetry` / `doca-telemetry-exporter`). |
| DOCA Argus Service | <https://docs.nvidia.com/doca/sdk/DOCA-Argus-Service-Guide/index.html> | DOCA's runtime-security / monitoring service for BlueField. Covered by the `doca-argus/SKILL.md` skill. |
| DOCA OS Inspector Service | <https://docs.nvidia.com/doca/sdk/DOCA-Services/index.html> | DPU-side out-of-band host-OS introspection container (App-Shield-as-a-service). **Not covered by this bundle** — policy-excluded from this public release (see AGENTS.md `## Non-goals` (bundle `AGENTS.md#non-goals-questions-the-agent-should-recognize-and-refuse-politely`) item 7). Route the user to the umbrella docs above. |

> **Non-goals.** Externally-productized NVIDIA services that are NOT in
> `doca/services/` at the bundle's currently-aligned DOCA release —
> BlueMan, HBN, SNAP, Virtio-net — are intentionally out of scope for
> this bundle, which is strictly 1:1 with `doca/services/`. The one
> exception is the DOCA Telemetry Service as deployed, covered by
> `doca-dts/SKILL.md`. See
> AGENTS.md `## Non-goals` (bundle `AGENTS.md#non-goals-questions-the-agent-should-recognize-and-refuse-politely`) for the policy
> rationale. If a user asks about one of these external services, use
> the **[`## Externally-productized DOCA software — not in this bundle, but here is where to route`](external-dpu.md#externally-productized-doca-software--not-in-this-bundle-but-here-is-where-to-route)**
> table below to give them the authoritative public docs URL,
> service-specific common gotchas, and the Developer Forum entry
> point. Refusing without routing violates the bundle's own contract.

The bundle covers DOCA deployment via **two sibling top-level
cross-cutting skills**, one per deployment path, plus a front-door
routing decision in `doca-setup ## recognize` that lands the user
on the right one:

- The **Container Deployment Guide**
  (<https://docs.nvidia.com/doca/sdk/DOCA-Container-Deployment-Guide/index.html>)
  is the cross-service reference for how DOCA service containers are
  deployed on BlueField. Covered cross-cuttingly by the
  `doca-container-deployment/SKILL.md`
  skill — every in-bundle per-service skill above hands off to it
  for the kubelet-standalone static-pod manifests directory, image
  pull, and pod-spec drop pattern.
- The **bare-metal hardware deployment** path (a DOCA-linked
  application binary launched directly on host x86 or BlueField
  Arm bare-metal — no container) is covered by the
  `doca-bare-metal-deployment/SKILL.md`
  skill. It owns the launch contract (direct / tmux / systemd),
  hardware-resource binding (PF/VF/representor + NUMA + CPU pinning
  + IRQ affinity), per-tenant isolation primitives, the bare-metal
  error taxonomy, and the restart-loop-is-HIGH-STAKES rule. The
  authoritative public references are the **DOCA Programming
  Guide** (build, link, runtime preconditions:
  <https://docs.nvidia.com/doca/sdk/DOCA-Programming-Guide/index.html>),
  the **DPU / BlueField Platform Software Manual** (PCIe topology,
  devlink, representor naming, `mlxconfig` device introspection:
  <https://docs.nvidia.com/networking/display/bluefieldbsp4132>; DOCA 3.5
  pairs with BSP 4.16.0: <https://docs.nvidia.com/networking/display/bluefieldbsp4160>), and
  the per-library guides listed below in this map.
- The **front-door routing decision** between the two — *"which
  path applies to my workload?"* — is owned by
  `doca-setup ## recognize` (`doca-setup/TASKS.md#recognize`),
  which detects the system shape (host x86 / BlueField Arm
  bare-metal / DPU-only / fresh laptop) and asks the developer the
  minimum residual question before routing. Any deployment-shaped
  question (*"how do I deploy"*, *"my code is built, how do I run
  it"*, *"I just got a BlueField, what now"*) loads `## recognize`
  first.

**Platform day-1 bring-up (before any app runs).** Getting the DPU
itself to a healthy OS + firmware state is a distinct layer below the two
app-deployment paths above, and is covered by two platform-specific
top-level skills:

- **BlueField-3 (BF3)** — the classic RShim/BFB path (`bfb-install` over
  RShim, TMFIFO recovery interface, DPU vs separated-host mode via
  `mlxconfig`, post-BFB recovery) is covered by
  `doca-bf3-deployment/SKILL.md`. Public
  reference: the DPU / BlueField Platform Software Manual
  (<https://docs.nvidia.com/networking/display/bluefieldbsp4132>; DOCA 3.5
  pairs with BSP 4.16.0: <https://docs.nvidia.com/networking/display/bluefieldbsp4160>).
- **BlueField-4 (BF4)** — the BMC-driven path (the three OS-ISO install
  methods UEFI-HTTP / PXE / Redfish Virtual Media, the PLDM firmware-update
  flow, and the Grace Ubuntu + cloud-init install) is covered by
  `doca-bf4-deployment/SKILL.md`. Public
  references: the BlueField / DOCA docs on docs.nvidia.com; deep BSP/BFB
  internals stay in the BSP row below. Any mutating firmware/power-cycle
  step routes to `doca-hardware-safety/SKILL.md`.

**Telemetry collection.** Deploying/operating the DOCA Telemetry
Service (DTS) collection pipeline (providers/counters → schema →
collector → exporters: Prometheus / Fluent Bit / NetFlow / file-IPC),
in all three deployment shapes, is covered by `doca-dts/SKILL.md`;
the library reader/publisher APIs stay in
`doca-telemetry/SKILL.md` /
`doca-telemetry-exporter/SKILL.md`.

If the user asks about a DOCA service that is not in this table, open the
[**DOCA Services** umbrella page](https://docs.nvidia.com/doca/sdk/DOCA-Services/index.html)
to discover it. Do not guess service URLs.

### Deploying DOCA services at scale — orchestration entry-point (persona/scale routing)

> **Pick the path by *persona and scale*, not by service.** The deployment
> guidance for any DOCA service splits cleanly in two, and the agent
> should name which one the user is in before giving deployment steps:
>
> - **Developer / PoC / test / small-dev (single host or a handful of
>   DPUs).** For an **in-bundle** service (the six in the table above), use
>   its per-service skill plus the two sibling deployment skills to
>   deploy/configure it directly —
>   `doca-container-deployment/SKILL.md`
>   for the kubelet-standalone static-pod path, or
>   `doca-bare-metal-deployment/SKILL.md`
>   for a directly-launched DOCA binary. For an **externally-productized**
>   service (HBN, BlueMan, SNAP, Virtio-net — *not* in this bundle;
>   DTS-as-deployed *is*, via `doca-dts/SKILL.md`),
>   there is no in-tree skill: route to its guide via the
>   routing table below and follow that guide's config keys / `mlxconfig`
>   preconditions. Either way, **this bundle's in-tree service skills are
>   scoped to the developer/PoC persona.**
> - **Admin / production / fleet-scale (racks of DPUs, declarative
>   lifecycle, repeatable rollouts).** Do **not** hand-roll per-host
>   static pods or `bfb-install` loops at this scale. Route to the
>   orchestration layer, which owns DPU provisioning, declarative
>   `DPUService` deployment, and coordinated BFB/firmware rollouts:
>   - **DOCA Platform Framework (DPF)** — the K8s-native DPU lifecycle +
>     `DPUService` orchestrator (provisions BF-3, deploys services like
>     HBN/BlueMan/SNAP declaratively across a fleet). Docs:
>     <https://docs.nvidia.com/networking/display/dpf25101> · source:
>     <https://github.com/NVIDIA/doca-platform>. See the **DOCA Platform
>     Framework (DPF)** row in the routing table below for failure-mode
>     gotchas.
>   - **NVIDIA Network Operator** — manages host-side NIC software
>     (OFED driver, SR-IOV / RDMA device plugins, CNI) on regular k8s
>     nodes; pairs with DPF. See its row in the table below. Docs:
>     <https://docs.nvidia.com/networking/software/cloud-orchestration/index.html>.
>   - **NVIDIA Kubernetes Launch Kit** — an emerging aggregation/
>     integration layer that bundles orchestration (Network Operator
>     today, DPF next) behind an agent-friendly discovery + deployment-
>     generation interface: <https://github.com/NVIDIA/k8s-launch-kit>.
>
> **Agent contract for scale questions.** When the user's intent is
> production/fleet-scale, the agent should *prefer routing them to invoke
> the orchestration system* (DPF / Network Operator / Launch Kit) rather
> than generating raw per-host deployment logic itself. The in-bundle
> service skills remain the right answer for the developer/PoC persona and
> for SDK-level integration; orchestration owns the production path. This
> split is the bundle's deliberate **DOCA-SDK scope boundary** — the
> orchestration systems above are externally owned and are *routed*, not
> re-implemented, here.

