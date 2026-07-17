# doca-public-knowledge-map — routing tables

Full enumerated routing tables moved out of `SKILL.md` to keep the loader within the per-file size budget. The agent opens this file when it needs the concrete docs.nvidia.com / on-disk / GitHub entries.

## Example questions this skill answers well

These are the question SHAPES this skill is designed to route, with one worked
example each. A productive A/B test against this skill probes the *shape*, not
the literal wording.

- **"Where can I read about DOCA &lt;library/service/tool/concept&gt;?"** —
  worked example: *"Where can I read about DOCA Flow Connection Tracking?"*
  Answered by walking the `## Library- and module-specific guides`,
  `## DOCA services`, or `## DOCA tools` routing tables to the matching
  `docs.nvidia.com/doca/sdk/...` entry.
- **"Which DOCA libraries do I have installed and at what version?"** —
  worked example: *"How do I confirm the box has DOCA Flow 3.3 installed?"*
  Answered by combining `## Layout of an installed DOCA package` and
  `## Where to find the version`, with cross-link to the layered
  version-detection rules in
  [doca-setup ## Capabilities and modes](../../doca-setup/CAPABILITIES.md).
- **"Where is sample &lt;X&gt; on disk and where is its source on GitHub?"** —
  worked example: *"Where is the doca_flow sample that exercises ACL pipes?"*
  Answered by combining `## Layout of an installed DOCA package` (local)
  with `## Public source code: GitHub` (remote) — both sections name the
  canonical paths.
- **"What does this on-disk path mean, what should I cite from it?"** —
  worked example: *"What's in `/opt/mellanox/doca/applications/` that the
  customer can actually run?"* Answered by `## Layout of an installed DOCA package`,
  plus the "no source-tree paths" ground rule above.
- **"This URL I have 404s — what's the new one?"** — worked example: *"The
  Comm Channel page is gone in DOCA 2.5+."* Answered by the URL-rename rule
  at the top of `## Public documentation entry points`, plus the
  `## URL audit` footer at the bottom of this file.
- **"Where is the customer-facing place to ask for help on this?"** —
  worked example: *"What's the developer forum's DOCA category?"* Answered
  from the developer-forum entry in `## Public documentation entry points`,
  never internal NVIDIA channels (see ground rule above).

If the question fits a different shape (how to write code, how to set up an
env, how to debug a crash), route to the matching skill instead — see
[`AGENTS.md`](../../../AGENTS.md) for the routing table.

## Library- and module-specific guides

Each DOCA library has its own subtree under `/doca/sdk/`. Use the matching
guide once the user's question is narrow enough to be about a single library.

> **First — the umbrella.** The DOCA SDK ships **dozens** of libraries; the
> table below names the ones agents most often need to route directly. If the
> user's library is not listed, the canonical first stop is the
> [**DOCA Libraries** umbrella page](https://docs.nvidia.com/doca/sdk/DOCA-Libraries/index.html),
> which lists every public library with its quality level (GA / Beta / Alpha)
> and links to its programming guide. Always consult the umbrella before
> telling a user a library does not exist or before guessing a URL. The
> umbrella is also the right answer for *"what DOCA libraries are available
> for X?"*-style discovery questions.

| Library | Guide | Typical questions it answers |
| --- | --- | --- |
| **DOCA Libraries** (umbrella index) | <https://docs.nvidia.com/doca/sdk/DOCA-Libraries/index.html> | Canonical list of every public DOCA library with its quality level. Always check here first when the user's library is not in the table below. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| DOCA Core (umbrella) | <https://docs.nvidia.com/doca/sdk/DOCA-Core/index.html> | The shared object-model every DOCA library is built on: `doca_dev`, `doca_devinfo`, `doca_pe` (progress engine), `doca_buf` / `doca_mmap`, `doca_ctx` lifecycle, the cross-library `DOCA_ERROR_*` taxonomy. Read this whenever the user is touching more than one library or asking *"how does DOCA in general work?"*. |
| DOCA Common | <https://docs.nvidia.com/doca/sdk/DOCA-Common/index.html> | The base utility library every DOCA program links against (`doca-common` `pkg-config` module). Pulled in transitively when you depend on any other DOCA library. |
| DOCA Flow | <https://docs.nvidia.com/doca/sdk/DOCA-Flow/index.html> | Port setup, device or representor selection, pipes, actions, actions memory, entry lifecycle. |
| DOCA Flow (incl. Connection Tracking) | <https://docs.nvidia.com/doca/sdk/DOCA-Flow/index.html> | Port setup, pipes, actions, action memory, entry lifecycle, validation, counters, traces. Folds Flow Connection Tracking (`doca_flow_ct.h`) as `## flow-ct` (connection-aware pipes, aging, NAT/SNAT/DNAT). Covered by the [`doca-flow`](../../libs/doca-flow/SKILL.md) skill. |
| DOCA Ethernet | <https://docs.nvidia.com/doca/sdk/DOCA-Ethernet/index.html> | RX/TX queues, packet I/O, `eth_rxq` / `eth_txq` lifecycle. Underpins the GPU Packet Processing app and most line-rate examples. Covered by the [`doca-eth`](../../libs/doca-eth/SKILL.md) skill. |
| DOCA RDMA | <https://docs.nvidia.com/doca/sdk/DOCA-RDMA/index.html> | DOCA's RDMA surface (send / recv / write / read patterns) on BlueField / ConnectX. Covered by the [`doca-rdma`](../../libs/doca-rdma/SKILL.md) skill. |
| DOCA Verbs | <https://docs.nvidia.com/doca/sdk/DOCA-Libraries/index.html> | Lower-level ibverbs-style API beneath DOCA RDMA / DOCA Eth, exposing raw QP / CQ / PD / MR / SRQ / AH primitives inside DOCA Core. Primary role is to route back to the higher-level library; takes the conversation only when the higher-level library does not expose the specific verb / opcode / WR flag the user needs. Covered by the [`doca-verbs`](../../libs/doca-verbs/SKILL.md) skill. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| DOCA DPA (incl. Comms + Verbs) | <https://docs.nvidia.com/doca/sdk/DOCA-DPA/index.html> | DPA host / device split-build, DPACC context, DPA annotation conventions. Folds DPA Comms (DPA-side communication primitives) as `## comms` and DPA Verbs (DPA-side verbs surface) as `## verbs`. Covered by the [`doca-dpa`](../../libs/doca-dpa/SKILL.md) skill. |
| DOCA Flow DPA Provider | <https://docs.nvidia.com/doca/sdk/DOCA-Libraries/index.html> | Bridges a `doca-flow` pipe into a DPA execution target so flow execution can run on the DPA processor instead of the host or DPU-CPU path. Covered by the [`doca-flow-dpa-provider`](../../libs/doca-flow-dpa-provider/SKILL.md) skill. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| DOCA GPUNetIO | <https://docs.nvidia.com/doca/sdk/DOCA-GPUNetIO/index.html> | GPU-initiated networking, CUDA + DOCA integration patterns. Covered by the [`doca-gpunetio`](../../libs/doca-gpunetio/SKILL.md) skill. |
| DOCA GPI | <https://docs.nvidia.com/doca/sdk/DOCA-Libraries/index.html> | GPU Programming Interface for kernel-launched RDMA operations directly from a CUDA thread. Distinct runtime surface from GPUNetIO; pairs with `doca-rdma` and `doca-gpunetio`. Covered by the [`doca-gpi`](../../libs/doca-gpi/SKILL.md) skill. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| DOCA Comch (formerly Comm Channel) | <https://docs.nvidia.com/doca/sdk/DOCA-Comch/index.html> | Host ↔ DPU control-plane messaging. **Library was renamed in DOCA 2.5**: the URL slug is `DOCA-Comch`, not `doca-comm-channel`. The `pkg-config` module on installed systems is `doca-comch`. Covered by the [`doca-comch`](../../libs/doca-comch/SKILL.md) skill. |
| DOCA Telemetry | <https://docs.nvidia.com/doca/sdk/DOCA-Telemetry/index.html> | DOCA's telemetry collection surface — schemas, sampling, integration with the DOCA Telemetry Service (DTS). Covered by the [`doca-telemetry`](../../libs/doca-telemetry/SKILL.md) skill. |
| DOCA Telemetry Exporter | <https://docs.nvidia.com/doca/sdk/DOCA-Telemetry-Exporter/index.html> | Application-side library used to *publish* telemetry from a DOCA program (distinct from `DOCA Telemetry`, which is the collection / consumption surface). Covered by the [`doca-telemetry-exporter`](../../libs/doca-telemetry-exporter/SKILL.md) skill. |
| DOCA DMA | <https://docs.nvidia.com/doca/sdk/DOCA-DMA/index.html> | Host ↔ DPU memory copy via the BlueField DMA engine. The DMA Copy reference application is the canonical example. Covered by the [`doca-dma`](../../libs/doca-dma/SKILL.md) skill. |
| DOCA Compress | <https://docs.nvidia.com/doca/sdk/DOCA-Compress/index.html> | Hardware-accelerated compression / decompression. Pairs with the File Compression reference application. Covered by the [`doca-compress`](../../libs/doca-compress/SKILL.md) skill. |
| DOCA AES-GCM | <https://docs.nvidia.com/doca/sdk/DOCA-AES-GCM/index.html> | Hardware-accelerated AES-GCM encryption / decryption. Member of the DOCA Crypto Acceleration family. Covered by the [`doca-aes-gcm`](../../libs/doca-aes-gcm/SKILL.md) skill. |
| DOCA SHA | <https://docs.nvidia.com/doca/sdk/DOCA-SHA/index.html> | Hardware-accelerated SHA hashing. Pairs with the File Integrity reference application. Covered by the [`doca-sha`](../../libs/doca-sha/SKILL.md) skill. |
| DOCA Erasure Coding | <https://docs.nvidia.com/doca/sdk/DOCA-Erasure-Coding/index.html> | Hardware-accelerated erasure coding (RS / similar). Used in storage workloads. Covered by the [`doca-erasure-coding`](../../libs/doca-erasure-coding/SKILL.md) skill. |
| DOCA App Shield (library) | <https://docs.nvidia.com/doca/sdk/DOCA-App-Shield/index.html> | Process-introspection primitives the App Shield Agent application is built on. **Not covered by this bundle** — DOCA App Shield is policy-excluded from this public release (see [AGENTS.md `## Non-goals`](../../../AGENTS.md#non-goals-questions-the-agent-should-recognize-and-refuse-politely) item 7). Route the user to the public docs above. |
| DOCA PCC (library) | <https://docs.nvidia.com/doca/sdk/DOCA-PCC/index.html> | Programmable congestion control library (DPA-hosted). Distinct from the PCC reference application and the `pcc_counters` (`pcc_counters.sh`) tool. Covered by the [`doca-pcc`](../../libs/doca-pcc/SKILL.md) skill. |
| DOCA PCC ZTR-RTTCC Algorithm | <https://docs.nvidia.com/doca/sdk/DOCA-Libraries/index.html> | The shipped reference Zero-Touch-RTT Congestion-Control algorithm that runs under `doca-pcc`. Pairs with `doca-pcc` (host) and `doca-pcc-counters` (observability). Covered by the [`doca-pcc-ztr-rttcc-algo`](../../libs/doca-pcc-ztr-rttcc-algo/SKILL.md) skill. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| DOCA UROM (library) | <https://docs.nvidia.com/doca/sdk/DOCA-UROM/index.html> | Unified Communication Remote Memory Operations library. Distinct from the DOCA UROM Service. Covered by the [`doca-urom`](../../libs/doca-urom/SKILL.md) skill. |
| DOCA Arg Parser | <https://docs.nvidia.com/doca/sdk/DOCA-Arg-Parser/index.html> | Argument parser used by every shipped DOCA sample and reference application. Worth knowing when the user adapts a sample's CLI surface. Covered by the [`doca-argp`](../../libs/doca-argp/SKILL.md) skill. |
| DOCA Device Emulation (umbrella) | <https://docs.nvidia.com/doca/sdk/DOCA-Device-Emulation/index.html> | Umbrella for the device-emulation libraries (PCI Generic, virtio, virtio-fs). Start here if the user is building emulated PCIe devices on BlueField. Covered by the [`doca-devemu`](../../libs/doca-devemu/SKILL.md) skill. |
| DOCA MGMT | <https://docs.nvidia.com/doca/sdk/DOCA-Libraries/index.html> | Programmatic management of DOCA device state (library-side). Pairs with `doca-dms` (service-side). Covered by the [`doca-mgmt`](../../libs/doca-mgmt/SKILL.md) skill. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| DOCA RDMI | <https://docs.nvidia.com/doca/sdk/DOCA-Libraries/index.html> | DOCA RDMA Initiator — accelerator-initiated (host or DPA-kernel) one-sided RDMA flow surface; pairs with `doca-rdma` (the general RDMA library) and `doca-dpa` / `doca-verbs` for the DPA-kernel-initiated path. Covered by the [`doca-rdmi`](../../libs/doca-rdmi/SKILL.md) skill. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| DOCA STA (Storage Target Acceleration) | <https://docs.nvidia.com/doca/sdk/DOCA-STA/index.html> | Target-side storage acceleration library — accelerates the NVMe-oF target data path over RDMA on BlueField / ConnectX. The [DOCA Storage Applications index](https://docs.nvidia.com/doca/sdk/DOCA-Storage-Applications/index.html) is the docs page that collects the storage-focused reference applications (Comch-to-RDMA zero-copy, GGA offload, SBC generator, initiator, target) built on it — it is a docs index, not a separate library. Start there for *"how do I move storage I/O across the BlueField?"* before drilling into a specific app guide. Covered by the [`doca-sta`](../../libs/doca-sta/SKILL.md) skill. |
| DOCA Rivermax | <https://docs.nvidia.com/doca/sdk/DOCA-Rivermax/index.html> | DOCA's Rivermax integration (media / streaming workloads). Covered by the [`doca-rmax`](../../libs/doca-rmax/SKILL.md) skill. |
| DOCA DPDK Bridge (API-only) | <https://docs.nvidia.com/doca/api/3.1.0/doca-libraries-api/modules.html#group__DOCA__DPDK__BRIDGE> | The interop layer that lets an existing DPDK application reach DOCA libraries (most commonly DOCA Flow) without rewriting its data-plane. **No standalone SDK doc page today** — documented as an API-reference module under DOCA Libraries API. Covered by the [`doca-dpdk-bridge`](../../libs/doca-dpdk-bridge/SKILL.md) skill. |
| DOCA Reference Applications | <https://docs.nvidia.com/doca/sdk/DOCA-Reference-Applications/index.html> | The shipped reference applications (PCC, DPI, IPsec gateway, file-compression, etc.) — what each one does, where its source lives under `/opt/mellanox/doca/applications/`, and how to recompile with `meson` + `ninja`. |

There is **no current single "DOCA Samples Overview" page** in the v3.x docs.
Samples are documented per-library inside each library's programming guide
(see "Sample" sections inside the URLs above) and ship on disk under
`/opt/mellanox/doca/samples/doca_<library>/`. Earlier (v1.x / v2.x) docs did
have a single overview page — those URLs are now archived and will return 404
on `docs.nvidia.com/doca/sdk/`. Do not link the archived page; route the user
to the per-library "Sample" sections plus the on-disk samples directory.

If the user asks about a DOCA library that is **not** in the table above, do
**not** guess the URL. Open the
[**DOCA Libraries** umbrella page](https://docs.nvidia.com/doca/sdk/DOCA-Libraries/index.html)
first to confirm the library exists and to find its canonical guide URL.
Only after that, fall back to the user's installed sample directory (see
"Layout of an installed DOCA package" below).

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
| DOCA Management Service (DMS) | <https://docs.nvidia.com/doca/sdk/DOCA-Management-Service-Guide/index.html> | Centralized configuration / operation of BlueField and ConnectX devices via gRPC (gNMI for config, gNOI for system ops). Covered by the [`doca-dms`](../../services/doca-dms/SKILL.md) skill. |
| DOCA Firefly Service | <https://docs.nvidia.com/doca/sdk/DOCA-Firefly-Service-Guide/index.html> | PTP / time synchronization service. Covered by the [`doca-firefly`](../../services/doca-firefly/SKILL.md) skill. |
| DOCA Flow Inspector Service | <https://docs.nvidia.com/doca/sdk/DOCA-Flow-Inspector-Service-Guide/index.html> | Mirrored-flow inspection service. **Not covered by this bundle** — policy-excluded from this public release (see [AGENTS.md `## Non-goals`](../../../AGENTS.md#non-goals-questions-the-agent-should-recognize-and-refuse-politely) item 7). Route the user to the public docs above. |
| DOCA UROM Service | <https://docs.nvidia.com/doca/sdk/DOCA-UROM-Service-Guide/index.html> | Unified Communication Remote Memory Operations service. Covered by the [`doca-urom-svc`](../../services/doca-urom-svc/SKILL.md) skill. |
| DOCA Argus Service | <https://docs.nvidia.com/doca/sdk/DOCA-Argus-Service-Guide/index.html> | DOCA's runtime-security / monitoring service for BlueField. Covered by the [`doca-argus`](../../services/doca-argus/SKILL.md) skill. |
| DOCA OS Inspector Service | <https://docs.nvidia.com/doca/sdk/DOCA-Services/index.html> | DPU-side out-of-band host-OS introspection container (App-Shield-as-a-service). **Not covered by this bundle** — policy-excluded from this public release (see [AGENTS.md `## Non-goals`](../../../AGENTS.md#non-goals-questions-the-agent-should-recognize-and-refuse-politely) item 7). Route the user to the umbrella docs above. |

> **Non-goals.** Externally-productized NVIDIA services that are NOT in
> `doca/services/` at the bundle's currently-aligned DOCA release —
> DOCA Telemetry Service (DTS) as-deployed, BlueMan, HBN, SNAP,
> Virtio-net — are intentionally out of scope for this bundle, which
> is strictly 1:1 with `doca/services/`. See
> [AGENTS.md `## Non-goals`](../../../AGENTS.md#non-goals-questions-the-agent-should-recognize-and-refuse-politely) for the policy
> rationale. If a user asks about one of these external services, use
> the **[`## Externally-productized DOCA software — not in this bundle, but here is where to route`](#externally-productized-doca-software-not-in-this-bundle-but-here-is-where-to-route)**
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
  [`doca-container-deployment`](../../doca-container-deployment/SKILL.md)
  skill — every in-bundle per-service skill above hands off to it
  for the kubelet-standalone static-pod manifests directory, image
  pull, and pod-spec drop pattern.
- The **bare-metal hardware deployment** path (a DOCA-linked
  application binary launched directly on host x86 or BlueField
  Arm bare-metal — no container) is covered by the
  [`doca-bare-metal-deployment`](../../doca-bare-metal-deployment/SKILL.md)
  skill. It owns the launch contract (direct / tmux / systemd),
  hardware-resource binding (PF/VF/representor + NUMA + CPU pinning
  + IRQ affinity), per-tenant isolation primitives, the bare-metal
  error taxonomy, and the restart-loop-is-HIGH-STAKES rule. The
  authoritative public references are the **DOCA Programming
  Guide** (build, link, runtime preconditions:
  <https://docs.nvidia.com/doca/sdk/DOCA-Programming-Guide/index.html>),
  the **DPU / BlueField Platform Software Manual** (PCIe topology,
  devlink, representor naming, `mlxconfig` device introspection:
  <https://docs.nvidia.com/networking/display/bluefieldbsp4132>), and
  the per-library guides listed below in this map.
- The **front-door routing decision** between the two — *"which
  path applies to my workload?"* — is owned by
  [`doca-setup ## recognize`](../../doca-setup/TASKS.md#recognize),
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
  RShim, TMFIFO `192.168.100.x`, DPU vs separated-host mode via
  `mlxconfig`, post-BFB recovery) is covered by
  [`doca-bf3-deployment`](../../doca-bf3-deployment/SKILL.md). Public
  reference: the DPU / BlueField Platform Software Manual
  (<https://docs.nvidia.com/networking/display/bluefieldbsp4132>).
- **BlueField-4 (BF4)** — the BMC-driven path (the three OS-ISO install
  methods UEFI-HTTP / PXE / Redfish Virtual Media, the PLDM firmware-update
  flow, and the Grace Ubuntu + cloud-init install) is covered by
  [`doca-bf4-deployment`](../../doca-bf4-deployment/SKILL.md). Public
  references: the BlueField / DOCA docs on docs.nvidia.com; deep BSP/BFB
  internals stay in the BSP row below. Any mutating firmware/power-cycle
  step routes to [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md).

**CollectX telemetry collection.** Deploying/operating a CollectX (clx)
telemetry-collection pipeline (providers/counters → schema → collector
daemon → exporters: Prometheus / Fluent Bit / NetFlow / file-IPC) is
covered by [`doca-collectx-deployment`](../../doca-collectx-deployment/SKILL.md);
the library reader/publisher APIs stay in
[`doca-telemetry`](../../libs/doca-telemetry/SKILL.md) /
[`doca-telemetry-exporter`](../../libs/doca-telemetry-exporter/SKILL.md), and
the productized DTS container is routed out as externally-productized.

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
>   [`doca-container-deployment`](../../doca-container-deployment/SKILL.md)
>   for the kubelet-standalone static-pod path, or
>   [`doca-bare-metal-deployment`](../../doca-bare-metal-deployment/SKILL.md)
>   for a directly-launched DOCA binary. For an **externally-productized**
>   service (HBN, BlueMan, SNAP, Virtio-net, DTS-as-deployed — *not* in
>   this bundle), there is no in-tree skill: route to its guide via the
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
>     <https://docs.nvidia.com/networking/display/kubernetes2611/index.html>.
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

## DOCA tools

DOCA ships a set of *tools* — small CLIs installed under
`/opt/mellanox/doca/tools/` on a real install, each documented on its
own public page. Per-tool skills (where they exist) live under
`skills/tools/<tool>`.

> **First — the umbrella.** When the user's tool is not listed below, the
> canonical first stop is the
> [**DOCA Tools** umbrella page](https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html),
> which lists every public DOCA tool and links to its tool guide.

| Tool | Guide | What it does |
| --- | --- | --- |
| **DOCA Tools** (umbrella index) | <https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html> | Canonical list of every public DOCA tool with its purpose and guide link. Always check here first when a tool is not in the table below. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| Capabilities Print Tool (`doca_caps`) | <https://docs.nvidia.com/doca/sdk/DOCA-Capabilities-Print-Tool/index.html> | Prints DOCA devices and the per-library capabilities they support. Side-effect-free; safe to call early. Covered by the [`doca-caps`](../../tools/doca-caps/SKILL.md) skill. |
| DOCA Bench | <https://docs.nvidia.com/doca/sdk/DOCA-Bench/index.html> | Performance evaluation harness for the built-in workload modes. Covered by the [`doca-bench`](../../tools/doca-bench/SKILL.md) skill. |
| DOCA Bench Extension | <https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html> | In-tree extension / plug-in framework that lets `doca-bench` measure workload classes its built-in modes do not cover. Reference exemplar `doca_bench_cuda` drives GPUNetIO RX / TX kernels. Covered by the [`doca-bench-extension`](../../tools/doca-bench-extension/SKILL.md) skill. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| Comm Channel Admin Tool | <https://docs.nvidia.com/doca/sdk/DOCA-Comm-Channel-Admin-Tool/index.html> | Admin CLI for Comch channels. Covered by the [`doca-comm-channel-admin`](../../tools/doca-comm-channel-admin/SKILL.md) skill. |
| Flow Tune | <https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html> | Unified visibility / analysis / recommendation tool for live `doca-flow` pipelines. The artifact is ONE binary with TWO internal roles (server role: snapshots and exposes pipe / counter / KPI state through local IPC; client / consumer role: dumps, analyzes, visualizes, recommends parameter changes). The historical "Flow Tune Tool" and "Flow Tune Server" split lives INSIDE this artifact — there is one skill: [`doca-flow-tune`](../../tools/doca-flow-tune/SKILL.md). **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| Flow Perf | <https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html> | Host / DPU-CPU control-plane rule-rate measurement (install / delete / query rate). Distinct from `Flow DPA Perf` (DPA-offloaded path) and `Flow Tune` (optimizes a deployed pipeline). Covered by the [`doca-flow-perf`](../../tools/doca-flow-perf/SKILL.md) skill. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| Flow DPA Perf | <https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html> | Flow performance measurement for the DPA-offloaded execution path. Covered by the [`doca-flow-dpa-perf`](../../tools/doca-flow-dpa-perf/SKILL.md) skill. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| Flow gRPC Server | <https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html> | gRPC remote-control server for `doca-flow` rule programming. Covered by the [`doca-flow-grpc-server`](../../tools/doca-flow-grpc-server/SKILL.md) skill. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| PCC Counter | <https://docs.nvidia.com/doca/sdk/DOCA-PCC-Counter-Tool/index.html> | PCC counter inspection. Covered by the [`doca-pcc-counters`](../../tools/doca-pcc-counters/SKILL.md) skill. |
| Socket Relay | <https://docs.nvidia.com/doca/sdk/DOCA-Socket-Relay/index.html> | Socket relay between host and DPU. Covered by the [`doca-socket-relay`](../../tools/doca-socket-relay/SKILL.md) skill. |
| App Shield Config | <https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html> | Generates the host-OS profile / symbol files that App Shield needs to interpret host kernel state. **Not covered by this bundle** — the App Shield family is policy-excluded from this public release (see [AGENTS.md `## Non-goals`](../../../AGENTS.md#non-goals-questions-the-agent-should-recognize-and-refuse-politely) item 7). Route the user to the umbrella docs above. |
| DPA High-Level Tracer | <https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html> | Captures DPA-side execution traces with higher-level events than raw cycle counts. Covered by the [`doca-dpa-hl-tracer`](../../tools/doca-dpa-hl-tracer/SKILL.md) skill. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| GPUNetIO ib_write_bw | <https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html> | RDMA-write bandwidth benchmark from the GPUNetIO framework. Covered by the [`doca-gpunetio-ib-write-bw`](../../tools/doca-gpunetio-ib-write-bw/SKILL.md) skill. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| GPUNetIO ib_write_lat | <https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html> | RDMA-write latency benchmark from the GPUNetIO framework. Covered by the [`doca-gpunetio-ib-write-lat`](../../tools/doca-gpunetio-ib-write-lat/SKILL.md) skill. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| SHA Offload Engine | <https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html> | OpenSSL ENGINE wrapping the DOCA SHA library; lets unmodified OpenSSL-based applications offload SHA without code changes. Covered by the [`doca-sha-offload-engine`](../../tools/doca-sha-offload-engine/SKILL.md) skill. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| SPCX-CC | <https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html> | Programmable Congestion-Control extension (next-gen). Pairs with `doca-pcc`, `doca-pcc-ztr-rttcc-algo`. Live-fabric safety implications: heavy use of `doca-hardware-safety`. Covered by the [`doca-spcx-cc`](../../tools/doca-spcx-cc/SKILL.md) skill. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| Telemetry Utils | <https://docs.nvidia.com/doca/sdk/DOCA-Telemetry-Utils/index.html> | Operator-side support CLI for a DOCA Telemetry exporter pipeline. Translates name ↔ Data ID, enumerates the diagnostic-counter schema, and probes per-device counter support before an exporter config commits to it. Covered by the [`doca-telemetry-utils`](../../tools/doca-telemetry-utils/SKILL.md) skill. |

> **Non-goals.** Externally-productized NVIDIA tools that are NOT in
> `doca/tools/` at the bundle's currently-aligned DOCA release —
> DOCA-DPACC-Compiler, DPA-Tools (DPA GDB Server / PS / Statistics),
> DOCA-DPU-CLI, DOCA-Ngauge, `doca-hugepages` helper — are
> intentionally out of scope for this bundle, which is strictly 1:1
> with `doca/tools/`. See
> [AGENTS.md `## Non-goals`](../../../AGENTS.md#non-goals-questions-the-agent-should-recognize-and-refuse-politely). If a user
> asks about one of these external tools, use the
> **[`## Externally-productized DOCA software — not in this bundle, but here is where to route`](#externally-productized-doca-software-not-in-this-bundle-but-here-is-where-to-route)**
> table below for the authoritative docs URL, per-tool common gotchas,
> and the Developer Forum entry point. Refusing without routing
> violates the bundle's own contract.

If the user asks about a DOCA tool that is not in this table, open the
[**DOCA Tools** umbrella page](https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html)
to discover it. Do not guess tool URLs.

## Externally-productized DOCA software — not in this bundle, but here is where to route

NVIDIA ships several DOCA-adjacent products **outside** the
`doca/{libs,services,tools}` monorepo this bundle is 1:1 with. They are
real, supported NVIDIA products — but because the bundle's strict
alignment contract (see [`SKILLS.md`](../../../SKILLS.md) and
[`AGENTS.md ## Non-goals`](../../../AGENTS.md#non-goals-questions-the-agent-should-recognize-and-refuse-politely)) forbids
synthesizing answers about them from training knowledge, the bundle's
job for these products is to **route, not answer**.

When the user asks about any product in the table below, the required
response shape is the three-part contract spelled out in
[`AGENTS.md ## Non-goals` #7](../../../AGENTS.md#non-goals-questions-the-agent-should-recognize-and-refuse-politely):

1. **Recognize the class** — name the product and state it is out of
   scope for this bundle, because it is not in the monorepo's
   `doca/{services,tools}/` at the currently-aligned DOCA release.
2. **Name the boundary honestly** — the bundle is strictly 1:1 with
   the monorepo; products productized outside it (e.g. shipped as
   separate NGC containers, separate packages, or separate BSP
   utilities) are intentionally excluded.
3. **Route with substance** — provide the authoritative public URL
   for that specific product (most often under `docs.nvidia.com/doca/sdk/`
   but, for BSP / BMC / k8s / fabric / firmware-tools / Cumulus /
   Spectrum-X / Rivermax-license-layer products, under
   `docs.nvidia.com/networking/`, `docs.nvidia.com/networking-ethernet-software/`,
   `docs.nvidia.com/datacenter/`, or `network.nvidia.com/products/`), name
   the most common gotcha class the user's symptom (if any) is likely
   hitting, and give the Developer Forum entry point as the
   escalation channel.

A refusal that names only (1) and (2) but skips (3) violates the
bundle's contract. Use this table.

> **Hardware-safety overlay still applies.** If the user's question
> about an externally-productized product touches live BlueField /
> NIC hardware state (`mlxconfig` writes, firmware burn, BFB reflash,
> SFC mode flip during HBN setup), load
> [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) alongside
> this routing step. The route covers *where to read*; safety covers
> *how to change live state* and is not waived by a non-goal refusal.

| Product | Authoritative public docs | One-line WHAT it is | Common-gotcha classes worth surfacing | Forum search hint |
| --- | --- | --- | --- | --- |
| **DOCA DPL Service** (Pipeline Language; in the public DOCA Services catalog — routed here, no deep skill authored in this bundle version) | <https://docs.nvidia.com/doca/sdk/DOCA-Pipeline-Language-Services-Guide/index.html> | The DOCA Pipeline Language (DPL) — a P4-16-*derived* domain-specific language plus a services framework for programming the BlueField packet-processing pipeline. Shipped as **two separately-deployed containers**: the **DPL Development Container** (the `p4c`-based compiler + tools) and the **DPL Runtime Service** (the BlueField backend that programs the DPU datapath via a **P4Runtime-compliant server on TCP 9559**). **Beta** as of DOCA 3.3. | "My standard P4 program behaves differently on BlueField" → DPL's *syntax* is derived from P4-16 but its **pipeline semantics target NVIDIA's DPU architecture (dRMT-style), not the standard P4 RMT model** — do not assume upstream-P4 execution behavior. "Where is the DPL C API / library?" → there isn't one; DPL is a **language + services** model, not an SDK/driver — you compile a DPL program and load it through the Runtime Service, you do not link a `libdpl`. "Runtime can't load my program" → the Development Container (compile) and Runtime Service (load/run) are **deployed separately**; a program compiled in one must be delivered to the other, and the P4Runtime client must reach TCP 9559 on the BF. | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "DPL" or "Pipeline Language" or "P4" |
| **OVS-DOCA** (ASAP² Open vSwitch offload; part of the `doca-networking` DOCA-Host profile — routed here, no deep skill authored in this bundle version) | <https://docs.nvidia.com/doca/sdk/OVS-DOCA-Hardware-Acceleration/index.html> | The DOCA-Flow-backed datapath-offload interface (DPIF) for Open vSwitch — a third OVS data-path alongside **OVS-Kernel** and **OVS-DPDK**. It uses NVIDIA **ASAP²** to offload the OVS data-plane into the NIC/DPU **eSwitch** via the **DOCA Flow** library while keeping the OVS control-plane (OpenFlow / `ovs-vsctl`) unmodified; it delivers the richest offload feature set of the three modes. Ships in the **`doca-networking`** install profile of DOCA-Host (it is **not** a container service). | "I enabled hw-offload but nothing is offloaded" → OVS-DOCA must be turned on explicitly: `ovs-vsctl --no-wait set Open_vSwitch . other_config:doca-init=true` **and** `other_config:hw-offload=true`, then **restart `openvswitch`** for the change to take effect. "Bridge won't offload" → the bridge must be created with **`datapath_type=netdev`** and the NIC must be in **switchdev** mode. "Connection-tracking sizing / IPv6 CT" → tune `other_config:hw-offload-ct-size` and `other_config:hw-offload-ct-ipv6-enabled`. Do not confuse OVS-DOCA with OVS-Kernel/OVS-DPDK — they share CLI but differ in offload path. | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "OVS-DOCA" or "ASAP2" or "hw-offload" |
| **DOCA SNAP Services** (umbrella: SNAP-4, SNAP-3, SNAP Virtio-fs) | <https://docs.nvidia.com/doca/sdk/DOCA-SNAP-Services/index.html> | Hardware-accelerated storage virtualization on BlueField — emulates local PCIe block/file devices to the host while forwarding I/O over a fabric. SNAP-4 = NVMe + virtio-block on BF-3; SNAP-3 = same on BF-2; SNAP Virtio-fs = file-system emulation on BF-3. | "Host can't enumerate the NVMe device" or "device shows up but won't boot" almost always traces to: (a) NIC firmware not configured for SNAP / virtio (need the protocol-specific `mlxconfig` keys per the *Firmware Configuration* section of the SNAP-4 / Virtio-fs guides; common keys: `INTERNAL_CPU_MODEL=1`, `PCI_SWITCH_EMULATION_ENABLE`, plus PF/VF hotplug keys); (b) SNAP container not running on BF (`crictl ps -a \| grep snap`); (c) no fabric target reachable from BF management VRF; (d) BF BSP / SNAP version mismatched against host DOCA release. SNAP Virtio-fs is **beta** as of DOCA 3.3. | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "SNAP" |
| **DOCA HBN Service** (Host-Based Networking) | <https://docs.nvidia.com/doca/sdk/DOCA-HBN-Service-Guide/index.html> | A "bump-in-the-wire" service that turns the BlueField into a BGP/EVPN L3 router for the host side of the network. Linux routing/bridging is accelerated into hardware tables by the `nl2docad` (Netlink-to-DOCA) daemon inside the HBN container. | "HBN routing not working" or "HBN container won't start" almost always traces to: (a) **Service Function Chaining (SFC) not enabled at BFB-install time** — HBN is bump-in-the-wire and requires SFC; you generally must reflash the BFB with SFC enabled or pass the right `bf-cfg.cfg` (see *HBN Service Requirements* and *Deploying BlueField DOCA with SFC*); (b) `br-hbn` OVS bridge missing on BF (auto-created when BFB is installed with HBN enabled — its absence is a tell that step (a) didn't happen); (c) FRR / `nl2docad` not running inside the HBN container (`docker exec` to inspect, or `crictl logs`); (d) NGC YAML config not applied / image not pulled. | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "HBN" |
| **DOCA BlueMan Service** (web dashboard for DPU health) | <https://docs.nvidia.com/doca/sdk/DOCA-BlueMan-Service-Guide/index.html> | A standalone web dashboard hosted on the BF that consolidates basic info, health, and telemetry counters. All data is pulled from the on-BF DOCA Telemetry Service (DTS). Default install path: `/opt/mellanox/doca/services/blueman/`. | "BlueMan UI shows red" or "BlueMan page is blank" almost always traces to: (a) **DTS not running on the BF** (BlueMan has no data source of its own; verify DTS pod with `crictl ps -a \| grep telemetry`); (b) the DOCA Privileged Executer (DPE) daemon not running (`systemctl status dpe`); (c) BFB image too old (BlueMan needs BFB ≥ 3.9.3.1); (d) accessing the UI from a host that isn't on the same network as the DPU OOB interface (BlueMan binds to the DPU OOB IP by default; for DPF deployments, you typically `iptables -t nat -A PREROUTING` to expose ports 443/10000). | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "BlueMan" |
| **DOCA Virtio-net Service** (BF-3 only) | <https://docs.nvidia.com/doca/sdk/DOCA-Virtio-net-Service-Guide/index.html> | A BF-3 service that exposes virtio-net PCIe devices to the host (PFs, hotplug PFs, SR-IOV VFs). Both data-plane and control-plane are offloaded to the BF; the host sees standard virtio-net devices with no QEMU / guest-driver dependency. Driven by the `virtio-net-controller` systemd service on the BF; configured via `mlxconfig` + optional JSON at `/opt/mellanox/mlnx_virtnet/virtnet.conf`. | "Host doesn't see the virtio-net device" or "`virtnet list` returns nothing" almost always traces to: (a) `VIRTIO_NET_EMULATION_ENABLE=1` not set in `mlxconfig` (plus the hotplug-PF / SR-IOV-VF keys per the *Configuring NIC Firmware* section of the guide); (b) `virtio-net-controller` systemd service not running on the BF (`systemctl status virtio-net-controller`); (c) `/opt/mellanox/mlnx_virtnet/virtnet.conf` malformed JSON (validate with `jq` first); (d) the *user is on BF-2* — Virtio-net Service is BF-3-only as of DOCA 3.x; for BF-2 the legacy `virtio-net-controller` path under Emulated Devices is the read; (e) `mlxconfig` change applied but BF not rebooted. | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "Virtio-net" or "virtnet" |
| **DOCA Telemetry Service (DTS) — as-deployed** (distinct from the `doca-telemetry` library skill in this bundle) | <https://docs.nvidia.com/doca/sdk/DOCA-Telemetry-Service-Guide/index.html> | DTS is shipped both as a built-in BF service (auto-started via `/etc/kubelet.d/doca_telemetry_standalone.yaml`) and as a container on NGC for hosts. It aggregates metrics from built-in providers (`sysfs`, `ethtool`, `tc`) and from external apps using the `doca-telemetry-exporter` library, and exports via Prometheus (pull) or Fluent Bit (push). Config lives at `/opt/mellanox/doca/services/telemetry/config/dts_config.ini` plus `fluent_bit_configs/`. | "DTS is up but Prometheus / Grafana shows nothing" almost always traces to: (a) provider not enabled in `dts_config.ini` (each provider has its own `enable=true` knob); (b) Prometheus exporter not enabled in `dts_config.ini` (it is off by default); (c) on BF, the BlueField storage write-protect means DTS data write is disabled by default (this is intentional); (d) FluentBit config in `fluent_bit_configs/` is missing the right input → filter → output chain. **Do not confuse DTS-as-deployed with the in-tree [`doca-telemetry`](../../libs/doca-telemetry/SKILL.md) library**; the library is the publisher-side API, the service is the collector/aggregator daemon. | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "DOCA Telemetry" or "DTS" |
| **DOCA DPACC Compiler** (`dpacc` + DPA toolchain) | <https://docs.nvidia.com/doca/sdk/DOCA-DPACC-Compiler/index.html> | High-level compiler for the BlueField DPA processor. Compiles C source targeted at the DPA into a DPA-executable + host library ("DPA program"), which the host application then links. Uses `dpa-clang` (LLVM-based cross-compiler) plus `dpa-ar`, `dpa-nm`, `dpa-objdump`. Installs alongside the DOCA DPA package. | "GDB can't read DPA debug info" almost always traces to: DPACC default debug standard is DWARFv5; pre-10.1 GDBs need `--devicecc-options="-gdwarf-4"`. "Where is my DPA ELF inside the host binary?" → use `dpacc-extract` from the DPACC package. "How do I link DPA code into my host app?" → use compile-and-link mode (single `dpacc` invocation with all sources + mandatory `--output-libname`). | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "DPACC" or "DPA" |
| **DPA Tools** (`dpa-gdbserver`, `dpa-ps`, `dpa-statistics`, `dpa-eu-mgmt-tool`) | <https://docs.nvidia.com/doca/sdk/DOCA-DPA-GDB-Server-Tool/index.html> (umbrella PDF: <https://docs.nvidia.com/doca/sdk/nvidia-dpa-tools.pdf>) | The DPA-side debug / introspection toolkit. `dpa-gdbserver` provides remote-GDB debugging of FlexIO DEV programs (default TCP port 1981, default EU 29); `dpa-ps` lists DPA processes; `dpa-statistics` dumps DPA counters; `dpa-eu-mgmt-tool` manages DPA Execution Units. As of DOCA 3.x the GDB-Server tool is **beta**. | "GDB can't connect to dpa-gdbserver" → check TCP 1981 is reachable from the GDB host AND not occupied by another `dpa-gdbserver`; multiple instances need different ports + EUs (default EU 29 conflicts the same way). "GDB attaches but symbols are mangled" → use `gdb-multiarch` 9.2 or RISC-V GDB ≥ 12.1, and provide source paths with GDB's `directory` command if the GDB host is not the build host. `dpa-ps` and `dpa-statistics` are read-only and safe to run live. | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "DPA GDB" or "dpa-gdbserver" |
| **DOCA DPU CLI** | <https://docs.nvidia.com/doca/sdk/DOCA-DPU-CLI/index.html> | A single-page cheat-sheet of useful one-liners for the BlueField environment — hugepage setup, `mlxconfig` flips, `devlink` queries, BSP utilities. Not a tool you "install"; it is a curated reference of commands that already exist on a properly-installed BF. | The cheat-sheet is **version-pinned to the SDK release that publishes it** — always verify the specific command on the actual installed BSP / DOCA version before suggesting it (some commands change syntax across BSP majors). For DPDK hugepages specifically, the cheat-sheet shows the raw sysfs writes (`echo '1024' \| sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages` + `mount -t hugetlbfs`); on BlueField Ubuntu 22.04, prefer the `doca-hugepages` helper (next row) for managed allocation. | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "DPU CLI" or "BlueField CLI" |
| **DOCA Ngauge** (`ngauge`) | <https://docs.nvidia.com/doca/archive/2-9-0/DOCA+Ngauge/index.html> | A NIC hardware-counter probing tool that stores measurements + metadata in HDF5 (`.h5`) format and prints live progress bars on the CLI. Driven by a single YAML config (output path + counter list are mandatory). Comes with a `simple_plot.py` plugin under `/usr/share/doc/ngauge/examples/plugins/`. | "`apt install ngauge` failed" → on a BlueField DPU, the package is `ngauge-dpu`, not `ngauge` (on x86 / arm64 hosts it is `ngauge`). "Empty `.h5` output" → the YAML's `id` field for each counter is the only mandatory counter field; the other fields default if omitted. "Plot won't render" → use `simple_text_plot.py` for SSH-only / no-X11 sessions. **Currently only an archived doc page exists (DOCA 2.9.0); the SDK index does not host a v3.x page yet — that's a known doc-side gap, not a tool gap.** | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "Ngauge" |
| **`doca-hugepages` helper** (`/usr/sbin/doca-hugepages`) | <https://docs.nvidia.com/doca/archive/3-2-2/DOCA-doca-hugepages-Tool/index.html> | BSP-side managed-allocation CLI for hugepages on BlueField — `config` / `reload` / `show` / `remove` subcommands manage a database of per-application hugepage requests and apply them on driver load. Per-application config drops into `/etc/mlnx.d/`. **Supported only on Ubuntu 22.04 on BlueField.** | "I set hugepages via `doca-hugepages config` but nothing changed" → `config` only updates the database; you must `doca-hugepages reload` to actually allocate. "Hugepages are inconsistent / fighting" → do NOT mix `doca-hugepages` with raw `/sys/kernel/mm/hugepages/...` writes or `/etc/default/grub` `hugepages=N` edits; pick one ownership model. Recommended pattern for boot-time pre-allocation: per-app config file with `"is_active": "inactive"` in `/etc/mlnx.d/`, let `doca-hugepages` activate it during driver load. **Currently the `docs.nvidia.com/doca/sdk/` SDK index does not have a v3.x page — the linked URL is the 3.2.2 LTS archive, which is the most-current public doc as of this writing.** | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "doca-hugepages" or "BlueField hugepages" |
| **BlueField BSP / BFB / `bfb-install` / RShim / TMFIFO / `bf.cfg`** | <https://docs.nvidia.com/networking/display/bluefieldbsp4132> | The BlueField platform software bundle (BFB image), the host-side `bfb-install` push tool, the `rshim` driver / daemon that exposes the BF-side `Linux up` console + the TMFIFO `192.168.100.2` recovery interface, and the `bf.cfg` install-time configuration file. This is the layer **below** every DOCA service / library / tool — without a healthy BSP, nothing else works. | "`bfb-install` exited 0 but DPU is dead" → the installer's exit code is **not** authoritative; always parse the actual console output for any `[MISC]` / `[ERR]` line whose text contains a failure verb ("failed", "error", "abort") and verify `Linux up` / `DPU is ready` markers appeared. The most common field-reported partial-failure signature is *"Ubuntu installation completed"* (or *"Ubuntu installation finished"*) followed by *"INFO[MISC]: NIC firmware update failed"* with `exit 0` — that means the OS image landed but the FW-update sub-step silently failed; bisect with `flint -d <bdf> q` (NIC FW stage) vs RShim console reads (OS-bring-up stage). "DPU never reaches `Linux up`" → after the install reports done, do a **real cold power cycle** (full host power-off, wait 30 s, power-on) before reflashing — RShim soft reset (`echo SW_RESET 1 > /dev/rshim0/misc`) does not reinitialize the BF PCIe complex and is often not enough. "TMFIFO `192.168.100.2` unreachable" → restart `rshim.service` on the host, verify `lspci | grep -i bluefield` still enumerates, then re-attempt; if TMFIFO stays dead the BFB image likely failed mid-install and the BF needs reflash via PXE / external recovery. "Should I use `bf.cfg` or post-install config?" → use `bf.cfg` for anything that must be present at first boot (DPU mode, SFC for HBN, BSP-time hugepage pre-allocation); post-install is for runtime knobs. **BFB version must match the host DOCA release per the Release Notes — never mix.** | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "bfb-install" or "RShim" or "BFB install failed" |
| **DOCA Platform Framework (DPF)** | <https://docs.nvidia.com/networking/display/dpf25101> (source on GitHub: <https://github.com/NVIDIA/doca-platform>) | DPU lifecycle and orchestration at Kubernetes scale: provisions BlueField-3 DPUs, deploys DPUServices (HBN, BlueMan, SNAP, etc.) declaratively, manages BFB rollouts across racks of DPUs. The K8s-native answer to "how do I run DOCA at fleet scale", distinct from single-host `doca-bare-metal-deployment` workflows. **Supports dual-port BF-3 DPUs.** | "DPF provisioning hangs at BFB-install step" → first check the DPF-side operator logs for the specific DPU's `DPUCluster` / `DPUSet` CR status; the underlying failure is almost always one of the BSP-row failure classes (FW update partial-failure, RShim unreachable, cold-power-cycle needed) surfaced through the operator. "DPUService won't reconcile" → version-skew between DPF release, BFB image, and host DOCA — DPF release notes pin the supported BFB + DOCA-Host versions; running a DPUService image that expects a different BFB will silently no-op. "etcd / API server unreachable from DPU" → DPF assumes the DPU's OOB management network has a route to the cluster control plane; verify on the BF arm side, not the host. **Not the same as Network Operator**: DPF manages **DPUs**, Network Operator manages **host NICs / drivers / CNI on regular k8s nodes**. | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "DPF" or "DOCA Platform Framework" |
| **NVIDIA Network Operator** (Kubernetes) | <https://docs.nvidia.com/networking/display/kubernetes2611/index.html> | A Kubernetes operator that installs and manages NVIDIA networking host software — MLNX_OFED driver, RDMA shared device plugin, SR-IOV device plugin, CNI plugins (Multus, SR-IOV-CNI, Macvlan), IPAM (whereabouts, NVIDIA IPAM), and the NicClusterPolicy CR that ties them together. Pairs with the GPU Operator for GPUDirect RDMA workloads. | "Network Operator deployed but pods can't get a NIC" → almost always `NicClusterPolicy` CR missing or malformed; the operator installs CRDs but a separate `kubectl apply` of the policy CR is what actually provisions device plugins. "Driver pod CrashLoopBackOff" → kernel-version skew vs the precompiled OFED container; switch to driver-build pattern or pin the OFED container tag matching your kernel. "Node Feature Discovery (NFD) conflicts" → if NFD is already running cluster-wide, set `nfd.enabled=false` in the Helm values (operator deploys its own NFD by default). "SR-IOV VF count not honored" → the operator does NOT change NIC firmware; you must `mlxconfig` SR-IOV-keys + reboot the node first, OR deploy the NIC Configuration Operator (separate row, below) to do that step in-band. | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "Network Operator" or "NicClusterPolicy" |
| **MLNX_OFED (when installed separately, not as DOCA-OFED via the DOCA installer)** | <https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/> (transition guide: <https://docs.nvidia.com/doca/sdk/MLNX_OFED-to-DOCA-OFED-Transition-Guide/index.html>) | The standalone MLNX_OFED driver package (kernel modules + libibverbs / libmlx5 / perftest / OpenSM client / firmware-tools). **As of DOCA 2.5+, NVIDIA's recommended path is DOCA-OFED installed via the DOCA installer** (covered by the in-tree `doca-version` four-source audit); MLNX_OFED-as-separate-package is the legacy / advanced path for users not yet on the unified DOCA installer or who specifically need an MLNX_OFED-only build. | "I installed MLNX_OFED **and** DOCA-Host" → that is the failure mode the `doca-version` skill exists to detect; the two are mutually exclusive on the same host, and the four-source audit will flag the conflict. "MLNX_OFED kernel modules won't load after kernel upgrade" → either rebuild via `mlnx-en-installer.sh --add-kernel-support` (DKMS-style) or switch to DOCA-OFED which handles this in the installer. "perftest / `ibv_devinfo` segfaults" → almost always userspace-vs-kernel-module mismatch when MLNX_OFED was partially upgraded; rerun the full installer, never `dpkg -i` individual packages. **For new deployments, route to the transition guide and recommend DOCA-OFED unless the user has a documented reason to stay on MLNX_OFED-only.** | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "MLNX_OFED" or "DOCA-OFED transition" |
| **NVIDIA UFM (Unified Fabric Manager)** (InfiniBand) | <https://docs.nvidia.com/networking/display/ufmenterpriseumv62311> | Centralized IB fabric management — subnet manager (or supervises the in-fabric SM), telemetry, alarms, topology, partition (PKey) management, firmware push to switches/HCAs, congestion control plumbing. **InfiniBand-only**; for Ethernet fabrics use NetQ + Cumulus (separate rows). | "RDMA verbs work on RoCE but my IB port is stuck in Init" → fabric has no active subnet manager; either start OpenSM on a host or, in UFM-managed fabrics, check that UFM's SM is running and that the port is in UFM's "active" set. "PKey configured on host but `ibv_devinfo` shows default" → UFM partition table not pushed; check UFM REST `/resources/pkeys` and re-push the partition. "Switch FW skew" → use UFM's bulk firmware push, do not `flint` switches individually unless you have a specific reason. "I just want a subnet manager for a tiny IB testbed" → don't deploy UFM, just run `opensm` on one host (covered by MLNX_OFED docs). | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "UFM" or "InfiniBand fabric manager" |
| **NVIDIA Cumulus Linux** (Spectrum switch OS) | <https://docs.nvidia.com/networking-ethernet-software/cumulus-linux/> | The full-featured Debian-bookworm-based switch OS for NVIDIA Spectrum switches — BGP / EVPN / VXLAN control plane, NVUE CLI, package upgrade and ONIE install. Used as the **leaf / spine OS** in fabrics where DOCA HBN is the host-side BGP speaker. | "HBN says BGP is up on the BF, but the leaf switch isn't forwarding" → first symptom-bisect: is the failure on the BF (HBN-row gotcha class) or on the switch (Cumulus-row)? On the switch: check `nv show vrf default router bgp peer <bf-ip>` and `net show bgp summary`; if BGP is down switch-side, the issue is leaf-side configuration (peer-group, ASN mismatch, EVPN family not enabled), not HBN. "Cumulus version vs Spectrum-X stack" → the Cumulus version is **pinned per Spectrum-X validated solution release** (see Spectrum-X stack row); free-running Cumulus upgrades break Spectrum-X validation. "NVUE vs net commands" → NVUE is the supported declarative API; `net` commands are legacy and deprecated for new automation. | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "Cumulus" or "Spectrum" |
| **NVIDIA Firmware Tools (MFT)** (`mst`, `flint`, `mlxconfig`, `mlxfwmanager`, `mlxlink`, `mlxcables`, `mstflint`) | <https://docs.nvidia.com/networking/display/mftv434118lts/General-Information> (product page: <https://network.nvidia.com/products/adapter-software/firmware-tools/>) | The host-side firmware/diagnostic toolkit for ConnectX NICs and BlueField DPUs. **Every** firmware burn, every `mlxconfig` write, every link-level debug query (`mlxlink`), every cable diagnostic (`mlxcables`) goes through MFT. **Required precondition**: load `mst` (`mst start`) and identify devices with `mst status` before any per-device command. | "`mlxconfig -d <bdf> set ...` fails with 'device not found'" → `mst start` not run, or device BDF is the PF function not the management interface (`mst status` shows the correct path, typically `/dev/mst/mt4129_pciconf0` for a BF-3). "I burned firmware with `flint` but the new version isn't active" → firmware burns require **either** a reboot **or** `mlxfwreset -d <bdf> reset` to activate, AND many `mlxconfig` keys additionally require BFB-time application — read the per-key activation method from `mlxconfig -d <bdf> show_confs`. "Online firmware fetch" → use `mlxfwmanager --online -d <bdf>` (NOT `flint`); `mlxfwmanager` validates the FW image against the OPN before flashing. **Hardware-safety overlay (`doca-hardware-safety`) is mandatory for any MFT mutating command** — never burn or reset live without the operator-window + OOB-console + rollback preconditions. | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "MFT" or "mlxconfig" or "flint" |
| **NVIDIA Rivermax SDK** (separately licensed media-streaming SDK; the in-tree `doca-rmax` library is a wrapper) | <https://docs.nvidia.com/doca/sdk/DOCA-Rivermax/index.html> (Prerequisites → License) | A separately licensed userspace library for high-precision media streaming (SMPTE 2110, NMOS) over NVIDIA NICs. The bundle ships an in-tree `doca-rmax` skill that **wraps** Rivermax — the wrapper compiles and links once the SDK is installed and a license is active, but the SDK + license are **external products** that ship outside the DOCA installer. | "`doca-rmax` builds fine but the app dies at create with `DOCA_ERROR_NOT_PERMITTED` / `RMAX_ERR_NO_LICENSE`" → the Rivermax SDK is not licensed on this host (or the license expired); the in-tree skill **cannot** make this work — route to the Rivermax SDK Prerequisites page to obtain / renew the license. "pkg-config shows `doca-rmax` but `rivermax.so` not found" → SDK not installed (the wrapper alone is not enough); install the Rivermax SDK package from NGC, then re-run the four-source audit (`doca-version`). "Rivermax license file location" → `/opt/mellanox/rivermax/lib/rivermax.lic` by default; check `RIVERMAX_LICENSE_PATH` env override. | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "Rivermax" or "rmax license" |
| **BlueField BMC Software** (out-of-band management) | <https://docs.nvidia.com/networking/display/bluefieldbmcv2601/Platform-Management-Interface> | The dedicated Baseboard Management Controller on BlueField cards, exposing Redfish + IPMI + serial-over-LAN over a dedicated OOB management port. The **only** path to recover a BlueField when the Arm OS won't boot, when SSH is dead, when `rshim` won't attach, or when a BFB install bricked the BF. **Mandatory precondition for any `doca-hardware-safety`-class mutating change.** | "BF SSH dead after a bad BFB install, no console output" → BMC console-over-Redfish is the recovery path; do not just power-cycle blind. "How do I get to BMC?" → BMC has its own IP on the OOB management port (separate from BF Arm management); default creds in the BMC docs, change them immediately. "Redfish API for batch BFB reflash" → BMC supports BFB push via Redfish `UpdateService` endpoint — preferred for fleet recovery vs per-host `bfb-install`. "BMC firmware version vs BFB version" → BMC FW is independent of BFB; check release notes for the supported BMC-FW + BFB pair before recovery operations. | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "BlueField BMC" or "Redfish BlueField" |
| **DOCA Privileged Executor (DPE)** | <https://docs.nvidia.com/doca/sdk/DOCA-Telemetry-Service-Guide/index.html> (DPE section) | A privileged daemon (`dpeserver` / `dpe`) on BlueField that lets containerized DOCA services (DTS, BlueMan, others) reach BF-only privileged data over gRPC. Without DPE running, every BF-side container that needs `sysfs`/`ethtool`/`devlink` data from outside its namespace returns empty. | "DTS provider 'sysfs' returns no rows" or "BlueMan shows empty dashboard" → first check `systemctl status dpe` on the BF; if DPE is stopped, the containerized services have no data path even though the containers themselves are up. "DPE service won't start" → almost always a SELinux / AppArmor profile blocking the privileged socket, or the BFB image is too old (DPE was added in BFB 3.9.x). "DPE vs Privileged Container pattern" → DPE is the **supported** pattern; running DTS as `privileged: true` works in isolation but breaks the BlueMan / DPF integration. **Already named as a sub-gotcha in the DTS-as-deployed and BlueMan rows above — this row exists because DPE is itself a separately-described component with its own failure modes.** | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "DPE" or "DOCA Privileged Executor" |
| **NIC Configuration Operator** (Kubernetes; part of Network Operator family) | <https://docs.nvidia.com/networking/display/kubernetes2611/nic-conf-operator/nic-configuration-operator.html> | A separate Kubernetes operator that applies coordinated NIC firmware-version + `mlxconfig`-class NVConfig changes across nodes in a cluster — fills the gap that Network Operator deliberately does not cover (Network Operator manages **drivers + plugins**, NIC Configuration Operator manages **firmware + NVConfig**). | "Network Operator deployed but SR-IOV VFs still not exposed" → SR-IOV requires `mlxconfig` keys + reboot per node; Network Operator does NOT do this. Either pre-stage with `mlxconfig` per node OR deploy NIC Configuration Operator and write a `NicConfigurationTemplate` CR. "NicConfigurationTemplate not reconciling" → check the operator pod's logs for per-node FW-update failures; the operator orchestrates `mlxfwmanager` and reboot, but a node that's `cordon`ed will hang the rollout. "BF-3 DPU firmware reset limitations" → the operator can update host NICs in-band; BF DPUs have stricter rules and may need OOB BMC-driven update (BlueField BMC row). "Operator vs in-band `mlxconfig` ownership" → pick one, do not mix; mixed ownership produces races and surprise reboots. | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "NIC Configuration Operator" or "NicConfigurationTemplate" |
| **NVIDIA NetQ** (fabric validation + telemetry for Cumulus / Ethernet fabrics) | <https://docs.nvidia.com/networking-ethernet-software/cumulus-netq/> | A scalable fabric-wide telemetry, validation, and root-cause-analysis tool for NVIDIA Cumulus Linux and Spectrum-Ethernet fabrics. Used alongside Cumulus / HBN to answer "is my whole fabric healthy?" rather than per-switch / per-host. | "HBN BGP looks fine on the BF, but a workload-to-workload ping fails" → NetQ's fabric trace is the right answer (it sees both BF + leaf + spine simultaneously); `nv show` per-switch is not enough. "NetQ agent not reporting" → first check the agent runs on every switch *and* every Cumulus-style host (`netq config show`); silent gaps in coverage produce phantom-pass validations. "NetQ vs UFM" → NetQ for Ethernet (Cumulus / Spectrum-Ethernet); UFM for InfiniBand. They do not overlap. | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "NetQ" or "fabric validation" |
| **NVIDIA NVOS** (switch OS for Spectrum-4 / NVLink / IB switches; parallel to Cumulus) | <https://docs.nvidia.com/networking/display/NVIDIANVOSUserManualforNVLinkSwitchesv25024282/Overview> | The switch OS for next-generation NVIDIA switches (Spectrum-4 Ethernet, NVLink-Switch, certain IB Quantum-X switches). NVUE CLI is shared with Cumulus, but the OS image, package set, and supported features differ. **Do not assume Cumulus syntax 1:1.** | "I followed a Cumulus guide on my NVLink switch" → wrong OS. Read the NVOS guide for that switch family; NVUE commands overlap but feature coverage differs. "Spectrum-4 vs Spectrum-3 OS" → Spectrum-4 ships with NVOS by default in Spectrum-X stacks; Spectrum-3 ships with Cumulus by default — check before you assume. "NVLink switch firmware update" → NVOS path, not Cumulus path; use the NVOS image update procedure. | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "NVOS" or "NVLink switch" |
| **NVIDIA Spectrum-X Validated Solution Stack** | <https://docs.nvidia.com/networking/software/spectrumx-solution-stack/index.html> | A version-pinned reference architecture tying together specific releases of: DOCA-Host, BlueField BFB, ConnectX firmware, MLNX_OFED, Cumulus Linux (or NVOS), Network Operator, NIC Configuration Operator, NetQ, HPC-X, NCCL, and the Spectrum / Spectrum-X switch firmware. Treats the whole stack as a single tested unit for AI / RoCE / GPUDirect deployments. | "I upgraded DOCA but my NCCL allreduce is now slower" → component-level upgrades break the Spectrum-X validated mix; either revert or upgrade the whole stack to a newer validated release. "Which Cumulus / BFB / DOCA versions go together?" → the validated solution stack release notes are the authoritative answer; do **not** infer from individual product release notes. "Spectrum-X tech-preview features" → the stack's release notes call out which features are tech-preview vs GA at the stack release; a feature can be GA in DOCA but tech-preview in the Spectrum-X release pinning that DOCA version. | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "Spectrum-X" or "Spectrum-X validated" |
| **NVIDIA GPU Operator** (Kubernetes) | <https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html> | The Kubernetes operator that installs and manages the NVIDIA GPU stack on cluster nodes — driver, container runtime hook, device plugin, DCGM exporter, MIG manager. **Pairs with Network Operator** to enable GPUDirect RDMA between GPU pods and NICs. | "I deployed Network Operator and DOCA-Host on the node, but `cuFile` / GPUDirect RDMA still doesn't work" → GPU Operator (or a manually installed NVIDIA driver matching the kernel) is the missing peer; verify `nvidia-peermem` module is loaded (`lsmod | grep nvidia_peermem`). "GPU Operator + Network Operator on the same node" → supported, but driver loading order matters; the GPU Operator's `nvidia.ko` should load before Network Operator's `nvidia_peermem` dependency check. "GDS (GPUDirect Storage) vs GPUDirect RDMA" → different features; GDS is `cuFile` to storage targets, GPUDirect RDMA is GPU to NIC. Don't conflate. | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> — search "GPU Operator" or "GPUDirect RDMA Kubernetes" |

**One worked example of the required response shape (for SNAP).** When
the user asks *"how do I use DOCA SNAP to provision a host server —
the host server never seems to be able to properly enumerate the NVMe
device, or if it does it never is able to actually boot"*, the agent
must answer with all three parts:

1. **Recognize the class.** *"DOCA SNAP Services (SNAP-4 on BF-3, SNAP-3
   on BF-2, SNAP Virtio-fs on BF-3) are externally-productized NVIDIA
   services — they ship as separate NGC containers / packages and are
   not in the `doca/services/` monorepo this bundle is 1:1 with, so I
   don't have an in-tree SNAP skill to load."*
2. **Name the boundary.** *"The bundle's strict-1:1 alignment with
   `doca/{services,tools}/` (see `AGENTS.md ## Non-goals` #7)
   intentionally excludes externally-productized services, because
   their cadence, packaging, and support model differ from the
   in-tree set the bundle was authored against."*
3. **Route with substance.** *"Your symptom — host can't enumerate
   the NVMe device, or enumerates but won't boot — is one of the
   four most-common SNAP gotcha classes: (a) NIC firmware not
   configured for SNAP / virtio (`mlxconfig` keys per the SNAP-4
   *Firmware Configuration* section); (b) SNAP container not running
   on the BF (`crictl ps -a \| grep snap`); (c) no fabric target
   reachable from the BF management VRF; (d) BF BSP / SNAP version
   mismatched against host DOCA. Authoritative read:
   <https://docs.nvidia.com/doca/sdk/DOCA-SNAP-Services/index.html>
   (drill into SNAP-4 for BF-3). For real-customer threads with
   NVIDIA-staff answers: <https://forums.developer.nvidia.com/c/infrastructure/doca/370>,
   search 'SNAP'."*

Skipping any of the three parts is a contract violation. The bundle's
own deep-E2E suite tests for all three.

## Public source code: GitHub

NVIDIA publishes a small, deliberately scoped set of DOCA-adjacent code on
GitHub. These are the public, customer-visible repositories.

| Repository | URL | What you find there |
| --- | --- | --- |
| DOCA Platform Framework | <https://github.com/NVIDIA/doca-platform> | DPU provisioning, Kubernetes operator pieces, deployment manifests. Read this when the user is operating DPUs at scale or running them under Kubernetes. |

**Important:** the bulk of DOCA — libraries, samples, applications, build
system — is **not on GitHub**. It ships as packages and, for licensed users,
as source archives downloaded from the Developer Zone. If the user asks for
"the DOCA library X source on GitHub", correct them: the published sample for
library X lives under `/opt/mellanox/doca/samples/doca_<library>/` on an
installed system, or in the downloadable source archive referenced from the
Downloads page.

## Layout of an installed DOCA package

When DOCA is installed from the official packages on a Linux host or on
BlueField, an agent can rely on this on-disk layout. Use it instead of asking
the user to share source code.

| Path | What is there | How to use it |
| --- | --- | --- |
| `/opt/mellanox/doca` | Install root. | Use as `${DOCA_DIR}` in any command you suggest. |
| `/opt/mellanox/doca/samples` | One subdirectory per library, each containing a self-contained sample (typical files: `<library>_main.c`, `meson.build`). | The authoritative example for that library on the installed version. Read these before answering "show me a sample of X". |
| `/opt/mellanox/doca/applications` | Full reference applications (e.g. `doca_pcc`, `doca_dpi`). | Larger, integrated examples. Inspect their `meson.build` to see how they declare DOCA dependencies. |
| `/opt/mellanox/doca/tools` | Shipped CLIs (e.g. `doca_caps`, `doca_socket_relay`). | Use them for runtime introspection before answering capability questions. |
| `/opt/mellanox/doca/infrastructure` *(legacy / split-profile installs only)* OR `/opt/mellanox/doca/lib/<arch>-linux-gnu/` + `/opt/mellanox/doca/include/` *(default on DOCA 3.3+)* | Headers, libraries, and `pkg-config` files used to build against DOCA. The `infrastructure/` subtree is present in some legacy / split installer profiles; on most current DOCA 3.3+ installs the layout has flattened and `infrastructure/` may NOT exist. ALWAYS resolve via `pkg-config --variable=includedir doca-common` and `pkg-config --variable=libdir doca-common` rather than hard-coding either path. | Inspect `*.pc` to verify the exact Meson dependency name for a library (`doca-flow`, `doca-common`, etc.). |
| `/opt/mellanox/doca/services` | Bundled services (DTS, telemetry agents, etc.). | Read service-specific README files inside each subdirectory before suggesting service-level changes. |

Useful enumeration commands to suggest to the user (read-only, safe):

```bash
ls /opt/mellanox/doca
ls /opt/mellanox/doca/samples
ls /opt/mellanox/doca/applications
pkg-config --list-all | grep -i doca
```

To build against DOCA from a user-owned directory, the canonical environment
hint is to expose the DOCA `pkg-config` directory before running `meson setup`:

```bash
PCDIR="$(dirname "$(find /opt/mellanox/doca -name doca-common.pc -print -quit 2>/dev/null)")"
export PKG_CONFIG_PATH="${PCDIR}:${PKG_CONFIG_PATH}"  # commonly /opt/mellanox/doca/lib/<arch>-linux-gnu/pkgconfig on DOCA 3.3+; or /opt/mellanox/doca/infrastructure/lib/pkgconfig on legacy / split-profile installs
```

If `pkg-config --modversion doca-common` fails after that, treat it as a real
environment problem (wrong package, wrong arch, missing dev package) and stop;
do not silently change the user's environment.

## Where to find the version

Pick whichever of these is available, in this order:

1. `cat /opt/mellanox/doca/applications/VERSION` (or the `VERSION` file at the
   install root, depending on package layout).
2. `pkg-config --modversion doca-common`
3. The DOCA Release Notes page header for the version the user says they
   installed.
4. The `--version` flag on any installed DOCA tool, for example
   `doca_caps --version`.

Always quote the version you actually observed; never assume "latest".

## URL audit

Every URL referenced by this routing map is HEAD-checked against the
public NVIDIA documentation surface on every commit, so bundle
releases always ship a routing map whose every URL was reachable at
release time. If a URL on this page no longer resolves, that is the
release's bug, not yours — fall back to the umbrella entry points
listed in *Public documentation entry points* (DOCA SDK index, DOCA
Libraries / Services / Tools umbrella pages) and search from there.

## Ground rules for any agent using this skill

1. **Prefer the local install over the web.** If DOCA is installed on the
   machine, the files in `/opt/mellanox/doca` (Linux) are the exact bits the
   user is running. Web docs describe a release; local files *are* the release.
2. **Always check the version.** Before quoting API names, options, or sample
   filenames, find the installed DOCA version (see "Where to find the
   version" below) and use the matching documentation release.
3. **Never invent URLs, file paths, package names, `pkg-config` modules,
   library names, or sample names.** If you do not see it in this map, in the
   user's local install, or in the official docs you fetched, say so and ask.
4. **Public sources only.** Reference NVIDIA documentation only on the
   public hosts listed in [`AGENTS.md` ground rule #1](../../../AGENTS.md).
   Anything else is not available to a customer agent and is rejected
   by an internal CI pipeline before the bundle ships.
5. **No source-tree paths.** Do not reference `devtools/...`, `docs/ai/...`,
   or any path that only exists inside the DOCA repository. Customers do not
   have those.

> **Hardware-safety meta-policy.** Every per-artifact `## Safety policy`
> in this bundle overlays the cross-cutting hardware-safety
> meta-policy. When the user's question touches a change to live DPU /
> NIC hardware state (`mlxconfig` writes, firmware burn, BlueField
> mode flip, BAR window change, IOMMU / hugepages kernel boot
> parameter, BFB reflash), load
> [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) alongside
> the per-artifact skill. Cross-cutting meta-policy lives there
> (pre-flight inventory, out-of-band access requirement, maintenance
> window, replica-first validation, rollback discipline,
> observability-before-workload, refuse-and-escalate when no rollback
> exists). Per-artifact overlays MUST NOT redefine it.

> **Upgrade / downgrade discipline.** When the user is contemplating
> or recovering a DOCA upgrade or downgrade — moving a host to the
> next release, refreshing the BFB, bumping an NGC container tag,
> rolling back, or reacting to an accidental `apt upgrade` drift —
> load [`doca-upgrade`](../../doca-upgrade/SKILL.md). Its headline
> contract is **detect → report → ASK → only-then guided upgrade;
> never auto-upgrade.** It routes version detection to
> [`doca-version`](../../doca-version/SKILL.md), every hardware /
> firmware / reboot step to
> [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md), and
> sunset / deprecation lifecycle lookups back to this map's
> Release Notes entry below. It does not redefine version detection
> or the hardware-touching discipline.

## Public documentation entry points

Start here for any conceptual question. These are the canonical NVIDIA-hosted
documents.

> **Rule when a URL in this skill returns 404.** NVIDIA periodically renames
> doc pages and library slugs (the most recent example: *Comm Channel* →
> *Comch* in DOCA 2.5). If a URL listed here 404s during a real session:
> (1) tell the user explicitly that the skill's URL is stale; (2) try
> `https://docs.nvidia.com/doca/sdk/` (the index always works) and look for a
> renamed link; (3) **do not invent a replacement URL** and do not silently
> point at an `archive/` URL — those are version-pinned to old releases.
> File a fix against this skill (see "URL audit" footer for the last
> verification date and DOCA version).

| Topic | URL | When to use |
| --- | --- | --- |
| DOCA SDK documentation index | <https://docs.nvidia.com/doca/sdk/index.html> | The top of the documentation tree. Always start here when the user asks an open-ended "how does DOCA do X?" question. |
| DOCA Overview | <https://docs.nvidia.com/doca/sdk/doca-overview/index.html> | High-level architecture: what DOCA is, what runs on the BlueField DPU vs. the host, libraries vs. applications vs. services vs. tools. |
| DOCA Installation Guide for Linux | <https://docs.nvidia.com/doca/sdk/NVIDIA+DOCA+Installation+Guide+for+Linux> | Install steps, supported OSes, package layout, post-install verification. Read this whenever the user is setting DOCA up for the first time. |
| DOCA Programming Guide (index) | <https://docs.nvidia.com/doca/sdk/doca-programming-guide/index.html> | SDK architecture, common patterns, how libraries connect, how to wire up a sample. Read this before drilling into a specific library guide. |
| DOCA Developer Quick Start Guide | <https://docs.nvidia.com/doca/sdk/doca-developer-quick-start-guide/index.html> | NVIDIA's official "how do I bring DOCA up and run my first reference application" walkthrough. Cite this for any beginner asking "how do I start?" *once they have BlueField + host hardware*. For users **without** hardware (macOS / Windows / Linux without a NIC), route via [`doca-setup ## no-install`](../../doca-setup/TASKS.md#no-install) (NGC container) instead — the Quick Start assumes hardware is already present. |
| DOCA Release Notes | <https://docs.nvidia.com/doca/sdk/doca-release-notes/index.html> | What changed in a release, supported hardware, supported OSes, known issues. Read whenever the user's symptom could be a known issue. |
| DOCA Compatibility Policy | <https://docs.nvidia.com/doca/sdk/doca-compatibility-policy/index.html> | NVIDIA's authoritative statement of DOCA version semantics — quarterly GA cadence, October LTS designation (3-year support, 7-update LTS train), the semantic-versioning scheme (X.Y.Z), the three compatibility types (source / binary / behavioral), and the two compatibility directions (backward / forward). Cite this whenever the user asks "what does the version string mean?", "is this LTS release still supported?", or any host ↔ DPU compatibility question. |
| Developer Zone — DOCA landing | <https://developer.nvidia.com/networking/doca> | Marketing-and-onboarding view (videos, tutorials, blog posts). Use only as a fallback when the official SDK docs do not have the topic yet. |
| DOCA Downloads | <https://developer.nvidia.com/doca-downloads> | Where customers actually download DOCA packages, BFB images, host packages. Use this to answer "which package do I need?". |
| NGC Catalog (containers and resources) | <https://catalog.ngc.nvidia.com> | Find DOCA containers, model artifacts, and DPU images. Search for `doca` to enumerate. |
| DOCA Developer Forum | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> | Last-resort discovery for undocumented behavior, real customer questions, NVIDIA staff answers. Always include the disclaimer that forum threads can age and may not match the user's installed version. |

