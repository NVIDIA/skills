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
| DOCA PCC ZTR-RTTCC Algorithm (`DOCA-PCC-ZTR-RTTCC-Algo`) | <https://docs.nvidia.com/doca/sdk/DOCA-Libraries/index.html> | The shipped reference Zero-Touch-RTT Congestion-Control algorithm that runs under `doca-pcc`. Pairs with `doca-pcc` (host) and `doca-pcc-counters` (observability). Covered by the [`doca-pcc-ztr-rttcc-algo`](../../libs/doca-pcc-ztr-rttcc-algo/SKILL.md) skill. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
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

