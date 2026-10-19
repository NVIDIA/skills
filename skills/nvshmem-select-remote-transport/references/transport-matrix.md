# NVSHMEM Transport Eligibility Matrix

## Contents

- [Evidence classes](#evidence-classes)
- [Selection model](#selection-model)
- [Common gates](#common-gates)
- [Transport matrix](#transport-matrix)
- [Selection configurations](#selection-configurations)
- [Candidate-specific guidance](#candidate-specific-guidance)
- [Operation and memory-mode gates](#operation-and-memory-mode-gates)
- [Source basis](#source-basis)

## Evidence Classes

- **Documented:** stated by the cited public NVSHMEM documentation for the displayed release.
- **Source-derived:** observed in the reviewed NVSHMEM source; verify against the target release.
- **Inference:** a recommendation derived from documented/source behavior and the supplied workload.
- **Unknown:** not established by available evidence; never treat as eligible.

## Selection Model

Evaluate three independent layers:

1. **Built:** the selected installation contains the required transport plugin and compatible device code.
2. **Runnable:** the plugin can load and all hardware, provider, library, driver, and memory-registration requirements are met.
3. **Suitable:** the kernel/API pattern is expected to fit the transport architecture.

A source-tree CMake option proves only a possible build. A plugin filename proves installation,
not loadability. A compatible NIC proves neither plugin availability nor operation coverage.

## Common Gates

### Version and Plugin Inventory

- Record the installed version from `nvshmem-info -n`, installed headers, a package manifest, or an explicit user value when readily available. Patch-level verification is required only for a decision that depends on a release-specific gate, known issue, setting change, or operation limitation.
- Look for `nvshmem_transport_<name>.so*` in the selected prefix. Do not merge plugins from different prefixes or `LD_LIBRARY_PATH` entries.
- Treat IBGDA specially: it is enabled with `NVSHMEM_IB_ENABLE_IBGDA=1`; it is not a value of `NVSHMEM_REMOTE_TRANSPORT`.
- Treat GPUNetIO GDAKI specially: it requires both the GPUNetIO remote plugin and GPUNetIO-capable device code.
- `NVSHMEM_REMOTE_TRANSPORT` accepts `ibrc`, `ibdevx`, `gpunetio`, `ucx`, `libfabric`, and `none`. Use these stable documented values unless there is evidence that the target release predates a candidate or changed the setting.

### Remote Path

- Remote transport selection affects network-reachable peers. Peer-reachable NVLink/PCIe traffic uses the local P2P path instead.
- Select `none` only when every relevant peer is local and peer-reachable and no API requires a remote-network fallback.
- A single-node launch does not alone prove that every PE pair is peer-reachable.

### Operation Coverage

- Verify release-specific RMA, AMO, signal, synchronization, host, on-stream, custom-QP, and registered-buffer restrictions when the application actually requires them. Ordinary RMA selection does not need an exhaustive exact-release operation audit.
- Do not generalize one successful put benchmark to atomics, gets, put-with-signal, host APIs, or user-registered buffers.
- Collectives may route through NCCL, NVLS, multicast, P2P, or point-to-point algorithms. Analyze the actual remote point-to-point path rather than assigning collective performance to the selected plugin automatically.

### GDRCopy Evidence

- Treat a loaded `gdrdrv` kernel module or visible `/dev/gdrdrv` device as sufficient evidence that GDRCopy is present for transport selection.
- `pkg-config` and loader-cache entries are optional supporting hints. Their absence does not disprove a custom GDRCopy installation and must not exclude IBRC or independently justify preferring IBDevX.

## Transport Matrix

| Candidate | Architecture | Primary fabric/provider gate | Important runtime/build gates | Expected fit | Important caveats |
| --- | --- | --- | --- | --- | --- |
| IBRC | CPU-proxy, verbs RC | InfiniBand or verified RoCE | IBRC plugin, verbs/OFED, GPUDirect RDMA registration path; GDRCopy for documented remote atomic coverage | Conservative Mellanox default; packed or lower-concurrency remote traffic | Single proxy serializes device requests; do not assume Ethernet means configured RoCE |
| IBDevX | CPU-proxy, direct mlx5 DevX RC | mlx5 InfiniBand or verified RoCE | IBDevX plugin, mlx5 DevX/`mlx5dv`, exact device/firmware support | Mellanox proxy path when DevX is explicitly available and required coverage is verified | Off by default in source builds reviewed; no general public performance superiority claim |
| IBGDA | GPU-initiated GDAKI plus separate host transport | Mellanox InfiniBand or supported RoCE | IBGDA plugin/device code, GDAKI prerequisites, host remote plugin, supported NIC handler | Many fine-grained parallel device operations; persistent GPU communication; DCI/DCT scale | Individual GPU WQE construction can have higher isolated latency; RC/DC resource and exact-release limitations matter |
| GPUNetIO CPU | CPU data path through GPUNetIO | Mellanox InfiniBand or verified RoCE | GPUNetIO plugin; GPUNetIO library bundled or installed | Unified GPUNetIO deployment without GPU data path | Verify exact operation and platform coverage |
| GPUNetIO GDAKI | GPU-initiated GPUNetIO path plus GPUNetIO CPU path | Mellanox InfiniBand or supported RoCE | GPUNetIO plugin/device code, GDAKI prerequisites; optional DOCA for advanced features | Fine-grained parallel device traffic when RC scale is acceptable; unified CPU/GPU path | RC-oriented; verify connection-scale coverage for the target release |
| UCX | CPU-proxy through UCP | UCX-supported fabric selected by configured transports | UCX plugin and compatible UCX built with required thread/device-memory support | UCX-managed sites or verified capability/fallback unavailable natively | Public docs describe UCX transport as experimental; GDRCopy/atomic path is configuration-specific |
| libfabric | CPU-proxy through libfabric | Native `efa`, `cxi`, or verified `verbs` provider | libfabric plugin, matching provider, provider-specific memory registration and launcher rules | First choice for AWS EFA and Slingshot/CXI when supported | Exact-release VMM, registered-buffer, operation, and EFA environment limitations are hard gates |
| none | No remote plugin | All relevant PE pairs local and peer-reachable | P2P path present; no remote-network requirement | PCIe/NVLink-only deployments | Disables remote fallback; do not choose from node count alone |

## Selection Configurations

Use lowercase values as documented. Preserve the user's configuration mechanism when it is a
config file rather than shell exports.

### IBRC

```bash
export NVSHMEM_REMOTE_TRANSPORT=ibrc
```

### IBDevX

```bash
export NVSHMEM_REMOTE_TRANSPORT=ibdevx
```

### IBGDA

```bash
export NVSHMEM_REMOTE_TRANSPORT=<resolved-host-transport>
export NVSHMEM_IB_ENABLE_IBGDA=1
```

The host transport must be independently eligible for host/on-stream remote operations. Do not
set `NVSHMEM_GPUNETIO_ENABLE_GDAKI=1` in the same configuration.

### GPUNetIO CPU Path

```bash
export NVSHMEM_REMOTE_TRANSPORT=gpunetio
export NVSHMEM_GPUNETIO_ENABLE_GDAKI=0
```

### GPUNetIO GDAKI

```bash
export NVSHMEM_REMOTE_TRANSPORT=gpunetio
export NVSHMEM_GPUNETIO_ENABLE_GDAKI=1
```

### UCX

```bash
export NVSHMEM_REMOTE_TRANSPORT=ucx
```

Do not synthesize `UCX_TLS` or other UCX tuning from transport selection alone.

### libfabric

```bash
export NVSHMEM_REMOTE_TRANSPORT=libfabric
export NVSHMEM_LIBFABRIC_PROVIDER=<cxi|efa|verbs>
```

Provider-specific mandatory compatibility settings belong in preconditions, not in the minimal
selection block, unless the user explicitly asks for a full launch configuration.

### No Remote Transport

```bash
export NVSHMEM_REMOTE_TRANSPORT=none
```

## Candidate-Specific Guidance

### IBRC

- **Documented:** IBRC is the default remote transport.
- **Documented:** proxy-based device operations are serviced by one NVSHMEM CPU proxy and benefit from packing small logical transfers into larger operations.
- **Documented:** IBRC uses GDRCopy for remote atomic support in the documented configuration.
- **Inference:** prefer IBRC when broad compatibility matters more than potential high-message-rate GDAKI benefits.
- Exclude or mark conditional when required non-native/float atomics lack exact-release coverage.

### IBDevX

- **Source-derived:** IBDevX is a CPU-access transport that directly creates/programs mlx5 objects through DevX/`mlx5dv`.
- **Source-derived:** reviewed builds default IBDevX support off, so require plugin evidence.
- **Source-derived:** its plugin does not use the common GDRCopy component in the reviewed build configuration.
- **Inference:** confirmed absence of `gdrdrv` may justify considering IBDevX for a GDRCopy-dependent requirement, but missing `pkg-config` or loader-cache hints alone may not.
- Never rank it above IBRC solely because it uses a lower-level interface.

### IBGDA

- **Documented:** GDAKI moves control and data-plane network submission to the GPU and can increase the message rate for many small transfers submitted in parallel.
- **Documented:** a GPU thread creates a WQE more slowly than a CPU thread, which can increase latency for an isolated message.
- **Documented:** IBGDA supports RC and DCI/DCT tuning, trading latency and connection memory/scale.
- **Documented:** Mellanox NIC, MOFED 5.0+, NVIDIA driver 510.40.3+, and an accepted DMA-BUF/peer-memory registration path are prerequisites.
- Check default GPU doorbell mapping versus CPU-assisted NIC-handler requirements. CPU-assisted IBGDA can introduce a GDRCopy requirement.
- Favor DCI/DCT when RC connections per peer would be prohibitive at the supplied PE scale; do not infer the threshold without resource evidence.

### GPUNetIO

- **Documented:** GPUNetIO provides an optional GDAKI path.
- **Documented:** `NVSHMEM_GPUNETIO_ENABLE_GDAKI=1` requires `NVSHMEM_REMOTE_TRANSPORT=gpunetio`.
- **Documented:** the open path does not require the full DOCA SDK; DOCA 3.4+ is optional for documented advanced features such as DDP.
- **Source-derived for the reviewed release:** GPUNetIO exposes CPU read/write/atomic capabilities and adds GPU capabilities when GDAKI is enabled.
- **Source/design limitation:** device initiation is RC-oriented, whereas IBGDA offers DCI/DCT. Prefer verified required DOCA functionality, unified CPU/GPU deployment, or installed availability.

### UCX

- **Documented:** UCX is a proxy-based transport and requires a compatible UCX build; public installation guidance specifies UCX 1.10+ with multithreading and device-memory support for the documented release.
- **Documented:** public documentation labels UCX transport experimental and records platform-specific known issues; consult the exact release.
- **Documented:** release notes describe a UCX socket-based atomic path for systems without InfiniBand in a specific PCIe scenario.
- **Source-derived:** atomic handling varies between remote UCX atomics and local proxy processing and may require GDRCopy.
- Prefer site/provider requirements or verified capabilities, not a generic belief that UCX is more portable or faster.

### libfabric

- **Documented:** provider values are `cxi`, `efa`, and `verbs`; the default is `cxi`.
- **Documented:** NVSHMEM supports Slingshot-11 through CXI and AWS EFA through libfabric.
- Treat the native provider as a fabric gate. Do not rank a Mellanox `verbs` provider over the native IB transports without a concrete site requirement.

## Operation and Memory-Mode Gates

Before ranking, make a row for every required operation family:

| Requirement | Evidence to resolve |
| --- | --- |
| Device put/get | Exact API/scope, message sizes, remote path, plugin functional coverage |
| Atomics | Type/width/operation, native versus staged/proxy path, GDRCopy/provider requirements |
| Put-with-signal | Exact transport support and completion/ack path |
| Host/on-stream APIs | Host transport capabilities; IBGDA alone is insufficient |
| Registered user buffers | Exact release and provider limitation |
| CUDA VMM heap | Exact release/provider support; libfabric limitation |
| Custom QPs | Exact release and plugin support; connection type and resource scale |
| Multi-NIC | Transport/version support and separate NIC-mapping workflow |
| Collectives | Actual chosen algorithm/backend; do not assume point-to-point plugin controls it |

If a mandatory application requirement is unknown, prefer a broadly supported fallback, emit its minimal selection configuration, and mark the specialized capability conditional. Withhold exports only when no candidate is known to satisfy a non-negotiable requirement.

## Source Basis

Primary public sources reviewed:

- NVSHMEM Device APIs: <https://docs.nvidia.com/nvshmem/release-notes-install-guide/best-practice-guide/apis.html>
- NVSHMEM Performance: <https://docs.nvidia.com/nvshmem/release-notes-install-guide/best-practice-guide/performance.html>
- NVSHMEM Installation Guide: <https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/abstract.html>
- Using NVSHMEM: <https://docs.nvidia.com/nvshmem/api/using.html>
- Environment Variables: <https://docs.nvidia.com/nvshmem/api/gen/env.html>
- NVSHMEM 3.7.0 Release Notes: <https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/release-3700.html>

Development-source areas reviewed for architecture and configuration shape:

- `src/host/transport/transport.cpp`
- `src/modules/transport/{ibrc,ibdevx,ibgda,gpunetio,ucx,libfabric}`
- `src/modules/transport/common/env_defs.h`
- transport plugin CMake definitions

Recheck source-derived claims when the recommendation depends on a release-sensitive detail. A patch-version difference alone does not require live documentation lookup for an ordinary stable transport setting.
