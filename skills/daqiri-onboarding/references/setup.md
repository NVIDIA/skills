# Setup and Runtime Selection

Use the [documentation map](documentation-map.md) for technical instructions and helper locations. Read this only when setup, runtime compatibility, or path selection is unresolved. Reuse supplied facts; do not run a setup interview for saved-output diagnosis or a hardware-free repair.

## Locate the current stage

Use information already supplied, then ask a small batch of unresolved questions:

- What should the first result prove: working install, GPU-accessible acquisition, cabled throughput, hardware loopback, UDP/TCP baseline, RoCE, or diagnosis of a failure?
- Is this the host, a container, or a remote system? Which checkout/image/install and benchmark binaries are available?
- Which GPU and NIC are available? Is there a confirmed cable/peer, and which ports belong to the intended test?

For configuration inputs, distinguish discovered values from user estimates or examples. Mark uncertain port pairing, peer addresses and CPU choices explicitly; verify them before dependent actions. Hardware enumeration alone does not identify cable endpoints.

| Existing state | Next step |
|---|---|
| No checkout | Use Getting Started to obtain source/submodules; do not assume SSH credentials |
| Source but no compatible image/build | Read the build guide, select required optional engines, and compile in the project container |
| Build/image but no usable runtime | Inspect existing containers and propose the required launch; preserve unrelated work |
| Runtime but no concrete config | Discover inputs and use the canonical configuration generator |
| Concrete config | Run native semantic validation, then verify hardware/path prerequisites |
| Prior benchmark output | Interpret or diagnose that run before proposing another |

Before reusing a runtime, compare available source provenance, library ABI, compiled engines and
required tools with the selected checkout. Distinguish confirmed incompatibility from unknown
provenance. A missing required validator or incompatible ABI blocks validation; a different source
revision requires assessing whether the relevant interfaces and behavior match. Propose a compatible
runtime or authorized rebuild without switching branches or borrowing another release's validator.

Do not build or launch another container when an adequate existing runtime already satisfies the request.

## Choose the path

Honor the user's stated transport and goal. Otherwise:

- ConnectX and a confirmed cable/peer: raw Ethernet using a compiled engine.
- ConnectX without a cable/peer: supported raw hardware loopback.
- Linux TCP/UDP compatibility or baseline: always-built socket engine.
- RoCE/RDMA goal: `roce://` endpoints and the optional ibverbs/RDMA engine.
- No usable NIC path or explicit hardware-free goal: software loopback smoke test.

Confirm compiled engine availability using the native validator's `--list-engines` when supported. Do not invent a `socket` build-option value. Read the benchmark overview for current engine choices.

On GB10/Spark, use the documented `host_pinned` shared-memory path; do not mistake it for discrete-GPU device-memory registration or require peermem. For IGX dGPU acquisition use `device` packet/payload and reorder-output buffers; preserve that memory path when tuning. On hybrid iGPU/dGPU systems, follow the platform guide for GPU UUID selection and container ordinals. Verify actual CPU affinity in the context that will execute the workload.
