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
| Flow Tune (`DOCA-Flow-Tune`) | <https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html> | Unified visibility / analysis / recommendation tool for live `doca-flow` pipelines. The artifact is ONE binary with TWO internal roles (server role: snapshots and exposes pipe / counter / KPI state through local IPC; client / consumer role: dumps, analyzes, visualizes, recommends parameter changes). The historical "Flow Tune Tool" and "Flow Tune Server" split lives INSIDE this artifact — there is one skill: [`doca-flow-tune`](../../tools/doca-flow-tune/SKILL.md). **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| Flow Perf (`DOCA-Flow-Perf`) | <https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html> | Host / DPU-CPU control-plane rule-rate measurement (install / delete / query rate). Distinct from `Flow DPA Perf` (DPA-offloaded path) and `Flow Tune` (optimizes a deployed pipeline). Covered by the [`doca-flow-perf`](../../tools/doca-flow-perf/SKILL.md) skill. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| Flow DPA Perf (`DOCA-Flow-DPA-Perf`) | <https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html> | Flow performance measurement for the DPA-offloaded execution path. Covered by the [`doca-flow-dpa-perf`](../../tools/doca-flow-dpa-perf/SKILL.md) skill. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| Flow gRPC Server (`DOCA-Flow-gRPC-Server`) | <https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html> | gRPC remote-control server for `doca-flow` rule programming. Covered by the [`doca-flow-grpc-server`](../../tools/doca-flow-grpc-server/SKILL.md) skill. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| PCC Counter | <https://docs.nvidia.com/doca/sdk/DOCA-PCC-Counter-Tool/index.html> | PCC counter inspection. Covered by the [`doca-pcc-counters`](../../tools/doca-pcc-counters/SKILL.md) skill. |
| Socket Relay | <https://docs.nvidia.com/doca/sdk/DOCA-Socket-Relay/index.html> | Socket relay between host and DPU. Covered by the [`doca-socket-relay`](../../tools/doca-socket-relay/SKILL.md) skill. |
| App Shield Config | <https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html> | Generates the host-OS profile / symbol files that App Shield needs to interpret host kernel state. **Not covered by this bundle** — the App Shield family is policy-excluded from this public release (see [AGENTS.md `## Non-goals`](../../../AGENTS.md#non-goals-questions-the-agent-should-recognize-and-refuse-politely) item 7). Route the user to the umbrella docs above. |
| DPA High-Level Tracer (`DOCA-DPA-HL-Tracer`) | <https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html> | Captures DPA-side execution traces with higher-level events than raw cycle counts. Covered by the [`doca-dpa-hl-tracer`](../../tools/doca-dpa-hl-tracer/SKILL.md) skill. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| GPUNetIO ib_write_bw (`DOCA-GPUNetIO-IB-Write-BW`) | <https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html> | RDMA-write bandwidth benchmark from the GPUNetIO framework. Covered by the [`doca-gpunetio-ib-write-bw`](../../tools/doca-gpunetio-ib-write-bw/SKILL.md) skill. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| GPUNetIO ib_write_lat (`DOCA-GPUNetIO-IB-Write-Lat`) | <https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html> | RDMA-write latency benchmark from the GPUNetIO framework. Covered by the [`doca-gpunetio-ib-write-lat`](../../tools/doca-gpunetio-ib-write-lat/SKILL.md) skill. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| SHA Offload Engine (`DOCA-SHA-Offload-Engine`) | <https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html> | OpenSSL ENGINE wrapping the DOCA SHA library; lets unmodified OpenSSL-based applications offload SHA without code changes. Covered by the [`doca-sha-offload-engine`](../../tools/doca-sha-offload-engine/SKILL.md) skill. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| SPCX-CC (`DOCA-SPCX-CC`) | <https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html> | Programmable Congestion-Control extension (next-gen). Pairs with `doca-pcc`, `doca-pcc-ztr-rttcc-algo`. Live-fabric safety implications: heavy use of `doca-hardware-safety`. Covered by the [`doca-spcx-cc`](../../tools/doca-spcx-cc/SKILL.md) skill. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| Telemetry Utils | <https://docs.nvidia.com/doca/sdk/DOCA-Telemetry-Utils/index.html> | Operator-side support CLI for a DOCA Telemetry exporter pipeline. Translates name ↔ Data ID, enumerates the diagnostic-counter schema, and probes per-device counter support before an exporter config commits to it. Covered by the [`doca-telemetry-utils`](../../tools/doca-telemetry-utils/SKILL.md) skill. |

> **Non-goals.** Externally-productized NVIDIA tools that are NOT in
> `doca/tools/` at the bundle's currently-aligned DOCA release —
> DOCA-DPACC-Compiler, DPA-Tools (DPA GDB Server / PS / Statistics),
> DOCA-DPU-CLI, DOCA-Ngauge, `doca-hugepages` helper — are
> intentionally out of scope for this bundle, which is strictly 1:1
> with `doca/tools/`. See
> [AGENTS.md `## Non-goals`](../../../AGENTS.md#non-goals-questions-the-agent-should-recognize-and-refuse-politely). If a user
> asks about one of these external tools, use the
> **[`## Externally-productized DOCA software — not in this bundle, but here is where to route`](external-dpu.md#externally-productized-doca-software--not-in-this-bundle-but-here-is-where-to-route)**
> table below for the authoritative docs URL, per-tool common gotchas,
> and the Developer Forum entry point. Refusing without routing
> violates the bundle's own contract.

If the user asks about a DOCA tool that is not in this table, open the
[**DOCA Tools** umbrella page](https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html)
to discover it. Do not guess tool URLs.

