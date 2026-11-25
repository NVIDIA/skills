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
| Capabilities Print Tool (`doca_caps`) | <https://docs.nvidia.com/doca/sdk/DOCA-Capabilities-Print-Tool/index.html> | Prints DOCA devices and the per-library capabilities they support. Side-effect-free; safe to call early. Covered by the `doca-caps/SKILL.md` skill. |
| DOCA Bench | <https://docs.nvidia.com/doca/sdk/DOCA-Bench/index.html> | Performance evaluation harness for the built-in workload modes, including GPUNetIO benchmarks (which also need the DOCA Bench Extension package and GDRCopy). Covered by the `doca-bench/SKILL.md` skill. |
| Comm Channel Admin Tool | <https://docs.nvidia.com/doca/sdk/DOCA-Comm-Channel-Admin-Tool/index.html> | Admin CLI for Comch channels. Covered by the `doca-comm-channel-admin/SKILL.md` skill. |
| Flow Tune (`DOCA-Flow-Tune`) | <https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html> | Unified visibility / analysis / recommendation tool for live `doca-flow` pipelines. The artifact is ONE binary with TWO internal roles (server role: snapshots and exposes pipe / counter / KPI state through local IPC; client / consumer role: dumps, analyzes, visualizes, recommends parameter changes). The historical "Flow Tune Tool" and "Flow Tune Server" split lives INSIDE this artifact — there is one skill: `doca-flow-tune/SKILL.md`. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| PCC Counter | <https://docs.nvidia.com/doca/sdk/DOCA-PCC-Counter-Tool/index.html> | PCC counter inspection. Covered by the `doca-pcc-counters/SKILL.md` skill. |
| Perftest (`DOCA-Perftest`) | <https://docs.nvidia.com/doca/archive/3-5-0/doca-perftest/index.html> | RDMA bandwidth, message-rate and latency benchmark — point-to-point or multi-node, optional DOCA Verbs and GPUNetIO engines. Covered by the `doca-perftest/SKILL.md` skill. |
| Socket Relay | <https://docs.nvidia.com/doca/sdk/DOCA-Socket-Relay/index.html> | Socket relay between host and DPU. Covered by the `doca-socket-relay/SKILL.md` skill. |
| App Shield Config | <https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html> | Generates the host-OS profile / symbol files that App Shield needs to interpret host kernel state. **Not covered by this bundle** — the App Shield family is outside this bundle's scope (see AGENTS.md `## Non-goals` (bundle `AGENTS.md#non-goals-questions-the-agent-should-recognize-and-refuse-politely`) item 7). Route the user to the umbrella docs above. |
| SHA Offload Engine (`DOCA-SHA-Offload-Engine`) | <https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html> | OpenSSL ENGINE wrapping the DOCA SHA library; lets unmodified OpenSSL-based applications offload SHA without code changes. Covered by the `doca-sha-offload-engine/SKILL.md` skill. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| SPCX-CC (`DOCA-SPCX-CC`) | <https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html> | Spectrum-X congestion-control (SPC-X CC) tool. Pairs with `doca-pcc`. Live-fabric safety implications: heavy use of `doca-hardware-safety`. Covered by the `doca-spcx-cc/SKILL.md` skill. **No standalone SDK doc page in the public index today — use the umbrella above to discover.** |
| Telemetry Utils | <https://docs.nvidia.com/doca/sdk/DOCA-Telemetry-Utils/index.html> | Operator-side support CLI for a DOCA Telemetry exporter pipeline. Translates name ↔ Data ID, enumerates the diagnostic-counter schema, and probes per-device counter support before an exporter config commits to it. Covered by the `doca-telemetry-utils/SKILL.md` skill. |
| ngauge | <https://docs.nvidia.com/doca/archive/3-5-0/doca-ngauge/index.html> | Analyze, visualize, and debug single-node network performance by probing NIC hardware counters and storing collected data in HDF5 format. Covered by the `doca-ngauge` skill. |
| DOCA Pipeline Language Developer Tools | <https://docs.nvidia.com/doca/archive/3-5-0/doca-pipeline-language-developer-tools/index.html> | Use this skill to inspect and debug DOCA Pipeline Language programs on BlueField DPUs with DPL Nspect, DPL Debugger, DPL Admin, P4 Runtime Shell, and DPL Runtime Controller SDK. Covered by the `doca-pipeline-language-developer-tools` skill. |
| DOCA Telemetry Traceback Application | <https://docs.nvidia.com/doca/archive/3-5-0/doca-telemetry-traceback-application/index.html> | Use this skill to configure, build, run, and debug the DOCA Telemetry Traceback application for local telemetry monitoring and event logging on BlueField DPUs or ConnectX NICs. Covered by the `doca-telemetry-traceback-application` skill. |

> **Non-goals.** NVIDIA tools outside the DOCA SDK tools this bundle
> covers — DOCA-DPACC-Compiler, DPA-Tools (DPA GDB Server / PS /
> Statistics), DOCA-DPU-CLI, `doca-hugepages` helper — are
> intentionally out of scope for this bundle. See
> AGENTS.md `## Non-goals` (bundle `AGENTS.md#non-goals-questions-the-agent-should-recognize-and-refuse-politely`). If a user
> asks about one of these external tools, use the
> **[`## Externally-productized DOCA software — not in this bundle, but here is where to route`](external-dpu.md#externally-productized-doca-software--not-in-this-bundle-but-here-is-where-to-route)**
> table below for the authoritative docs URL, per-tool common gotchas,
> and the Developer Forum entry point. Refusing without routing
> violates the bundle's own contract.

If the user asks about a DOCA tool that is not in this table, open the
[**DOCA Tools** umbrella page](https://docs.nvidia.com/doca/sdk/DOCA-Tools/index.html)
to discover it. Do not guess tool URLs.

