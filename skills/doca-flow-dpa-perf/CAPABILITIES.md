# DOCA Flow DPA Perf — Capabilities

**Where to start:** `doca_flow_dpa_perf` measures the DPA-offloaded
Flow path's rule update / disable rate; the pattern overview below
names the recurring `doca_flow_dpa_perf`-class questions. Pick the
pattern first, then drill into the H2 that owns the substance. For
the *how* of executing each pattern, jump to [TASKS.md](TASKS.md).
For the `doca-flow` API behind the pipeline this tool drives, see
[`doca-flow CAPABILITIES.md`](../../libs/doca-flow/CAPABILITIES.md);
for the DPA programming model behind the device-side execution
engine, see
`doca-dpa CAPABILITIES.md`.

This file is loaded by [`SKILL.md`](SKILL.md). It documents *what
`doca_flow_dpa_perf` is*, *what it measures (DPA path) vs what
it doesn't (host / DPU-CPU path)*, *what hardware preconditions
gate it*, *what versions it ships in*, *what its layered error
and observability surfaces look like*, and *the safety posture*
its hardware-driving role forces.

## Pattern overview

Every `doca_flow_dpa_perf` question this skill teaches resolves
into one of SIX patterns. The patterns are CLASSES — they apply
across every DPA-capable device and workload class, not one
specific board.

| `doca_flow_dpa_perf` pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Pick the right tool | DPA-vs-host path is the load-bearing decision. This tool measures the DPA-offloaded Flow update / disable path; [`doca-flow-perf`](../doca-flow-perf/SKILL.md) measures the host / DPU-CPU Flow path. Quoting a number without naming the tool that produced it is the cross-tool apples-to-oranges failure. | [`## Capabilities and modes`](#capabilities-and-modes) DPA-vs-host bullet + [TASKS.md ## configure](TASKS.md#configure) |
| 2. Check DPA preconditions | The DPA-Provider library uses the DPA device. Hardware is gated: ConnectX-7 and above (ConnectX-8 recommended-or-later per the shipped README), BlueField-3. SFs are not supported on the DPA path. VNF Flow mode is required per the shipped README. | [`## Capabilities and modes`](#capabilities-and-modes) device-preconditions table + [TASKS.md ## configure](TASKS.md#configure) |
| 3. Pick the active / passive device split | Two-port boards run the active device for steering + the passive device for outgoing traffic; one-port boards run active only. Picking the wrong split is a configuration error, not a measurement error. | [`## Capabilities and modes`](#capabilities-and-modes) device-split bullet + [TASKS.md ## configure](TASKS.md#configure) |
| 4. Pick the workload-shape axes | Burst size, queue size, completion threshold, number of PSL tables, table size, number of workers, hash pipe algorithm, work policy, operation (update vs disable). Each axis is documented in the shipped README on the user's install and in the public guide; picking one set silently narrows the answer. | [`## Capabilities and modes`](#capabilities-and-modes) workload-shape table + [TASKS.md ## configure](TASKS.md#configure) |
| 5. Smoke before bulk | Confirm a single small-operation run completes with sensible iteration stats (median ≈ max, low standard deviation, warmup applied if enabled) before kicking off a long run or a parameter sweep. | [TASKS.md ## run](TASKS.md#run) smoke flow + [TASKS.md ## test](TASKS.md#test) eval loop |
| 6. Diagnose zero-ops / hung / failed self-test | Walk the layered error taxonomy in [`## Error taxonomy`](#error-taxonomy) — config-syntax → device-binding → dpa-precondition → workload-precondition → measurement-soundness → self-test → version → cross-cutting — instead of guessing. | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **DPA-path vs host-path is the load-bearing decision.** A
  Kops/sec number from `doca_flow_dpa_perf` and a Kops/sec
  number from `doca_flow_perf` are not the same number — they
  measure different paths through different processors. An
  agent that reports either without naming which tool produced
  it has silently conflated two surfaces.
- **DPA preconditions are HARD, not soft.** The DPA path is
  not available on every DOCA install; it requires DPA-capable
  hardware, VNF Flow mode, and (per the shipped README)
  PF or VF — never SF — for the device target. The right
  response to *"the tool reports zero ops"* on a BlueField-2 is
  *"this hardware is not DPA-capable"*, not a parameter sweep.

## Capabilities and modes

`doca_flow_dpa_perf` is shipped as a **single CLI binary** on
every DOCA install where the Flow DPA Perf package profile is
present. The tool has a host-side control half (per the shipped
`flow_dpa_perf.c` / `flow_dpa_perf_core.c`) and a DPA-side
device half (per the shipped `dpa_side.c` and the DPA-side
attributes the build defines) — but the operator interacts with
*one binary*. There is no separately-shipped DPA worker daemon
the operator runs.

### DPA-vs-host path — the load-bearing boundary

The DPA-Provider library (per `doca-flow`) is what lets a
`doca-flow` application offload its rule update / disable path
to the DPA execution engine on a DPA-capable device. This tool
measures *that* path specifically.

| Surface | What it measures | When to reach for it |
| --- | --- | --- |
| **`doca_flow_dpa_perf`** (this skill) | The update / disable rate of the DPA-offloaded Flow rule path on a DPA-capable device | Question is *"how fast can the DPA program path-selector entries"* on hardware that supports it |
| [`doca-flow-perf`](../doca-flow-perf/SKILL.md) | The throughput / latency / rule-install rate of the host / DPU-CPU Flow path | Question is *"how fast does the host / DPU CPU drive the Flow library"*; works on a broader hardware set |

The downstream rule: name which tool produced which number.
Two Kops/sec numbers without a tool name are not comparable.

### Device preconditions

The Flow DPA Provider uses the DPA device; the shipped README
states the hardware support set explicitly. The agent's rule:
*do not invoke this tool on hardware outside the supported set;
the right response is to route to host-side
[`doca-flow-perf`](../doca-flow-perf/SKILL.md) instead*.

| Hardware class | Supported by this tool? | Notes from the shipped README |
| --- | --- | --- |
| ConnectX-7 | Supported, lower bound | Tool will run; ConnectX-8 is the recommended-or-later starting point per the shipped README |
| ConnectX-8 | Supported, recommended starting point | The shipped README documents ConnectX-8 as the first generation recommended for routine use |
| ConnectX-9 | Supported | Single-port operation documented in the shipped README example invocations |
| BlueField-3 | Supported | Two-port operation documented in the shipped README example invocations |
| BlueField-2 | NOT supported | The DPA-Provider library does not run on BlueField-2; the right tool here is host-side `doca-flow-perf` |
| Earlier ConnectX / BlueField | NOT supported | Same as above |

Two additional preconditions per the shipped README:

- **VNF Flow mode is required.** The tool only supports VNF
  mode; switch / SWITCH-EXPRESS modes are out of scope.
- **PF recommended; VF works; SF does NOT.** Per the shipped
  README, *"the DPA devices will not run if SFs are used"*.
  Recommend PF; accept VF; refuse SF.

### Active / passive device split

The tool runs a steering pipeline on the **active device** and
optionally sends outgoing traffic out the **passive device**
(two-port mode). On a single-port board (e.g. the shipped
ConnectX-9 example), only the active device is configured and
traffic is dropped on the steering pipeline.

The shipped README documents the invocation shapes for each:

- **BlueField-3 (two-port):** active + passive PCI device pair.
- **ConnectX-8 (two-port):** active + passive PCI device pair.
- **ConnectX-9 (one-port):** active device only.

The agent must commit to the split before invoking the tool;
running a two-port invocation on a one-port board is a
[`## Error taxonomy`](#error-taxonomy) layer 2 (device-binding)
failure.

### Workload-shape axes

Per the shipped README, the tool's workload is parameterized
along the following axes. The class is the load-bearing piece;
specific defaults belong to the README on the user's install,
not to this skill.

| Axis | What it picks | Why naming it matters |
| --- | --- | --- |
| **Operation** | Update vs disable-enable. The two operations exercise different DPA-side paths. | A *"DPA Kops/sec for update"* number and a *"DPA Kops/sec for disable"* number are not the same number. |
| **Burst size** | Number of update / disable WQE requests per burst. | Must be a divisor of `num_tables * table_size`; the README documents the divisibility rule. |
| **Queue size** | Capacity of the DPA-side send queue. | Bounds the worker's outstanding-request count before it must wait for a completion. |
| **Completion threshold** | Number of operations after which a completion is polled. | The README defaults this to `queue_size / 2`; smaller thresholds raise polling overhead, larger ones risk queue stall. |
| **Number of PSL tables (per worker)** | Number of path-selector pipes per DPA worker. | Sizes the worker's working set; default is 1 per the shipped README. |
| **PSL table size** | Number of entries per PSL table. | Sizes the per-pipe entry count; default is 256 per the shipped README. |
| **Number of workers** | Number of DPA worker threads. | Supported range is 1-8 per the shipped README. |
| **Hash pipe algorithm** | Random vs identity per the shipped README. | The identity algo only supports update; the random algo supports update / disable / enable. |
| **Work policy** | Sequential vs random per the shipped README. | The README notes sequential is ~4% better due to memory locality; the choice should match the workload being characterized. |
| **Self-test** | When enabled, the tool replaces one path-selector value with a sentinel and pauses for traffic-side verification per tcpdump. | Useful for end-to-end correctness; a separate concern from the throughput measurement. |
| **Warmup** | A short warmup round excluded from time measurements. | Disabled by default per the shipped README; enabling it surfaces a steadier number. |
| **Pipeline policy** | Entries vs resources per the shipped README. | Determines which DOCA Flow pipeline policy the tool exercises. |

For exact flag names, default values, and ranges, read the
shipped `README.md` under the tool's source on the user's
install plus `--help` on the installed binary; do not invent
values from prose.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way
match rule, NGC container semantics, and the headers-win-over-
docs rule, see [`doca-version`](../../doca-version/SKILL.md).
The body lives there; this skill does not duplicate it.

**The `doca_flow_dpa_perf`-specific overlay** is:

- **The tool rides the `doca-flow` and `doca-dpa` library
  versions it links against.** The binary, the Flow library
  the install ships, and the DPA programming runtime the
  install ships must all come from the same DOCA install.
  Cross-train pairings are partial-install hazards per
  [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility);
  the right answer for *"the tool runs but reports
  implausible numbers"* is to confirm the four-way match
  first per
  [`doca-version TASKS.md ## test`](../../doca-version/TASKS.md#test).
- **The DPA execution engine is firmware-version-sensitive.**
  Per the shipped README, the LAG / multi-port-Eswitch
  preconditions named there are firmware-gated, and DPA-side
  workloads exercise firmware code paths that move across
  releases. Do not copy a Kops/sec number from one firmware
  band to another without re-running the smoke.
- **Per-platform support varies.** ConnectX-7 is the lower
  bound; BlueField-3 is supported; BlueField-2 is not. The
  agent must consult the public DOCA Flow DPA Perf page and
  the shipped README on the user's install before quoting a
  hardware support claim.
- **Output format stability is not contractually frozen.** The
  stdout layout for iteration statistics is documented in the
  shipped README's example output, but the exact field names
  and column ordering can shift across releases. Agents that
  consume the output programmatically should re-verify against
  the user's installed binary.

## Error taxonomy

`doca_flow_dpa_perf`'s error surface is broader than a pure
read-only tool because the binary configures devices, allocates
buffers, drives the DPA engine, and produces measured numbers
each of which has its own failure mode. The error layers the
agent should distinguish, in escalating order:

1. **Config-syntax.** Invocation does not parse: unknown flag,
   malformed value, missing required argument, conflicting
   flags. Cause: the operator wrote a flag string that does
   not exist in `--help` on the installed version. Routing:
   re-read `--help`, the shipped README, and the public DOCA
   Flow DPA Perf guide via
   [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools);
   do not guess.
2. **Device-binding.** Invocation parses; the tool cannot
   bind the active / passive devices. Cause: the PCI address
   does not exist, the operator gave a two-port invocation
   to a one-port board, the device is SF (unsupported on
   DPA per the shipped README), or the driver stack is not
   loaded. Routing: confirm the device is visible to DOCA at
   all via [`doca-caps ## run`](../doca-caps/TASKS.md#run);
   confirm the device class is in the supported set per
   [`## Capabilities and modes`](#capabilities-and-modes);
   confirm the `mlx5_core` driver stack is loaded per
   [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug).
3. **DPA-precondition.** Device bound; the DPA path refuses
   to start. Cause: the device is not DPA-capable
   (BlueField-2, older ConnectX), VNF Flow mode is not
   active, the DPA execution resources are not available,
   or the firmware band does not satisfy the LAG / multi-
   port-Eswitch preconditions the shipped README documents.
   Routing: re-read the shipped README's setup requirements
   section; cross-check with
   `doca-dpa CAPABILITIES.md ## Capabilities and modes` for the DPA-side
   preconditions; do not invent a workaround.
4. **Workload-precondition.** DPA is exercisable; the
   workload shape is invalid. Cause: a burst size that is
   not a divisor of `num_tables * table_size`, a queue size
   that is not a power of 2, a workers count outside the
   1-8 range per the shipped README, an identity hash pipe
   algo combined with a disable operation it does not
   support. Routing: re-walk axis 4 (workload-shape) of the
   model in
   [`## Capabilities and modes`](#capabilities-and-modes)
   and the README's per-axis rules.
5. **Measurement-soundness.** The run completes and reports
   numbers, but they are unsound and must not be quoted
   as-is. Three sub-layers:
    - *Warmup not applied / too short.* First iteration is
      cold-cache / cold-pipeline; the median is below
      steady-state. Fix: enable warmup per the README;
      re-run.
    - *Iteration statistics show high standard deviation.*
      The run is in a transient region or under contention.
      Fix: re-run with a larger `num-iterations` per the
      README; investigate concurrent workload on the device.
    - *Self-test path not exercised.* When the user's
      question is end-to-end correctness, a throughput-only
      run is insufficient. Fix: enable self-test and follow
      the tcpdump-side verification step per the shipped
      README.
6. **Self-test.** The run reports numbers but the optional
   self-test step fails. Cause: traffic was not sent to the
   device during the self-test pause, the path-selector
   sentinel value did not appear on the wire, or the disable
   step did not take effect. Routing: re-walk the README's
   self-test section; confirm the operator's traffic
   generator targets the configured ports and is running
   during the pause.
7. **Version.** Cross-cutting partial-install / mixed-
   version layer per
   [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).
   Symptoms: tool version disagrees with
   `pkg-config --modversion doca-flow`, DPA programming
   runtime version disagrees with the tool's expectations,
   firmware band changed underneath the install. Routing:
   walk [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   before any further investigation.
8. **Cross-cutting.** The cause is below DOCA — driver,
   firmware, NUMA, BlueField mode, hugepages. Routing:
   hand off to [`doca-debug ## debug`](../../doca-debug/SKILL.md)
   and [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug);
   the tool's surface has reached its limit.

`doca_flow_dpa_perf` does not itself participate in the
cross-library `DOCA_ERROR_*` taxonomy in the way an
application-linked library does; the tool is a CLI driving
libraries. For the cross-library `DOCA_ERROR_*` family and the
program-side debug order, see
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).

## Observability

`doca_flow_dpa_perf`'s observability surface is **the
measurement output itself**, supplemented by the optional
self-test path. Specifically:

- **Iteration statistics.** Per the shipped README's example
  output, every run prints per-iteration timings plus an
  *"Overall time measurement statistics"* block with average,
  median, max, and standard deviation. The agent's rule: quote
  median + max + stddev; the average alone hides outliers.
- **Kops/sec rate.** Derived from the median round time and
  the configured `num-operations`. The shipped README example
  output is *illustrative* (e.g. "793 Kops/sec" on the
  example platform) — not a baseline this skill quotes.
- **Self-test path verification.** When self-test is enabled,
  the tool replaces one path-selector value with the sentinel
  `65432` and pauses for traffic-side verification via
  tcpdump per the README. Useful as an end-to-end correctness
  signal; the README documents the tcpdump invocation shape.
- **Run echo.** The tool prints its full parameter set at run
  start per the README ("Tool parameters are: ..."); preserve
  this echo in any captured baseline so the (command line +
  effective defaults) pair is documented.
- **Operator-side env primitives.** The tool does not emit
  cross-cutting env metrics of its own; those live in
  [`doca-setup CAPABILITIES.md ## Observability`](../../doca-setup/CAPABILITIES.md#observability)
  (representor enumeration, `devlink dev show`, `mlxconfig`).
  Capture them alongside the tool's output for downstream
  diagnosis.

For the program-side observability surface
(`DOCA_LOG_LEVEL`, `--sdk-log-level`) see
[`doca-programming-guide CAPABILITIES.md ## Observability`](../../doca-programming-guide/CAPABILITIES.md#observability).
For the DPA-side execution-engine observability see
`doca-dpa CAPABILITIES.md ## Observability`.

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

`doca_flow_dpa_perf` *does* drive hardware: it binds the active
(and optionally passive) device, allocates DPA execution
resources, and programs path-selector entries. The safety rules:

- **Not for production deployments.** The shipped README plus
  the public DOCA Flow DPA Perf guide position the tool as a
  performance characterization tool, not a production
  workload. The agent must surface this whenever a user
  proposes running it on a host that also carries production
  traffic.
- **Smoke-before-bulk; never sweep first.** A swept run on
  the wrong workload-shape axis consumes hours and produces
  unusable data. The agent's rule is the
  [`TASKS.md ## run`](TASKS.md#run) smoke step (small
  `num-operations`, small `num-iterations`, warmup enabled)
  before any sweep.
- **Quote the four-tuple AND the tool name.** A
  `doca_flow_dpa_perf` number is only meaningful with the
  (command line + DOCA version + device + as-deployed
  environment) tuple AND the tool name. Comparing a number
  from this tool to a number from
  [`doca-flow-perf`](../doca-flow-perf/SKILL.md) without
  naming both tools is the cross-tool apples-to-oranges
  failure.
- **Do not invent flags or default values.** The documented
  invocations and the shipped README on the user's installed
  version are the joint authoritative surface; prose-derived
  flag strings are the most common hallucination failure for
  this skill.
- **SFs are out of scope; do not workaround.** Per the
  shipped README, the DPA path does not run if SFs are used.
  The right answer for *"my device target is an SF"* is to
  switch to PF or VF, not to invent a workaround.

## Public-source pointer

The single canonical public source for `doca_flow_dpa_perf` is
the **DOCA Flow DPA Perf** page on `docs.nvidia.com`, reachable
through
[`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools).
The second source on the user's install is the shipped
`README.md` under the tool's source tree. Do not invent flags,
default values, workload-axis identifiers, or output column
names beyond what those pages document — and re-verify against
`--help` on the user's installed binary, since granular-build
support means the *available* surface is install-specific
within the *documented* surface. For the Flow library behind
the pipeline this tool drives, see
[`doca-flow`](../../libs/doca-flow/SKILL.md); for the DPA
programming model behind the execution engine, see
`doca-dpa`.
