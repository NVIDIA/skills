# DOCA Bench — Capabilities

**Where to start:** `doca_bench` is a single CLI binary (plus a
companion app for the remote half of remote-memory / RDMA / Eth
scenarios); the pattern overview below names the recurring
`doca_bench`-class questions. Pick the pattern first, then drill
into the H2 that owns the substance. For the *how* of executing
each pattern, jump to [TASKS.md](TASKS.md).

This file is loaded by [`SKILL.md`](SKILL.md). It documents *what
`doca_bench` is*, *what it can measure across DOCA libraries*,
*what versions it ships in*, *what its layered error and
observability surfaces look like*, and *the safety posture* the
public guide stakes out (notably the *"not for production
deployments"* warning on the companion-app channel). For
step-by-step invocations and the smoke-before-bulk workflow, see
[`TASKS.md`](TASKS.md).

## Pattern overview

Every `doca_bench`-class question this skill teaches resolves into
one of SIX patterns. The patterns are CLASSES — they apply across
every DOCA library bench can drive, not just one.

| `doca_bench` pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Pick the target library | Cross-library — bench can drive AES-GCM, Comch, Compress, DMA, EC, Eth, RDMA, SHA, GPUNetIO. Picking the target library is axis 1 of the three-axis configuration. | [`## Capabilities and modes`](#capabilities-and-modes) cross-library table + [TASKS.md ## configure](TASKS.md#configure) step 1 |
| 2. Pick the workload shape | The pipeline-of-steps model is axis 2 — what *operation* is the device being asked to do on each iteration (e.g. compress vs decompress, send vs receive, encrypt vs decrypt), with the data provider, batching, and remote-memory choices that go with it. | [`## Capabilities and modes`](#capabilities-and-modes) workload-shape rules + [TASKS.md ## configure](TASKS.md#configure) step 3 |
| 3. Pick the measurement axis | Throughput vs bulk-latency vs precision-latency vs max-bandwidth is axis 3 — the shipped `doca_bench` binary defines FOUR benchmark modes in `tools/bench/doca_bench/configuration.hpp`. The public guide documents the modes as **not interchangeable**; a single number reported without naming the mode is ambiguous and the agent must surface that. | [`## Capabilities and modes`](#capabilities-and-modes) measurement-axis table + [TASKS.md ## configure](TASKS.md#configure) step 4 |
| 4. Smoke-before-bulk | Confirm the bench can talk to the target device with a trivial workload (small job count, short duration, minimum data provider) before kicking off a long or swept run. | [TASKS.md ## run](TASKS.md#run) smoke flow + [TASKS.md ## test](TASKS.md#test) eval-loop overlay |
| 5. Diagnose a bench failure | Walk the layered error taxonomy in [`## Error taxonomy`](#error-taxonomy) — config-syntax / device-binding / library-precondition / workload-precondition / measurement-soundness / version / cross-cutting — instead of guessing at causes from a stack trace. | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |
| 6. Interpret a bench number | A bench number is only meaningful with the (command line + DOCA version + device + as-deployed environment) four-tuple. Quoting one without the other three is the cross-version regression-hunt failure mode. | [`## Observability`](#observability) + [TASKS.md ## test](TASKS.md#test) baseline-capture rule |

Two cross-cutting rules that apply to *every* pattern above:

- **`doca_bench` is cross-library, not single-library.** Recommendations
  that frame it as *"the doca_rdma benchmarker"* or *"the
  doca_compress benchmarker"* are categorically wrong; bench is
  the harness, the library is the unit under test. An agent
  picking one DOCA library and never naming the others is
  silently narrowing the answer.
- **Warm-up is part of the measurement, not part of the bug.**
  The public guide documents a warm-up period (a non-zero number
  of jobs run through the pipeline before measurement starts).
  Treating a first-iteration number as steady-state, or stripping
  the warm-up to *"get a faster answer"*, is the canonical
  measurement-soundness failure mode and the reason the bench
  ships the concept in the first place.

## Capabilities and modes

`doca_bench` is shipped as a single CLI binary at
`/opt/mellanox/doca/tools/doca_bench` on every DOCA install (host
or BlueField Arm) since DOCA 2.7.0, plus a separate **companion
app** that is part of the same install and provides the remote
half of remote-memory / RDMA / Eth scenarios. The two halves
communicate over an out-of-band channel (TCP/IP sockets or a DOCA
Comch channel per the public guide). There is no daemon, no
library to link against, and no programmatic API for the harness
itself — the entire interaction model is *configure the
invocation, run the binary, read the printed and/or CSV output*.

**Three-axis configuration model — the load-bearing concept.**
Every `doca_bench` invocation commits to a point in this space;
omitting any axis produces an ambiguous result.

| Axis | What it picks | Why the agent must name it |
| --- | --- | --- |
| 1. Target library | Which DOCA library is the unit under test (AES-GCM / Comch / Compress / DMA / EC / Eth / RDMA / SHA / GPUNetIO per the public guide's documented surface). | The cross-library scope is the bench's defining feature; an answer that picks one library and never names the others has silently narrowed the bench to a single-library tool. |
| 2. Workload shape | The pipeline-of-steps the device runs each iteration — direction (e.g. send vs receive for RDMA / Eth, encrypt vs decrypt for AES-GCM, compress vs decompress for COMPRESS), the data provider (file / file-set / random-data per the public guide), batching, and whether remote-memory input / output is in play. | Two runs against the *same* library can report wildly different numbers if the workload shape differs; quoting "DOCA Compress throughput" without naming compress-vs-decompress, the data provider, and the buffer sizing is ambiguous. |
| 3. Measurement axis | Throughput vs bulk-latency vs precision-latency vs max-bandwidth per the four documented operating modes (the shipped binary's `benchmark_mode` enum has four values; the public guide enumerates them). | The modes are explicitly **not interchangeable** — precision-latency disables batching, bulk-latency uses bucketing, throughput maximizes pipeline occupancy, max-bandwidth holds the pipeline at the saturation point. Comparing a precision-latency number to a throughput-mode number (or a max-bandwidth number to either) is the cross-mode apples-to-oranges failure. |

**Cross-library scope — what bench can drive.** The public DOCA
Bench guide enumerates the supported DOCA libraries explicitly;
the bench's own built-in query system reports which subset is
*actually installed and exposed on the running host* (the
"granular build support" model). The library set on any given
install is the intersection of "documented by bench" and
"installed by this DOCA package selection" — the agent must
probe, not assume.

**Pipeline of steps — the workload primitive.** Bench expresses
a workload as a pipeline of steps, where each step is a
documented operation on a DOCA library (e.g. an Ethernet
receive, a SHA hash, a compress). The public guide reports that
DOCA currently supports running a single pipeline at a time per
invocation. Multi-step pipelines (e.g. receive → decompress →
hash) are explicitly in scope; multi-pipeline-per-invocation is
out of scope and the agent must not propose it.

**Multi-core / multi-thread scaling.** Bench creates execution
threads with CPU affinities the operator picks (the public guide
documents `--core-mask`, `--core-list`, `--core-count`,
`--threads-per-core` as the family — exact flag names should be
re-confirmed against `--help` on the installed version). Each
thread independently runs the configured pipeline against its
own jobs pool. The number of cores / threads and their NUMA
placement is part of axis 2 (workload shape) for any number a
later run is going to be compared against.

**Operating modes.** Per the public guide:

| Mode | What it optimizes for | What it sacrifices |
| --- | --- | --- |
| `throughput` (default) | Maximum pipeline occupancy and aggregate bandwidth / op-rate | Per-job latency precision; results are aggregate, not per-job |
| `bulk-latency` | A balance — submits batches, measures the time from first-submit to last-complete per batch, uses bucketing to report latency distribution | Per-job precision; latency is per-batch + bucketed, not per-job |
| `precision-latency` | Per-job latency precision (min / max / median / percentiles) by submitting one job at a time | Throughput; the pipeline is deliberately serialized and capacity is wasted by design |
| `max-bandwidth` | Pushes the workload at maximum sustainable bandwidth — separate operating mode from `throughput` in the parser (`tools/bench/doca_bench/configuration.hpp` defines four `benchmark_mode` enum values: `throughput`, `bulk_latency`, `precision_latency`, **`max_bandwidth`**; the CLI parser at `tools/bench/doca_bench/impl/configuration_parser.cpp` accepts the kebab-case spelling `max-bandwidth`). Use when the question is about the achievable bandwidth ceiling rather than aggregate op-rate. | Per-job latency precision; bandwidth-pinned runs deliberately hold the pipeline at the saturation point. |

**Built-in query system.** Bench ships a documented query
surface that reports which DOCA libraries are *installed* on
the running host and the supported sweep attributes. The agent
treats the query output as the install-side ground truth for
*"is library X benchable on this box"*; absence in the query
output is the canonical *"granular build does not include X"*
answer.

**Companion app.** Remote-memory / RDMA / Eth scenarios require
the companion app on the far side, communicating with the bench
host over the documented out-of-band channel (TCP/IP sockets or
DOCA Comch). The agent's rule for remote scenarios: a companion
app is not optional — bench will not synthesize the remote half
on its own.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way
match rule, NGC container semantics, and the headers-win-over-docs
rule, see [`doca-version`](../../doca-version/SKILL.md). The body
lives there; this skill does not duplicate it.

**The `doca_bench`-specific overlay** is:

- **`doca_bench` is available since DOCA 2.7.0** per the public
  DOCA Bench guide's prerequisites section. On older installs
  the binary is not present; the right answer for *"I can't
  find doca_bench"* is to confirm the installed version per
  [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure)
  and, if `< 2.7.0`, route to
  [`doca-setup`](../../doca-setup/SKILL.md) for an upgrade
  rather than recommending alternative tools.
- **Granular build means the *available library* set is
  install-specific.** The public guide documents granular build
  support — bench probes the install and only exposes libraries
  that are actually present. A flag / scenario reachable on one
  install can be silently absent on another even at the same
  DOCA version. The agent's rule: re-run the built-in query
  per [`TASKS.md ## configure`](TASKS.md#configure) step 2
  before quoting which libraries are benchable.
- **Companion app version must match the bench version.** Bench
  and the companion app are shipped together; mixing a bench
  binary from one DOCA version with a companion app from
  another is unsupported and falls into the partial-install
  layer of [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).
- **Output format stability is not contractually frozen.** The
  documented operating modes are stable; the exact textual
  layout of stdout, the CSV column order, and the screen-mode
  histogram rendering can shift across releases. Agents that
  need to consume bench output programmatically should prefer
  the structured helper per
  [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md#schemas)
  when present and re-verify the textual layout against the
  user's installed version when absent.
- **Per-platform support matrix per the public guide.** The
  documented BlueField generation support varies per library
  (e.g. some library operations are documented for BlueField-3
  but not BlueField-2). Do not copy a library-availability
  claim from one BlueField generation to another; re-read the
  public matrix per
  [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools).

## Error taxonomy

`doca_bench`'s error surface is broader than `doca_caps` because
the tool *does* configure devices, allocate buffers, drive
hardware, and produce measured numbers — each of which has its
own failure mode. The error layers the agent should distinguish,
in escalating order:

1. **Config-syntax.** The invocation itself does not parse:
   unknown flag, malformed value (e.g. a unit suffix bench does
   not accept), missing required argument for the chosen mode,
   conflicting flags. Cause: the operator wrote a flag string
   that does not exist in `--help` on the installed version
   (often a flag taken from prose / blog / older release).
   Routing: re-read `--help` on the installed binary, and the
   public DOCA Bench guide via
   [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools);
   do not guess.
2. **Device-binding.** Invocation parses; bench cannot bind the
   target device. Cause: device PCIe address / IB name /
   interface name does not exist on this host, NUMA placement
   of the chosen cores is wrong, or the underlying driver stack
   (`mlx5_core`, IB stack, etc.) is not loaded. Routing: first
   verify the device is visible to DOCA at all via
   [`doca-caps ## run`](../doca-caps/TASKS.md#run) and
   [`doca-setup ## test`](../../doca-setup/TASKS.md#test); only
   re-attempt the bench invocation once the device is on the
   capability snapshot.
3. **Library-precondition.** Device bound; bench refuses to
   exercise the requested DOCA library on this install. Cause:
   the granular-build setup did not include the library (its
   `pkg-config` module is missing, its samples are missing);
   the library is present but the device does not support the
   requested operation per its capability surface; the library
   is present but the BlueField generation is outside the
   public guide's per-platform support matrix. Routing: bench's
   own query system per [`TASKS.md ## configure`](TASKS.md#configure)
   step 2 + the per-library skill (e.g.
   [`doca-comch`](../../libs/doca-comch/SKILL.md),
   [`doca-compress`](../../libs/doca-compress/SKILL.md)) for
   the library-internal capability rules.
4. **Workload-precondition.** Library exercisable; the workload
   shape is invalid for the library. Cause: a data provider
   the library does not accept (e.g. random-data fed to a
   decompression scenario that needs structured input), a
   buffer / job sizing the device does not support, a remote-
   memory choice without the companion app on the far side, a
   pipeline-of-steps the library cannot chain. Routing: re-walk
   axis 2 (workload shape) of the three-axis model in
   [`## Capabilities and modes`](#capabilities-and-modes);
   bench will not silently substitute a workload, and the
   agent must not either.
5. **Measurement-soundness.** The run completes and reports
   numbers, but the numbers are unsound and must not be quoted
   as-is. Three sub-layers, all documented by the public guide:
    - *Warm-up not applied / too short.* Reported numbers
      include cold-cache / cold-pipeline iterations and are
      lower than steady-state. Fix: confirm the warm-up
      configuration matches the public guide's documented
      default and is appropriate for the chosen library /
      mode.
    - *Steady-state not reached.* Run duration / job count is
      too small for the pipeline to settle, and the reported
      number is in the transient region. Fix: lengthen the
      run via the documented duration / job-count limits and
      re-iterate per [`TASKS.md ## test`](TASKS.md#test).
    - *Outliers / distribution unreported.* A single
      throughput average hides a heavy tail; a precision-
      latency mean hides a 99.99-percentile spike that the
      consumer workload will actually feel. Fix: report the
      distribution (the bulk-latency histogram or the
      precision-latency percentile breakdown) alongside any
      single number.
6. **Version.** Cross-cutting partial-install / mixed-version
   layer per [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).
   Symptoms: bench binary version disagrees with `doca_caps
   --version` or `pkg-config --modversion doca-common`,
   companion-app version disagrees with the bench-host version,
   public guide version the operator is reading disagrees with
   the install. Routing: walk
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   before any further investigation.
7. **Cross-cutting.** The cause is below DOCA — driver /
   firmware / NUMA / hugepages / OS. Symptoms that do not fit
   layers 1-6 (e.g. throughput numbers that fall sharply only
   on one NUMA node, latency spikes correlated with kernel-
   thread scheduling, throughput tied to firmware version
   independent of DOCA version). Routing: hand off to
   [`doca-debug ## debug`](../../doca-debug/SKILL.md) and
   [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug);
   the bench surface has reached its limit.

`doca_bench` does not *itself* participate in the cross-library
`DOCA_ERROR_*` taxonomy that DOCA libraries return through their
C API; bench is a CLI driving libraries, not a library call. For
the cross-library `DOCA_ERROR_*` taxonomy and the program-side
debug order, see
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).

## Observability

`doca_bench`'s observability surface is **the measurement output
itself**, plus the built-in query system. Specifically:

- **Stdout summary.** Every run prints an aggregate summary on
  stdout per the public guide — the documented examples include
  duration, enqueued / dequeued job counts, throughput / ingress
  / egress rates for throughput mode, and a histogram + bucket
  breakdown for bulk-latency mode, and min / max / median / mean
  / percentile rows for precision-latency mode. The exact
  textual layout is install-specific; re-verify against the
  user's run.
- **CSV output.** The public guide documents a CSV output path
  with options to append, separate dynamic values, and emit
  environment information alongside the numbers. CSV is the
  documented machine-readable surface — the agent's rule for
  any baseline that will be re-read by a later run: capture
  CSV, not just stdout.
- **Real-time stats interval.** The public guide documents a
  real-time stats interval so the operator can see the run
  progress without waiting for completion. Useful when
  diagnosing whether a long run has reached steady state.
- **Query subsystem.** The bench's `Queries` section in the
  public guide documents a built-in mechanism for reporting
  device capabilities and supported sweep attributes. This is
  the install-side ground truth for *"can this run even
  attempt library X"* before committing to a full run.
- **Reported invocation echo.** Per the public guide, bench
  echoes the full set of configured values at the start of the
  run so a captured log self-documents the (command line +
  effective defaults) the numbers belong to. The agent must
  preserve this echo in any captured baseline; it is the
  *"what command produced this number"* leg of the four-tuple
  in [`## Pattern overview`](#pattern-overview) pattern 6.

For the cross-cutting env-side observability primitives
(representor enumeration, `devlink dev show`, `mlxconfig`) see
[`doca-setup CAPABILITIES.md ## Observability`](../../doca-setup/CAPABILITIES.md#observability).
For the program-side observability surface (DOCA log levels,
`DOCA_LOG_LEVEL`, `--sdk-log-level`) see
[`doca-programming-guide CAPABILITIES.md ## Observability`](../../doca-programming-guide/CAPABILITIES.md#observability).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

`doca_bench` is a measurement tool and a more powerful surface
than `doca_caps`; it *does* allocate buffers, bind devices, and
drive hardware, including remote-memory operations over an
out-of-band channel. The safety rules:

- **Not for production deployments.** The public guide carries
  an explicit warning that the bench is not intended for
  production deployment, and that the companion-app out-of-band
  channel can carry sensitive information. The agent must
  surface this warning whenever the user proposes running bench
  on a host that also serves production traffic, and must not
  recommend running the companion app over an untrusted network
  segment without the public guide's secure-channel guidance.
- **Smoke-before-bulk; never run a long sweep first.** A swept
  run on the wrong device, wrong workload, or wrong mode
  consumes hours and produces unusable data. The agent's rule
  is the [`TASKS.md ## run`](TASKS.md#run) smoke step (trivial
  workload, short duration) before any sweep or long run.
- **Quote the (command line + version + device + environment)
  four-tuple, not just the number.** A bench number quoted
  without the four-tuple is unreplicable and unfalsifiable.
  This rule applies to every output of this skill — the *most
  common* downstream misuse of bench is quoting a screenshot
  from one platform as if it described another.
- **Do not invent flags, scenario names, attribute names, or
  metric names.** The documented invocations and the installed
  `--help` are the authoritative surface. Prose-derived flags
  are the most common hallucination failure for this skill;
  see the cross-cutting rule in
  [`TASKS.md ## Command appendix`](TASKS.md#command-appendix).
- **Host vs BlueField Arm execution rule.** Per the public
  guide the binary is the same on both sides; the *measured*
  numbers differ because the environment differs. An agent
  comparing a host-side number to a BlueField-Arm-side number
  without naming where each ran is making the cross-platform
  apples-to-oranges mistake.

## Public-source pointer

The single canonical public source for `doca_bench` is the
**DOCA Bench** page on `docs.nvidia.com`, reachable through
[`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools).
Do not invent flags, scenario names, attribute names, metric
names, or supported library entries beyond what that page
documents — and re-verify against `--help` on the user's
installed binary, since granular-build support means the
*available* surface is install-specific within the *documented*
surface.
