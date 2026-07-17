---
license: Apache-2.0
name: doca-bench
description: >
  Use this skill when the user is running `doca_bench` (DOCA ≥
  2.7.0) — the cross-library micro-benchmark harness — to measure
  throughput, bulk-latency, precision-latency, or max-bandwidth
  (the four `benchmark_mode` enum values the binary ships) of a
  DOCA library (RDMA, COMPRESS, AES-GCM, SHA, DMA, EC, ETH,
  Comch, GPUNetIO) on host or BlueField Arm, probe the granular-
  build query for which libraries the install exposes, capture a
  baseline four-tuple (command, version, device, environment), or
  diagnose a bench failure in the config-syntax, device-binding,
  library/workload-precondition, or measurement-soundness layer.
  Trigger even without the word "doca_bench" — typical implicit
  forms include "how fast is compress on my BlueField", "what
  RDMA throughput can this NIC do", "measure AES-GCM latency",
  or "baseline before a firmware update". Refuse and route
  elsewhere for application-level end-to-end timing, custom
  benchmark programs, DOCA install/upgrade, or patching the
  binary.
metadata:
  kind: tool
compatibility: >
  Requires DOCA SDK ≥ 2.7.0 installed at /opt/mellanox/doca on
  Linux (Ubuntu 22.04/24.04 or RHEL/SLES) with a BlueField DPU or
  ConnectX NIC attached and the `doca_bench` binary present at
  /opt/mellanox/doca/tools/doca_bench. Companion app must run on
  the far side for remote-memory / RDMA / Eth scenarios; host and
  BlueField-Arm execution both supported.
---

# DOCA Bench (`doca_bench`)

**Where to start:** This is a tool skill for invoking `doca_bench`,
the cross-library micro-benchmark harness. Open
[`TASKS.md`](TASKS.md) and start at
[`## configure`](TASKS.md#configure) for the three-axis decision
(target library × workload shape × measurement axis), then
[`## run`](TASKS.md#run) for the smoke-before-bulk flow. Open
[`CAPABILITIES.md`](CAPABILITIES.md) when the question is *what
`doca_bench` can measure*, *which DOCA libraries it can drive*, or
*how to interpret throughput / latency / op-rate output without
fooling yourself on warm-up or steady-state*. If DOCA is not
installed yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first; if the install
version is < 2.7.0, `doca_bench` is not shipped on this host.

## Example questions this skill answers well

The CLASSES of `doca_bench` questions this skill is built to answer,
each with one worked example. The class is the load-bearing piece;
the worked example is one instance.

- **"What does this DOCA library actually deliver on this device?"** —
  worked example: *"throughput of DOCA Compress on my BlueField-3"*.
  Answered by the three-axis configuration in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the smoke-before-bulk flow in
  [`TASKS.md ## run`](TASKS.md#run). The *same* shape answers
  *"send-side throughput of DOCA RDMA"* — `doca_bench` is
  cross-library, not single-library.
- **"Which DOCA libraries can `doca_bench` actually drive on this
  install?"** — worked example: *"is doca_sha enumerable on a
  granular-build install"*. Answered by the built-in query system
  surfaced in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + [`TASKS.md ## configure`](TASKS.md#configure) step 2
  (probe-before-bench). Empty enumeration = library not installed,
  not bench failure.
- **"Is this number reliable, or did I miss the warm-up?"** —
  worked example: *"why does my first-second number differ from my
  steady-state number"*. Answered by the measurement-soundness
  overlay in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  layer 5 + [`TASKS.md ## test`](TASKS.md#test) (the eval-loop
  overlay treats warm-up / steady-state / outliers as
  re-iteration triggers, not one-shot facts).
- **"Bench reports zero throughput / hangs at start / disagrees
  with the public docs."** — worked example: *"`doca_bench` shows
  zero ops for AES-GCM but `doca_caps` says the device supports
  it"*. Answered by the layered error taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  (config-syntax → device-binding → library-precondition →
  workload-precondition → measurement-soundness → version →
  cross-cutting) + [`TASKS.md ## debug`](TASKS.md#debug).
- **"How do I capture a baseline I can later regression-test
  against?"** — worked example: *"snapshot decompress throughput
  on this BlueField + DOCA version before a firmware update"*.
  Answered by the CSV output + version-overlay rule in
  [`TASKS.md ## test`](TASKS.md#test) (capture command line +
  version + device + as-deployed environment alongside the
  numbers; quoting numbers without the four-tuple is the
  cross-version regression-hunt failure mode).
- **"`doca_bench` returns nothing for library X — what does that
  mean?"** — worked example: *"empty output for DOCA SHA"*.
  Answered by the empty-output interpretation rules in
  [`TASKS.md ## debug`](TASKS.md#debug) +
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).
  Re-route through
  [`doca-caps`](../doca-caps/SKILL.md) for the coarse
  per-device per-library capability ground truth, then back
  into bench once the capability is confirmed present.

## Audience

This skill serves **external operators, developers, and AI agents
who need a reproducible, vendor-supported way to measure DOCA
library performance on the user's actual install and device**.
Concretely:

- An external developer choosing between DOCA libraries (e.g.
  COMPRESS vs SHA vs DMA throughput) before committing an
  application design.
- A platform operator validating a tuning change (NUMA pinning,
  driver upgrade, firmware burn) by re-running a captured
  `doca_bench` baseline against the new state.
- An SRE / performance engineer producing a *"this is what the
  device delivers today"* artifact that downstream consumers
  (capacity planning, regression bisection) can cite.
- An AI agent answering *"what throughput / latency should I
  expect from DOCA library X on device Y?"* honestly — with a
  measured number, the command line that produced it, and the
  version + device + environment that scopes it — instead of
  guessing from datasheet headlines.

It is **not** for users debugging the `doca_bench` source code,
and **not** a substitute for the live public DOCA Bench guide on
`docs.nvidia.com`.

`doca_bench` is shipped as a **tool** (a single CLI binary plus a
companion app for the remote half of remote-memory / RDMA / Eth
scenarios), not a library you link against. The skill uses the
same `kind: tool` three-file shape as the rest of the bundle so
the agent's task-verb contract
(`configure / build / modify / run / test / debug`) is uniform
across libraries, services, and tools — even when individual
verbs collapse to a routing stub for a shipped binary.

## When to load this skill

Load this skill when the user is — or the agent needs to — invoke
`doca_bench` on a real host with DOCA ≥ 2.7.0 installed (or
inside the public NGC DOCA container with the equivalent version)
to measure performance of a DOCA library. Concretely:

- Picking *which* DOCA library to benchmark for a candidate
  workload (RDMA vs COMPRESS vs DMA, etc.).
- Picking *which* measurement axis to ask for (throughput vs bulk
  latency vs precision latency vs max-bandwidth) — the four modes
  defined in `tools/bench/doca_bench/configuration.hpp` are not
  interchangeable.
- Probing the install's granular-build state so the agent can
  honestly report *"this library is not exposed on this install"*
  instead of inventing a workload.
- Capturing a documented baseline (command line + version + device
  + as-deployed environment + numbers) for later regression hunts.
- Diagnosing why a bench run reported zero / unstable / unexpected
  results (the error-taxonomy walk in
  [`TASKS.md ## debug`](TASKS.md#debug)).

Do **not** load this skill for general DOCA orientation, library
API work, or installation. For those, use
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md),
the matching `libs/<library>` skill, or
[`doca-setup`](../../doca-setup/SKILL.md). Do not load it for
*application-level* end-to-end benchmarking either — `doca_bench`
measures the DOCA library surface, not the user's application
above it.

## What this skill provides

This is a **thin loader**. Substantive material lives in two
companion files:

- `CAPABILITIES.md` — what `doca_bench` can measure (the
  cross-library scope, the three-axis configuration model, the
  documented operating modes, the warm-up / pipeline / multi-core
  concepts that constrain measurement soundness), the version
  overlay (`doca-bench`-specific facts on top of the canonical
  `doca-version` rules), the layered error taxonomy
  (config-syntax / device-binding / library-precondition /
  workload-precondition / measurement-soundness / version /
  cross-cutting), the observability surface (screen + CSV
  output, real-time stats, query system), and the safety
  posture (the public guide's *"not for production"* warning,
  the host vs BlueField execution rule, the companion-app
  attack surface).
- `TASKS.md` — step-by-step workflows for the in-scope task
  verbs: `configure` (the three-axis decision + the
  probe-before-bench step), `build` (route to install — the
  binary is shipped, the companion app is shipped), `modify`
  (refuse — do not patch the bench binary; modify the bench
  *invocation* instead), `run` (the smoke-before-bulk flow),
  `test` (the eval loop — warm-up, steady-state, outliers,
  cross-version), `debug` (walk the error taxonomy layer by
  layer), plus a `Deferred task verbs` block routing
  out-of-scope questions and a `Command appendix` of
  `doca_bench`-specific invocation classes.

The skill assumes a host where DOCA ≥ 2.7.0 is already installed
(or the public NGC DOCA container is running at an equivalent
version) and the operator has whatever permissions the public
guide requires for `doca_bench` to bind devices and allocate
resources on their platform.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or scripts
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Specific flag strings or scenario / metric / attribute names
  beyond what the public DOCA Bench guide documents.** The flag
  surface evolves and is install-specific; the documented
  invocations + `--help` on the installed version are the
  authoritative answer. Inventing a flag is the most common
  hallucination failure for this skill.
- **Pre-baked example output or expected throughput numbers.**
  Bench output is device-, version-, firmware-, NUMA-, and
  tuning-specific. A captured number pinned to one platform and
  one DOCA version misleads operators on a different
  platform / version.
- **Wrappers, parsers, or scripts** in any language that consume
  `doca_bench` CSV or stdout. The output formats are documented;
  if a user wants to script against them, the right answer is
  "read the live guide, write the parser against your installed
  version".
- **A `samples/` or `reference/` subtree.** This is a thin
  loader for a documented CLI; substantive material lives on
  the public page and in `--help`.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is
   in scope (the user actually wants to invoke `doca_bench` for
   measurement, not learn about a DOCA library in general).
2. **For what `doca_bench` measures, the three-axis model, the
   version overlay, the error taxonomy, observability surface,
   and safety posture, see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For the documented invocations and the smoke-before-bulk
   workflow — `configure`, `build`, `modify`, `run`, `test`,
   `debug` — see [TASKS.md](TASKS.md).**

## Related skills

- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  — routing to the public DOCA Bench page on `docs.nvidia.com`
  and the rest of the public DOCA documentation set.
- [`doca-version`](../../doca-version/SKILL.md) — the canonical
  version-detection chain, four-way match rule, NGC container
  semantics, and headers-win-over-docs rule. The
  `## Version compatibility` section in this skill is a thin
  overlay on top of `doca-version`; the body lives there.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md)
  — the bundle-wide contract for structured-output helper tools.
  Bench-runner / bench-snapshot executables that satisfy the
  detect-prefer-fallback-report loop are deferred to PR2; the
  contract is consumed here in advance so the
  `## Command appendix` in [`TASKS.md`](TASKS.md) is infra-aware
  from PR1.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, hugepages, NUMA awareness, and the *I
  have no install yet* path with the public NGC DOCA container.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder. Bench surfaces *its own* error taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy);
  when the cause turns out to be below DOCA (driver, firmware,
  NUMA), the bench taxonomy hands off to `doca-debug`.
- [`doca-caps`](../doca-caps/SKILL.md) — the sibling DOCA tool
  for the coarse per-device per-library capability snapshot.
  Bench probes capability at finer grain via its own query
  system; `doca_caps` is the cheaper first step to confirm the
  device is even visible to DOCA.
- The matching `libs/<library>` skill — e.g.
  [`doca-comch`](../../libs/doca-comch/SKILL.md),
  [`doca-compress`](../../libs/doca-compress/SKILL.md) — for
  the workload-side preconditions, capability-query rules, and
  error-taxonomy overlays of the library under test. Bench
  drives the library; the library skill explains *what
  "healthy" means for it*.
