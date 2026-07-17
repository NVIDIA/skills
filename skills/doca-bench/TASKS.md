# DOCA Bench — Tasks

**Where to start:** The verbs that carry real workflow content are
`## configure`, `## run`, `## test`, and `## debug`. The other two
substantive verbs (`build`, `modify`) carry routing stubs because
`doca_bench` is a shipped binary, not a source artifact the user
compiles or patches. The `## test` verb is an iterative loop, not
a one-shot pass — see the eval-loop overlay in `## test` below.

This file is loaded by [`SKILL.md`](SKILL.md) after
[`CAPABILITIES.md`](CAPABILITIES.md). It walks the agent through
the six task verbs every artifact in this bundle exposes
(`configure / build / modify / run / test / debug`), then
explicitly defers task verbs that do not belong here.

For `doca_bench`, the verbs that carry real workflow content are
`configure`, `run`, `test`, and `debug`. The other two verbs
*exist as anchors* because the agent's task-verb contract is
uniform across libraries, services, and tools — and each one
carries a meaningful **routing stub** that names where the user's
question really belongs.

## configure

The bench's *configuration* is the invocation: there is no
separate config file, no daemon, no env knob the public guide
documents as required (DOCA-wide env vars like `DOCA_LOG_LEVEL`
still apply, but they are owned by
[`doca-programming-guide CAPABILITIES.md ## Observability`](../../doca-programming-guide/CAPABILITIES.md#observability),
not by bench). What the agent has to *configure* is the three-axis
decision documented in
[`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).

Steps the agent should walk the user through, in order:

1. **Confirm DOCA ≥ 2.7.0 is installed and the binary is
   present at `/opt/mellanox/doca/tools/doca_bench`.** If
   not, route to
   [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure)
   for the detection chain and
   [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
   for the upgrade path. Do not propose alternative tools; on
   `< 2.7.0` installs the bench is genuinely not there.
2. **Probe the install's granular build first.** Use the
   bench's built-in query system (per
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability))
   to enumerate which DOCA libraries are *actually exposed* on
   this install before committing to a target. An empty
   enumeration for the user's target library means the
   granular build did not include it — route to
   [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
   for the install profile, not to a bench invocation.
3. **Axis 1 — pick the target library.** Per the cross-library
   table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   commit explicitly to which DOCA library is the unit under
   test (e.g. COMPRESS, RDMA, AES-GCM, SHA, DMA, EC, Eth,
   Comch, GPUNetIO). An answer that picks one without naming
   the alternatives has silently narrowed the bench to a
   single-library tool; surface that to the user.
4. **Axis 2 — pick the workload shape.** Direction (e.g.
   send vs receive, encrypt vs decrypt, compress vs
   decompress), data provider (file / file-set / random-data
   per the public guide), buffer / job sizing, batching, NUMA
   placement of the chosen cores, and whether remote-memory
   input / output is in play (which pulls in the companion
   app — see step 6). Re-cross-check against the per-platform
   support matrix in
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility);
   not every workload shape is documented for every BlueField
   generation.
5. **Axis 3 — pick the measurement mode.** Throughput vs
   bulk-latency vs precision-latency per the public guide's
   three documented modes. Quote back to the user *why* you
   picked the mode — *"throughput because the user asked
   about bandwidth"*, *"precision-latency because the user
   asked about per-job tail latency"* — so the user can
   challenge the framing if it does not match intent. The
   modes are not interchangeable; see
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   for the trade-off table.
6. **If the workload is remote (RDMA, Eth, remote-memory),
   plan the companion app deployment.** The companion app is
   shipped with bench but runs on the far side, with the
   out-of-band channel between the two halves. Surface the
   public guide's *"not for production"* warning per
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
   and confirm a non-production segment is available.

For the canonical DOCA universal lifecycle that underlies
program-side configuration (which `doca_bench` itself runs
internally per library), see
[`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure).
This skill is concerned with the *operator*-side configuration
of the bench invocation, not the program-side lifecycle of the
library under test.

## build

`doca_bench` and its companion app are **shipped pre-built** as
part of every DOCA install since 2.7.0
(`/opt/mellanox/doca/tools/doca_bench`). There is no source tree
the external user is expected to compile, no build flags, no
`meson` or `make` workflow for the bench itself.

Routing for nearby "build" questions:

- *"The binary isn't there — do I need to build it?"* → no.
  Route to
  [`doca-setup ## install`](../../doca-setup/TASKS.md#configure).
  The fix is to install (or re-install) DOCA at ≥ 2.7.0, or
  use the public NGC DOCA container per
  [`doca-setup ## no-install`](../../doca-setup/TASKS.md#no-install)
  at an equivalent version.
- *"I want to build my own benchmark program against DOCA
  library X."* → not a `doca_bench` question. Route to
  [`doca-programming-guide ## build`](../../doca-programming-guide/TASKS.md#build)
  for the cross-library build pattern and the matching
  `libs/<library>` skill (e.g.
  [`doca-comch ## build`](../../libs/doca-comch/TASKS.md#build),
  [`doca-compress`](../../libs/doca-compress/SKILL.md)) for
  the library-specific build overlay. The bench is the
  shipped harness; the user's bespoke harness is a
  different artifact.
- *"I want to extend the bench with a new scenario."* →
  out of scope here; this skill is for external operators
  consuming the shipped bench, not for contributors
  extending it.

The `## What this skill deliberately does not ship` block in
[`SKILL.md`](SKILL.md) explicitly forbids adding a build recipe
or wrappers for `doca_bench`; revisit that policy before
changing this section.

## modify

**Do not modify the shipped `doca_bench` binary.** It is an
NVIDIA-shipped CLI; there is no documented public way to change
its behavior, output format, scenario set, or attribute surface,
and none should be invented.

What the agent *does* modify, every time, is the **bench
invocation** — the flags, the chosen library, the workload
shape, the mode, the core / thread layout. That is the
configuration loop in [`## configure`](#configure) above and the
iteration loop in [`## test`](#test) below; treat *modify the
invocation, not the binary* as the operating mode.

Routing for nearby "modify" questions:

- *"The output format is inconvenient — can I change it?"* →
  the documented surfaces are stdout, CSV, and the real-time
  stats interval per
  [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability).
  If those are insufficient, the right answer is *"write a
  parser against the documented CSV format on your installed
  version"* — not a binary patch — and even that scripting is
  out of scope per
  [`SKILL.md ## What this skill deliberately does not ship`](SKILL.md#what-this-skill-deliberately-does-not-ship).
- *"Can I patch `doca_bench` to add scenario X?"* → out of
  scope for external users; this skill is for consumers of
  the shipped tool, not contributors to it.
- *"I need a *different measurement* than `doca_bench`
  reports."* → re-examine axis 3 (measurement mode) in
  [`## configure`](#configure) first; the three documented
  modes cover throughput, latency-distribution, and
  per-job-latency. If the question is genuinely outside
  bench's surface (e.g. application-level end-to-end timing),
  route to
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md)
  and the matching `libs/<library>` skill — the user's own
  program is the right place to measure end-to-end.

## run

The smoke-before-bulk flow — every bench session goes through
it, no exceptions. The full invocation surface lives in the
public DOCA Bench guide; this section names the *shape* of the
flow, not the verbatim command lines (per
[`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
*"do not invent flags"*).

> **Do-not-invent guard (specific flag names).** Real downstream
> agents have hallucinated the following plausibly-named but
> non-existent flags for `doca_bench`: `--pipeline`, `--mode`,
> `--device`, `--csv-output`, `--target-library`. None of these
> appear in the bundle or in `doca_bench --help` on the public
> DOCA release. The flag inventory the bundle DOES name verbatim
> is `--core-mask`, `--core-list`, `--core-count`,
> `--threads-per-core`, the `--run-limit-*` family, `--sweep`,
> and the `--csv-*` family — and those are named as *classes*,
> not as ready-to-paste literals; the agent MUST read
> `doca_bench --help` on the installed bench and use only the
> names that appear there. Any example invocation written in
> this skill MUST keep tokens like `<flag-from-help>` or
> `<chosen-mode-from-help>` as placeholders for the
> per-install-variable inventory.

1. **Confirm the binary, version, and granular-build
   inventory.** Per [`## configure`](#configure) steps 1-2;
   without this the next four steps will burn the operator's
   time on a configuration that the install does not support.
2. **Smoke run — trivial workload, short duration.** Pick the
   smallest defensible workload for the chosen target library
   and mode (e.g. a tiny job count, a short time limit, the
   minimum data provider that the library will accept). The
   goal is *"bench can bind the device, start the pipeline,
   and emit numbers"*, not a usable measurement. The public
   guide documents the duration / job-count limit families
   (`--run-limit-*` per the user's installed `--help`); use
   the smallest defensible value of one of them.
3. **Read the echoed invocation.** Per
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability)
   the bench prints the full set of configured values at the
   start; this is the user's chance to catch a defaulted
   value that does not match intent (e.g. wrong NUMA node,
   wrong data provider, warm-up turned off) before the run
   completes.
4. **Inspect the summary and reject silently-bad runs.** Exit
   0 + a number is not enough — verify the number is in a
   defensible order of magnitude for the library / device,
   verify warm-up actually happened, verify the mode in the
   summary matches the requested mode. If anything looks off,
   loop back to [`## debug`](#debug) before sinking time
   into a longer run.
5. **Plan the bulk / swept run** only after the smoke is
   green. The public guide documents the `--sweep` family for
   parameter sweeps; the agent's rule for sweep planning is
   *enumerate the swept dimension explicitly, estimate the
   total run time, and confirm the operator is OK with the
   wall-clock cost* before committing.
6. **For remote scenarios, start the companion app first.**
   The companion app must be listening before the bench host
   tries to drive it; per
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   the companion app is part of the install, not optional.

When recording the run for downstream consumers (the *baseline*
pattern), write down: the DOCA version, the host platform (host
vs BlueField Arm, OS, kernel, firmware), the exact command line
used (the bench's own echo line covers this), the device target
(PCIe address / IB name / interface name), and the full
unredacted summary + CSV. The downstream `## test` and `## debug`
workflows depend on those five fields.

## test

`doca_bench` is **a measurement tool**, so its `## test` verb is
about *testing the measurement* — i.e. confirming the numbers
are sound and reproducible — not unit-testing the bench itself.

**`## test` is an iterative loop, not a one-shot pass.** A bench
run that completes is not the same as a bench run that produced
a defensible number; each iteration tightens one axis of
measurement soundness (warm-up, steady-state, outliers, NUMA
placement, cross-run reproducibility, cross-version delta) and
loops back to [`## run`](#run).

The eval-loop overlay (rows apply to every bench run, not just
one library × mode):

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| Smoke completed; number is far below datasheet headline | Could be cold pipeline, wrong workload shape, wrong NUMA, or actually-right for this install. Do not assume datasheet first. | Confirm warm-up applied per [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy) layer 5; re-check axis 2 (workload shape) in [`## configure`](#configure) step 4; only then question hardware. |
| Throughput-mode number swings > X% across short re-runs | Steady-state not reached; outlier-dominated run | Lengthen the run via the documented duration / job-count limit; re-run; if still volatile, switch to bulk-latency mode to surface the distribution. |
| Precision-latency mean looks good; 99.99th percentile is huge | Tail-latency story is the actual answer; the mean is misleading | Quote the percentile breakdown, not the mean, per [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability) the precision-latency mode's reported distribution. |
| Same invocation produces different numbers on two hosts at the same DOCA version | NUMA / firmware / driver delta below DOCA | Walk axis 2 environment (cores / threads / NUMA) and the version layer per [`doca-version TASKS.md ## test`](../../doca-version/TASKS.md#test) before blaming bench. |
| Same invocation produces different numbers on the same host across DOCA versions | This *is* a regression signal — provided both four-tuples are captured | Cross-link the two baselines, name the changed fields, route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) for the version-delta diagnosis. |
| `--sweep` of a parameter shows a discontinuity | Could be a real performance cliff (capacity / cache / queue depth) or a measurement artefact at the swept value | Re-run the boundary points without the sweep to confirm; if real, that is the answer the user came for. |
| Bench reports zero / hung after extended wait | Device-binding or library-precondition layer; not a measurement-soundness issue | Stop iterating on the workload shape; jump to [`## debug`](#debug) layers 2-3. |

The agent's rule: every change to the invocation re-opens the
loop. Re-running with a tweaked flag and quoting the new
number without re-checking warm-up / steady-state / outliers /
distribution is exactly the failure mode this loop replaces.

**Baseline-capture rule.** When the goal of the bench session
is a baseline (vs an ad-hoc question), the captured artifact
must include the *four-tuple* per
[`## Pattern overview`](CAPABILITIES.md#pattern-overview)
pattern 6 — command line + DOCA version + device target +
as-deployed environment (firmware, kernel, NUMA, hugepages) —
alongside the summary and CSV. Without all four, the baseline
cannot be regression-tested later; quoting a number without the
four-tuple is the cross-version regression-hunt failure mode.

Loop termination: stop iterating once two consecutive runs do
not change the picture — the answer is now *"this is what the
device delivers on this DOCA version / firmware / driver
stack"*. Escalate cross-version or cross-host comparisons to
[`doca-version TASKS.md ## test`](../../doca-version/TASKS.md#test)
or [`doca-debug ## debug`](../../doca-debug/SKILL.md) with the
captured four-tuples as evidence.

This skill does **not** ship a "test fixture" or pre-recorded
expected output. The expected output is install-, device-,
firmware-, and tuning-specific; pinning one would mislead
operators on a different platform / version. See
[`SKILL.md ## What this skill deliberately does not ship`](SKILL.md#what-this-skill-deliberately-does-not-ship).

## debug

When `doca_bench` fails to start, fails to produce numbers, or
produces numbers that do not look defensible, walk the
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
layers in order. The shape of the diagnosis:

1. **Config-syntax.** Invocation does not parse. Confirm the
   flag exists in `--help` on the *installed* binary (not a
   blog or older release). Confirm units / values are in the
   documented form. Confirm the flag is not mutually
   exclusive with another in the same invocation. Route to
   the public DOCA Bench guide via
   [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools);
   do not infer from generic CLI knowledge.
2. **Device-binding.** Invocation parses; bench cannot bind
   the target device. Confirm the device is visible to DOCA
   at all via
   [`doca-caps ## run`](../doca-caps/TASKS.md#run); confirm
   the driver stack is loaded
   ([`doca-setup ## debug`](../../doca-setup/TASKS.md#debug)
   layer Driver); confirm the chosen NUMA / core layout
   matches the device's actual NUMA node. Re-attempt the
   bench invocation only after the device is on the
   capability snapshot.
3. **Library-precondition.** Device bound; bench refuses to
   exercise the requested DOCA library. Re-run the bench
   query system per [`## configure`](#configure) step 2 to
   confirm the library is actually exposed; re-cross-check
   the per-platform support matrix per
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility);
   walk the per-library skill for library-internal
   capability rules (e.g.
   [`doca-comch CAPABILITIES.md ## Capabilities and modes`](../../libs/doca-comch/CAPABILITIES.md#capabilities-and-modes),
   [`doca-compress`](../../libs/doca-compress/SKILL.md) for
   compress-specific capability boundaries).
4. **Workload-precondition.** Library exercisable; the
   workload shape is invalid. Re-walk axis 2 of the
   three-axis model per [`## configure`](#configure) step 4;
   the most common failure here is feeding random-data to a
   structured workload (e.g. decompression / decryption)
   that needs valid input, or sizing buffers below / above
   what the library accepts.
5. **Measurement-soundness.** The run completes and reports
   numbers, but the numbers are unsound. Walk the three
   sub-layers per
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   layer 5 — warm-up applied? steady-state reached?
   distribution reported alongside the single number? —
   before quoting any number.
6. **Version.** Cross-cutting partial-install / mixed-version.
   Walk [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   end-to-end; common bench-specific symptom is a
   companion-app version that does not match the bench host
   version.
7. **Cross-cutting.** Cause is below DOCA. Hand off to
   [`doca-debug ## debug`](../../doca-debug/SKILL.md) for
   the cross-cutting debug ladder and
   [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug)
   for the env-side layers (driver / firmware / hugepages /
   NUMA).

In every case: **quote what the bench reported.** Do not
paraphrase the summary, do not reorder fields, do not
"summarize" a histogram into a single number. The bench is in
the loop precisely to break the agent out of the
inference-from-datasheet trap.

## Deferred task verbs

The four verbs below are not `doca_bench` work and should be
routed out before the agent does any of them under this skill's
name.

- **install** ⇒ [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
  (and [`## no-install`](../../doca-setup/TASKS.md#no-install)
  for the public NGC DOCA container path). The bench is
  shipped by the install at ≥ 2.7.0; this skill does not own
  the install workflow.
- **build a custom DOCA benchmark or application** ⇒
  [`doca-programming-guide ## build`](../../doca-programming-guide/TASKS.md#build)
  for the cross-library pattern, plus the matching
  `libs/<library>` skill for the library-specific build
  details. `doca_bench` is the shipped harness; a bespoke
  harness is a different artifact.
- **library-internal benchmarking** (e.g. an application-level
  end-to-end measurement, or a library-internal performance
  counter the bench does not expose) ⇒ the matching
  `libs/<library>` skill plus
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
  Bench is a uniform harness; library-internal performance
  questions go to the library.
- **streaming telemetry / live metrics from a production
  workload** ⇒ not a bench feature, and bench is explicitly
  *not for production* per
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
  The DOCA Telemetry Service (DTS) is the documented telemetry
  surface; routing belongs in
  [`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).

## Command appendix

`doca_bench`-specific invocation classes the verbs above reach
for. Every row is a CLASS — the agent must not invent flags
beyond `--help` on the installed binary and the public DOCA
Bench guide. The five-class symmetry below is the load-bearing
piece; one worked example per class is shown.

**Infra-aware preamble (every row below).** Per the bundle's
detect → prefer → fall back → report contract documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent should:

1. Probe for the matching structured helper FIRST (`doca-env --json`
   for version + devices + libraries + drivers + hugepages in one
   shot; `doca-capability-snapshot` for per-device capability
   flags; `version-matrix.json` for *"available since"* lookups;
   a future bench-runner / bench-snapshot helper for the
   four-tuple-capturing baseline pattern when it lands per
   [`doca-structured-tools-contract ## Relationship to PR2 executables`](../../doca-structured-tools-contract/SKILL.md#relationship-to-pr2-executables)).
2. If the probe succeeds, the structured tool's output is the
   authoritative answer and the agent SHOULD NOT also run the
   manual command in the row below. Report *"using structured
   `<tool>`"*.
3. If the probe fails, fall back to the manual command in the
   row. Report *"falling back to manual chain"*.
4. The schemas the structured tools emit are defined in
   [`doca-structured-tools-contract ## Schemas`](../../doca-structured-tools-contract/SKILL.md#schemas);
   the version-handling semantics (four-way match, NGC,
   headers-win) are owned by
   [`doca-version`](../../doca-version/SKILL.md).

| Purpose (class) | Invocation (shape) | Owning step | Reads as healthy when … |
| --- | --- | --- | --- |
| Discover the documented flag surface | `doca_bench --help` (and the public DOCA Bench guide for the long-form documentation) | [`## configure`](#configure) step 1; [`## debug`](#debug) layer 1 | Prints the documented flag inventory the agent uses as the only source of truth for flag names; the public guide is the secondary source. |
| Probe the granular-build inventory | The bench's built-in query family for installed libraries + supported sweep attributes per [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability) | [`## configure`](#configure) step 2 | Reports the libraries this install actually exposes; absence of the target library is the canonical *"granular build does not include X"* answer. |
| Drive a single-library micro-benchmark | The documented bench invocation for the chosen (target library × workload shape × measurement mode) — flag names re-confirmed against `--help` on the installed binary | [`## run`](#run) steps 2-4; [`## test`](#test) eval loop | The bench echoes the invocation, applies the documented warm-up, completes in the requested duration / job count, and prints a summary in the chosen mode's documented format. |
| Capture a CSV baseline alongside stdout | The documented CSV-output family per [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability) — `--csv-*` family on the installed `--help` | [`## test`](#test) baseline-capture rule | A CSV file is written, the stdout summary matches the CSV aggregate, and the captured four-tuple (command line + version + device + environment) accompanies the CSV. |
| Sweep a parameter across a planned range | The documented sweep family (`--sweep` per the public guide); the agent estimates total wall-clock cost before committing | [`## run`](#run) step 5 | The smoke run for the boundary values passed first; the sweep completes; the resulting series has no implausible discontinuities that disappear when re-running the boundary point without the sweep. |
| Drive a remote scenario with the companion app | The bench invocation paired with the companion app on the far side, communicating over the documented out-of-band channel | [`## configure`](#configure) step 6; [`## run`](#run) step 6 | The companion app is up and reachable, the channel comes up, the bench drives it without the *"not for production"* warning being triggered on a production segment. |

Three cross-cutting rules for this appendix:

- **Never invent a `doca_bench` flag, scenario name, attribute
  name, or metric name.** `--help` on the installed binary and
  the public DOCA Bench guide are the joint contract;
  prose-derived flag strings are the most common hallucination
  failure for this skill.
- **Smoke before bulk.** Every row above presumes the smoke
  row succeeded first; running a sweep or a long single-library
  run without the smoke is the canonical operator-time-waste
  failure mode.
- **Cross-link instead of duplicate.** Cross-cutting commands
  (`pkg-config --modversion`, `doca_caps --list-devs`,
  `dmesg`, `mlxconfig -d <bdf> q`, `numactl --hardware`) live
  in [`doca-debug ## debug`](../../doca-debug/SKILL.md) and
  [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug);
  this appendix names only `doca_bench`-specific invocation
  classes.

## Cross-cutting

A few rules that apply across every verb in this file, restated
here so they are visible at the point of action and not buried
in [`SKILL.md`](SKILL.md):

- The **public DOCA Bench guide** plus the installed `--help`
  are the joint source of truth. When they disagree (e.g. a
  flag landed in a release this skill was not written
  against), the *installed* `--help` wins for the user's
  actual run.
- `doca_bench` *does* drive hardware and *does* allocate
  resources; smoke-before-bulk is mandatory, and re-running
  a long sweep "to confirm" without the smoke step is
  exactly the failure mode this skill is here to prevent.
- **Quote the four-tuple, not just the number.** Command line
  + DOCA version + device target + as-deployed environment
  is the minimum unit a bench number is meaningful in. The
  agent must surface all four whenever reporting a number to
  the user.
- This skill **assumes a healthy DOCA install at ≥ 2.7.0**
  (or the public NGC DOCA container at an equivalent
  version). If the install is in doubt, route to
  [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure)
  and
  [`doca-setup`](../../doca-setup/SKILL.md) before running
  anything else here.
