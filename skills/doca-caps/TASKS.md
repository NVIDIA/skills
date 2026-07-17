# DOCA Capabilities Print Tool — Tasks

**Where to start:** The verbs that carry real workflow content are
`## run`, `## test`, and `## debug`. The other three (`configure`,
`build`, `modify`) are documented routing stubs that exist because
the bundle's verb contract is uniform. The `## test` verb is an
iterative loop, not a one-shot pass — see the eval-loop overlay in
`## test` below.

This file is loaded by [`SKILL.md`](SKILL.md) after
[`CAPABILITIES.md`](CAPABILITIES.md). It walks the agent through the
six task verbs every artifact in this bundle exposes
(`configure / build / modify / run / test / debug`), then explicitly
defers task verbs that do not belong here.

For `doca_caps`, the verbs that carry real workflow content are
`run`, `test`, and `debug`. The other three verbs *exist as anchors*
because the agent's task-verb contract is uniform across libraries,
services, and tools — and each one carries a meaningful **routing
stub** that names where the user's question really belongs.

## configure

`doca_caps` takes **no configuration**. It is a flag-driven
read-only CLI; there are no config files, no daemons, no
environment knobs the public guide documents as required.

If the user is asking *"how do I configure `doca_caps`?"*, the
question they almost certainly mean is one of:

- *"How do I install DOCA so `doca_caps` shows up?"* → route to
  [`doca-setup ## install`](../../doca-setup/TASKS.md#configure). The
  binary is shipped with every DOCA install since 2.6.0; configuring
  is install.
- *"How do I make `doca_caps` see device X?"* → not a configuration
  question; it is a hardware / driver / permission question. Run
  `doca_caps --list-devs` first (see [`## run`](#run)) and let the
  empty-output diagnosis flow in [`## debug`](#debug) guide the next
  step.
- *"How do I configure DOCA logging that affects what `doca_caps`
  prints?"* → `doca_caps`'s own output is fixed by its flags;
  general DOCA log filtering belongs to
  [`doca-programming-guide CAPABILITIES.md ## Observability`](../../doca-programming-guide/CAPABILITIES.md#observability).

Do not invent configuration files or environment variables for this
tool. If the public guide does not document a config knob, it does
not exist.

## build

`doca_caps` is **shipped pre-built** as part of every DOCA install
(`/opt/mellanox/doca/tools/doca_caps`). There is no source tree the
external user is expected to compile, no build flags, no `meson` or
`make` workflow.

Routing for nearby "build" questions:

- *"The binary isn't there — do I need to build it?"* → no. Route to
  [`doca-setup ## install`](../../doca-setup/TASKS.md#configure). The
  fix is to install (or re-install) DOCA, or use the public NGC DOCA
  container per
  [`doca-setup ## no-install`](../../doca-setup/TASKS.md#no-install).
- *"I want to build a tool that **uses** `doca_caps`-style capability
  introspection from inside my application"* → not a `doca_caps`
  question. Route to the matching `libs/<library>` skill — DOCA
  libraries expose programmatic capability checks; `doca_caps` is a
  CLI wrapper around (a subset of) those checks for operator use.

The `## What this skill deliberately does not ship` block in
[`SKILL.md`](SKILL.md) explicitly forbids adding a build recipe for
`doca_caps` or shipping wrappers around it; revisit that policy
before changing this section.

## modify

**Do not modify the shipped `doca_caps` binary.** It is an NVIDIA-
shipped read-only CLI; there is no documented public way to change
its behavior, output format, or capability surface, and none should
be invented.

Routing for nearby "modify" questions:

- *"The output format is inconvenient — can I change it?"* → no, not
  inside this skill. The documented surface is the surface. If the
  user wants structured output, the right answer is *"write a
  parser against the documented format on your installed version"* —
  and even that is out of scope per
  [`SKILL.md ## What this skill deliberately does not ship`](SKILL.md#what-this-skill-deliberately-does-not-ship).
- *"Can I patch `doca_caps` to add flag X?"* → out of scope for
  external users; this skill is for consumers of the shipped tool,
  not contributors to it.
- *"I need different *information* than `doca_caps` reports"* →
  route to the matching `libs/<library>` skill or to
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md)
  — the right answer is to call the underlying library API
  programmatically, not to modify the CLI wrapper.

## run

The two invocations the public guide explicitly walks through are:

```text
/opt/mellanox/doca/tools/doca_caps --list-devs
/opt/mellanox/doca/tools/doca_caps --list-rep-devs
```

Either can be scoped to a single PCIe address with `--pci-addr`,
which the public guide documents as supported on both `--list-devs`
and `--list-rep-devs`.

Recommended flow when the user asks the agent to *"check what DOCA
sees"*:

1. **Confirm the binary is present** at
   `/opt/mellanox/doca/tools/doca_caps`. If absent, route to
   [`## configure`](#configure) above.
2. **Run `--list-devs` first.** This is the canonical first
   invocation in both [`doca-setup ## test`](../../doca-setup/TASKS.md#test)
   and [`doca-programming-guide ## debug`](../../doca-programming-guide/TASKS.md#debug).
   Capture the full output verbatim.
3. **If the user cares about representors / SR-IOV / SF setups,
   also run `--list-rep-devs`.** Otherwise this is noise.
4. **If the user has a specific PCIe address in mind, scope with
   `--pci-addr <addr>`** to keep output focused; otherwise list
   everything.
5. **For exact, current flag inventory** (e.g. logger listing, OS
   target listing, library-list flag, library-capabilities flag),
   read `--help` on the installed version. Do **not** invent flags
   from generic CLI knowledge — the public guide and `--help` are
   the source of truth, see
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   for the documented capability families and
   [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools)
   for the public guide URL.

When recording the run for downstream consumers (the *capability
snapshot* pattern), write down: the DOCA version, the host platform
(host vs BlueField Arm, OS, kernel), the exact command line used,
and the full unredacted output. The downstream `## test` and
`## debug` workflows depend on those four fields.

## test

`doca_caps` is **the canonical install smoke-test** prescribed by
[`doca-setup ## test`](../../doca-setup/TASKS.md#test).

**`## test` is an iterative loop, not a one-shot pass.** The agent
re-runs `doca_caps` after every state-changing action that
*should* have affected DOCA's view of the host (driver reload,
firmware change, BlueField mode flip, container `--device` change).
Treating the smoke as a one-shot pass is the failure mode this loop
replaces.

The eval-loop overlay (rows apply to every DOCA install, not just
one):

| Step | Why this is a loop, not a step | Where the substance lives |
| --- | --- | --- |
| 1 → ## debug | Empty / unexpected output is a finding, not a tool failure; the agent must walk the debug ladder, then re-run step 1 | [`## debug`](#debug) |
| 1 → driver reload → 1 | After `modprobe` / driver reload, re-run step 1 to confirm DOCA's view caught up | [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) layer Driver |
| 1 → firmware / mode change → 1 | After `mlxconfig` or BlueField mode change, re-run step 1 to confirm the new capability surface | [`doca-setup ## configure`](../../doca-setup/TASKS.md#configure) |
| 1 → container `--device` change → 1 | After remounting / `--device`-mapping a container, re-run step 1 inside the container | [`doca-setup ## no-install`](../../doca-setup/TASKS.md#no-install) |
| 1 (clean) → save → debug session | Once clean, the snapshot is saved and consumed by `doca-debug ## test` step 3 (read-only triple) | [`doca-debug ## test`](../../doca-debug/TASKS.md#test) |

The agent's rule: every state-changing action that *could* affect
DOCA's view re-opens the smoke. Saving a stale snapshot from before a
mutation is exactly the failure mode this loop is here to prevent.

The pattern the rest of the bundle expects:

1. Run `doca_caps --list-devs`.
2. **Exit code 0 + at least one DOCA device listed** ⇒ the install
   can see hardware. The install is *probably* healthy at the level
   `doca_caps` can verify; routing for the next health check is
   library-specific (e.g.
   [`doca-flow CAPABILITIES.md ## Capabilities and modes`](../../libs/doca-flow/CAPABILITIES.md#capabilities-and-modes)
   for whether DOCA Flow can program the device).
3. **Exit code 0 + zero devices listed** ⇒ the install is on a host
   with no DOCA-supported PCIe device, *or* the user is inside a
   container without PCIe passthrough, *or* the driver stack is not
   loaded. This is a *finding*, not a tool failure — see
   [`## debug`](#debug) for the layered diagnosis.
4. **Non-zero exit code** ⇒ the tool failed (permission, missing
   library, broken install). See
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   for the layered taxonomy.

The output of step 1 is the *capability snapshot* that subsequent
debug steps consume. Save it (file, paste buffer, conversation
artifact) — without it, downstream debug starts guessing. This
matters more than it looks: the most common reason a Flow-style
*"why isn't this working"* debug session goes off the rails is that
the agent skipped this step and inferred capability from device
model number instead of from `doca_caps` ground truth.

This skill does **not** ship a "test fixture" or pre-recorded
expected output. The expected output is install- and
hardware-specific; pinning one would mislead operators on a
different platform / version. See
[`SKILL.md ## What this skill deliberately does not ship`](SKILL.md#what-this-skill-deliberately-does-not-ship).

## debug

When `doca_caps` returns nothing useful or fails, walk the
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
layers in order. The shape of the diagnosis:

1. **Tool-not-installed.** `doca_caps` does not exist at
   `/opt/mellanox/doca/tools/doca_caps`. Confirm DOCA is installed
   (e.g. `pkg-config --modversion doca-common`,
   `cat /opt/mellanox/doca/applications/VERSION`) and that the
   version is ≥ 2.6.0. Route to
   [`doca-setup ## install`](../../doca-setup/TASKS.md#configure) if
   not.
2. **Permission / driver layer.** Tool runs but cannot enumerate
   devices. Check that the underlying driver stack (`mlx5_core`, IB
   stack) is loaded, that the user has whatever privileges the
   public install profile expects, and that the BlueField mode is
   compatible with the requested capability family. The tool's own
   message and `dmesg` are ground truth; do not guess at causes.
3. **Empty / partial output, exit 0.** *Capability finding*, not a
   tool failure:
   - Zero devices listed in `--list-devs`: no DOCA-supported PCIe
     device on this host, or the container lacks PCIe passthrough,
     or the user is inside the public NGC DOCA container without
     `--device` mappings. Route to
     [`doca-setup ## no-install`](../../doca-setup/TASKS.md#no-install)
     for the container path, or to
     [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
     for hardware questions.
   - Fewer devices than expected: re-run without `--pci-addr` to
     confirm the scope is not the cause; check the install version
     against the device's first-supported DOCA release in the
     public release notes (route through
     [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)).
4. **Library-capability mismatch.** Tool runs, capabilities printed,
   but the user's library question (*"why does library X say it
   doesn't support Y on my device?"*) goes beyond what
   `doca_caps` reports. Route to the matching `libs/<library>`
   skill — for Flow, that is
   [`doca-flow CAPABILITIES.md ## Capabilities and modes`](../../libs/doca-flow/CAPABILITIES.md#capabilities-and-modes).
5. **Output schema confusion.** The user is reading the capability
   snapshot and a field name does not look right. Re-fetch the
   public **DOCA Capabilities Print Tool** page (via
   [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools))
   on the user's installed DOCA version and quote it; the field
   inventory in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   reflects the documented surface but is not contractually frozen.

In every case: **quote what the tool said.** Do not paraphrase the
output, do not reorder fields, do not "summarize" capability lines
into prose. The whole point of `doca_caps` in a debug context is to
break the agent out of the inference-from-model-number trap.

## Command appendix

`doca_caps`-specific invocations the verbs above reach for. Every
row is a class — the agent must not invent flags beyond `--help`
on the installed binary. The four-family symmetry below is the
load-bearing piece; one worked example per family is shown.

| Purpose (class) | Invocation (shape) | Owning step | Reads as healthy when … |
| --- | --- | --- | --- |
| Discover available flags | `doca_caps --help` | [`## run`](#run) | Prints the documented flag inventory; the agent uses this as the only source of truth for flag names. |
| Enumerate DOCA devices | `doca_caps --list-devs` | [`## run`](#run) + [`## test`](#test) step 1 | Exit 0 and at least one device row, with the expected PCIe addresses present. |
| Enumerate representor devices | `doca_caps --list-rep-devs` | [`## run`](#run) | Exit 0; the representor topology matches what `devlink dev show` reports. |
| Scope to one PCIe address | `doca_caps --pci-addr <bdf>` | [`## run`](#run) + [`## debug`](#debug) layer "Library-capability mismatch" | The output is restricted to one device; rules out scope as the cause of an unexpectedly empty answer. |
| Scope to one DOCA library | The documented per-library flag (see `--help`) | [`## run`](#run) | Returns the library's supported capabilities; empty output = *not supported on this device/version*. |
| Save a snapshot for debug | `doca_caps --list-devs > caps.txt` (and equivalents for the families above) | [`## test`](#test) step 4 + [`doca-debug ## test`](../../doca-debug/TASKS.md#test) step 3 | The saved file is consumed by the debug session's read-only triple. |
| Re-confirm after a state change | Any of the above, re-run after driver / firmware / mode / container change | [`## test`](#test) eval loop | The post-change output reflects the change; a stale snapshot is the failure mode. |

Three cross-cutting rules for this appendix:

- **Never invent a `doca_caps` flag.** `--help` is the contract;
  prose-derived flags are the most common hallucination failure for
  this skill.
- **Empty output is an answer.** Re-running the same invocation
  hoping for different output is the wrong move; route to
  [`## debug`](#debug) and walk the layers instead.
- **Cross-link instead of duplicate.** Cross-cutting commands
  (`pkg-config --modversion`, `dmesg`, `mlxconfig -d <bdf> q`) live
  in [`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
  this appendix names only `doca_caps`-specific invocations.

## Deferred task verbs

The four verbs below are not `doca_caps` work and should be routed
out before the agent does any of them under this skill's name.

- **install** ⇒ [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
  (and [`## no-install`](../../doca-setup/TASKS.md#no-install) for
  the public NGC DOCA container path). `doca_caps` is shipped by
  the install; this skill does not own the install workflow.
- **build a custom DOCA application** ⇒
  [`doca-programming-guide ## build`](../../doca-programming-guide/TASKS.md#build)
  for the cross-library pattern, plus the matching `libs/<library>`
  skill for library-specific build details. `doca_caps` is not a
  template; it is a tool the user runs against an existing install.
- **library-internal capability check** (e.g. Flow's pipe-creation
  capability matrix, RDMA's verbs-level features) ⇒ matching
  `libs/<library>` skill. `doca_caps` only exposes the *coarse*
  per-device per-library capability surface.
- **streaming telemetry / live metrics** ⇒ not a `doca_caps`
  feature. The DOCA Telemetry Service (DTS) is the documented
  telemetry surface; routing belongs in
  [`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).

## Cross-cutting

A few rules that apply across every verb in this file, restated
here so they are visible at the point of action and not buried in
[`SKILL.md`](SKILL.md):

- The **public DOCA Capabilities Print Tool guide** plus the
  installed `--help` are the joint source of truth. When they
  disagree (e.g. a flag landed in a release this skill was not
  written against), the *installed* `--help` wins for the user's
  actual run.
- `doca_caps` is **read-only and side-effect-free**. Re-running it
  is always safe; an agent uncertain whether to run it should run it.
- **Quote, do not paraphrase.** The capability snapshot is the
  artifact downstream debug consumes; reformatting it loses
  fidelity that the rest of the bundle's procedures depend on.
- This skill **assumes a healthy DOCA install** (or the public NGC
  DOCA container) at the management endpoint. If the install is in
  doubt, route to
  [`doca-setup`](../../doca-setup/SKILL.md) before running anything
  else here.
