# DOCA Pipeline Language Developer Tools — Tasks

**Where to start:** The verbs that carry real workflow content are
`## inspect`, `## debug`, `## modify`, `## build`, and `## shell`.
The other task verbs from the bundle's uniform contract
(`configure`, `run`, `test`) carry routing stubs because the DPL
Developer Tools are a shipped suite in the DPL Development
Container, not a source artifact the external operator compiles or
patches — the *thing the agent modifies* is the runtime state and
configuration of `dpl_rtd`, not the tools themselves. The
`## debug` verb is an iterative loop (prerequisite check →
read-only inspection → visualize → escalate), not a one-shot pass —
see the eval-loop overlay in `## debug` below.

This file is loaded by [`SKILL.md`](SKILL.md) after
[`CAPABILITIES.md`](CAPABILITIES.md). It walks the agent through the
five task verbs every artifact in this bundle exposes
(`inspect / debug / modify / build / shell`), then explicitly
defers task verbs that do not belong here, and ends with the
`Command appendix` honoring the bundle's
`doca-structured-tools-contract/SKILL.md`
preamble.

For the DPL Developer Tools, the verbs that carry real workflow
content are `inspect`, `debug`, `modify`, `build`, and `shell`. The
other verbs *exist as anchors* because the agent's task-verb
contract is uniform across libraries, services, and tools — and
each one carries a meaningful **routing stub** that names where the
user's question really belongs.

## inspect

The DPL Developer Tools' *inspection* is the read-only DPL Nspect
workflow documented in
[`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes) —
a command-line interface for inspecting DPL runtime state,
including tables, keys, counters, and more — layered on top of the
precondition that the DPL Runtime Service container is deployed and
running on the DPU, the `dpl_rtd` process is active, and the DPU is
configured to operate in DPU mode. DPL Nspect does not change
runtime state; it is the gate every state-changing operation must
pass through first.

Steps the agent should walk the user through, in order:

1. **Confirm the environment prerequisites.** Before invoking DPL
   Nspect, verify the three prerequisites named in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes):
   the DPL Runtime Service container is deployed and running on the
   DPU, the `dpl_rtd` process is active, and the DPU is configured
   to operate in DPU mode. If any of these is not true, the tools
   cannot interact with `dpl_rtd` at all — route to
   `doca-setup ## configure` (`doca-setup/TASKS.md#configure`)
   before proceeding.
2. **Identify what to inspect.** DPL Nspect exposes runtime state —
   tables, keys, counters, and more. The exact invocation, flag
   names, and output column names live in the public DOCA Pipeline
   Language Developer Tools guide for the user's release and in the
   installed tool help; the agent must not quote commands not
   present there. Ask the user which runtime state they need to
   inspect, then point them at the documented entry point.
3. **Run DPL Nspect against the running `dpl_rtd`.** This is a
   read-only operation. The agent should recommend the user run the
   tool's help first to confirm the installed version's exact
   syntax, then run the inspection command.
4. **Interpret the output against expected DPL program state.**
   Compare the observed tables, keys, and counters to what the DPL
   program should contain. A mismatch is a symptom, not a cause —
   do not modify anything yet; move to
   [`## debug`](#debug).
5. **Record the current state before any modification.** Per
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   a clean read-only inspection is the gate for any state-changing
   operation. Capture the observed state so a later modification can
   be compared against a known baseline.

## debug

The DPL Developer Tools' *debug* is the DPL Debugger workflow
documented in
[`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes) —
a graphical interface for visualizing and debugging packet flow
through the DPL program — with two additional prerequisites beyond
the base environment: the DPL program must be compiled for
debugging with the DPL Compiler debug flag `-g` enabled, and the
compiled binary must be loaded into `dpl_rtd` using the P4Runtime
controller. DPL Debugger is read-only; it does not change runtime
state.

Steps the agent should walk the user through, in order:

1. **Check the debug-session prerequisites first.** Before assuming
   the tool is broken, confirm the two debug-specific prerequisites
   from
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes):
   the DPL program was compiled with the debug flag `-g` enabled,
   and the compiled binary was loaded into `dpl_rtd` using the
   P4Runtime controller. If either is missing, the debug session
   cannot start — route to the DOCA Pipeline Language Service Guide
   for compiling and loading DPL programs, not to this skill.
2. **Confirm the base environment prerequisites.** The same three
   prerequisites from [`## inspect`](#inspect) apply: DPL Runtime
   Service container deployed and running, `dpl_rtd` process active,
   DPU configured to operate in DPU mode. A missing base
   prerequisite produces a silent break — the debugger cannot see
   the daemon.
3. **Start the DPL Debugger graphical interface.** The exact launch
   command lives in the public guide and installed tool help; the
   agent must not invent it. The debugger visualizes packet flow
   through the DPL program.
4. **Visualize packet flow and compare to expected behavior.**
   Observe the path packets take through the DPL program and compare
   it to the expected table matches. A deviation is a symptom to
   investigate, not a state change to make blindly.
5. **Map the symptom to the error taxonomy before any
   state-changing intervention.** Use
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   to place the symptom at the right layer: environment /
   prerequisites, debug-session prerequisite, tool-specific
   operation, or below DOCA. If the cause is below DOCA, route to
   `doca-debug/SKILL.md`; any hardware or firmware state change
   routes through `doca-hardware-safety/SKILL.md`.
6. **Iterative loop.** After any state-changing operation (for
   example, a DPL Admin modification or a reload of the compiled
   binary), re-run the read-only inspection and debug visualization
   before concluding the issue is resolved. The debug verb is not a
   one-shot pass; it is the loop that re-opens the smoke after every
   change.

## modify

The DPL Developer Tools' *modify* is the DPL Admin workflow
documented in
[`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes) —
a CLI tool for dynamically modifying configuration items in the DPL
Runtime Daemon without restarting it. This is a state-changing
operation and is gated on a clean read-only inspection first.

Steps the agent should walk the user through, in order:

1. **Establish the current state with DPL Nspect.** Per
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   run the read-only inspection from [`## inspect`](#inspect) and
   record the current configuration items before any modification.
   Do not skip this gate.
2. **Identify the configuration item to modify.** DPL Admin
   dynamically modifies configuration items in the running DPL
   Runtime Daemon. The exact item names and modification syntax live
   in the public guide and installed tool help; the agent must not
   invent them. Ask the user which configuration item they need to
   change and why.
3. **Run DPL Admin to apply the modification.** This changes the
   running daemon's configuration without restarting it. The agent
   should recommend the user run the tool's help first to confirm
   the installed version's exact syntax, then apply the change.
4. **Re-run DPL Nspect to confirm the change took effect.** Compare
   the post-modification state to the pre-modification baseline
   captured in step 1. If the change did not take effect or had an
   unexpected side effect, do not apply further changes blindly —
   move to [`## debug`](#debug).
5. **Plan the rollback path.** Because DPL Admin modifies the
   running daemon, the agent should record the previous
   configuration item value so it can be restored if the change
   causes a data-path break. If a rollback requires a hardware or
   firmware state change, route through
   `doca-hardware-safety/SKILL.md`.

## build

The DPL Developer Tools' *build* is the DPL Runtime Controller SDK
workflow documented in
[`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes) —
an SDK for building custom P4Runtime applications or integrating
with DPL shared memory APIs. The SDK is build-time; it does not
change running state by itself.

Steps the agent should walk the user through, in order:

1. **Confirm the DPL Development Container is available.** The
   developer tools, including the SDK, are included in the DPL
   Development Container, which is publicly available on NGC.
   Installation instructions refer to the DPL Installation Guide.
   If the container is not installed, route to
   `doca-setup ## install` (`doca-setup/TASKS.md`).
2. **Identify the build target.** The DPL Runtime Controller SDK
   supports building custom P4Runtime applications or integrating
   with DPL shared memory APIs. The exact APIs, headers, and build
   commands live in the public guide and installed SDK
   documentation; the agent must not invent them.
3. **Build the custom application or integration.** This is a
   build-time operation. The agent should point the user at the
   documented SDK entry point and the installed tool help.
4. **Loading the result is a separate, state-changing step.** If
   the user builds a custom P4Runtime application that will load a
   compiled binary into `dpl_rtd`, that loading step is
   state-changing and must be gated on a clean read-only inspection
   per
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
   The debug-session prerequisites from [`## debug`](#debug) also
   apply if the loaded program is intended for debugging.
5. **Authoring, compiling, and deploying DPL programs themselves
   route to the DOCA Pipeline Language Service Guide.** The DPL
   Compiler is listed in the tool inventory but is not driven by
   this skill.

## shell

The DPL Developer Tools' *shell* is the P4 Runtime Shell workflow
documented in
[`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes) —
an open-source control plane manager implementing the P4Runtime
API. It is a read-only control-plane interaction with `dpl_rtd`.

Steps the agent should walk the user through, in order:

1. **Confirm the environment prerequisites.** The same three base
   prerequisites from [`## inspect`](#inspect) apply: DPL Runtime
   Service container deployed and running, `dpl_rtd` process active,
   DPU configured to operate in DPU mode.
2. **Use the P4 Runtime Shell to drive the P4Runtime API.** The
   shell is an open-source control plane manager implementing the
   P4Runtime API. The exact commands and flags live in the public
   guide and installed tool help; the agent must not invent them.
3. **Read-only control-plane interaction.** The shell itself is a
   read-only control-plane interaction. However, loading a compiled
   binary into `dpl_rtd` using the P4Runtime controller is a
   state-changing operation and must be gated on a clean read-only
   inspection per
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
4. **If the shell cannot reach `dpl_rtd`**, map the symptom to
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   before any state-changing intervention.

## Deferred task verbs

The bundle's uniform task-verb contract includes `configure`, `run`,
and `test`. For the DPL Developer Tools, these verbs carry routing
stubs because the tools are a shipped suite, not a source artifact
the operator compiles or patches.

- **`configure`** — The DPL Developer Tools do not carry a separate
  config file or daemon contract of their own beyond the DPL Runtime
  Service container and DPU mode prerequisites. Configuration
  questions route to `doca-setup ## configure`
  (`doca-setup/TASKS.md#configure`) and the DPL Installation Guide.
- **`run`** — The tools are invoked from the DPL Development
  Container against the running `dpl_rtd`; there is no separate run
  verb beyond invoking the specific tool. Run questions route to the
  relevant tool section above (`## inspect`, `## debug`,
  `## modify`, `## build`, or `## shell`).
- **`test`** — There is no separate test verb for the DPL Developer
  Tools. Validation is the inspect-before-modify loop in
  [`## inspect`](#inspect) and [`## debug`](#debug). Test questions
  route to those sections.

## Command appendix

This appendix honors the bundle's
`doca-structured-tools-contract/SKILL.md`
preamble: detect → prefer → fall back → report.

The public DOCA Pipeline Language Developer Tools guide for the
user's release and the installed tool help are the joint source of
truth for every command, flag, path, and output column name. This
skill deliberately does not pin verbatim commands because they can
change between releases — see
[`SKILL.md ## What this skill deliberately does not ship`](SKILL.md#what-this-skill-deliberately-does-not-ship).

For each tool, the agent must:

1. **Detect** the installed tool in the DPL Development Container.
2. **Prefer** the documented invocation from the public guide for
   the user's DOCA release.
3. **Fall back** to the installed tool's help when the public guide
   and the installed version disagree.
4. **Report** the exact command used, including the source
   (public guide or installed help), so the operator can reproduce
   it.

| Tool | Documented entry point | Risk class | Notes |
| --- | --- | --- | --- |
| DPL Nspect | Public guide + installed tool help | Read-only | Inspect tables, keys, counters, and more |
| DPL Debugger | Public guide + installed tool help | Read-only | Requires debug flag `-g` and P4Runtime load |
| DPL Admin | Public guide + installed tool help | State-changing | Gate on clean inspection |
| DPL Runtime Controller SDK | Public guide + installed SDK documentation | Build-time | Build custom P4Runtime apps / shared memory APIs |
| P4 Runtime Shell | Public guide + installed tool help | Read-only control-plane | Open-source P4Runtime API manager |
| DPL Compiler | Public guide + installed tool help | Out of scope | Route to DOCA Pipeline Language Service Guide |

## Cross-cutting

- **Version questions** route through `doca-version/SKILL.md`. The
  public guide is versioned; this skill's facts are drawn from the
  DOCA 3.5.0 archive.
- **Installs** route through `doca-setup/SKILL.md`. The DPL
  Development Container is publicly available on NGC; installation
  instructions refer to the DPL Installation Guide.
- **Hardware or firmware state changes** route through
  `doca-hardware-safety/SKILL.md`. This skill does not authorize any
  step that changes hardware or firmware state.
- **Authoring, compiling, and deploying DPL programs** route to the
  DOCA Pipeline Language Service Guide, not this skill.
- **Below-DOCA debugging** routes through `doca-debug/SKILL.md` when
  the error taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  places the cause below DOCA.
- **Public guide pointer** is reached via
  `doca-public-knowledge-map ## DOCA tools`
  (`doca-public-knowledge-map/SKILL.md#doca-tools`).
