# DOCA Pipeline Language Developer Tools — Capabilities

**Where to start:** The DPL Developer Tools are a suite of six
artifacts that all interact with the DPL Runtime Daemon (`dpl_rtd`)
on a BlueField DPU. The pattern overview below names the recurring
developer-tools questions. Pick the pattern first, then drill into
the H2 that owns the substance. For the *how* of executing each
pattern, jump to [TASKS.md](TASKS.md). For authoring, compiling,
and deploying DPL programs, see the DOCA Pipeline Language Service
Guide; for install and bring-up, see
`doca-setup/CAPABILITIES.md`.

This file is loaded by [`SKILL.md`](SKILL.md). It documents *what
the tools inspect and change*, *which tool shapes are documented*,
*what versions they ship in*, *the layered error and observability
surfaces*, and *the high-stakes safety policy that gates
state-changing operations*. For step-by-step invocations and the
inspect-before-modify workflow, see [`TASKS.md`](TASKS.md).

## Pattern overview

Every DPL Developer Tools question this skill teaches resolves into
one of SIX patterns. The patterns are CLASSES — they apply across
every DOCA install that ships the tools, not just one platform.

| Developer-tools pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Inspect DPL runtime state | Use DPL Nspect to read tables, keys, counters, and other runtime state from `dpl_rtd` without changing it | [`## Capabilities and modes`](#capabilities-and-modes) read-only split + [TASKS.md ## inspect](TASKS.md#inspect) |
| 2. Debug packet flow through a DPL program | Use DPL Debugger to visualize and debug packet flow; requires the program to be compiled with the debug flag `-g` and loaded into `dpl_rtd` using the P4Runtime controller | [`## Capabilities and modes`](#capabilities-and-modes) prerequisites + [TASKS.md ## debug](TASKS.md#debug) |
| 3. Dynamically modify configuration items in `dpl_rtd` | Use DPL Admin to change configuration items in the running DPL Runtime Daemon without restarting it | [`## Capabilities and modes`](#capabilities-and-modes) state-changing split + [TASKS.md ## modify](TASKS.md#modify) |
| 4. Build a custom P4Runtime application or integrate with DPL shared memory APIs | Use the DPL Runtime Controller SDK to build custom control-plane applications instead of the P4 Runtime Shell | [`## Capabilities and modes`](#capabilities-and-modes) build split + [TASKS.md ## build](TASKS.md#build) |
| 5. Use the P4 Runtime Shell as a control plane manager | Drive the P4Runtime API through the open-source P4 Runtime Shell to manage the DPL program | [`## Capabilities and modes`](#capabilities-and-modes) tool inventory + [TASKS.md ## shell](TASKS.md#shell) |
| 6. Diagnose why the tools cannot interact with `dpl_rtd` | Map the symptom (tools cannot see the daemon, debug session cannot start) to the prerequisite layer before any state-changing intervention | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **The tools sit on top of the DPL Runtime Daemon.** Every tool in
  the suite interacts with `dpl_rtd` running on the BlueField DPU.
  A missing DPL Runtime Service container, an inactive `dpl_rtd`
  process, or a DPU not configured to operate in DPU mode produces
  a *silent* break — the tools cannot inspect, debug, or modify
  anything. The agent must say so before any state-changing
  operation, and must recommend the prerequisite check documented
  in [`## Safety policy`](#safety-policy) before assuming a tool is
  broken.
- **Read-only inspection and state-changing modification are
  different risk classes.** DPL Nspect and DPL Debugger read
  runtime state; DPL Admin modifies configuration items in the
  running daemon; loading a compiled binary into `dpl_rtd` via the
  P4Runtime controller changes the runtime program. An agent that
  recommends a modification without first inspecting current state
  is skipping the gate documented in
  [`## Safety policy`](#safety-policy).

## Capabilities and modes

The DOCA Pipeline Language Developer Tools are a documented DOCA
suite that ships in the DPL Development Container, which is
publicly available on NGC. The exact installed location, binary
names, flag inventories beyond the documented debug flag `-g`,
default paths, and per-release availability live in the public
DOCA Pipeline Language Developer Tools guide reachable via
`doca-public-knowledge-map ## DOCA tools`
(`doca-public-knowledge-map/SKILL.md#doca-tools`)
and on the installed tool help; the skill deliberately does not
pin them — see
[`SKILL.md ## What this skill deliberately does not ship`](SKILL.md#what-this-skill-deliberately-does-not-ship).

**Use case — what the tools are *for*.** The DPL Developer Tools
are the documented suite for **validating and debugging DPL
applications** on BlueField DPUs. An operator with a DPL program
loaded in `dpl_rtd` uses the tools to inspect runtime state,
visualize packet flow, dynamically adjust configuration, or build
a custom control-plane application. The tools do not author,
compile, or deploy DPL programs; those activities belong to the
DOCA Pipeline Language Service Guide.

**Tool inventory — the load-bearing concept.** Every
developer-tools question commits to one of the six tools below;
omitting the tool produces an ambiguous answer the agent must flag
back to the user.

| Tool | What it does | Risk class |
| --- | --- | --- |
| DPL Compiler | Converts DPL source code into a binary format that can be loaded by `dpl_rtd` | Out of scope for this skill; route to the DOCA Pipeline Language Service Guide |
| P4 Runtime Shell | Open-source control plane manager implementing the P4Runtime API | Read-only control-plane interaction |
| DPL Nspect | Command-line interface for inspecting DPL runtime state, including tables, keys, counters, and more | Read-only |
| DPL Debugger | Graphical interface for visualizing and debugging packet flow through the DPL program | Read-only |
| DPL Admin | CLI tool for dynamically modifying configuration items in the DPL Runtime Daemon | State-changing |
| DPL Runtime Controller SDK | SDK for building custom P4Runtime applications or integrating with DPL shared memory APIs | Build-time; does not change running state by itself |

**Prerequisites — the gate every pattern must pass.** Before any
tool can interact with `dpl_rtd`, the following must be true:

- The DPL Runtime Service container is deployed and running on the
  DPU.
- The `dpl_rtd` process is active.
- The DPU is configured to operate in DPU mode.

Starting a debug session adds two more prerequisites:

- The DPL program is compiled for debugging with the DPL Compiler
  debug flag `-g` enabled.
- The compiled binary is loaded into `dpl_rtd` using the P4Runtime
  controller.

**Availability.** The developer tools are included in the DPL
Development Container, which is publicly available on NGC.
Installation instructions refer to the DPL Installation Guide; DOCA
install and DPL Runtime Service bring-up requests route to `doca-setup`.

## Version compatibility

This skill's facts are drawn from the DOCA 3.5.0 archive of the
public DOCA Pipeline Language Developer Tools guide. The public
guide is versioned; for any other release, route version questions
through `doca-version/SKILL.md` and read the public guide for that
release. The skill does not pin binary names, flags beyond the
documented debug flag `-g`, default paths, or output column names
because those can change between releases.

## Error taxonomy

Map a DPL Developer Tools symptom to the right layer before any
state-changing intervention. The layers are ordered from the most
common environmental cause to the deepest DOCA cause.

| Layer | Symptom class | What to check first |
| --- | --- | --- |
| 0 — Environment / prerequisites | Tools cannot interact with `dpl_rtd` at all | DPL Runtime Service container deployed and running; `dpl_rtd` process active; DPU configured to operate in DPU mode |
| 1 — Debug-session prerequisite | Debug session cannot start | DPL program compiled with the debug flag `-g`; compiled binary loaded into `dpl_rtd` using the P4Runtime controller |
| 2 — Tool-specific operation | Nspect does not show expected tables/keys/counters; Debugger does not visualize expected packet flow; Admin modification fails or has unexpected effect | Re-run the read-only inspection first; confirm the tool is pointed at the correct `dpl_rtd` and the program state matches expectations |
| 3 — Below DOCA | Symptoms persist after layers 0–2 are clean | Route to `doca-debug/SKILL.md`; any hardware or firmware state change routes through `doca-hardware-safety/SKILL.md` |

## Observability

The DPL Developer Tools are themselves the observability surface
for a running DPL program. DPL Nspect exposes runtime state —
tables, keys, counters, and more — as a command-line interface.
DPL Debugger exposes packet flow through the DPL program as a
graphical interface. P4 Runtime Shell exposes the P4Runtime API
control plane. DPL Admin exposes and changes configuration items
in the DPL Runtime Daemon. No separate logging or metrics surface
is documented in the public DPL Developer Tools guide; the agent must not invent
one.

## Safety policy

- **Read-only tools first.** DPL Nspect and DPL Debugger do not
  change runtime state. Use them to establish the current state
  before any modification.
- **State-changing operations are gated.** DPL Admin dynamically
  modifies configuration items in the running DPL Runtime Daemon;
  loading a compiled binary into `dpl_rtd` via the P4Runtime
  controller changes the runtime program. Both require a clean
  read-only inspection first and must not be performed blindly.
- **Hardware and firmware state changes route through
  `doca-hardware-safety/SKILL.md`.** This skill does not authorize
  any step that changes hardware or firmware state.
- **Out-of-scope authoring, compiling, and deploying route to the
  DOCA Pipeline Language Service Guide.** The DPL Compiler is
  listed in the tool inventory but is not driven by this skill.

## Public-source pointer

The only public-source pointer documented in the public DPL
Developer Tools guide is the P4 Runtime Shell, described as an open-source control plane
manager implementing the P4Runtime API. The DPL Development
Container is publicly available on NGC as a distribution channel;
no source repository for the other tools is documented in the
guide. The agent must not invent source URLs.
