---
license: Apache-2.0
name: doca-pipeline-language-developer-tools
description: >
  Use this skill when the operator is driving the DOCA Pipeline
  Language (DPL) Developer Tools to inspect or debug DPL programs on
  a BlueField DPU — DPL Nspect for runtime state such as tables,
  keys, and counters, DPL Debugger for packet flow, DPL Admin for
  dynamic configuration changes in the DPL Runtime Daemon, P4
  Runtime Shell as a P4Runtime control plane manager, or DPL Runtime
  Controller SDK for custom P4Runtime applications and DPL shared
  memory APIs. Trigger even when the user does not explicitly mention
  "DPL Developer Tools" — implicit phrasings include "inspect the DPL
  tables on the DPU", "why can't I start a DPL debug session",
  "change a dpl_rtd configuration item without restarting", "build my
  own P4Runtime app for DPL", or "visualize packet flow through my
  DPL program". Refuse and route elsewhere: authoring, compiling and
  deploying DPL programs to the DOCA Pipeline Language Service Guide,
  DOCA install/bring-up to doca-setup,
  and hardware or firmware state changes to doca-hardware-safety.
metadata:
  author: NVIDIA <doca-feedback@nvidia.com>
  kind: tool
  doca_version: "3.5"
  tags:
    - doca
    - dpl
    - developer-tools
    - p4runtime
    - debug
    - nspect
compatibility: >
  Requires a BlueField DPU configured to operate in DPU mode with the
  DPL Runtime Service container deployed and running, the dpl_rtd
  process active, and the DPL Development Container (publicly
  available on NGC) providing the developer tools. Debug sessions
  additionally require a DPL program compiled with the DPL Compiler
  debug flag -g enabled and loaded into dpl_rtd using the P4Runtime
  controller. The validation steps require a live DOCA install.
---

# DOCA Pipeline Language Developer Tools

## Scope boundaries

These requests are out of scope: say so, name the skill that owns
the request, and stop.

- installing DOCA and bringing up the DPL Runtime Service container:
  `doca-setup/SKILL.md`. Name `doca-setup` in the answer and route the
  whole install/bring-up there; do not walk the install steps from this
  skill or from the DPL Installation Guide.
- DPU mode or firmware state changes: `doca-hardware-safety/SKILL.md`.
- which DOCA version is installed, version compatibility: `doca-version/SKILL.md`.
- authoring, compiling and deploying DPL programs (DOCA Pipeline Language Service Guide): `doca-public-knowledge-map/SKILL.md`.

## Examples

- **"How do I inspect the DPL runtime tables on the DPU?"** — this is
  in scope. Use DPL Nspect; do not author or compile DPL programs here.
- **"I want to visualize packet flow through my DPL program."** — this
  is in scope. Use DPL Debugger; ensure the program was compiled with
  the debug flag `-g` and loaded into `dpl_rtd` using the P4Runtime
  controller.
- **"How do I dynamically change a configuration item in dpl_rtd?"** —
  this is in scope. Use DPL Admin; inspect current state first.
- **"Can I build a custom P4Runtime application for DPL?"** — this is
  in scope. Use the DPL Runtime Controller SDK.
- **"My DPL program is loaded but I cannot start a debug session."** —
  check the debug-flag prerequisite and the P4Runtime load path before
  assuming the tool is broken.

## Quick facts

The authoritative source for the tool names, flags, and paths is the public
DOCA Pipeline Language Developer Tools guide for your release; confirm them
with the installed tool's help before quoting any of them. As that guide
describes it:

- The developer tools are a suite for validating and debugging DPL
  applications: DPL Compiler, P4 Runtime Shell, DPL Nspect, DPL Debugger,
  DPL Admin, and DPL Runtime Controller SDK.
- All tools interact with the DPL Runtime Daemon (`dpl_rtd`) running on the
  BlueField DPU.
- DPL Compiler converts DPL source code into a binary format that can be
  loaded by `dpl_rtd`.
- P4 Runtime Shell is an open-source control plane manager implementing the
  P4Runtime API.
- DPL Nspect is a command-line interface for inspecting DPL runtime state,
  including tables, keys, counters, and more.
- DPL Debugger is a graphical interface for visualizing and debugging packet
  flow through the DPL program.
- DPL Admin is a CLI tool for dynamically modifying configuration items in
  the DPL Runtime Daemon.
- DPL Runtime Controller SDK is an SDK for building custom P4Runtime
  applications or integrating with DPL shared memory APIs.
- Prerequisites: the DPL Runtime Service container must be deployed and
  running on the DPU, the `dpl_rtd` process must be active, and the DPU must
  be configured to operate in DPU mode.
- Starting a debug session requires the DPL program to be compiled for
  debugging: compile with the DPL Compiler debug flag `-g` enabled, then load
  the compiled binary into `dpl_rtd` using the P4Runtime controller.
- The developer tools are included in the DPL Development Container, which is
  publicly available on NGC. Installation instructions refer to the DPL
  Installation Guide.
- For instructions on creating, compiling, and deploying DPL programs, refer
  to the DOCA Pipeline Language Service Guide.
- Support and feedback: visit the NVIDIA DOCA Enterprise Networking Forum.
  Registration with the NVIDIA Developer Program is required to post;
  registration is not required to view existing discussions. For help with
  registration, refer to the FAQ.

## Instructions

Follow Loading order below. Open the tasks companion for
inspect / debug / modify / build / shell. Open the capabilities companion for
what the tools inspect and change. Do not open every companion file.

**Where to start:** This is a tool skill for invoking the DOCA Pipeline
Language Developer Tools — the suite for inspecting and debugging DPL
programs on BlueField DPUs. Open the tasks companion and start at
`## inspect` for DPL Nspect runtime-state inspection, `## debug` for DPL
Debugger packet-flow visualization, `## modify` for DPL Admin dynamic
configuration changes, `## build` for DPL Runtime Controller SDK custom
P4Runtime applications, and `## shell` for P4 Runtime Shell control-plane
use. Open the capabilities companion when the question is *what state can
the tools inspect, what do they change, and what prerequisites gate their
use*. If the user has not installed DOCA yet, route to
`doca-setup/SKILL.md` first. If the user is asking about DPL program
authoring, compiling, or deploying rather than inspecting or debugging a
loaded program, route to the DOCA Pipeline Language Service Guide, not this
skill. Any step that changes hardware or firmware state routes through
`doca-hardware-safety/SKILL.md`. Version questions route through
`doca-version/SKILL.md`; installs route through `doca-setup/SKILL.md`.

**How to use this skill.** Read this file, then search the companion file
for the `## <name>` heading the question needs and read that whole section.
Answer completely from it, quoting only its symbols and commands. Do not
browse docs; use the entry points in `doca-public-knowledge-map`. Give
environment checks as commands for the user's DOCA host, quoting only
commands documented in the public guide or installed tool help. For
out-of-scope requests, name the skill that owns them and stop.

## Example questions this skill answers well

The CLASSES of DPL developer-tools questions this skill is built to answer,
each with one worked example. The class is the load-bearing piece; the
worked example is one instance.

- **"What tools are available for inspecting and debugging DPL programs?"**
  — worked example: *"I have a DPL program loaded on the DPU and need to
  see its tables, keys, counters, or packet flow"*. Answered by the tool
  inventory in the capabilities companion + the tasks companion.
- **"How do I inspect DPL runtime state?"** — worked example: *"I want to
  list the tables and counters currently active in dpl_rtd"*. Answered by
  the DPL Nspect workflow in the tasks companion.
- **"How do I debug packet flow through a DPL program?"** — worked example:
  *"packets are not matching the expected table entries; I need to visualize
  the path through the DPL program"*. Answered by the DPL Debugger
  prerequisites (debug flag `-g`, P4Runtime load) in the capabilities
  companion + the tasks companion.
- **"How do I dynamically change configuration items in dpl_rtd?"** —
  worked example: *"I need to update a configuration item without restarting
  the DPL Runtime Daemon"*. Answered by the DPL Admin workflow in the tasks
  companion.
- **"How do I build a custom P4Runtime application or integrate with DPL
  shared memory APIs?"** — worked example: *"I want to write my own
  control-plane application for DPL instead of using the P4 Runtime Shell"*.
  Answered by the DPL Runtime Controller SDK workflow in the tasks companion.
- **"What are the prerequisites for using the developer tools?"** — worked
  example: *"the tools cannot see dpl_rtd; is the DPL Runtime Service
  container running and is the DPU in DPU mode?"*. Answered by the
  prerequisites in the capabilities companion + the tasks companion.

## When to load this skill

Load this skill when the user is — or the agent needs to — drive the DOCA
Pipeline Language Developer Tools on a real BlueField DPU with the DPL
Runtime Service running. Concretely:

- Inspecting DPL runtime state (tables, keys, counters) with DPL Nspect.
- Visualizing or debugging packet flow through a DPL program with DPL
  Debugger.
- Dynamically modifying configuration items in `dpl_rtd` with DPL Admin.
- Building custom P4Runtime applications or integrating with DPL shared
  memory APIs using the DPL Runtime Controller SDK.
- Using the P4 Runtime Shell as a P4Runtime control plane manager.
- Preparing a debug session: ensuring the DPL program was compiled with the
  debug flag `-g` and loaded into `dpl_rtd` using the P4Runtime controller.
- Diagnosing why the developer tools cannot interact with `dpl_rtd` (DPL
  Runtime Service container not running, `dpl_rtd` not active, DPU not in
  DPU mode).

Do **not** load this skill for general DOCA orientation, DPL program
authoring/compilation/deployment, DOCA install, or hardware/firmware state
changes. For those, route to
`doca-public-knowledge-map/SKILL.md`,
the DOCA Pipeline Language Service Guide,
`doca-setup/SKILL.md`, or
`doca-hardware-safety/SKILL.md`.

## What this skill provides

This is a **thin loader**. Substantive material lives in two companion
files:

- the capabilities companion — what the developer tools inspect and change:
  the tool inventory (DPL Compiler, P4 Runtime Shell, DPL Nspect, DPL
  Debugger, DPL Admin, DPL Runtime Controller SDK), the read-only vs
  state-changing operation split (Nspect and Debugger read-only; Admin
  modifies; Compiler and SDK build), the prerequisites (DPL Runtime Service
  container, `dpl_rtd` active, DPU mode, debug flag `-g`, P4Runtime load),
  availability via the DPL Development Container on NGC, support/feedback via
  the NVIDIA DOCA Enterprise Networking Forum, and the high-stakes safety
  policy that dynamic configuration changes through DPL Admin affect the
  running DPL Runtime Daemon and must be gated on a clean inspection.
- the tasks companion — step-by-step workflows for the in-scope task verbs:
  `inspect` (DPL Nspect runtime state), `debug` (DPL Debugger packet flow,
  including the debug-flag prerequisite), `modify` (DPL Admin dynamic
  configuration), `build` (DPL Runtime Controller SDK custom P4Runtime
  applications / shared memory integration), `shell` (P4 Runtime Shell
  control plane), plus a `Deferred task verbs` block and a `Command appendix`
  that honors the bundle's
  `doca-structured-tools-contract/SKILL.md`
  preamble.

The skill assumes a BlueField DPU where the DPL Runtime Service container is
already deployed and running, `dpl_rtd` is active, and the DPU is configured
to operate in DPU mode.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or scripts bundle. To keep
the boundary clean, it deliberately does not contain — and pull requests
should not add:

- **Verbatim subcommand names, flag inventories beyond the documented debug
  flag `-g`, default paths, port numbers, or output column names.** The
  public DOCA Pipeline Language Developer Tools guide on `docs.nvidia.com`
  and the installed tool help on the user's version are the joint source of
  truth; copying them here pins the skill to one release and silently rots
  when the tools evolve. The skill routes the agent at those sources instead.
- **Pre-baked example output.** Output is install-, version-, and
  deployment-specific. A captured example will mislead an operator on a
  different platform / state.
- **Wrappers, parsers, or scripts** in any language that consume the tools'
  output. The output format is documented; users who want to script against
  it should read the live guide and write the parser against their installed
  version.
- **A `samples/` or `reference/` subtree.** This is a thin loader for a
  documented DOCA artifact; substantive material lives on the public page and
  in the installed tool help.

## Loading order

1. Read this skill first to confirm the user's question is in scope (the
   user wants to inspect or debug DPL programs on a BlueField DPU, not
   author, compile, or deploy DPL programs).
2. **For what the tools inspect and change, the tool inventory, the
   read-only vs state-changing split, prerequisites, availability, support,
   and safety posture, see the capabilities companion.**
3. **For the documented invocations and the workflows — `inspect`, `debug`,
   `modify`, `build`, `shell`, plus the `Command appendix` — see the tasks
   companion.**

## Related skills

- `doca-public-knowledge-map/SKILL.md`
  — routing to the public DOCA Pipeline Language Developer Tools guide and
  the rest of the public DOCA documentation set. The canonical URL is reached
  via `doca-public-knowledge-map ## DOCA tools`
  (`doca-public-knowledge-map/SKILL.md#doca-tools`).
- `doca-version/SKILL.md` — canonical DOCA
  version-handling rules. Version questions route here; the capabilities
  companion redirects to this skill for the body.
- `doca-setup/SKILL.md` — env preparation,
  install verification, DPL Runtime Service container deployment, and the *I
  have no install yet* path with the public NGC DPL Development Container.
  This skill assumes its preconditions are satisfied.
- `doca-hardware-safety/SKILL.md`
  — any step that changes hardware or firmware state routes here.
- `doca-structured-tools-contract/SKILL.md`
  — the bundle's detect → prefer → fall back → report contract for structured
  helper tools. The Command appendix in the tasks companion honors this
  contract.
- `doca-debug/SKILL.md` — the cross-cutting
  debug ladder. The DPL developer tools slot in at the *runtime* layer as
  inspection/debug tooling whose layered error surface escalates to the
  cross-cutting ladder when the cause is below DOCA.
