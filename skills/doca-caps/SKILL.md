---
license: Apache-2.0
name: doca-caps
description: >
  Use this skill when the user wants to invoke the read-only doca_caps
  CLI to ask what DOCA sees on this host — listing DOCA devices and PCIe
  addresses, listing representor devices, asking which DOCA libraries
  are available on the current OS, checking per-device per-library
  capabilities, scoping output to a specific PCIe address, or capturing
  a side-effect-free capability snapshot for debug or install
  smoke-test. Trigger even when the user does not explicitly mention
  "doca_caps" or "capabilities print tool" — typical implicit phrasings
  include "what does DOCA actually see on this box", "is my BlueField PF
  visible to DOCA", "is Flow available on my RHEL host", "enumerate VF
  representors for pf0", "doca_caps: command not found", or "empty
  output for RDMA, is the tool broken". Refuse and route elsewhere for
  DOCA installation (doca-setup owns it; never walk the steps),
  library-internal capability matrices (Flow pipe creation, RDMA verbs
  features), streaming telemetry / DTS, or modifying the shipped binary.
metadata:
  author: NVIDIA <doca@nvidia.com>
  kind: tool
  tags:
    - doca
    - caps
    - tools
compatibility: >
  Requires DOCA SDK ≥ 2.6.0 installed at /opt/mellanox/doca on Linux
  (Ubuntu 22.04/24.04 or RHEL/SLES) with a BlueField DPU or ConnectX
  NIC; runs identically on the host or on the BlueField Arm side.
  Invokes /opt/mellanox/doca/tools/doca_caps and reads `pkg-config
  doca-common` to confirm the install. The validation steps require a
  live DOCA install.
---

# DOCA Capabilities Print Tool (`doca_caps`)

## Examples

- **"What DOCA devices does this host see?"** — this is in
  scope. Stay on `doca_caps`; do not start a library-internal
  capability matrix here.
- **"Is Flow available on my RHEL host?"** — list libraries
  then scope per-device if needed.

## Instructions

Follow Loading order below. Open the tasks companion for run /
test. Open the capabilities companion for what the CLI reports.
Do not open every companion file.

**Where to start:** This is a tool skill for invoking `doca_caps`,
a side-effect-free CLI. Open the tasks companion and start at
`## run` for the documented invocations, or
`## test` when using `doca_caps` as an install
smoke-test. Open the capabilities companion when the
question is *what kinds of capability families `doca_caps` reports*.

> **Boundary: this skill never installs DOCA.** `doca_caps`
> *verifies* an install and never performs one. A request to install
> DOCA — choosing packages, adding repositories, running a package
> manager — is out of scope here, and stays out of scope even though
> this skill's `## test` verb uses the tool as an install smoke-test
> *afterwards*. Hand the whole request to
> [`doca-setup`](../../doca-setup/SKILL.md) by name. Naming the owner
> and then listing the steps anyway is the failure mode to avoid: do
> not walk the installation, not even as a summary or a caveat.
>
> This narrows *installation only*. It is not a reason to answer
> narrowly about tooling: when the user's question reaches past
> `doca_caps` — what else can I ask this hardware, how do I measure it,
> how do I inspect a running pipeline — name the sibling tool that owns
> it (`doca_bench`, `doca_flow_tune`, `pcc_counters.sh`,
> `doca_comm_channel_admin`, …) and tie it to the question. Pointing at
> the rest of the catalog is this skill's job, not a boundary breach.

## Example questions this skill answers well

The CLASSES of `doca_caps` questions this skill is built to answer,
each with one worked example. The class is the load-bearing piece;
the worked example is one instance.

- **"What DOCA devices does this host see?"** — worked example: *"is
  my BlueField PF visible to DOCA"*. Answered by the device
  enumeration in
  the capabilities companion
  + the `--list-devs` invocation in
  the tasks companion.
- **"Which DOCA libraries are available on this OS?"** — worked
  example: *"is Flow available on my RHEL host"*. Answered by the
  library-availability surface in
  the capabilities companion
  + the library-listing invocation in
  the tasks companion.
- **"Does this device support library X capability Y?"** — worked
  example: *"does this device support Flow hairpin?"*. Answered by
  the per-device per-library capability matrix in
  the capabilities companion
  + the `--pci-addr`-scoped invocation in
  the tasks companion.
- **"What representors are visible to DOCA?"** — worked example:
  *"enumerate VF representors for pf0"*. Answered by the representor
  enumeration in
  the capabilities companion
  + the `--list-rep-devs` invocation in
  the tasks companion.
- **"I want a snapshot of state to attach to my debug session."** —
  worked example: *"save device + library + capability output to a
  file"*. Answered by the snapshot workflow in
  the tasks companion and consumed by
  [`doca-debug`](../../doca-debug/SKILL.md) step 3
  (read-only triple) and
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
- **"`doca_caps` returned nothing for capability Y — what does that
  mean?"** — worked example: *"empty output for RDMA"*. Answered by
  the empty-output interpretation rules in
  the tasks companion +
  the capabilities companion.

## Audience

This skill serves **external operators, developers, and AI agents who
need a side-effect-free way to ask "what does DOCA see on this host?"**
before doing anything that changes state. Concretely:

- An external developer who installed DOCA (or is using the public NGC
  DOCA container per [`doca-setup`](../../doca-setup/SKILL.md))
  and wants to confirm the install can see hardware before writing code.
- A platform operator deploying a DOCA service who wants a documented,
  read-only snapshot of *which DOCA libraries this host actually
  supports* and *which capabilities each DOCA device offers*.
- An AI agent producing a *capability snapshot* artifact during the
  documented setup or programming-guide debug procedures (it's listed
  as the canonical first step in
  [`doca-setup`](../../doca-setup/SKILL.md) and
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md)).

It is **not** for users debugging `doca_caps` itself, and **not** a
substitute for the live public Capabilities Print Tool guide.

`doca_caps` is shipped as a **tool** (a single CLI binary), not a
library you link against. The skill uses the same `kind: tool`
three-file shape as the rest of the bundle so the agent's task-verb
contract (`configure / build / modify / run / test / debug`) is uniform
across libraries, services, and tools — even when individual verbs
collapse to a routing stub for a shipped read-only binary.

## When to load this skill

Load this skill when the user is — or the agent needs to — invoke
`doca_caps` on a real host with DOCA installed (or inside the public
NGC DOCA container). Concretely:

- Running `doca_caps --list-devs` to enumerate DOCA devices.
- Running `doca_caps --list-rep-devs` to enumerate representor
  devices.
- Scoping output to a specific PCIe address with `--pci-addr`.
- Listing the DOCA libraries the install reports as available on the
  current OS.
- Listing the available DOCA logger names.
- Capturing a documented, side-effect-free **capability snapshot** as
  prerequisite evidence for later `## debug` workflows.

Do **not** load this skill for general DOCA orientation, library API
work, or installation. For those, use
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md),
the matching `libs/<library>` skill, or
[`doca-setup`](../../doca-setup/SKILL.md).

## What this skill provides

This is a **thin loader**. Substantive material lives in two companion
files:

- the capabilities companion — what `doca_caps` reports (the five documented
  capability families: devices, representors, libraries, library
  capabilities, loggers), version availability and execution
  environment, the tool's narrow error surface, its observability role
  inside other skills' workflows, and its read-only safety posture.
- the tasks companion — step-by-step workflows for the in-scope task verbs:
  `configure` (route to install), `build` (route to install), `modify`
  (refuse), `run` (the documented invocations), `test` (capability
  snapshot as install smoke-test), `debug` (what to do when the tool
  reports nothing or fails), plus a `Deferred task verbs` block routing
  out-of-scope questions.

The skill assumes a host where DOCA is already installed (or the public
NGC DOCA container is running) and the operator has whatever
permissions the public guide requires for `doca_caps` to enumerate
devices on their platform.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or scripts bundle. To
keep the boundary clean, it deliberately does not contain — and pull
requests should not add:

- **Pre-baked example output.** Output is install- and
  hardware-specific. A captured example pinned to one platform and
  one DOCA version misleads operators on a different platform / version.
- **Wrappers, parsers, or scripts** in any language that consume
  `doca_caps` output. The output format is documented; if a user
  wants to script against it, the right answer is "read the live
  guide, write the parser against your installed version".
- **A `samples/` or `reference/` subtree.** This is a thin loader for
  a documented CLI; substantive material lives on the public page and
  in `--help`.

## Loading order

1. Read this skill first to confirm the user's question is in
   scope (the user actually wants to invoke `doca_caps`, not learn
   about DOCA in general).
2. **For what `doca_caps` reports, version availability, error
   surface, and safety posture, see the capabilities companion.**
3. **For the documented invocations and the capability-snapshot
   workflow — `configure`, `build`, `modify`, `run`, `test`, `debug` —
   see the tasks companion.**

## Related skills

- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  — routing to the public Capabilities Print Tool guide and the rest
  of the public DOCA documentation set.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation, install
  verification (`doca_caps` is the canonical first step there), and
  the *I have no install yet* path with the public NGC DOCA container.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md) —
  cross-library programming patterns, including the `## debug`
  procedure where the saved `doca_caps` snapshot is consumed.
- The matching `libs/<library>` skill — for fine-grained,
  library-specific capability questions that go beyond what
  `doca_caps` exposes.
