---
license: Apache-2.0
name: doca-pcc-counters
description: >
  Use this skill when the user is invoking the DOCA PCC Counters tool
  — the `pcc_counters.sh` bash script under the DOCA tools directory —
  to arm and read the fixed firmware/hardware PCC (Programmable
  Congestion Control) diagnostic counters (CNP, RTT, WRED-drop, etc.)
  on a ConnectX / BlueField device via mst + the mlx5 debugfs
  `diag_cnt` interface. The script takes two positional args —
  `set | query` and an mst device path — with no `--help` or
  subcommands. Trigger even without "pcc_counters.sh" or "PCC
  counters": "how do I read the CNP / RTT / WRED-drop counters",
  "PCC counter stuck at zero", "the script says Bad Device", or "is
  congestion control dropping packets on this port?". Route elsewhere
  for writing a custom PCC algorithm (doca-pcc), factory firmware PCC
  config, DOCA install, or fleet-wide CC tuning.
metadata:
  kind: tool
compatibility: >
  Requires a DOCA/MFT environment on Linux (Ubuntu 22.04/24.04 or
  RHEL/SLES) with a ConnectX-6+/BlueField device, mst tools (`mst status
  -v` must resolve it), debugfs mounted, and root/sudo (reads/writes
  /sys/kernel/debug/mlx5/{pci}/diag_cnt/). `pcc_counters.sh` is a bash
  script under the DOCA tools dir; it reads FW/HW diagnostic counters,
  independent of any custom DPA/doca-pcc kernel. The set op reconfigures
  collected counters (privileged write) and is fleet-impacting.

---

# DOCA PCC Counters (`pcc_counters.sh`)

**Where to start:** This is a tool skill for invoking
`pcc_counters.sh` — a small bash script that arms and reads the
device's fixed set of firmware / hardware **PCC diagnostic
counters** (CNP count, RTT-perf, WRED-drop, RTT-gen, handled
events) through the mlx5 debugfs `diag_cnt` interface. Open
[`TASKS.md`](TASKS.md) and start at [`## run`](TASKS.md#run) for
the canonical `set`-then-`query` sequence, or
[`## debug`](TASKS.md#debug) when the user reports
*"`ERROR: Bad Device`"*, *"counter stuck at zero"*, or *"the
dump is empty"*. Open [`CAPABILITIES.md`](CAPABILITIES.md) when
the question is *which counters the script reports and how it
reaches them*. If the user has not installed DOCA / MFT yet,
route to [`doca-setup`](../../doca-setup/SKILL.md) first.

This skill is the **firmware / HW PCC counter readout** surface.
It is NOT the host-side control library that loads custom
congestion-control kernels onto the DPA (that is
[`doca-pcc`](../../libs/doca-pcc/SKILL.md)) and it is NOT the
firmware PCC *algorithm* configuration (that path is firmware
configuration, routed via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)).
The counters this script reads are device / firmware
diagnostic counters that exist **regardless of whether a custom
`doca-pcc` DPA kernel is running** — do not condition them on a
custom kernel being loaded.

## Example questions this skill answers well

The CLASSES of `pcc_counters.sh` questions this skill is built
to answer, each with one worked example. The class is the
load-bearing piece; the worked example is one instance.

- **"How do I read the PCC diagnostic counters on this
  device?"** — worked example: *"arm and dump the CNP / RTT /
  WRED-drop counters for `/dev/mst/mt41692_pciconf0`"*.
  Answered by the fixed counter set in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the `set`-then-`query` invocation in
  [`TASKS.md ## run`](TASKS.md#run).
- **"What is the smallest legal invocation?"** — worked
  example: *"what exactly do I type?"*. Answered by the
  two-positional-argument contract
  (`set | query` + an mst device path) in
  [`TASKS.md ## run`](TASKS.md#run).
- **"The script printed `ERROR: Bad Device` — what's wrong?"**
  — worked example: *"my device path is not matching"*.
  Answered by the device-resolution layer in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + [`TASKS.md ## debug`](TASKS.md#debug).
- **"A counter is stuck at zero — is the device idle, the
  counters not armed, or genuinely no events?"** — worked
  example: *"`PCC_CNP_COUNT` reads 0 after `query`"*. Answered
  by the arm-before-read rule and the layered diagnosis in
  [`TASKS.md ## debug`](TASKS.md#debug) +
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).
- **"Is this script on my install, and where?"** — worked
  example: *"is `pcc_counters.sh` present and where does the
  install put it"*. Answered by the install overlay in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility),
  which redirects to the canonical
  [`doca-version`](../../doca-version/SKILL.md) rules.

## Audience

This skill serves **operators, developers, and AI agents who
need to read a ConnectX / BlueField device's firmware PCC
diagnostic counters** to reason about congestion-control
behaviour (CNP generation, RTT requests/responses, WRED
drops) on a port. Concretely:

- A network operator confirming whether congestion-control
  events (CNPs, RTT, WRED drops) are occurring on a device.
- A developer correlating a custom `doca-pcc` algorithm's
  effect with the device-level PCC diagnostic counters
  (the script reads the firmware counters; the custom
  algorithm itself is a separate surface owned by
  [`doca-pcc`](../../libs/doca-pcc/SKILL.md)).
- An AI agent producing a *PCC counter snapshot* as evidence
  for a congestion-control investigation.

It is **not** for users debugging the script's bash itself,
**not** the place to learn how to write a custom PCC
algorithm — that audience belongs in
[`doca-pcc`](../../libs/doca-pcc/SKILL.md) — and **not** the
place for users who want to configure the factory firmware PCC
algorithm (route via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)).

`pcc_counters.sh` is shipped as a **plain bash script**
installed under the DOCA tools directory (per `install_data` in
`tools/pcc_counters/meson.build`), not a compiled binary and
not a library you link against. The skill uses the bundle's
`kind: tool` three-file shape (`SKILL.md` + `CAPABILITIES.md`
+ `TASKS.md`) so the agent's task-verb contract
(`configure / build / modify / run / test / debug`) is uniform
across the bundle.

## When to load this skill

Load this skill when the user is — or the agent needs to —
arm and read the device PCC diagnostic counters on a host or
BlueField Arm with the mst tools available and debugfs
mounted. Concretely:

- Arming the diagnostic counters with `set` on a target mst
  device.
- Reading the armed counters with `query` and quoting the
  named counter lines verbatim.
- Capturing a counter readout as evidence for a
  congestion-control investigation.

Do **not** load this skill for general DOCA orientation,
custom-PCC algorithm design, the host-side `doca-pcc` library
API, the factory firmware PCC algorithm, or DOCA / MFT install.
For those, route to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md),
[`doca-pcc`](../../libs/doca-pcc/SKILL.md), or
[`doca-setup`](../../doca-setup/SKILL.md).

## What this skill provides

This is a **thin loader**. Substantive material lives in two
companion files:

- `CAPABILITIES.md` — what `pcc_counters.sh` does: the exact
  two-operation surface (`set` arms the device's diagnostic
  counters by writing counter IDs + params to debugfs;
  `query` reads the `diag_cnt/dump` and prints the named
  counters), the FIXED firmware / HW counter set it knows
  (`PCC_CNP_COUNT`, the `MAD_RTT_PERF_CONT_*`, the
  `*_EVENT_WRED_DROP` family, `HANDLED_*_EVENTS`, the
  `DROP_RTT_PORT*`/`RTT_GEN_PORT*` families), how it resolves
  an mst device to a PCI address (`mst status -v` + `lspci`)
  and reaches `/sys/kernel/debug/mlx5/<pci>/diag_cnt/`, the
  install-availability overlay that redirects to
  [`doca-version`](../../doca-version/SKILL.md), the layered
  error taxonomy (script-not-present / bad-device /
  not-armed-before-query / debugfs-or-permission /
  counter-stuck-at-zero / cross-cutting), and the safety
  policy that flags `set` as a privileged debugfs write and
  any CC tuning decision derived from a reading as
  high-stakes.
- `TASKS.md` — step-by-step workflows for the in-scope task
  verbs: `configure` (route to install + confirm mst /
  debugfs / sudo), `build` (route to install; nothing to
  compile — it is a script), `modify` (refuse — do not patch
  the shipped script), `run` (the `set`-then-`query`
  sequence), `test` (confirm the dump contains the named
  counters with finite values), `debug` (the layered
  diagnosis ladder), plus a `Deferred task verbs` block and a
  `Command appendix`.

The skill assumes a host or BlueField where DOCA / MFT is
already installed (mst tools present, debugfs mounted, sudo
available) and the target device is visible to `mst status -v`.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or scripts
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Invented flags or subcommands.** `pcc_counters.sh` has
  exactly two operations (`set`, `query`), takes exactly two
  positional arguments, and has NO `--help`, `--version`,
  `list`, `snapshot`, `watch`, or `diff`. Do not invent any.
- **Pre-baked example counter values.** Counter values are
  device-, firmware-, and traffic-state-specific; a captured
  value will mislead an operator on a different device.
- **Wrappers, parsers, or rewritten copies of the script.**
  The script is the contract; modifying or re-implementing it
  is out of scope.
- **A specific congestion-control tuning recommendation
  derived from a counter reading.** That is a high-stakes
  domain question — the skill prescribes how to *capture* the
  counters; it refuses to translate a delta into a CC
  parameter change without the user's own domain analysis.
- **A `samples/` or `reference/` subtree.** This is a thin
  loader for a documented script; substantive material lives
  in the script and the public PCC documentation.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question
   is in scope (reading the device's firmware PCC diagnostic
   counters; not designing or loading a custom algorithm).
2. **For what the script does, the fixed counter set, the
   debugfs mechanism, the install-availability overlay, the
   layered error surface, and the safety posture, see
   [CAPABILITIES.md](CAPABILITIES.md).**
3. **For the documented invocations — `configure`, `build`,
   `modify`, `run`, `test`, `debug`, plus the `Command
   appendix` — see [TASKS.md](TASKS.md).**

## Related skills

- [`doca-pcc`](../../libs/doca-pcc/SKILL.md) — the host-side
  library for writing and loading custom congestion-control
  kernels onto the DPA. It is a SEPARATE surface: the
  firmware PCC diagnostic counters `pcc_counters.sh` reads
  exist independently of any custom `doca-pcc` kernel, but an
  operator tuning a custom algorithm may read these counters
  to observe device-level CC behaviour. Conflating the script
  (firmware counter readout) with the library (custom
  algorithm load/control) is the most common PCC first-touch
  error.
- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  — routing to the public DOCA / PCC documentation set,
  including the firmware PCC algorithm configuration.
- [`doca-version`](../../doca-version/SKILL.md) — canonical
  DOCA version-handling rules. The `## Version compatibility`
  section in [`CAPABILITIES.md`](CAPABILITIES.md) is a concise
  overlay that redirects here.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, mst tools, debugfs, and the
  *I have no install yet* path with the public NGC DOCA
  container. This skill assumes its preconditions are
  satisfied.
- [`doca-debug`](../../doca-debug/SKILL.md) — the
  cross-cutting debug ladder. The PCC counter readout slots
  in as a read-only device-state evidence source before any
  congestion-control tuning recommendation is made.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md)
  — general DOCA programming patterns shared across the
  bundle.
