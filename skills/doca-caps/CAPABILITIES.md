# DOCA Capabilities Print Tool — Capabilities

**Where to start:** `doca_caps` is a single CLI binary; the pattern
overview below names the recurring `doca_caps`-class questions. Pick
the pattern first, then drill into the H2 that owns the substance.
For the *how* of executing each pattern, jump to
[TASKS.md](TASKS.md).

This file is loaded by [`SKILL.md`](SKILL.md). It documents *what
`doca_caps` is*, *what it reports*, *what versions it ships in*, *what
its narrow error and observability surfaces look like*, and *the
read-only safety posture* that makes it the canonical first step in
other skills' workflows. For step-by-step invocations and the
capability-snapshot workflow, see [`TASKS.md`](TASKS.md).

## Pattern overview

Every `doca_caps`-class question this skill teaches resolves into one
of FIVE patterns. The patterns are CLASSES — they apply across every
DOCA install, not just one specific platform.

| `doca_caps` pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Enumerate (devices / representors) | List what DOCA sees so the agent can refer to real PCIe addresses, not made-up ones | [`## Capabilities and modes`](#capabilities-and-modes) device + representor families + [TASKS.md ## run](TASKS.md#run) |
| 2. Library availability snapshot | Confirm a specific DOCA library is available on this OS before recommending its skill | [`## Capabilities and modes`](#capabilities-and-modes) library family + [TASKS.md ## run](TASKS.md#run) |
| 3. Per-device capability lookup | Confirm a specific feature is supported on a specific device before instructing the user to program it | [`## Capabilities and modes`](#capabilities-and-modes) per-device per-library family + [TASKS.md ## run](TASKS.md#run) scoped invocation |
| 4. Side-effect-free smoke-test | Use `doca_caps` as the first step of an install smoke-test; no state changes, no risk | [TASKS.md ## test](TASKS.md#test) + [`## Safety policy`](#safety-policy) read-only stance |
| 5. Interpret empty / missing output | Empty output means *not supported on this device / version*, not *the tool is broken* | [TASKS.md ## debug](TASKS.md#debug) + [`## Error taxonomy`](#error-taxonomy) |

Two cross-cutting rules that apply to *every* pattern above:

- **Read-only always; never recommend a state-changing alternative.**
  `doca_caps` is uniquely safe to run early in any DOCA workflow.
  The agent must not propose a state-changing workaround when
  `doca_caps` answers the question.
- **Empty output is an answer, not a failure.** When `doca_caps`
  prints nothing for a capability the user asked about, the answer
  is *"this device on this DOCA version does not expose that
  capability"*. The agent must report that conclusion, not retry the
  invocation hoping for different output.

## Capabilities and modes

`doca_caps` is shipped as a single read-only CLI binary at
`/opt/mellanox/doca/tools/doca_caps` on every DOCA install (host or
BlueField Arm) since DOCA 2.6.0. There is no daemon, no library to
link against, and no programmatic API. The user's entire interaction
model is *invoke the binary, read the printed output*.

The tool reports **five documented capability families**, per the
public DOCA Capabilities Print Tool guide:

1. **DOCA device list** — for every DOCA device, prints the PCIe
   address and per-device attributes (`ibdev_name`, `iface_name`,
   `iface_index`, `pci_func_type` (PF / VF / SF), `uplink_ib_port`,
   `mac_addr`, `ipv4_addr`, `ipv6_addr`).
2. **DOCA representor device list** — for every DOCA device, prints
   the PCIe address of every available DOCA representor and
   per-representor attributes (`ib_port`, `host_index`, `pf_index`,
   `vf_index`, `pci_func_type`, `hotplug`, `vuid`, `iface_name`,
   `iface_index`).
3. **DOCA library list** (CLI flag: `--list-libs`) — prints the
   DOCA libraries supported by the running OS and their availability
   for specific OS targets. This is the documented authoritative
   answer to *"which DOCA libraries does this install actually
   support on my OS?"*
4. **DOCA library capabilities** — for every DOCA device, prints the
   capabilities it supports in every DOCA library. This is the
   documented authoritative answer to *"can DOCA library X actually
   do Y on this device?"*
5. **DOCA logger list** (CLI flag: `--list-loggers`) — prints the
   available logger names of DOCA libraries. Useful when configuring
   `doca_log_*` filters.

The flags the binary registers in `tools/caps/main.c` are
`--list-devs`, `--list-rep-devs`, `--list-libs`, `--list-loggers`,
`--pci-addr` (scope to a single PCIe address), and `--lib`. Any of
the `--list-*` flags above can be combined with `--pci-addr` /
`--lib` to narrow the output. The exact, current flag inventory
and example output live in the public guide and in the tool's own
`--help` on the installed version — see
[`TASKS.md ## run`](TASKS.md#run).

The tool has **no execution modes** beyond the flag-driven
selection of which capability family to print. There is no
`--watch`, no streaming subscription, no JSON output mode in the
documented surface; if a feature like that lands in a future DOCA
release, treat the public guide and `--help` as ground truth and do
not assume legacy generic-CLI flags work.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way match rule, NGC container semantics, and the headers-win-over-docs rule, see [`doca-version`](../../doca-version/SKILL.md). The body lives there; this skill does not duplicate it.

**The doca_caps-specific overlay** is:

- **`doca_caps` is available since DOCA 2.6.0.** On older installs the binary is not present; the right answer for *"I can't find doca_caps"* is to confirm the installed version per [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure) and, if `< 2.6.0`, route to [`doca-setup`](../../doca-setup/SKILL.md) for an upgrade rather than recommending alternative tools.
- **Where it runs:** on the x86 / Arm host that has DOCA installed, *or* on the BlueField Arm side. Same binary, same flags; the set of devices it sees differs by execution context.
- **Output format stability:** the documented capability families (the five listed in [`## Capabilities and modes`](#capabilities-and-modes)) are stable across the recent DOCA train. The exact textual / column layout of the output is **not** contractually frozen — agents that need to consume the output programmatically should prefer the structured `doca-capability-snapshot` helper per [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md#schemas) when present, and re-verify the textual layout against the user's installed version when absent.
- **Per-OS library support:** capability family 3 ("DOCA library list") explicitly varies with the OS the install runs on; do not copy a library-availability claim from one host to another.

## Error taxonomy

`doca_caps`'s error surface is narrow because the tool is read-only,
takes no configuration, and does no orchestration. The error layers
the agent should distinguish, in escalating order:

1. **Tool-not-installed.** `doca_caps` does not exist at
   `/opt/mellanox/doca/tools/doca_caps`. Causes (distinguish before
   routing):
   - (a) DOCA is not installed on this host at all — `pkg-config
     --modversion doca-common` errors with `Package 'doca-common' was
     not found`. Routing: full install via
     [`doca-setup ## install`](../../doca-setup/TASKS.md#configure).
   - (b) The install is < DOCA 2.6.0 (the first release that ships
     `doca_caps`). `pkg-config --modversion doca-common` returns
     `2.5.x` or earlier. Routing: upgrade to ≥ 2.6.0, or use the
     per-library `doca_<library>_cap_*` C API as the cap-discovery
     surface until upgraded.
   - (c) **Partial install on a host that DOES have `doca-common`**
     — `pkg-config --modversion doca-common` returns a version ≥
     2.6.0 but the per-tool package that ships `doca_caps` (on
     DOCA 3.3+ this is the granular `doca-caps` package; on older
     releases it shipped via the legacy `doca-tools` meta-package
     that no longer exists on 3.3+) was not installed alongside
     `doca-common`. This is the most common cause on lab boxes and
     CI containers that pinned a minimal install. Confirm via:
     `pkg-config --modversion doca-common` succeeds AND `ls
     /opt/mellanox/doca/tools/doca_caps` errors. Routing: confirm
     the package name with `apt-cache policy doca-caps` FIRST, then
     run `apt install doca-caps` (or, on older releases, the legacy
     `apt install doca-tools`) — NOT a full `doca-all` reinstall,
     and NOT a version downgrade. See
     [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)
     partial-install table.
   - (c2) **Package installed, binary off-PATH (DOCA 3.3+ default).**
     `dpkg -l doca-caps` shows the package as `ii` AND
     `ls /opt/mellanox/doca/tools/doca_caps` succeeds, but
     `command -v doca_caps` and a bare `doca_caps --version` both
     report "not found." On DOCA 3.3+, per-tool binaries land under
     `/opt/mellanox/doca/tools/` and are NOT symlinked into
     `/usr/bin/`. This is NOT a partial install — it is a `$PATH`
     hygiene issue. Routing: invoke by absolute path
     (`/opt/mellanox/doca/tools/doca_caps --version`) or extend
     `PATH` (`export PATH=/opt/mellanox/doca/tools:$PATH`) once at
     the start of the session. Do NOT re-`apt install`.
   - (d) The install path was changed by the operator. Confirm via
     `pkg-config --variable=prefix doca-common` and check whether
     `doca_caps` exists under that prefix's `tools/`. Routing: ask
     the operator for the correct path or restore the canonical
     install layout.
2. **Permission / driver layer.** Tool runs but cannot enumerate
   devices because the underlying driver stack (`mlx5_core`, IB
   stack, etc.) is not loaded, the user lacks the privileges the
   install profile expects, or the BlueField mode is incompatible
   with the requested capability family. The tool's own message
   (and `dmesg` for the driver layer) is ground truth; do not guess.
3. **Empty / partial output, no error.** The tool exits 0 but reports
   zero devices or a representor list shorter than the operator
   expects. Cause: no DOCA-supported devices on this host (e.g. the
   public NGC DOCA container with no PCIe passthrough), or the device
   the user expected is excluded by the chosen `--pci-addr` scope.
   This is a *capability-snapshot finding*, not a tool failure —
   route the answer to the consumer of the snapshot
   ([`doca-setup ## test`](../../doca-setup/TASKS.md#test) or
   [`doca-programming-guide ## debug`](../../doca-programming-guide/TASKS.md#debug)).
4. **Library-capability mismatch.** Tool runs successfully and prints
   capabilities, but the user is asking *"why does library X say it
   doesn't support feature Y on my device?"*. That question belongs
   in the matching `libs/<library>` skill — `doca_caps` only reports
   the coarse per-device per-library capability surface, not the
   library-internal reasoning.

`doca_caps` does **not** participate in the cross-library
`DOCA_ERROR_*` taxonomy that DOCA libraries return through their C
API; it is a CLI, not a library call. For that taxonomy and the
program-side debug order, see
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).

## Observability

`doca_caps` is itself an **observability primitive** for the rest of
the bundle — it is *what other skills load to observe* the install
and the hardware before doing anything that changes state.
Specifically:

- [`doca-setup ## test`](../../doca-setup/TASKS.md#test) prescribes
  running `doca_caps --list-devs` (and `--list-rep-devs` where
  representors are in scope) as the documented install smoke-test.
- [`doca-programming-guide ## debug`](../../doca-programming-guide/TASKS.md#debug)
  prescribes running `doca_caps` early and **preserving its output
  as a capability snapshot** so subsequent debug steps have ground
  truth instead of guesses.
- The matching `libs/<library>` skills (e.g.
  [`doca-flow CAPABILITIES.md ## Capabilities and modes`](../../libs/doca-flow/CAPABILITIES.md#capabilities-and-modes))
  point back at `doca_caps` as the documented source of *coarse*
  per-device per-library capability claims, while reserving
  fine-grained library-internal capability checks for the library's
  own programmatic API.

The tool does not emit metrics, traces, or logs of its own beyond
the printed output. For the program-side observability surface (DOCA
log levels, `doca_log_*`, `DOCA_LOGGER_*` env vars) see
[`doca-programming-guide CAPABILITIES.md ## Observability`](../../doca-programming-guide/CAPABILITIES.md#observability).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

`doca_caps` is the **safest tool the bundle prescribes**:

- **Read-only.** The tool prints; it does not configure, allocate,
  claim, or modify any DOCA, kernel, firmware, or device state. This
  property is what makes it the canonical first step before any
  workflow that *does* mutate state.
- **No persistent side effects on the host.** No files written to
  `/etc`, no daemons started, no devices reserved. Re-running it is
  free.
- **Safe to run inside the public NGC DOCA container.** Because it
  takes no configuration and writes nothing, an agent can run it as
  the very first action even on a host where the operator has not
  yet decided whether to keep DOCA installed; running it inside the
  NGC container is the documented zero-install path.
- **Quote what the tool said. Do not paraphrase capability claims.**
  If the user later asks *"does my setup support feature Y?"*, the
  correct answer is to point at the line of the snapshot that says
  so. If the snapshot does not show feature Y, the answer is *"this
  install / device combination does not report support for Y"* —
  not *"it should work, try it"*.
- **Do not invent flags.** The documented invocations are the
  authoritative surface. If the user asks for an output format or
  a flag the public guide does not list, the safe answer is "the
  installed `--help` is the source of truth — let me check it
  there", not a guess based on generic CLI conventions.

## Public-source pointer

The single canonical public source for `doca_caps` is the **DOCA
Capabilities Print Tool** page on `docs.nvidia.com`, reachable
through
[`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools).
Do not invent flags, output formats, or capability families beyond
what that page documents.
