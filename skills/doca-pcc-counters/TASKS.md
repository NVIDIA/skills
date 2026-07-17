# DOCA PCC Counters (`pcc_counters.sh`) — Tasks

**Where to start:** The verbs that carry real workflow content
for this tool are `## configure`, `## run`, `## test`, and
`## debug`. The other two (`build`, `modify`) are routing
stubs because `pcc_counters.sh` is a shipped bash script — the
user does not compile it and must not patch it. The canonical
flow is dead simple: `set <mst-device>` to arm the counters,
then `query <mst-device>` to read them.

This file is loaded by [`SKILL.md`](SKILL.md) after
[`CAPABILITIES.md`](CAPABILITIES.md). It walks the agent through
the task verbs every artifact in this bundle exposes
(`configure / build / modify / run / test / debug`), defers
verbs that do not belong here, and ends with the `Command
appendix`.

For the host-side custom-PCC control surface (a SEPARATE
surface — the firmware counters this script reads are
independent of any custom kernel), see
[`doca-pcc`](../../libs/doca-pcc/SKILL.md).

## configure

Goal: confirm the environment preconditions `pcc_counters.sh`
needs — DOCA / MFT installed, the mst tools available, debugfs
mounted, root / sudo — and identify the target mst device path
BEFORE running `set` / `query`.

Steps the agent should walk the user through, in order:

1. **Confirm DOCA / MFT is installed and the script is
   present.** If the script is absent under the DOCA tools
   directory, route to
   [`doca-setup TASKS.md ## install`](../../doca-setup/TASKS.md#configure)
   (or [`doca-setup TASKS.md ## no-install`](../../doca-setup/TASKS.md#no-install)
   for the public NGC DOCA container path).
2. **Start mst and identify the device path.** Run
   `sudo mst start` (if not already running) then
   `sudo mst status -v`, and read off the exact device path
   for the target NIC (e.g. `/dev/mst/mt41692_pciconf0`). The
   script's second positional argument MUST be a path that
   appears in `mst status -v`; otherwise it prints
   `ERROR: Bad Device`.
3. **Confirm debugfs and privileges.** The script reads/writes
   `/sys/kernel/debug/mlx5/<pci>/diag_cnt/`, so debugfs must be
   mounted (`mount | grep debugfs`), the mlx5 driver loaded,
   and the user must have root / sudo.

`pcc_counters.sh` takes **no configuration of its own** — there
is no config file, no environment knob. "Configure" here means
*confirming the preconditions above*, not setting script
options.

## build

`pcc_counters.sh` is **a shipped bash script, not a compiled
artifact**. It is installed by `install_data` in
`tools/pcc_counters/meson.build`; there is nothing to compile,
no `meson` / `make` step, no build flags. Do not recompile or
re-implement it.

Routing for nearby "build" questions:

- *"The script isn't there — do I need to build it?"* → no.
  Route to
  [`doca-setup TASKS.md ## install`](../../doca-setup/TASKS.md#configure).
- *"How do I build a custom PCC algorithm whose behaviour
  these counters reflect?"* → that is the host-side +
  DPA-side compile path owned by
  [`doca-pcc`](../../libs/doca-pcc/SKILL.md) and the DPACC
  compiler — a separate surface from this firmware-counter
  readout script.

## modify

**Do not modify the shipped `pcc_counters.sh`.** It is an
NVIDIA-shipped diagnostic script with a fixed two-operation
contract and a fixed counter set. There is no supported way to
add a counter, a flag, or a subcommand by editing it, and none
should be invented.

- *"I want a counter the script doesn't report"* → the script
  reads a fixed firmware diagnostic counter set; it is not
  user-extensible here. If you need device counters outside
  that set, that is a firmware / driver question — route via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
- *"I want different output formatting / a parser"* → out of
  scope; the script's printed `Counter: ... Value: ...` lines
  are the contract. Parse them on your side if needed, but the
  parser is not shipped by this skill per
  [`SKILL.md ## What this skill deliberately does not ship`](SKILL.md#what-this-skill-deliberately-does-not-ship).

## run

The canonical flow is **`set` then `query`** against the same
mst device path. The script takes exactly two positional
arguments and nothing else.

1. **Confirm the configure baseline.** Per
   [`## configure`](#configure): script present, mst device
   path known, debugfs mounted, sudo available.
2. **Arm the counters with `set`.** Run:

```bash
sudo ./pcc_counters.sh set /dev/mst/mt41692_pciconf0
```

   This resolves the device to a PCI address and writes the
   fixed counter-ID list + sampling params to
   `/sys/kernel/debug/mlx5/<pci>/diag_cnt/`, arming the
   device's diagnostic counter collection.
3. **Read the counters with `query`.** Run:

```bash
sudo ./pcc_counters.sh query /dev/mst/mt41692_pciconf0
```

   This reads `diag_cnt/dump` and prints each known counter as
   `Counter: <NAME>   Value: <n>` (e.g. `PCC_CNP_COUNT`, the
   `*_EVENT_WRED_DROP` family, `RTT_GEN_PORT0_REQ`, …). Quote
   the lines verbatim.
4. **For a before / after comparison**, leave the counters
   armed (one `set`), run `query` before the change, apply the
   controlled change, run `query` after, and diff the two
   captured outputs with standard text tooling — the script
   has no built-in diff / watch.

The `query` value of a counter is the device's firmware
diagnostic value; it is independent of any custom `doca-pcc`
DPA kernel. Do not attribute a value to a custom
algorithm without independent evidence.

When recording the run for downstream consumers, write down:
the device path used, whether `set` was run first, the full
verbatim `query` output, and the host / BlueField side the
script ran on.

## test

"Test" for this script means *confirm the readout is
meaningful*, not unit-testing the bash. The smoke:

1. **Arm-then-read smoke.** Run `set` then `query` on the
   target device. The `query` output must contain the named
   counter lines (`Counter: <NAME>   Value: <n>`) and the
   values must be finite integers. If `query` prints nothing
   or errors, walk [`## debug`](#debug).
2. **Sanity against traffic.** On a device carrying RoCE
   traffic that should generate CNPs / RTT / WRED activity,
   re-run `query` after a controlled interval; the relevant
   counters should advance. A counter staying at 0 on a port
   with no such events is expected, not a bug — confirm there
   is relevant traffic before treating a zero as a finding.

This skill does **not** ship pre-recorded expected counter
values; they are device-, firmware-, and traffic-state-specific.

## debug

When `pcc_counters.sh` errors or prints values that do not look
defensible, walk the layers in
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
IN ORDER:

1. **Script-not-present.** The script is not under the DOCA
   tools directory. Confirm DOCA / MFT is installed and route
   to [`doca-setup TASKS.md ## install`](../../doca-setup/TASKS.md#configure).
2. **Bad-device (`ERROR: Bad Device`).** The mst device path
   does not match `sudo mst status -v`. Run `sudo mst start`,
   re-list with `sudo mst status -v`, and use the exact path.
3. **Bad-request (`Bad Request: choose 'set' or 'query'`).**
   The first argument was not `set` or `query` (e.g. a guessed
   `list` / `--help`). Use exactly `set` or `query`.
4. **Not-armed-before-query.** `query` returns stale / zero
   values because `set` was not run first. Run `set` then
   `query`.
5. **Debugfs-or-permission.** The script cannot access
   `/sys/kernel/debug/mlx5/<pci>/diag_cnt/`. Confirm sudo,
   `mount | grep debugfs`, and that the mlx5 driver is loaded;
   route env-side issues to
   [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug).
6. **Counter-stuck-at-zero.** Values print but a counter reads
   0 — usually correct (no such events). Confirm relevant
   traffic before treating it as a finding.
7. **Cross-cutting.** The question is really about the custom
   `doca-pcc` algorithm, firmware PCC configuration, or
   driver / firmware behaviour. Route to
   [`doca-pcc TASKS.md ## debug`](../../libs/doca-pcc/TASKS.md#debug),
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md),
   or [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).

In every case: **quote what the script printed.** Do not
paraphrase or reorder the counter lines.

## Deferred task verbs

The verbs below are not `pcc_counters.sh` work and should be
routed out before the agent does any of them under this
skill's name.

- **install** ⇒
  [`doca-setup TASKS.md ## install`](../../doca-setup/TASKS.md#configure)
  (and
  [`doca-setup TASKS.md ## no-install`](../../doca-setup/TASKS.md#no-install)
  for the NGC DOCA container path).
- **writing / loading a custom PCC algorithm** ⇒
  [`doca-pcc`](../../libs/doca-pcc/SKILL.md). That is the
  host-side library + DPA-side compile surface; this script
  only reads the device's firmware diagnostic counters.
- **firmware PCC algorithm configuration / fleet-wide CC
  tuning** ⇒ out of scope; route the documentation lookup via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
  Any CC decision derived from a counter reading must go back
  through the user's own domain analysis per
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
- **long-term retention / analytics on PCC counters** ⇒ not a
  feature of this script. The DOCA Telemetry Service (DTS) is
  the documented telemetry surface; route via
  [`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).

## Command appendix

`pcc_counters.sh`-specific invocations the verbs above reach
for. The script has exactly two operations and two positional
arguments — the agent MUST NOT invent flags, subcommands, or
counter names beyond the fixed set in
[`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).

| Purpose | Invocation | Owning step | Reads as healthy when … |
| --- | --- | --- | --- |
| Find the target device path | `sudo mst start` then `sudo mst status -v` | [`## configure`](#configure) step 2 | The target NIC appears with a `/dev/mst/...` device path the script can take as its second argument |
| Arm the diagnostic counters | `sudo ./pcc_counters.sh set <mst-device>` | [`## run`](#run) step 2 | Exit without `ERROR: Bad Device`; the device's `diag_cnt` is now configured |
| Read the counters | `sudo ./pcc_counters.sh query <mst-device>` | [`## run`](#run) step 3; [`## test`](#test) | Prints `Counter: <NAME>   Value: <n>` lines for the fixed counter set with finite integer values |
| Confirm debugfs is reachable | `mount \| grep debugfs` and `ls /sys/kernel/debug/mlx5/` | [`## debug`](#debug) layer 5 | debugfs is mounted and the mlx5 per-PCI directory exists with a `diag_cnt/` subtree |

Cross-cutting rules for this appendix:

- **Never invent a subcommand, flag, or counter name.** The
  script accepts only `set` / `query` plus one device path and
  reports only the fixed counter set; the script source
  (`tools/pcc_counters/pcc_counters.sh`) is the contract.
- **`set` before `query`.** Every read presumes the counters
  were armed first.
- **Cross-link instead of duplicate.** Cross-cutting commands
  (`mst status`, `lspci`, `dmesg`) and their interpretation
  live in
  [`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix)
  and
  [`doca-setup TASKS.md ## Command appendix`](../../doca-setup/TASKS.md#command-appendix);
  this appendix names only `pcc_counters.sh`-specific
  invocations on top.

## Cross-cutting

A few rules that apply across every verb in this file:

- The **script source plus the device's actual `query`
  output** are the joint source of truth. The script has no
  `--help`; its behaviour is fully determined by
  `tools/pcc_counters/pcc_counters.sh`.
- **`query` is read-only; `set` is a privileged debugfs
  write.** Neither changes dataplane forwarding, but any CC
  tuning decision derived from a reading is high-stakes per
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
- **Quote, do not paraphrase** the counter lines.
- This skill **assumes a healthy DOCA / MFT install** (mst
  tools, debugfs, sudo). If in doubt, route to
  [`doca-setup`](../../doca-setup/SKILL.md) first.
