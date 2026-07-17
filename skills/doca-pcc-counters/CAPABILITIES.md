# DOCA PCC Counters (`pcc_counters.sh`) — Capabilities

**Where to start:** `pcc_counters.sh` is a small bash script
with exactly two operations — `set` (arm the device's PCC
diagnostic counters) and `query` (read and print them). The
pattern overview below names the recurring questions. Pick the
pattern first, then drill into the H2 that owns the substance.
For the *how* of executing each pattern, jump to
[TASKS.md](TASKS.md).

This file is loaded by [`SKILL.md`](SKILL.md). It documents
*what the script does*, *the fixed firmware / HW counter set it
reports*, *how it reaches the counters through mst + debugfs*,
*what install / environment it needs*, *the layered error
surface*, and *the safety posture* that flags the `set`
operation as a privileged debugfs write and any CC tuning
decision derived from a reading as high-stakes.

## Pattern overview

Every `pcc_counters.sh` question this skill teaches resolves
into one of FOUR patterns. The patterns are CLASSES — they
apply across every device the script can address.

| Pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Arm the diagnostic counters | Run `set <mst-device>` to program which counters the device collects (writes the fixed counter-ID list + params to `diag_cnt`); this MUST happen before a meaningful `query` | [`## Capabilities and modes`](#capabilities-and-modes) `set` row + [TASKS.md ## run](TASKS.md#run) step 2 |
| 2. Read the counters | Run `query <mst-device>` to read `diag_cnt/dump` and print the named counters with their values | [`## Capabilities and modes`](#capabilities-and-modes) `query` row + [TASKS.md ## run](TASKS.md#run) step 3 |
| 3. Resolve the device | The single positional device argument is an mst path (e.g. `/dev/mst/mt41692_pciconf0`); the script maps it to a PCI address via `mst status -v` + `lspci` before touching debugfs | [`## Capabilities and modes`](#capabilities-and-modes) device-resolution row + [`## Error taxonomy`](#error-taxonomy) layer 2 |
| 4. Diagnose missing / silent output | Map symptom (`ERROR: Bad Device`, `Bad Request`, empty dump, counter pinned at zero, permission/debugfs error) to the right layer | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **`set` arms, `query` reads — in that order.** `query`
  without a prior `set` reads whatever the device's `diag_cnt`
  was last configured for (possibly nothing). The documented
  sequence in the script's own header comment is *"always
  `set` first then `query`"*.
- **These are firmware / HW diagnostic counters.** They exist
  at the device level and are INDEPENDENT of any custom
  `doca-pcc` DPA kernel. Do not condition a reading on a
  custom kernel being loaded, and do not attribute a counter
  value to a custom algorithm without independent evidence.

## Capabilities and modes

`pcc_counters.sh` is a **bash script** installed under the
DOCA tools directory (`install_data` in
`tools/pcc_counters/meson.build`, installed when the build is
not an internal-tool build). There is no daemon, no library to
link against, no `--help`, no `--version`, and no subcommands
beyond the two operations below. The entire interaction model
is *run the script with two positional arguments, read the
printed output*.

**The two operations — exact contract.**

| Operation | Positional form | What it does |
| --- | --- | --- |
| `set` | `sudo ./pcc_counters.sh set <mst-device>` | Resolves the device to a PCI address, then writes the fixed counter-ID list and the sampling params to `/sys/kernel/debug/mlx5/<pci>/diag_cnt/counter_id` and `/diag_cnt/params`, and writes `set` to `/diag_cnt/dump`. This ARMS the device's diagnostic counter collection. It is a privileged debugfs write. |
| `query` | `sudo ./pcc_counters.sh query <mst-device>` | Resolves the device to a PCI address, reads `/sys/kernel/debug/mlx5/<pci>/diag_cnt/dump`, and prints each known counter as `Counter: <NAME>   Value: <n>`. |

Any first argument other than `set` or `query` makes the
script print `Bad Request: choose 'set' or 'query'`. A device
the script cannot resolve via `mst status -v` makes it print
`ERROR: Bad Device` and exit.

**Device argument.** The second positional argument is an
**mst device path** as shown by `sudo mst status -v` (e.g.
`/dev/mst/mt41692_pciconf0`). The script greps `mst status -v`
for that device to obtain the bus address, then resolves it to
a domain PCI address via `lspci -D` (filtering for
Mellanox / NVIDIA) to build the
`/sys/kernel/debug/mlx5/<pci>/diag_cnt/` path.

**The fixed counter set.** Unlike a custom-kernel surface, the
counters are a FIXED list baked into the script's `query`
case-statement (firmware / HW PCC diagnostic counters). The
agent may name these because they are constant in the source:

| Counter | Meaning (class) |
| --- | --- |
| `PCC_CNP_COUNT` | Congestion Notification Packets counted |
| `MAD_RTT_PERF_CONT_REQ` | RTT-perf continuous request counter |
| `MAD_RTT_PERF_CONT_RES` | RTT-perf continuous response counter |
| `SX_EVENT_WRED_DROP` | WRED drops on SX events |
| `SX_RTT_EVENT_WRED_DROP` | WRED drops on SX RTT events |
| `ACK_EVENT_WRED_DROP` | WRED drops on ACK events |
| `NACK_EVENT_WRED_DROP` | WRED drops on NACK events |
| `CNP_EVENT_WRED_DROP` | WRED drops on CNP events |
| `RTT_EVENT_WRED_DROP` | WRED drops on RTT events |
| `HANDLED_SXW_EVENTS` | Handled SXW events |
| `HANDLED_RXT_EVENTS` | Handled RXT events |
| `DROP_RTT_PORT0_REQ` / `DROP_RTT_PORT1_REQ` | Dropped RTT requests, port 0 / 1 |
| `DROP_RTT_PORT0_RES` / `DROP_RTT_PORT1_RES` | Dropped RTT responses, port 0 / 1 |
| `RTT_GEN_PORT0_REQ` / `RTT_GEN_PORT1_REQ` | Generated RTT requests, port 0 / 1 |
| `RTT_GEN_PORT0_RES` / `RTT_GEN_PORT1_RES` | Generated RTT responses, port 0 / 1 |

The script reads these via the mlx5 `diag_cnt` debugfs
interface; the set is not user-extensible without editing the
script (which this skill forbids — see
[`SKILL.md ## What this skill deliberately does not ship`](SKILL.md#what-this-skill-deliberately-does-not-ship)).

## Version compatibility

For the canonical DOCA version-detection chain and NGC
container semantics, see
[`doca-version`](../../doca-version/SKILL.md). The body lives
there; this skill does not duplicate it.

**The `pcc_counters.sh`-specific overlay** is:

- **Confirm the script is present before assuming
  availability.** It is installed under the DOCA tools
  directory by `install_data` in
  `tools/pcc_counters/meson.build` (and only when the build is
  not an internal-tool build). If the script is absent, route
  to [`doca-setup`](../../doca-setup/SKILL.md); do NOT
  re-implement the script or quote counter values from memory.
- **mst / MFT and debugfs are the real preconditions.** The
  script depends on `mst status -v` resolving the device and
  on `/sys/kernel/debug/mlx5/<pci>/diag_cnt/` being present
  (debugfs mounted, mlx5 driver loaded). These are
  environment facts, not DOCA-version facts; confirm them per
  [`doca-setup`](../../doca-setup/SKILL.md).
- **The counter set is fixed in the script.** Counter names
  do not vary by flag or version at the script level — they
  are the constant case-statement above. What CAN vary is
  whether the underlying firmware / driver exposes the
  `diag_cnt` interface for a given device generation; confirm
  on the user's device rather than assuming.

## Error taxonomy

The error layers the agent should distinguish, in escalating
order:

1. **Script-not-present.** `pcc_counters.sh` is not under the
   DOCA tools directory. Cause: DOCA / MFT not installed, or an
   internal-tool build that did not install it. Routing:
   [`doca-setup ## install`](../../doca-setup/TASKS.md#configure).
2. **Bad-device.** The script prints `ERROR: Bad Device`.
   Cause: the mst device path does not match an entry in
   `sudo mst status -v` (wrong path, mst service not started,
   device not present). Fix: run `sudo mst start` then
   `sudo mst status -v` and use the exact device path it
   lists.
3. **Bad-request.** The script prints
   `Bad Request: choose 'set' or 'query'`. Cause: the first
   positional argument was something other than `set` or
   `query` (e.g. a guessed `list` / `snapshot` / `--help`).
   Fix: use exactly `set` or `query`.
4. **Not-armed-before-query.** `query` runs but the values are
   stale / zero because `set` was never run for this device
   (or the device's `diag_cnt` was reconfigured by something
   else). Fix: run `set` first, then `query`, per the script's
   documented order.
5. **Debugfs-or-permission.** The script cannot read/write
   `/sys/kernel/debug/mlx5/<pci>/diag_cnt/`. Cause: not running
   as root / sudo, debugfs not mounted, or the mlx5 driver does
   not expose `diag_cnt` for this device. Fix: run under sudo,
   confirm `mount | grep debugfs`, confirm the mlx5 driver is
   loaded. Route env-side issues to
   [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug).
6. **Counter-stuck-at-zero.** The script runs, the dump prints,
   but a counter reads 0. Cause is usually correct behaviour —
   the device genuinely saw no such events (no congestion, no
   RTT, no drops) — NOT a tool bug. Confirm there is relevant
   traffic before treating a zero as a finding.
7. **Cross-cutting.** Script runs cleanly, counters are
   readable, but the user's question is really about the
   custom `doca-pcc` algorithm, firmware PCC configuration,
   BlueField mode, or driver / firmware behaviour. Route to
   [`doca-pcc`](../../libs/doca-pcc/SKILL.md),
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md),
   or [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).

## Observability

`pcc_counters.sh` IS an observability primitive — it is the
documented way to read the device's firmware PCC diagnostic
counters from the command line. Its only output is the printed
`Counter: <NAME>   Value: <n>` lines on stdout; it emits no
metrics, traces, or DOCA logs of its own.

- Save the full `query` output (file or paste buffer) when
  capturing evidence; quote the counter lines verbatim rather
  than paraphrasing.
- For a before / after comparison, the agent runs `set` once,
  then `query` before the change and `query` after, and diffs
  the two captured outputs with standard text tooling (the
  script has no built-in diff).
- For the custom-algorithm host-side report stream (a separate
  surface), see
  [`doca-pcc CAPABILITIES.md ## Observability`](../../libs/doca-pcc/CAPABILITIES.md#observability).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

`query` is read-only; `set` writes to the device's debugfs
`diag_cnt` interface to reconfigure which diagnostic counters
are collected. Neither changes dataplane forwarding, but both
require root / sudo, and the downstream decisions an agent
might derive from a reading are high-stakes.

- **`set` is a privileged debugfs write.** It reprograms the
  device's diagnostic counter selection. It does not change
  congestion-control behaviour, but it is a root-level write
  to a kernel debug interface — run it deliberately, on the
  intended device, not speculatively across devices.
- **The reading is evidence; the tuning decision it informs is
  high-stakes.** A misread counter that drives a
  congestion-control parameter change can destabilize the
  RDMA / RoCE fabric for every node on the port. The skill
  captures counters; it refuses to translate a delta into a CC
  parameter change without the user's own domain analysis.
- **Confirm traffic before treating a zero as a finding.** A
  counter at 0 usually means the device saw no such events,
  not that the tool failed.
- **Quote what the script printed.** Do not paraphrase or
  reorder the `Counter: ... Value: ...` lines; the verbatim
  output is the artifact downstream debug consumes.
- **Never invent a counter name, flag, or subcommand.** The
  script has exactly `set` / `query`, two positional
  arguments, and the fixed counter set above; anything beyond
  that is a hallucination.

## Public-source pointer

The canonical public source for device PCC behaviour and the
firmware congestion-control surface is the DOCA / PCC
documentation on `docs.nvidia.com`, reachable through
[`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools).
The authoritative source for the script's exact behaviour is
the script itself (`tools/pcc_counters/pcc_counters.sh`) on the
user's install. For the `doca-pcc` library that loads custom
congestion-control kernels (a separate surface), see
[`doca-pcc`](../../libs/doca-pcc/SKILL.md).
