# DOCA DPDK Bridge capabilities, version overlay, errors, observability, safety

**Where to start:** Pick the H2 anchor that matches your question
(bridge-vs-native / port-handover / mbuf conversion / capabilities /
errors / safety) and read that section end-to-end. The tables in
each section are the load-bearing content; the prose around them
is interpretation.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the *how* of executing each pattern
(the verbs `configure / build / modify / run / test / debug`),
jump to [TASKS.md](TASKS.md). For the canonical DOCA
version-handling rules that this skill layers a bridge overlay
on top of, see [`doca-version`](../../doca-version/SKILL.md).
For the start-fresh alternative (no DPDK in the picture), defer
to [`doca-eth`](../doca-eth/SKILL.md); for the steering library
the bridge most commonly composes with, defer to
[`doca-flow`](../doca-flow/SKILL.md).

## Pattern overview

Every DOCA DPDK Bridge question this skill teaches resolves into
one of FIVE patterns. The patterns are CLASSES — they apply
across every bridge release and every host / BlueField pair.

| Pattern | When it applies (class shape) | Where the substance lives |
| --- | --- | --- |
| 1. Bridge or go native | Decide whether the user has an EXISTING DPDK app (use this bridge) or is starting fresh (use native `doca-eth` instead); the bridge is for interop, not for new projects | [`## Capabilities and modes`](#capabilities-and-modes) bridge-vs-native table + [TASKS.md ## configure](TASKS.md#configure) step 2 |
| 2. Bind a DPDK port id to a DOCA device | Map a DPDK port id ↔ `doca_dev` (via `doca_dpdk_port_probe()` from a `doca_dev`, or `doca_dpdk_port_as_dev()` from a port id) so DOCA Core / DOCA Flow — which operate on the `doca_dev` — can drive the same physical port the DPDK app uses | [`## Capabilities and modes`](#capabilities-and-modes) bridge-objects table + [TASKS.md ## configure](TASKS.md#configure) steps 3-4 |
| 3. Convert at the data-plane boundary | Translate `rte_mbuf` ↔ DOCA-buf only at the points where DOCA libraries actually need to operate on the payload; do not convert in the inner loop unnecessarily | [`## Capabilities and modes`](#capabilities-and-modes) mbuf-conversion subsection + [TASKS.md ## modify](TASKS.md#modify) |
| 4. Discover capabilities | Query `doca_dpdk_cap_is_rep_port_supported()` on the active `doca_devinfo` to learn whether the device supports representors for `doca_dpdk_port_probe()` — the bridge's single public capability query on this device + this DOCA + this DPDK | [`## Capabilities and modes`](#capabilities-and-modes) capability-query rule + [TASKS.md ## configure](TASKS.md#configure) step 5 |
| 5. Diagnose a bridge error | Map symptom (`DOCA_ERROR_BAD_STATE`, `_NOT_FOUND`, `_INVALID_VALUE`, `_NOT_PERMITTED`, `_NOT_SUPPORTED`) to root cause without leaving the bridge layer prematurely; in particular, do not invent a code fix for a DPDK ↔ DOCA version mismatch | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **The bridge is an interop layer, not a migration step.** Its
  purpose is to let an existing DPDK data-plane reach DOCA
  libraries (especially DOCA Flow) WITHOUT rewriting the
  data-plane. Recommending a port-then-rewrite plan when the
  user already has a working DPDK app is almost always the
  wrong tradeoff. Recommending the bridge for a fresh project
  with no DPDK code is also wrong — that user wants
  [`doca-eth`](../doca-eth/SKILL.md).
- **DPDK and DOCA are a matched pair, not independent
  installs.** The bridge is the layer that reads both versions
  and assumes a compatibility window the user did not pick.
  The single most common bridge failure mode is *"the bridge
  loads but every operation returns confusing errors"* caused
  by a DPDK install that does not match the DPDK the bridge
  was built against. The agent must surface this coupling
  *before* recommending any code change — see
  [`## Version compatibility`](#version-compatibility).

## Capabilities and modes

The two orthogonal selection axes for any DOCA DPDK Bridge
design are *path* (bridge vs native) and, for the bridge path,
*port handover and conversion shape*. Choose the path first,
then drill into the relevant capability-query.

**Bridge vs native — the path-selection rule.** This is the
single most consequential decision the agent makes for a user
asking *"how do I do packet I/O with DOCA?"*.

| Path | When it fits | When it does NOT fit | Owning skill |
| --- | --- | --- | --- |
| `doca-dpdk-bridge` (this skill) | The user has an EXISTING DPDK app and wants to reach DOCA libraries (most commonly DOCA Flow for HW steering) WITHOUT rewriting the data-plane. The DPDK port and EAL stay owned by DPDK; the bridge surfaces the port to DOCA. | The user is starting fresh; or the user is host-only and DPDK alone meets their needs; or the user only wants DOCA libraries and has no DPDK lock-in. | This skill (interop overlay on top of [`doca-eth`](../doca-eth/SKILL.md) and [`doca-flow`](../doca-flow/SKILL.md)) |
| Native `doca-eth` | The user is starting fresh, has no DPDK code yet, and wants line-rate raw packet I/O against DOCA queues directly. | The user already has a working DPDK app and migrating it to `doca-eth` is a multi-week rewrite for a feature (HW steering) that the bridge can deliver without it. | [`doca-eth`](../doca-eth/SKILL.md) |
| Pure DPDK, no DOCA | Host-only line-rate packet processing with no DOCA library in the picture. | Any DOCA library is needed (Flow / Compress / AES-GCM / Telemetry / …). | Out of scope for this bundle; defer to upstream DPDK docs reachable via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md). |

The agent's rule on this axis: **ask the user whether DPDK is
already in their codebase before recommending a path**. The
bridge is a good answer for *"yes, DPDK is in"* and almost
never the right answer for *"no DPDK yet"*.

**Bridge object surface.** There is **no `doca_dpdk_port`
handle/object**. The bridge instead establishes a 1:1
association between a *DPDK port id* and a *`doca_dev`*, and a
separate memory-pool object for mbuf conversion. DOCA Flow and
the other DOCA libraries operate on the `doca_dev` (and the
DPDK port id), not on any bridge-specific port handle.

| Surface | What it does | Owner of the underlying state | Key inputs |
| --- | --- | --- | --- |
| DPDK port id ↔ `doca_dev` mapping | `doca_dpdk_port_probe(doca_dev, devargs)` attaches a DPDK port for an opened `doca_dev`; `doca_dpdk_port_as_dev(port_id, &dev)` returns the `doca_dev` for a DPDK port id; `doca_dpdk_get_first_port_id()` / `doca_dpdk_get_port_ids()` go the other way (`doca_dev` → port id(s)) | DPDK keeps owning the `rte_eth_dev` (start, stop, queue config); the `doca_dev` is the DOCA-side identity DOCA Flow / Core program against | An opened `doca_dev` (for probe) or a DPDK port id (for `as_dev`); the bridge does not re-size queues |
| `doca_dpdk_mempool` (mbuf ↔ DOCA-buf conversion) | `doca_dpdk_mempool_create(rte_mempool, &mempool)` wraps a DPDK mempool; `doca_dpdk_mempool_mbuf_to_buf(mempool, inventory, mbuf, &buf)` acquires a `doca_buf` that references the same memory as an `rte_mbuf`, using a `doca_buf_inventory` | DPDK owns the originating `rte_mempool`; the `doca_dpdk_mempool` + `doca_buf_inventory` own the DOCA-side bookkeeping | Per-packet, at the data-plane boundary where a DOCA library needs to operate on the payload — *not* in DPDK's inner loop |

The agent must NOT invent additional bridge objects beyond what
the headers and the `pkg-config --exists doca-dpdk-bridge`
install advertise. When the user asks about *"any other DPDK ↔
DOCA helper"*, the right answer is to walk the headers under
$(pkg-config --variable=includedir doca-common) (`doca_dpdk.h`)
and the bridge usage inside the shipped DOCA Flow samples at
`/opt/mellanox/doca/samples/doca_flow/` (e.g. `flow_common.c`,
which calls `doca_dpdk_port_probe()` / `doca_dpdk_port_as_dev()`).

**mbuf ↔ DOCA-buf conversion — the only rule.** Convert at the
*boundary*, not in the inner loop:

- The bridge exists so DPDK keeps doing the work it is good at
  (line-rate mbuf shuffling) and DOCA keeps doing the work it
  is good at (HW steering / accelerator offload / per-port
  control). The conversion is the cost of crossing the model
  boundary; it is not free, and the agent should not recommend
  doing it on every packet of a high-rate path unless the user
  has measured.
- If the user's pattern is *"every mbuf gets converted, fed to
  one DOCA call, then the result is converted back"*, that is
  often a sign the user actually wants the DOCA library inline
  (e.g. DOCA Flow programs the steering once and then DPDK
  delivers already-steered packets — no conversion needed in
  the data-plane). Surface this before recommending a hot-path
  conversion.

**Capability discovery — the only rule.** Unlike most DOCA
libraries, the bridge ships **only one** device-capability
query: `doca_dpdk_cap_is_rep_port_supported()`, which reports
whether the device supports representors for
`doca_dpdk_port_probe()`. There is no `doca_dpdk_*_cap_*`
family; the agent must not invent per-conversion or per-offload
cap symbols. The agent should:

| Capability | Query | Why the agent must ask |
| --- | --- | --- |
| Bridge presence on this DOCA install | `pkg-config --exists doca-dpdk-bridge` | Some DOCA releases use a slightly different module name; the agent must confirm spelling against the user's install before quoting any bridge symbol |
| Representor support for `port_probe` on this device | `doca_dpdk_cap_is_rep_port_supported()` against the active `doca_devinfo` (call with root privileges) | This is the bridge's only public device-capability query; call it before requesting representors in the `doca_dpdk_port_probe()` devargs — see [TASKS.md ## configure](TASKS.md#configure) step 5 |
| DPDK version the bridge expects | `pkg-config --modversion libdpdk` (DPDK side) compared against the bridge's documented compatibility window | Mismatched DPDK ↔ DOCA = confusing failures at runtime; the agent must surface the coupling before any code change. Do not quote a DPDK version pin from agent memory; read the host |
| Any other conversion / port-role support | Walk `doca_dpdk.h` shipped with this DOCA | The bridge exposes no further cap-query symbols; the header function set itself is the authoritative statement of what the bridge supports per-release — read it, do not assume |

**Configuration shape.** *Mandatory* preconditions before the
bridge call sequence: DPDK is installed and matched-pair-
compatible with this DOCA; DPDK EAL is initialized
(`rte_eal_init`) by the user's DPDK app; the target port is
bound under DPDK and started (`rte_eth_dev_start`); a
`doca_dev` is opened against the same physical device, and the
DPDK port id is bound to that `doca_dev` via
`doca_dpdk_port_probe()` (or resolved with
`doca_dpdk_port_as_dev()`). *Optional* configurations
(representors in the probe devargs) gate on
`doca_dpdk_cap_is_rep_port_supported()`.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way
match rule, NGC container semantics, and the
headers-win-over-docs rule, see
[`doca-version`](../../doca-version/SKILL.md). The body lives
there; this skill does not duplicate it.

**The bridge-specific overlay** is:

- **DPDK and DOCA are a matched pair.** The bridge is built and
  shipped against a specific DPDK version on each DOCA release;
  the user's installed DPDK must fall within that window.
  `pkg-config --modversion libdpdk` on the host is the runtime
  authority on which DPDK is in use. Disagreement between the
  user's DPDK and the bridge's expected DPDK is the canonical
  reason for *"the bridge loads but every operation returns
  confusing errors"* and the agent MUST surface this coupling
  before any code change. Do not quote a DPDK version pin from
  agent memory; do not assume *"the latest DPDK works"*; read
  the host. Per the cross-cutting headers-win rule in
  [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility),
  the bridge's installed headers are the authoritative
  statement of which DPDK it expects.
- **`doca-dpdk-bridge.pc` and `doca-common.pc` must both match
  `doca_caps --version`** at the four-way-match check (per
  [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)).
  Use `pkg-config --modversion doca-dpdk-bridge` as the
  build-time anchor; disagreement with `doca_caps --version` is
  a partial-install hazard and must be routed to
  [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
  layer 2 before any bridge-layer diagnosis. If
  `pkg-config --exists doca-dpdk-bridge` returns failure, the
  bridge is either not installed on this DOCA, or the module
  name in this DOCA release differs — confirm via
  `ls /opt/mellanox/doca/infrastructure/lib/pkgconfig/ | grep
  -i dpdk` before claiming the bridge is missing.

## Error taxonomy

Bridge-specific overlays on the cross-library `DOCA_ERROR_*`
taxonomy. The cross-library taxonomy itself lives in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy);
the rows below are the *bridge surface* meaning that the
agent must disambiguate before falling back to the cross-
library response.

| Error | DOCA DPDK Bridge context where it shows up | Bridge-specific cause |
| --- | --- | --- |
| `DOCA_ERROR_BAD_STATE` | Any bridge call before the underlying DPDK port is started, before `rte_eal_init` returned, or before `doca_ctx_start()` on the bridge-using DOCA context; also returned after teardown of either the DPDK port or the DOCA context | Lifecycle violation that crosses the DPDK ↔ DOCA seam. Walk the bridge's lifecycle order in [TASKS.md ## configure](TASKS.md#configure); the most common case is registering the DPDK port with the bridge before `rte_eth_dev_start` — DPDK has to be live first. |
| `DOCA_ERROR_NOT_FOUND` | Port-registration / handle-acquire calls on a DPDK port id that the bridge does not see | The DPDK port id is wrong (typo, wrong port index), the port is not bound under DPDK on this host, or the port id is from a different EAL instance. Confirm the user's `rte_eth_dev_get_port_by_name` / DPDK port enumeration first; do not modify the bridge call. |
| `DOCA_ERROR_INVALID_VALUE` (or `DOCA_ERROR_NO_MEMORY`) | `doca_dpdk_mempool_mbuf_to_buf()` conversion calls | The `rte_mbuf` does not represent memory from the originating `rte_mempool` that this `doca_dpdk_mempool` was created from (mbufs from external memory or a different pool are not convertible), or the supplied `doca_buf_inventory` has no free elements (`DOCA_ERROR_NO_MEMORY`). Walk the conversion's preconditions in [`## Capabilities and modes`](#capabilities-and-modes); the fix is at the mempool / inventory layer, not the inner-loop call. |
| `DOCA_ERROR_NOT_PERMITTED` | Bridge create / open calls when the process lacks DPDK-side privileges (no hugepage access, no PCIe port access) OR DOCA-side privileges (cannot open `doca_dev`) | The bridge inherits both privilege sets. Confirm `id` for group membership AND that DPDK's standard preconditions (`dpdk-devbind.py`, hugepages mounted) are met; route to [`doca-setup`](../../doca-setup/SKILL.md) for the env-side fix. |
| `DOCA_ERROR_NOT_SUPPORTED` | Representors requested in `doca_dpdk_port_probe()` on a device that does not support them, OR DPDK ↔ DOCA version mismatch surfacing as a runtime miss | Re-run `doca_dpdk_cap_is_rep_port_supported()` against the active `doca_devinfo`. If it says false, that is the answer. If it says true but the call still fails, suspect the DPDK ↔ DOCA matched-pair window per [`## Version compatibility`](#version-compatibility) before any code change. |
| `DOCA_ERROR_DRIVER` | Any bridge call that crosses into the kernel mlx5 driver | The layer below DOCA + DPDK reported failure. Capture state (`dmesg | tail`, `mlxconfig -d <pcie> q`) and route to env-class debug ([`doca-setup ## debug`](../../doca-setup/TASKS.md#debug)) — the layer below DOCA is the suspect, not the bridge program. |

The agent's rule: **never recommend a retry loop on a bridge
`DOCA_ERROR_*` without first identifying which of the rows
above is the cause**. None of the bridge-class errors above are
*"would-block, retry"* errors; they are configuration,
permission, or version errors. And *"my DPDK app sees packets
but my DOCA Flow rules don't apply"* is not a `DOCA_ERROR_*` —
it is a steering / port-attachment gap; route to
[`doca-flow`](../doca-flow/SKILL.md), not to the bridge error
taxonomy.

## Observability

DOCA DPDK Bridge observability is **doubled** — the agent must
read both the DPDK side and the DOCA side. The bridge owns
neither; it surfaces both.

Three primary signals the agent should reach for:

1. **DPDK-side port and queue counters.** `rte_eth_stats_get`
   on the DPDK port (or whatever stats path the user's DPDK
   app already plumbs) reports rx / tx / errors / drops on the
   DPDK queues that the bridge surfaces to DOCA. If the
   DPDK-side counters are zero, DOCA never had a chance — the
   problem is upstream of the bridge. The agent must check
   DPDK counters BEFORE diagnosing any bridge symptom.
2. **DOCA-side progress engine events for any bridge-using
   context.** The DOCA Core PE
   ([`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes))
   surfaces task completions for the DOCA libraries that
   operate on the `doca_dev` bound to the DPDK port (DOCA Flow
   rule installs, DOCA accelerator submissions, …). Absence of completions on a
   started DOCA context is *always* either a missing
   `doca_pe_progress()` call or — for steering operations — a
   port-attachment gap on the bound `doca_dev`.
3. **Capability snapshot at configure time.** The output of
   `doca_dpdk_cap_is_rep_port_supported()` is a snapshot of
   *what the bridge said was possible* before any task was
   submitted.
   Save it as the baseline; if a bridge call later returns
   `DOCA_ERROR_NOT_SUPPORTED` the diff against this snapshot
   is the bug — and the most common reason for that diff is a
   DPDK install change underneath, not a DOCA change.

For the cross-library debug-time observability
(`DOCA_LOG_LEVEL=trace`, `--sdk-log-level`, the trace build
flavor) see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).
For DPDK's own logging surface (`rte_log`, `--log-level`), the
upstream DPDK documentation is the authority — route via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

DOCA DPDK Bridge's safety surface is **matched-pair version
coupling, lifecycle ordering across the DPDK ↔ DOCA seam, and
the inherited DPDK privilege set**.

The **precondition matrix** the agent must walk for any new
bridge setup:

| Precondition | What must be true before the first bridge call | How the agent verifies | Where to fix |
| --- | --- | --- | --- |
| DOCA + DPDK matched pair | `pkg-config --modversion doca-dpdk-bridge` agrees with `doca_caps --version`; `pkg-config --modversion libdpdk` falls within the window the bridge's installed headers / docs declare; `pkg-config --exists doca-dpdk-bridge` succeeds | The three pkg-config queries above; if the bridge module name differs in this release, `ls /opt/mellanox/doca/infrastructure/lib/pkgconfig/ | grep -i dpdk` to find it | [`doca-version`](../../doca-version/SKILL.md) for the canonical four-way match; reinstall the matched pair if it does not hold — do not patch around it |
| DPDK runtime preconditions met | The user has hugepages mounted, the target PCIe port is bound under DPDK (e.g. via `dpdk-devbind.py`), `rte_eal_init` is called by the user's app, `rte_eth_dev_start` succeeded on the target port id | DPDK's own diagnostics (`dpdk-devbind.py --status`, hugepage `cat /proc/meminfo | grep Huge`); the agent does NOT walk DPDK setup itself — defer to upstream DPDK docs via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) | Upstream DPDK documentation; [`doca-setup`](../../doca-setup/SKILL.md) for the hugepage / `mlx5` env preconditions DOCA shares with DPDK |
| DOCA-side device access | A `doca_dev` opens against the same physical device DPDK is using (typically requires sudo or `mlnx`-group membership) | `id` for group membership; the open call failing with `DOCA_ERROR_NOT_PERMITTED` is the runtime symptom | [`doca-setup`](../../doca-setup/SKILL.md) for the env-side; do not modify the bridge program |

**Lifecycle order across the DPDK ↔ DOCA seam.** The bridge
inherits from BOTH lifecycles; the agent must keep the order
straight:

1. DPDK side: `rte_eal_init` → port configure → `rte_eth_dev_start`.
2. DOCA side: open `doca_dev` against the same device; create the
   DOCA Core context that will use the bridge.
3. Bridge: bind the DPDK port id to the `doca_dev`
   (`doca_dpdk_port_probe()` from the `doca_dev`, or
   `doca_dpdk_port_as_dev()` from the port id); pass that
   `doca_dev` to the DOCA library that needs it (DOCA Flow, etc.).
4. `doca_ctx_start()` on the DOCA side, then run.
5. Teardown in reverse: DOCA contexts stop / destroy first, then
   release the bridge binding, then DPDK port stop / EAL cleanup.

Out-of-order is caught as `DOCA_ERROR_BAD_STATE` (DOCA side) or
DPDK-side error (DPDK side), but the user-visible symptom does
not name the conflation; the agent must walk the order
explicitly.

**Smoke before scale-up.** Before exercising the full
DPDK + DOCA Flow path at line rate, the agent must walk the
user through a single trivial smoke (one DPDK port up; one
trivial DOCA Flow rule installed on the bound `doca_dev`; verify
the rule is programmed and that one packet steers as expected).
A failure here narrows cleanly: DPDK-side, bridge-handover, or
DOCA-side. A failure at scale-up *without* the smoke pass is a
much harder bisection — and on this stack the bisection has to
cross two version surfaces, which makes guessing
catastrophically expensive.
