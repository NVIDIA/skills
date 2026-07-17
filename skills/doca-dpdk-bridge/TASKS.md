# DOCA DPDK Bridge workflows

**Where to start:** The verbs run `configure → build → modify →
run → test → debug`. Skip ahead only when the user is already
past a verb. The `## test` verb is an iterative loop (bridge
install probe → DPDK preconditions → single-rule smoke → multi-
rule run → cross-version), not a one-shot pass — see the
eval-loop overlay in `## test` below. Every workflow below
assumes the *interop* case: the user has an EXISTING DPDK app
and wants to add DOCA libraries on top. If the user is starting
fresh (no DPDK code yet), the right answer is *not* this skill —
route to [`doca-eth`](../doca-eth/SKILL.md) and stop here.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the bridge capability surface,
bridge-vs-native path selection, port-handover and
mbuf-conversion shape, capability-query rules, error taxonomy,
observability, and safety policy, see
[CAPABILITIES.md](CAPABILITIES.md). For the cross-library DOCA
patterns layered under everything below (the universal
lifecycle, the cross-library `DOCA_ERROR_*` taxonomy, the
modify-a-shipped-sample workflow), see
[`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
For the steering library that is the most common reason to use
this bridge, see [`doca-flow`](../doca-flow/SKILL.md). For the
matched-pair DPDK ↔ DOCA version coupling that this skill
overlays on top of the canonical four-way match, see
[`doca-version`](../../doca-version/SKILL.md).

Each verb below describes the **shape of the workflow**, not a
copy-paste recipe. The agent's job is to walk the user through
the steps in order, verifying preconditions before recommending
the next call.

## configure

Goal: bind an already-running DPDK port id to an opened
`doca_dev` (the bridge's DPDK-port-id ↔ DOCA-device association),
with both the DPDK side and the DOCA side aware of their
lifecycle responsibilities, before any DOCA library (most
commonly DOCA Flow) is asked to operate on that `doca_dev`.

Steps the agent should walk the user through, in order:

1. **Confirm the bridge is installed and the DPDK ↔ DOCA pair
   is consistent.** Per the precondition matrix in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
   and the version overlay in
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility),
   run three pkg-config probes BEFORE writing any code:
   `pkg-config --exists doca-dpdk-bridge` (the bridge is
   present on this DOCA install — if it returns failure, run
   `ls /opt/mellanox/doca/infrastructure/lib/pkgconfig/ | grep
   -i dpdk` before claiming the bridge is missing, because some
   releases use a slightly different module name);
   `pkg-config --modversion doca-dpdk-bridge` agrees with
   `doca_caps --version` (no partial-install drift); and
   `pkg-config --modversion libdpdk` falls within the
   compatibility window the bridge expects. Quote all three to
   the user. Disagreement on the third probe is the canonical
   *"the bridge loads but every operation returns confusing
   errors"* failure mode and must be surfaced before any code
   change — route to
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   layer 2.
2. **Confirm this is the interop case, not the migration
   case.** Per the bridge-vs-native path-selection table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   the bridge is the right answer only when the user has an
   EXISTING DPDK app. Ask explicitly: *"is there already DPDK
   code that drives `rte_eth_*` ports in this codebase?"* If
   the answer is no, stop here and route to
   [`doca-eth`](../doca-eth/SKILL.md); the bridge is not the
   right starting point for fresh DOCA-native packet I/O.
3. **Verify the DPDK runtime preconditions on the user's
   host.** Per the precondition matrix in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   four DPDK-side conditions must already be true before the
   first bridge call: hugepages are mounted
   (`cat /proc/meminfo | grep Huge` reports non-zero free
   pages); the target PCIe port is bound under DPDK
   (`dpdk-devbind.py --status`); `rte_eal_init` is called by
   the user's app; and `rte_eth_dev_start` succeeded on the
   target port id. The agent does NOT walk DPDK setup itself —
   defer the env-side fix to upstream DPDK docs reachable via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
   and the shared hugepage / mlx5 env preconditions to
   [`doca-setup TASKS.md ## configure`](../../doca-setup/TASKS.md#configure).
4. **Open the DOCA-side device and bind it to the DPDK port.**
   Per the bridge-objects table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   open a `doca_dev` against the same physical device (DOCA-side
   privileges via sudo or `mlnx`-group membership), then
   establish the DPDK-port-id ↔ `doca_dev` association: if the
   DPDK port has not been probed yet, call
   `doca_dpdk_port_probe(doca_dev, devargs)` to attach it; if the
   user already has a DPDK port id, call
   `doca_dpdk_port_as_dev(port_id, &dev)` to resolve its
   `doca_dev` (and `doca_dpdk_get_first_port_id()` /
   `doca_dpdk_get_port_ids()` go from a `doca_dev` to its DPDK
   port id(s)). There is no `doca_dpdk_port` handle — the
   `doca_dev` is the DOCA-side identity, and DPDK keeps owning
   `rte_eth_dev` start / stop / queue config. The agent must walk
   `doca_dpdk.h` under
   $(pkg-config --variable=includedir doca-common) for the exact
   signatures on this DOCA release rather than quote them from
   memory.
5. **Run capability discovery against the active `doca_devinfo`
   BEFORE requesting representors.** Per the capability-query
   rule in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   the bridge exposes a single device-capability query:
   `doca_dpdk_cap_is_rep_port_supported()` on the active
   `doca_devinfo` (call with root privileges) tells you whether
   the device supports representors for `doca_dpdk_port_probe()`.
   There is no `doca_dpdk_*_cap_*` family — do not invent
   per-conversion or per-offload cap symbols; `doca_dpdk.h`
   itself is the authority on the rest of the surface. Quote the
   queried value back to the user; do not assume from prior
   installs.
6. **Start the DOCA context that will use the bound device.**
   Pass the `doca_dev` to the DOCA library that needs it
   (DOCA Flow is the canonical adopter; see
   [`doca-flow TASKS.md ## configure`](../doca-flow/TASKS.md#configure)),
   then call `doca_ctx_start()` on the DOCA side and progress
   the PE (`doca_pe_progress`). The DPDK side stays running
   throughout — the bridge does not stop the DPDK port. For
   the canonical DOCA universal lifecycle that underlies the
   DOCA-side calls, see
   [`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure);
   this skill adds only the bridge overlay (steps 2-5 above).

## build

Goal: compile a bridge-using consumer against the user's
installed DOCA AND the user's installed DPDK, with `pkg-config`
as the source of truth for include + link flags on BOTH sides.

The build pattern for any DOCA C/C++ consumer is fully
documented in
[`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).
This skill carries only the bridge-specific overlay — note that
the bridge is the one library in the bundle where the link line
spans TWO independent pkg-config namespaces (DOCA and DPDK), and
*both* must agree with the installed runtime:

| Slot | Value | Why it matters |
| --- | --- | --- |
| `pkg-config` module name (DOCA side) | `doca-dpdk-bridge` on the user's install — confirm with `pkg-config --exists doca-dpdk-bridge`; if it returns failure, run `ls /opt/mellanox/doca/infrastructure/lib/pkgconfig/ | grep -i dpdk` before claiming the bridge is missing, because some DOCA releases use a slightly different module name | Wrong module name = `pkg-config: Package 'doca-dpdk-bridge' was not found` at build time. Do not guess — read the install. |
| `pkg-config` module name (DPDK side) | `libdpdk` from the user's DPDK install | The bridge's link line depends on DPDK symbols; without `libdpdk` on the line the build fails at link time with unresolved `rte_*` references |
| Include flags | `pkg-config --cflags doca-dpdk-bridge libdpdk` | Resolves DOCA-side headers under $(pkg-config --variable=includedir doca-common) AND DPDK-side headers wherever DPDK is installed |
| Link flags | `pkg-config --libs doca-dpdk-bridge libdpdk` | Pulls in whatever `pkg-config --libs` resolves on this install (do not predict the `-l<name>` form by hand — `.so` basenames use underscores, `.pc` names use hyphens, and `pkg-config` is the only correct translator) on the DOCA side plus the DPDK shared libs (`-lrte_*`) on the DPDK side |
| Companion libraries | `doca-flow` if the consumer programs steering on the bound `doca_dev` (the common adopter); `doca-argp` for argument parsing if the consumer uses the standard DOCA arg style. **Do not** add other DOCA libraries unless the consumer actually calls them | Adding unnecessary companion libs bloats the link line and obscures real partial-install issues |
| Build-time version anchors | `pkg-config --modversion doca-dpdk-bridge` AND `pkg-config --modversion libdpdk` — record BOTH in the build log | Cross-version build / runtime mixing breaks per [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility); the DPDK side is the under-tested seam |

For non-C consumers (Rust, Go, Python), DPDK itself is a C-shaped
library and a non-C DPDK app is rare; if the user is in that
minority, the build-time concern is the same — the FFI wrapper
must see *both* `libdoca_dpdk_bridge.so` and `librte_*.so` at the
same versions the bridge was built against. The role-split and
capability-discovery rules from [`## configure`](#configure) still
apply per the runtime version rule in
[`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).

## modify

Goal: take the closest-fitting shipped DOCA Flow sample that
already uses the bridge (or the closest reference application
that pairs DPDK + DOCA Flow) and apply a **minimum diff** to make
it match the user's intent, without rewriting from scratch and
without inventing a build manifest of your own.

The universal modify-a-shipped-sample workflow is in
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify);
this skill provides the bridge-specific slot fill — the slots
the agent must elicit from the user before recommending any
code-level edit:

| Slot | What the agent asks the user | Bridge-specific consideration |
| --- | --- | --- |
| 1. Starting sample | Which DOCA Flow sample under `/opt/mellanox/doca/samples/doca_flow/` that already drives the bridge (e.g. ones using `flow_common.c`, which calls `doca_dpdk_port_probe()` / `doca_dpdk_port_as_dev()`)? Or which reference application that already pairs DPDK + DOCA Flow (per the reference-applications index in [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md))? | Pick the closest *adopter shape* — bridge + DOCA Flow vs bridge + DOCA accelerator — to the user's intent. The sample tree is C; a *"clean rewrite"* in another language is almost always slower to first green |
| 2. Port-handover code | Are you wiring a new DPDK port id into the bridge, or reusing the sample's port-handover code as-is? | The `doca_dpdk_port_probe()` / `doca_dpdk_port_as_dev()` binding and the `doca_dev` open are the load-bearing edits. Do NOT change the DPDK port's start / stop / queue-config calls — those stay owned by the DPDK side. The bridge associates a port id with a `doca_dev`; it does not transfer ownership of the port |
| 3. Conversion sites | Where in the data plane are `rte_mbuf` ↔ `doca_buf` conversions actually needed? | Per the mbuf-conversion rule in [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes), convert at the boundary, NOT in the inner loop. If the diff puts a conversion call on every packet of a high-rate path, surface the cost before recommending the edit — the user often actually wants DOCA Flow steering installed once and DPDK to keep delivering already-steered packets, with no per-packet conversion |
| 4. Capability re-validation | Did the diff start requesting representors on the probe that the program did not assume before? | Re-run `doca_dpdk_cap_is_rep_port_supported()` from [`## configure`](#configure) step 5 against the active `doca_devinfo`; the device cap is the authority, not the prior install's |
| 5. Build manifest | Are you keeping the sample's existing `meson.build`, or hand-rolling a Makefile? | Keep the sample's manifest. It already wires `pkg-config doca-dpdk-bridge libdpdk` and pins both versions; switching to a hand-rolled build removes the version-check rail per [`## build`](#build) and re-opens the DPDK ↔ DOCA mismatch failure mode |

The agent's anti-pattern alert: a *"clean rewrite"* from
scratch is almost always slower to first green than a minimum-
diff modify on a shipped sample, and removes the user's ability
to bisect against a known-good baseline. On this stack the
bisection has to cross TWO version surfaces (DOCA AND DPDK),
which makes guessing especially expensive — the known-good
sample anchors both.

## run

Goal: execute the bridge-using program end-to-end and confirm
the DPDK port reaches DOCA AND that one trivial DOCA-library
operation works on the bound `doca_dev`, before any
production traffic.

Steps the agent should walk the user through:

1. **Bring up the DPDK side first.** The user's existing DPDK
   app starts as it normally does — `rte_eal_init`, port
   configure, `rte_eth_dev_start`. Confirm the DPDK port is
   live BEFORE the DOCA side is started (`rte_eth_stats_get`
   on the port or whatever stats path the DPDK app already
   plumbs reports the port up; rx / tx counters increment
   under expected traffic). If the DPDK side is not live, the
   bridge binding may resolve but the first DOCA operation
   returns `DOCA_ERROR_BAD_STATE` per the lifecycle row in
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).
2. **Capture the structured log.** Set `DOCA_LOG_LEVEL=trace`
   for the first run (see
   [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability));
   the bridge calls and the underlying DOCA Core lifecycle
   transitions appear on stderr. For DPDK's own logging
   surface (`rte_log`, `--log-level`), the upstream DPDK docs
   are the authority — route via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md);
   the agent must not paraphrase DPDK log knobs.
3. **Send one trivial DOCA-library operation through the
   bridge handle.** For the canonical adopter (DOCA Flow): one
   trivial flow rule that matches a wide pattern (e.g. drop or
   count all incoming packets on the port) installed on the
   `doca_dev` bound to the DPDK port, with verification that the rule is
   programmed (a Flow-side validate / counter read per
   [`doca-flow TASKS.md ## configure`](../doca-flow/TASKS.md#configure)).
   Per the smoke-before-scale-up rule in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   this single-rule smoke is non-negotiable: a failure here
   narrows cleanly to DPDK-side, bridge-handover, or DOCA-side;
   a failure at scale-up without the smoke pass is a much
   harder bisection across the two version surfaces.
4. **Tear down in the right order.** Per the lifecycle table in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   reverse the configure order: stop / destroy the DOCA
   contexts first, then release the bridge binding, then stop
   the DPDK port and clean up EAL. Out-of-order teardown
   surfaces as either DOCA-side `DOCA_ERROR_BAD_STATE` or DPDK-
   side errors and masks the next run's first-failure signal.

For the runtime version + `LD_LIBRARY_PATH` cross-checks that
underlie *"the program built but does nothing"* (especially the
DPDK side, which is the under-tested seam on this skill), see
[`doca-version TASKS.md ## run`](../../doca-version/TASKS.md#run).

## test

Goal: prove a bridge consumer is correct end-to-end, on the
user's installed DOCA + installed DPDK + device + permissions,
before claiming the *"add DOCA to my DPDK app"* journey is done.

This is **a loop, not a one-shot pass.** Each iteration narrows
either the install / version surface (DOCA-side or DPDK-side),
the port-handover, the conversion-site set, or the
single-library smoke shape. The loop terminates when either (a)
the user's intended DOCA library composes cleanly with the
existing DPDK app at the rate they care about, or (b) the agent
has narrowed the failure cause to a layer outside the bridge
(DPDK app itself, driver, firmware, env) and escalated to the
matching skill.

Iteration shape:

1. **Install + version re-check.** Re-run the three pkg-config
   probes from [`## configure`](#configure) step 1 plus
   `doca_caps --version`. The DPDK ↔ DOCA pair is the most
   common failure mode and the cheapest one to detect — quote
   all four numbers back. If they disagree, that is the answer;
   fix the install per
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   and re-loop. Do not move to step 2 with a mismatched pair.
2. **DPDK-side precondition re-check.** Walk the four DPDK-side
   conditions from [`## configure`](#configure) step 3
   (hugepages, devbind, EAL up, port started). A silent bridge
   handle (no error, but the DOCA library never sees packets)
   is *almost always* a DPDK-side gap, not a bridge bug — the
   DPDK-side counters from [`## run`](#run) step 1 are the
   first thing to read.
3. **Single-rule smoke.** As in [`## run`](#run) step 3 — one
   trivial DOCA Flow rule (or one trivial accelerator
   submission) on the bound `doca_dev`, with verification.
   If yes, advance. If no, narrow: a bridge-call `DOCA_ERROR_*`
   narrows per [`## debug`](#debug); a DPDK-side counter that
   stops incrementing after the rule is installed narrows to
   the steering itself — route to
   [`doca-flow TASKS.md ## debug`](../doca-flow/TASKS.md#debug).
4. **Multi-rule / multi-conversion run.** Once single-rule is
   green, scale to the user's actual rule set (or actual
   per-packet conversion shape) and confirm every expected
   operation completes. Catches sizing / queue-depth gaps that
   the single-rule smoke does not exercise. The PE
   (`doca_pe_progress`) must be driven at the rate of submits
   per the cross-library rule in
   [`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes).
5. **Cross-version / cross-host run** (if the user has multiple
   installs). Re-run steps 1-4 on each install; quote both the
   DOCA version AND the DPDK version per
   [`doca-version TASKS.md ## test`](../../doca-version/TASKS.md#test)
   on each host. The bridge fails differently on each pair, so
   *"works on host A, fails on host B"* without quoted versions
   on both sides is an unactionable report.

Eval-loop overlay — why this is a loop, not a one-shot pass:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| pkg-config probes pass but every bridge call returns confusing errors | Bridge loads; first DOCA Flow / accelerator call returns `DOCA_ERROR_NOT_SUPPORTED` or `DOCA_ERROR_BAD_STATE` with no obvious code-side cause | The DPDK ↔ DOCA matched-pair window per [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility) is the prime suspect. Re-check `pkg-config --modversion libdpdk` against the bridge headers; do not patch around with retries |
| Single-rule smoke passes; multi-rule run drops completions | Rule installs succeed; under load completions stop firing | The PE is not being progressed at the rate of submits, OR the bridge's per-port queue sizing is at its default. Drive `doca_pe_progress()` between submits; do not assume slow-path patterns transfer to scale-up |
| Bridge binding is in place but no packets ever reach the DOCA library | No `DOCA_ERROR_*`, no completions, DPDK-side counters increment normally | A steering / port-attachment gap — the DOCA Flow rule has not actually been programmed on the `doca_dev` bound to this DPDK port. Route to [`doca-flow TASKS.md ## debug`](../doca-flow/TASKS.md#debug); this is NOT a bridge error |
| `DOCA_ERROR_INVALID_VALUE` on every conversion call | Mbuf ↔ DOCA-buf conversion always fails | The mbuf points outside any registered `doca_mmap`, or the conversion target buffer is too small. Walk the conversion preconditions in [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes); the fix is at the buffer-management layer, not the inner-loop call |
| Same code passes on host A, fails on host B | One host has a different DPDK version, a different DOCA version, or a different `doca_dev` permission | Re-run steps 1-2 on host B + four-way match per [`doca-version TASKS.md ## test`](../../doca-version/TASKS.md#test); the bridge's failure shape is per-host because the matched pair is per-host |

Loop termination: stop iterating once two consecutive iterations
do not change the picture — the cause is below the bridge (DPDK
app code, kernel driver, firmware, env). Escalate to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
with the captured trace + version state (BOTH DOCA and DPDK
sides) as evidence.

## debug

Goal: when a bridge session fails — whether by `DOCA_ERROR_*`
return, silent DOCA library on top of a working DPDK port, or
matched-pair version drift — isolate the cause to a single
layer before recommending any code change.

The cross-library debug ladder lives in
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).
Walk through it in order — install → version → build → link →
runtime → program → driver. This skill provides the bridge
overlay at the *version*, *runtime*, and *program* layers; the
DPDK side overlays on all of them and is the under-tested seam.

**Layer 2 (version) — bridge overlay.**

- The four-way match in
  [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)
  becomes a FIVE-way match on this skill: add
  `pkg-config --modversion libdpdk` against the bridge's
  expected DPDK window per
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility).
  Disagreement here is the canonical *"the bridge loads but
  every operation returns confusing errors"* failure; surface
  it before any code change.
- `pkg-config --exists doca-dpdk-bridge` returning failure is
  not always *"the bridge is not installed"* — some DOCA
  releases use a slightly different module name. Confirm via
  `ls /opt/mellanox/doca/infrastructure/lib/pkgconfig/ | grep
  -i dpdk` before claiming the bridge is missing.

**Layer 5 (runtime) — bridge overlay.**

- A silent bridge binding (resolves cleanly, no `DOCA_ERROR_*`,
  but the DOCA library never sees packets) is *almost always*
  a DPDK-side precondition gap or a steering / port-attachment
  gap, not a bridge code issue. Read the DPDK-side counters per
  [`## run`](#run) step 1 BEFORE any code change. If DPDK
  counters are zero, the problem is upstream of the bridge
  (DPDK app, env). If DPDK counters increment but the DOCA
  library still does not see packets, route the steering side
  to [`doca-flow TASKS.md ## debug`](../doca-flow/TASKS.md#debug).
- `DOCA_ERROR_NOT_PERMITTED` from a bridge create / open call
  is the inherited DPDK + DOCA privilege set per
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).
  Confirm BOTH `id` for `mlnx`-group membership AND the DPDK-
  side hugepage / devbind preconditions before recommending a
  code change; route the env fix to
  [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug).
- `DOCA_ERROR_DRIVER` from any bridge call that crosses into
  the kernel mlx5 driver: capture `dmesg | tail` and
  `mlxconfig -d <pcie> q` and route to
  [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug);
  the layer below DOCA + DPDK is the suspect, not the bridge
  program.

**Layer 6 (program) — bridge overlay.**

- Lifecycle order across the DPDK ↔ DOCA seam (per
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)):
  DPDK side up → DOCA `doca_dev` opened → bridge port
  registered → DOCA `doca_ctx_start()`. Out-of-order returns
  `DOCA_ERROR_BAD_STATE` from the DOCA side; the most common
  case is registering the DPDK port with the bridge before
  `rte_eth_dev_start` succeeded. The DPDK side has to be live
  first.
- `DOCA_ERROR_NOT_FOUND` on port-registration / handle-acquire
  is a wrong DPDK port id, a port not bound under DPDK, or a
  port id from a different EAL instance — NOT a bridge bug.
  Confirm the user's `rte_eth_dev_get_port_by_name` or DPDK
  port enumeration result before modifying any bridge call.
- `DOCA_ERROR_NOT_SUPPORTED` after the cap query returned true
  is a strong signal the DPDK ↔ DOCA matched-pair window has
  drifted at runtime — return to layer 2. Do not invent a
  code workaround; the cap query is *not* lying, the matched
  pair is.

Once the layer is identified, route to the matching debug verb
on the matching skill: install / build / link / driver to
[`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug);
version (DOCA side AND DPDK side) to
[`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug);
steering to
[`doca-flow TASKS.md ## debug`](../doca-flow/TASKS.md#debug);
cross-cutting runtime to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug);
program-layer Core-context patterns to
[`doca-programming-guide TASKS.md ## debug`](../../doca-programming-guide/TASKS.md#debug).

## Deferred task verbs

The following verbs are out of scope for this skill but are
commonly asked in the same conversations. Route them as follows
so the agent does not invent guidance:

- **install (DOCA side).** Installing DOCA, choosing packages,
  post-install verification, `pkg-config` wiring — defer to
  [`doca-setup`](../../doca-setup/SKILL.md) and to the
  install-tree layout in
  [doca-public-knowledge-map ## Layout of an installed DOCA package](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package).
  This skill assumes DOCA is already installed.
- **install / learn DPDK itself.** Installing DPDK, mounting
  hugepages, binding PCIe ports, learning the DPDK programming
  model, choosing a DPDK release — out of scope. DPDK is its
  own upstream project; this skill treats DPDK as a
  precondition and routes the user to upstream DPDK docs via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
- **migrate to native `doca-eth`.** Rewriting the user's DPDK
  data-plane to `doca_eth_rxq` / `doca_eth_txq` — out of scope
  here, and almost always the wrong tradeoff for a working
  DPDK app per the bridge-vs-native rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
  For fresh DOCA-native projects with no DPDK lock-in, route to
  [`doca-eth`](../doca-eth/SKILL.md).
- **deploy.** Deploying DPDK + DOCA applications at scale
  across many hosts / BlueFields, Kubernetes operator
  workflows for line-rate data planes — out of scope for
  Phase 1 and reserved for a future platform skill.
- **firmware burn / reset / kernel driver install.** Installing
  `mlx5_core`, burning new ConnectX firmware, modifying
  `mlxconfig` parameters that require a reset — out of scope.
  Route to
  [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug)
  layer 5 (driver), then to the upstream MLNX OFED / firmware
  documentation reachable through
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Command appendix

Every command below is **cross-cutting on DOCA DPDK Bridge** —
it answers a recurring class of question that comes up in the
verbs above. The agent should treat the *class* as load-bearing;
the worked example is a single instance. Run-as user is the
unprivileged user unless noted. Sudo is called out per row. The
bridge is unusual in this bundle because every row reads BOTH
the DOCA side AND the DPDK side; do not skip the DPDK column
when reporting back.

**Infra-aware preamble (every row below).** Per the bundle's
detect → prefer → fall back → report contract documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent should:

1. Probe for the matching structured helper FIRST (`doca-env --json`
   for version + devices + libraries + drivers + hugepages in one
   shot; `doca-capability-snapshot` for per-device capability flags;
   `version-matrix.json` for *"available since"* lookups).
2. If the probe succeeds, the structured tool's output is the
   authoritative answer and the agent SHOULD NOT also run the
   manual command in the row below. Report *"using structured
   `<tool>`"*.
3. If the probe fails, fall back to the manual command in the
   row. Report *"falling back to manual chain"*.
4. The schemas the structured tools emit are defined in
   [`doca-structured-tools-contract ## Schemas`](../../doca-structured-tools-contract/SKILL.md#schemas);
   the version-handling semantics (four-way match, NGC,
   headers-win) are owned by
   [`doca-version`](../../doca-version/SKILL.md).

| Command (worked example) | Owning step | Class of question it answers | What healthy output looks like |
| --- | --- | --- | --- |
| `pkg-config --exists doca-dpdk-bridge && echo present` | `## configure` step 1; `## build` slot 1 | Is the bridge module present on this DOCA install (and under this spelling)? | `present`. Empty / non-zero exit = either the bridge is not installed on this DOCA, or the module name in this release differs — confirm via the `ls` row below before claiming the bridge is missing |
| `ls /opt/mellanox/doca/infrastructure/lib/pkgconfig/ | grep -i dpdk` | `## configure` step 1; `## debug` layer 2 | If `pkg-config --exists doca-dpdk-bridge` fails, what is the bridge module actually called in this DOCA release? | One `.pc` file containing `dpdk` in its name. Empty = the bridge truly is not installed on this DOCA; route to [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug) |
| `pkg-config --modversion doca-dpdk-bridge` | `## configure` step 1; `## build` slot 6 | What is the build-time DOCA DPDK Bridge version? | A semver string matching `doca_caps --version`. Disagreement = partial install (route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2) |
| `pkg-config --modversion libdpdk` | `## configure` step 1; `## build` slot 6; `## debug` layer 2 | What DPDK version is installed on this host, and does it fall within the bridge's matched-pair window? | A semver string; the agent quotes it against the bridge's installed headers / docs declared window. Disagreement = the canonical *"the bridge loads but every operation returns confusing errors"* failure; do not assume *"latest DPDK works"* |
| `pkg-config --cflags --libs doca-dpdk-bridge libdpdk` | `## build` | What include + link flags does the linker need for a bridge consumer? | DOCA-side includes resolve under $(pkg-config --variable=includedir doca-common); libs include whatever `pkg-config --libs doca-dpdk-bridge` resolves (do not predict the `-l<name>` form by hand); DPDK-side includes resolve under the user's DPDK install; libs include the `-lrte_*` set |
| `ls /opt/mellanox/doca/samples/doca_flow/` | `## modify` slot 1 | Which DOCA Flow samples ship in this install (the bridge API has no samples dir of its own — it is used inside the DOCA Flow samples, e.g. `flow_common.c`), and which is the closest starting point? | A list of DOCA Flow sample directories; pick one whose adopter shape (bridge + Flow steering) is closest. Empty = the user's install does not ship DOCA Flow samples; route to the reference-applications index in [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) |
| `dpdk-devbind.py --status` | `## configure` step 3; `## debug` layer 5 | Is the target PCIe port bound under DPDK? | The target port appears under the `Network devices using DPDK-compatible driver` section. Anything else means the port is not on the DPDK side and the bridge cannot see it |
| `cat /proc/meminfo | grep Huge` | `## configure` step 3; `## debug` layer 5 | Are hugepages mounted on this host (DPDK precondition)? | `HugePages_Total` non-zero, `HugePages_Free` non-zero before the DPDK app starts |
| `doca_caps --list-devs` | `## configure` step 4; `## debug` layer 5 | Which devices on this host can be opened as a `doca_dev` and used as the DOCA side of the bridge binding? | One row per visible device with PCIe address; the device the DPDK port is bound to must appear here |
| `dmesg | tail -n 40` (sudo) | `## debug` layer 7 | What did the kernel / driver log around the last bridge call? | Empty or recent benign messages. Repeated `mlx5` / IB errors → driver-layer bug; route to [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug) |
| `DOCA_LOG_LEVEL=trace ./<binary>` | `## run` step 2 | What did the structured DOCA logger emit for the first failing bridge call? | A trace-level line on every DOCA Core lifecycle transition and every bridge call. Silence after a bridge call = the DOCA-side PE is not progressed; DPDK-side silence in addition = the DPDK app is not running |

For commands shared across libraries (`pkg-config --modversion`,
`doca_caps`, `cat /opt/mellanox/doca/applications/VERSION`,
`DOCA_LOG_LEVEL`) the cross-library overlay is in
[`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
this table adds the bridge-specific rows (every one of which
reads BOTH the DOCA side and the DPDK side) on top.
