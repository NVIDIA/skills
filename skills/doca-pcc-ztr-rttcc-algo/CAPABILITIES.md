# DOCA PCC ZTR RTTCC Algorithm capabilities, version overlay, errors, observability, safety

**Where to start:** Pick the H2 anchor that matches your
question (what the algorithm is and which side it runs on /
the public DPA-side API surface / the variant set the
DPA-side source ships / the parameter and counter surface /
the algorithm's relationship to `doca-pcc` and to
`doca-pcc-counters` / when this algorithm is the right
baseline vs writing a custom one) and read that section
end-to-end. The tables in each section are the load-bearing
content; the prose around them is interpretation.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the *how* of executing each
pattern (the verbs `install / configure / build / modify /
run / test / debug / use`), jump to [TASKS.md](TASKS.md).
For the canonical DOCA version-handling rules that this
skill layers on top of, see
[`doca-version`](../../doca-version/SKILL.md). For the
host-side PCC framework that loads this algorithm, see
[`doca-pcc`](../doca-pcc/SKILL.md). For the read-only
counter-inspection side, see
[`doca-pcc-counters`](../../tools/doca-pcc-counters/SKILL.md).

## Pattern overview

Every ZTR RTTCC question this skill teaches resolves into
one of SIX patterns. The patterns are CLASSES — they apply
across every DOCA release and every host + BlueField-3 +
DPACC + firmware combination.

| Pattern | When it applies (class shape) | Where the substance lives |
| --- | --- | --- |
| 1. Decide: shipped reference vs custom algorithm | The library is the no-config baseline; users either deploy it as-is, tune the documented parameters, or write a custom algorithm against `doca-pcc` when latency / fairness / convergence requirements diverge | [`## Capabilities and modes`](#capabilities-and-modes) shipped-vs-custom rule + [TASKS.md ## install](TASKS.md#install) |
| 2. Three distinct artifacts: library, host framework, counter tool | The shipped algorithm is consumed BY `doca-pcc` (host-side framework); the canonical observability path is `doca-pcc-counters` (read-only inspection); conflating any two is the most common first-touch error | [`## Capabilities and modes`](#capabilities-and-modes) artifact-triangle table + [TASKS.md ## configure](TASKS.md#configure) |
| 3. Variant selection at DPA-side compile time | The DPA-side source ships variants (vanilla, path-migration, RX-rate, RX-rate + PM, multipath, multipath + credits, window-probeless); the *public* symbol the user calls is one of them, chosen at DPACC build time per the README | [`## Capabilities and modes`](#capabilities-and-modes) variants table + [TASKS.md ## modify](TASKS.md#modify) |
| 4. Patch the shipped DOCA PCC application to dispatch to the algorithm | The README documents the EXACT in-place edits: include the device header, dispatch from `doca_pcc_dev_user_algo()` under a chosen algo slot, init from `doca_pcc_dev_user_init()`, seed parameters from `doca_pcc_dev_user_set_algo_params()`, plus the build-script and dependency-meson updates | [`## Capabilities and modes`](#capabilities-and-modes) integration-edits table + [TASKS.md ## modify](TASKS.md#modify) |
| 5. Tune the algorithm's parameters from the host without rebuilding the DPA-side image | The algorithm exposes documented parameters (e.g. BW_G, ALPHA, MAX_DEC, MAX_INC, AI, HAI, CONGESTION_DELAY_THRESHOLD, MAX_DELAY, RATE_ON_FIRST_CONGESTION, FAST_SCHED, RTT_TIMEOUT_THRESHOLD per the shipped DPA-side header set) settable via `doca_pcc_dev_set_ztr_rttcc_params` | [`## Capabilities and modes`](#capabilities-and-modes) parameter-surface table + [TASKS.md ## use](TASKS.md#use) |
| 6. Confirm the algorithm is modulating traffic, not just loaded | Loading the algorithm does not guarantee it is shaping flows; the canonical check is per-port + per-flow PCC counters via `doca-pcc-counters` AND a sustained-load smoke | [`## Observability`](#observability) + [TASKS.md ## test](TASKS.md#test) |

Two cross-cutting rules that apply to *every* pattern above:

- **The shipped algorithm is one artifact in a triangle of
  three.** `doca-pcc` (host-side framework that loads
  algorithms onto the BlueField DPA), this algorithm
  library (one specific algorithm), and `doca-pcc-counters`
  (read-only inspection CLI) are THREE distinct surfaces.
  A buggy first-touch design conflates the library with
  the framework ("just use `doca-pcc-ztr-rttcc-algo` to
  load my own algorithm"), the library with the tool
  ("inspect counters with the algorithm"), or the
  framework with the tool ("write algorithms with the
  counter CLI"). The agent's discipline is to confirm
  which artifact the user actually needs BEFORE prescribing
  any action.
- **The algorithm modulates EXISTING RoCE-v2 traffic.** Per
  the shipped README ("ZTR RTTCC algorithm over
  BlueField-3's DPA processor"), the algorithm runs on the
  BlueField-3 DPA and shapes flows the host already has on
  the wire. If there is no RoCE-v2 traffic on the attached
  port, there is nothing for the algorithm to do — the
  loaded-but-no-effect failure mode is most often a
  traffic-absence failure mode, not an algorithm bug.

## Capabilities and modes

The two orthogonal selection axes for any ZTR RTTCC
deployment are *which BlueField-3 port whose RoCE-v2
traffic the algorithm controls* (inherited from the
host-side `doca-pcc` per-port-context rule documented in
[`doca-pcc CAPABILITIES.md ## Capabilities and modes`](../doca-pcc/CAPABILITIES.md#capabilities-and-modes))
and *which variant of the algorithm the DPA-side
translation unit compiled against*. Choose both before
prescribing any code change.

**Three-artifact triangle — algorithm library, host-side
framework, counter tool.**

| Artifact | What it is | This skill's role | Pkg-config module |
| --- | --- | --- | --- |
| **`doca-pcc-ztr-rttcc-algo`** (this skill) | The shipped reference PCC algorithm; one of several algorithm bodies that could be loaded onto the DPA, except this one is the no-config-required baseline NVIDIA ships | Walk deployment / tuning / evaluation of this specific algorithm | `doca-pcc-ztr-rttcc-algo` |
| **`doca-pcc`** | Host-side framework that loads ANY PCC algorithm onto the BlueField DPA via the `doca_pcc` Core context and the `doca_pcc_app` image | Cross-link only — the host-side lifecycle lives there; this skill assumes it is already understood | `doca-pcc` (covered by [`doca-pcc`](../doca-pcc/SKILL.md)) |
| **`doca-pcc-counters`** | Read-only diagnostic CLI shipped under `/opt/mellanox/doca/tools/` that inspects per-port / per-flow PCC counters | Cross-link only — the observability workflow lives there; this skill names what counters this algorithm emits and routes the inspection to the tool | (covered by [`doca-pcc-counters`](../../tools/doca-pcc-counters/SKILL.md)) |

The agent's rule: when the user asks *"how do I deploy this
algorithm onto my port"*, this skill is the right one. When
the user asks *"how do I write a different algorithm"*, that
is the host-side `doca-pcc` plus the public PCC programming
guide. When the user asks *"why are my flows still
behaving badly"*, that is the read-only inspection via
`doca-pcc-counters` first, then this skill if the
algorithm's tunables need adjusting.

**Public DPA-side API surface — what the user actually
calls from the algorithm-host DPA translation unit.** Per
the installed header `doca_pcc_dev_ztr_rttcc_algo.h`:

| Call | Purpose | Where in the PCC application's DPA-side code |
| --- | --- | --- |
| `doca_pcc_dev_ztr_rttcc_init(algo_idx)` | Initialize the algorithm structure on the DPA, assigning it a unique `algo_idx` | Called from the `doca_pcc_dev_user_init()` user-init callback per the README; the user picks the `algo_idx` |
| `doca_pcc_dev_ztr_rttcc_algo(algo_ctxt, event, attr, results)` | The per-event algorithm body the framework dispatches to whenever a CC event lands on the selected algo slot; outputs the per-event `results` (e.g. updated rate) | Dispatched from `doca_pcc_dev_user_algo()` under the chosen algo slot per the README |
| `doca_pcc_dev_set_ztr_rttcc_params(port_num, algo_slot, param_id_base, param_num, new_param_values, params)` | Apply a parameter update from the host-side set-algo-params path; writes into the per-port / per-slot parameter array after range-checking | Called from `doca_pcc_dev_user_set_algo_params()` per the README |
| `doca_pcc_dev_ztr_rttcc_get_param_num(void)` | Total number of parameters the algorithm exposes on this build (includes the common parameter block plus any per-feature blocks the variant compiled in) | Read at init time when the application reserves parameter storage |
| `doca_pcc_dev_ztr_rttcc_get_counter_num(void)` | Total number of counters the algorithm exposes on this build (includes the common counter block plus any per-feature blocks) | Read at init time when the application reserves counter storage |
| `doca_pcc_dev_ztr_rttcc_get_num_of_histograms(void)` | Number of RTT / rate histograms the algorithm supports on this build (zero if the histogram feature was not compiled in) | Read at init time when the application reserves histogram storage |

These are the ONLY symbols the public header exposes; the
agent does not invent additional ones. The four `get_*`
helpers are how the application sizes its per-algorithm
storage without hardcoding numbers that drift between
variants.

**Variants — chosen at DPA-side compile time.** The DPA-side
source ships several variants; the public symbol
`doca_pcc_dev_ztr_rttcc_algo` resolves to ONE of them based
on which the DPA-side translation unit was compiled against
(documented in the shipped DPA-side header set, e.g.
`ztr_rttcc.h`). The agent names the set so the user can
pick deliberately:

| Variant | Class shape | When to consider it |
| --- | --- | --- |
| Vanilla | RTT-based congestion signal; additive-increase / hyper-additive-increase on the recovery side; multiplicative-decrease on the congestion side; no path migration, no multipath | The no-config-required default; the right starting point unless the user has a specific reason otherwise |
| Path Migration (PM) mode | Adds path-migration logic that exposes an entropy / path-switch decision to the framework | When the deployment uses multiple paths between endpoints and the user wants the algorithm to participate in path selection |
| RX-rate mode | Uses an RX-rate signal in addition to RTT | When the workload's congestion signal is dominated by receive-side rate rather than RTT, per the user's measurement |
| RX-rate + PM mode | Combination of the two | When both apply |
| Multipath (MP) mode | Multipath-aware variant | When the deployment is multipath and the user wants multipath-specific shaping |
| Multipath + credits | Multipath with credit-based pacing | When multipath shaping needs credit-based pacing |
| Window-probeless | Probeless windowed multipath variant | Specialized; route to the public PCC programming guide via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) for the design rationale |

The variant is selected at DPACC build time; switching
variants is a rebuild, not a runtime knob. The agent
surfaces this explicitly when the user asks "can I switch
modes without rebuilding" — the answer is no for the
variant choice, yes for the documented host-set
parameters (which are runtime-settable via
`doca_pcc_dev_set_ztr_rttcc_params`).

**Integration edits — what the user changes in the shipped
DOCA PCC application.** Per the README's "Building DOCA PCC
ZTR application" section, the integration is a SMALL set of
in-place edits on the shipped sample:

| Edit point | What changes | Why |
| --- | --- | --- |
| `/opt/mellanox/doca/applications/pcc/device/rp/rtt_template/rp_rtt_template_dev_main.c` — `#include` block | Add `#include "doca_pcc_dev_ztr_rttcc_algo.h"` | The header that exposes the public DPA-side API for this algorithm |
| Same file — `doca_pcc_dev_user_algo()` body | Add a dispatch case for the chosen algo slot that calls `doca_pcc_dev_ztr_rttcc_algo(algo_ctxt, event, attr, results)` | Routes CC events on that slot to this algorithm. Per the README, algo slot 0 is normally assigned to the sample rtt-template; if the user reuses slot 0, the sample's rtt-template dispatch must be removed |
| Same file — `doca_pcc_dev_user_init()` body | Add `doca_pcc_dev_ztr_rttcc_init(algo_idx)` for the chosen algo index | Initializes the algorithm structure |
| Same file — `doca_pcc_dev_user_set_algo_params()` body | Add `doca_pcc_dev_set_ztr_rttcc_params(param_id_base, param_num, new_param_values, params)` | Routes host-set parameter updates into the algorithm's parameter array (matching the public set-params signature in `doca_pcc_dev_ztr_rttcc_algo.h`) |
| `/opt/mellanox/doca/applications/pcc/build_device_code.sh` — DPACC compilation section | Add `-ldoca_pcc_ztr_rttcc_algo_dev` to the `-device-libs` option | Links the DPA-side static library into the device-side image |
| `/opt/mellanox/doca/applications/pcc/dependencies/meson.build` | Add `app_doca_depends += ['pcc_ztr_rttcc_algo']` | Wires the dependency into the meson build of the DOCA PCC application |
| Build invocation | `cd /opt/mellanox/doca/applications` then `meson <BUILD_DIR> -D=enable_all_applications=false -Denable_pcc=true && ninja -C <BUILD_DIR>` | Builds the PCC application with the algorithm included |

The agent's discipline: walk the user through these edits
*in place on the installed application sample* and verify
each one before triggering the rebuild. Do not author the
edited file from scratch — the surrounding sample code is
the verified base; only the listed lines change.

**Parameter surface — host-settable tunables.** The shipped
algorithm exposes a documented set of parameters
(declared in the DPA-side parameter table shipped with the
algorithm and validated by `doca_pcc_dev_set_ztr_rttcc_params`
before any value lands). The agent's role is to NAME the
parameter SHAPE (a bounded array of unsigned 32-bit values
indexed by parameter id, range-checked per parameter on
set) and to route the canonical list of parameter
identifiers to the installed DPA-side header set under the
algorithm's source tree. The README and the public DOCA
PCC programming guide on `docs.nvidia.com` are the
authoritative source for the parameter semantics; the
agent quotes them from the user's installed headers, not
from agent memory.

Categories the agent should surface when the user asks "what
can I tune":

- **Bandwidth and rate envelope.** Maximum bandwidth, maximum
  decrement / increment factors that bound how aggressively
  the algorithm changes rate per event.
- **Additive-increase pacing.** AI (active increment) /
  HAI (hyper-active increment) and the period at which
  HAI fires.
- **Congestion threshold.** Per-event congestion thresholds
  in the algorithm's documented units (delay / RTT band)
  and the rate to use when congestion is first detected.
- **Mode toggles for advanced features.** Booleans
  (delay-only mode, advanced-features enable, topology-
  aware mode, tracer enable for debugging) that flip
  algorithm behavior without changing the variant.
- **Per-variant blocks.** When the DPA-side variant is RX-
  rate or multipath or path-migration or window-probeless,
  additional per-variant parameters become reachable; the
  `get_param_num` helper reports the total reachable count
  on this build.

The agent's rule: when the user wants to change a parameter,
quote the parameter identifier verbatim from the installed
header set and let `doca_pcc_dev_set_ztr_rttcc_params` do
the range-check at apply time. Inventing parameter
identifiers or quoting them from agent memory is the
canonical baseline-agent hallucination failure mode.

**Counter surface — what the algorithm emits.** The algorithm
emits a documented set of counters (per the shipped DPA-side
counter table). The categories the agent should surface
(without inventing exact counter names beyond what the
header set documents): CC event handle counters (CNP / NACK
events handled), increment counters (AI, HAI), decrement
counters (regular, hyper, TX-side), RTT statistics counters
(min, max, sum, count, not-valid-RTT), rate envelope
counters (min, max), empty-system-RTT counter, RTT-timeout
counter, and an advanced-features counter for runs with the
advanced-features toggle on. The canonical inspection path
for these counters is `doca-pcc-counters` (route to that
skill); this skill names the counter surface so the user
knows what to look for in the tool's output.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way
match rule, NGC container semantics, and the
headers-win-over-docs rule, see
[`doca-version`](../../doca-version/SKILL.md). The body
lives there; this skill does not duplicate it.

**The algorithm-specific overlay** is:

- **The algorithm is a DOCA SDK component that ships and
  versions with DOCA itself.** Per the shipped README,
  the build chain is DPACC → flexio-sources →
  doca-sdk-common → doca-sdk-pcc → doca-sdk-pcc-ztr-rttcc-algo
  → doca-sdk-argp → doca-samples → DOCA PCC application
  build. The agent surfaces `pkg-config --modversion
  doca-pcc-ztr-rttcc-algo`, `pkg-config --modversion
  doca-pcc`, `pkg-config --modversion doca-common`, and
  the installed `dpacc` version, and cross-checks them
  against the DOCA Compatibility Policy at
  <https://docs.nvidia.com/doca/sdk/doca-compatibility-policy/index.html>.
- **DOCA and DPACC must match per the DOCA Compatibility
  Policy.** Inherited from
  [`doca-pcc CAPABILITIES.md ## Version compatibility`](../doca-pcc/CAPABILITIES.md#version-compatibility)
  and
  [`doca-dpa CAPABILITIES.md ## Version compatibility`](../doca-dpa/CAPABILITIES.md#version-compatibility):
  the algorithm is DPA-side code compiled by DPACC; a
  DOCA + DPACC version skew fails the algorithm at link
  time or at runtime in confusing ways.
- **The README is the authoritative install sequence on
  this version.** The DPACC → flexio-sources → SDK-common
  → SDK-pcc → SDK-pcc-ztr-rttcc-algo → SDK-argp → samples
  ordering documented in the shipped README is the
  load-bearing build dependency chain. The agent must
  surface that subset of the chain when the user is
  building from source rather than from the installed
  binary packages; otherwise the install verb routes
  through [`doca-setup`](../../doca-setup/SKILL.md).
- **BlueField generation is part of the version axis.** Per
  the README ("over BlueField-3's DPA processor"), the
  shipped library targets BlueField-3-class hardware. The
  agent surfaces that an older BlueField that does not
  expose the DPA cannot host this algorithm regardless of
  how recent the DOCA install is — the same cap-query
  rule from
  [`doca-pcc CAPABILITIES.md ## Capabilities and modes`](../doca-pcc/CAPABILITIES.md#capabilities-and-modes)
  applies.
- **Public-doc routing.** Search
  <https://docs.nvidia.com/doca/sdk/> for the *DOCA PCC*
  programming guide and the *DOCA PCC Application Guide*
  (the latter referenced explicitly by the shipped README
  at
  <https://docs.nvidia.com/doca/sdk/DOCA+PCC+Application+Guide.md>).
  Route through
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  rather than quoting URLs from memory.

## Error taxonomy

This algorithm sits on top of `doca-pcc` and inherits its
error taxonomy on the host side. The algorithm-specific
overlay is the DPA-side return-code surface defined by the
public header `doca_pcc_dev_ztr_rttcc_algo.h`:

| Status | Where it shows up | Meaning |
| --- | --- | --- |
| `DOCA_PCC_DEV_STATUS_OK` | `doca_pcc_dev_ztr_rttcc_init`, `doca_pcc_dev_ztr_rttcc_algo`, `doca_pcc_dev_set_ztr_rttcc_params` | The algorithm call returned successfully. The framework treats this as the "continue" signal |
| `DOCA_PCC_DEV_STATUS_FAIL` | Same calls | The algorithm call hit an error condition. For `_init`, most often a duplicate or invalid `algo_idx`. For `_algo`, an internal precondition was not met (e.g. event from an algo slot the algorithm was not assigned to). For `_set_..._params`, one or more parameter values are out of the per-parameter documented range — per the public header, when the set-params call fails NO parameters are changed |

For host-side errors that surface BEFORE the DPA-side
algorithm runs (e.g. `DOCA_ERROR_NOT_PERMITTED` from
`doca_pcc` because the firmware custom-PCC slot is
disabled; `DOCA_ERROR_DRIVER` from DOCA + DPACC skew;
`DOCA_ERROR_INVALID_VALUE` from an attempt to load an
algorithm image built for a different BlueField), the
authoritative taxonomy is in
[`doca-pcc CAPABILITIES.md ## Error taxonomy`](../doca-pcc/CAPABILITIES.md#error-taxonomy)
and the agent walks that ladder first.

For runtime observations that the algorithm "loaded but is
not affecting flows", the failure mode is most often NOT a
status code at all — it is the absence of counter movement
on `doca-pcc-counters`. That is an observability problem,
not an error-taxonomy problem; route to
[`## Observability`](#observability) and to the counter
tool.

The agent's rule: a `DOCA_PCC_DEV_STATUS_FAIL` from
`_set_ztr_rttcc_params` is *always* the user's parameter
shape or value being outside the documented range — fix the
parameters, do not retry the same values. A
`DOCA_PCC_DEV_STATUS_FAIL` from `_algo` is a sign that the
algorithm was dispatched against an event it cannot handle
(typically because the algo slot was assigned to a
different algorithm) — fix the dispatch in
`doca_pcc_dev_user_algo()`, not the algorithm.

## Observability

ZTR RTTCC observability surface is **multi-tier**: there
is a host-side surface (the `doca-pcc` framework's host
observability per
[`doca-pcc CAPABILITIES.md ## Observability`](../doca-pcc/CAPABILITIES.md#observability)
plus the DOCA logger), a per-port-counter surface (the
canonical `doca-pcc-counters` CLI), an algorithm-internal
counter surface (the counter table the algorithm emits as
documented in the installed DPA-side header), and a
histogram surface (when the variant has histograms
compiled in, `doca_pcc_dev_ztr_rttcc_get_num_of_histograms()`
returns non-zero and per-bin counters land in the same
counter inspection path).

Three primary signals the agent should reach for, in order:

1. **The `doca-pcc-counters` CLI output.** This is the
   *operator-side* canonical view of whether the running
   algorithm is shaping flows. Route to
   [`doca-pcc-counters`](../../tools/doca-pcc-counters/SKILL.md);
   that skill walks the list → snapshot → watch → diff
   loop. The agent's job here is to surface that this is
   the *first* check whenever the user says "the
   algorithm loaded but the flows look unchanged" — not
   to redefine the tool's surface.
2. **The algorithm's own counters** (CNP / NACK handle
   counters, AI / HAI / decrement counters, RTT
   statistics, rate envelope counters, RTT-timeout
   counter, advanced-features counter — categories per
   the algorithm's shipped header set). A monotonically
   growing handle counter combined with a stalled
   AI / decrement counter is the canonical signal that
   the algorithm is *receiving* CC events but is *not
   acting* on them; the agent surfaces that this points
   at parameter tuning or variant selection rather than
   at framework wiring.
3. **The host-side `doca-pcc` framework's reporting and
   the DOCA logger.** Inherited from
   [`doca-pcc CAPABILITIES.md ## Observability`](../doca-pcc/CAPABILITIES.md#observability);
   `DOCA_LOG_LEVEL=trace` on the application binary
   surfaces lifecycle transitions and the host-side
   visibility of the algorithm load.

For DPA-side hangs (the algorithm body itself is stuck on
the DPA processor and never returns to the framework
dispatch), the DPA-side developer tools inherited from
[`doca-dpa CAPABILITIES.md ## Observability`](../doca-dpa/CAPABILITIES.md#observability)
apply. Stuck-DPA cases are not redefined here; the agent
names them and routes.

For cross-cutting observability primitives
(`--sdk-log-level`, the `DOCA_LOG_LEVEL` env var, the trace
build flavor) see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

This algorithm modulates RoCE-v2 flows on a production
BlueField port. Misconfiguration does not just produce a
bad benchmark — it can throttle production traffic the
operator did not intend to throttle, or fail to apply
congestion control on a fleet that previously had it.
The five most common ZTR RTTCC first-app failures the
agent's discipline prevents are (1) deploying the
algorithm onto a production port without the
[`doca-pcc CAPABILITIES.md ## Safety policy`](../doca-pcc/CAPABILITIES.md#safety-policy)
preconditions verified first (firmware slot, DPA-capable
BlueField, matched DOCA + DPACC); (2) tuning parameters
without running through
`doca_pcc_dev_set_ztr_rttcc_params` (which enforces the
documented ranges) — direct memory writes bypass the
range-check; (3) deploying on a port without RoCE-v2
traffic flowing, then debugging the "no effect" as an
algorithm bug instead of a traffic-absence problem; (4)
partial-rebuilding only one side of the DPACC / host
build (inherited two-side-program rule from
[`doca-dpa CAPABILITIES.md ## Safety policy`](../doca-dpa/CAPABILITIES.md#safety-policy)
and
[`doca-pcc CAPABILITIES.md ## Safety policy`](../doca-pcc/CAPABILITIES.md#safety-policy));
and (5) skipping the replica-first validation when
deploying onto a fleet that carries production RoCE-v2
traffic.

The **algorithm-specific safety overlay** the agent must
walk:

| Precondition | What must be true | How the agent verifies | Where to fix |
| --- | --- | --- | --- |
| The host-side `doca-pcc` preconditions are satisfied | DPA-capable BlueField, firmware custom-PCC slot enabled, matched DOCA + DPACC versions, `doca_dev` access available | Walk [`doca-pcc TASKS.md ## configure`](../doca-pcc/TASKS.md#configure) step 1 (the triple-axis precondition check); do NOT proceed without all three axes green | [`doca-pcc`](../doca-pcc/SKILL.md) and [`doca-setup`](../../doca-setup/SKILL.md) — NOT a fix in this skill |
| The attached port carries RoCE-v2 traffic | The BlueField port the `doca_pcc` is attached to has live RoCE-v2 flows | Confirm with the user; if no, route to [`doca-rdma`](../doca-rdma/SKILL.md) for bringing up RoCE-v2 traffic; the algorithm has nothing to shape on an idle port | [`doca-rdma`](../doca-rdma/SKILL.md) — NOT a fix in this skill |
| Parameters set via `doca_pcc_dev_set_ztr_rttcc_params`, never by direct writes | Every parameter update flows through the framework's set-params path so the algorithm's range-check runs | Walk the parameter-set workflow in [TASKS.md ## use](TASKS.md#use); the public header documents the call signature | Program-side fix; do NOT bypass the framework call |
| Variant choice is deliberate and rebuilt cleanly | The DPA-side translation unit was compiled against the intended variant (vanilla / PM / RX-rate / MP / etc.); switching variants is a rebuild | Walk the variant selection in [`## Capabilities and modes`](#capabilities-and-modes); the README documents the relevant linkage line in `build_device_code.sh` | DPA-side rebuild via `dpacc`, then host rebuild that embeds it |
| Replica-first deployment | The algorithm has been deployed and observed under representative RoCE-v2 load on a non-prod replica with matching hardware class BEFORE production exposure | Inherited from [`doca-hardware-safety CAPABILITIES.md ## Safety policy`](../../doca-hardware-safety/CAPABILITIES.md#safety-policy); the canonical risk is *"the algorithm shapes my benchmark fine but does the wrong thing under production traffic mix"* | Schedule the replica run; do NOT skip the replica because *"it's the default algorithm, what could go wrong"* — the default algorithm still has tunables that interact with the workload |
| Rollback path documented | The operator can revert to the previous PCC state (either a previously-running custom algorithm, the default firmware PCC, or no custom PCC) on a single command without inventing one | The rollback path is *destroy the `doca_pcc` context that loaded this algorithm and re-attach the previous one*, OR stop the host-side application that loaded the algorithm. Document it explicitly before the deployment | Same pattern as [`doca-pcc CAPABILITIES.md ## Safety policy`](../doca-pcc/CAPABILITIES.md#safety-policy) — destroy and re-attach; do not invent a "rollback" sub-command |

**Do not partial-rebuild one side.** Inherited from
[`doca-pcc CAPABILITIES.md ## Safety policy`](../doca-pcc/CAPABILITIES.md#safety-policy)
and
[`doca-dpa CAPABILITIES.md ## Safety policy`](../doca-dpa/CAPABILITIES.md#safety-policy):
rebuilding only the DPA-side image while the host-side
application stays at the prior version (or vice versa) is
the canonical way to introduce `DOCA_ERROR_DRIVER` at
load time. Per the README's build sequence, rebuild the
DOCA PCC application AFTER any change to the DPA-side
linkage in `build_device_code.sh` or to the dependency
list in `dependencies/meson.build`.

**Treat tuning as a hardware-touching change.** Adjusting
parameters that affect rate-update aggressiveness on a
production port is in the same risk class as a `mlxconfig`
write — the change takes effect on the next CC event on
that port and shapes real traffic. The maintenance-window
discipline from
[`doca-hardware-safety CAPABILITIES.md ## Safety policy`](../../doca-hardware-safety/CAPABILITIES.md#safety-policy)
applies; the agent does not silently recommend
aggressive parameter retunes "right now" during business
hours.
