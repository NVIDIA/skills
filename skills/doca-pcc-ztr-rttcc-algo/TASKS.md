# DOCA PCC ZTR RTTCC Algorithm task workflows

**Where to start:** Pick the H2 anchor that matches the
user's verb. Each verb is its own workflow; do not collapse
them. The workflows are deliberately written as numbered
checklists so the agent can read straight through them
without inventing intermediate steps. Cross-links return
the agent to [SKILL.md](SKILL.md), [CAPABILITIES.md](CAPABILITIES.md),
the host-side framework at
[`doca-pcc`](../doca-pcc/SKILL.md), the read-only counter
tool at
[`doca-pcc-counters`](../../tools/doca-pcc-counters/SKILL.md),
the DPA two-side-program rule at
[`doca-dpa`](../doca-dpa/SKILL.md), and the public DOCA
PCC programming guide via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
whenever a question is out of scope for this skill.

**Universal precondition.** Every workflow below assumes:
DOCA is installed; the BlueField is a generation that
exposes the DPA processor (per the shipped README, the
shipped algorithm targets BlueField-3 and runs on its DPA);
the BlueField firmware has the custom-PCC slot enabled
(verified through
[`doca-pcc TASKS.md ## configure`](../doca-pcc/TASKS.md#configure));
the matched-version DPACC compiler is installed; the
attached port carries (or is about to carry) RoCE-v2
traffic; and the host-side `doca-pcc` framework is
understood at the lifecycle level covered by
[`doca-pcc`](../doca-pcc/SKILL.md). If any precondition is
not met, route to that skill first; do not work around it
here.

## install

The algorithm is shipped as a DOCA SDK component;
installation is part of the standard DOCA install. The
agent walks four checks, then stops:

1. **Is DOCA installed at all?** Run the canonical detection
   chain documented in
   [`doca-version`](../../doca-version/SKILL.md). If
   nothing is present, route to
   [`doca-setup`](../../doca-setup/SKILL.md) for the
   from-scratch install (host package OR public NGC
   container at `nvcr.io/nvidia/doca/`) and STOP — that
   skill is the install authority, not this one.
2. **Is the algorithm library present on this install?**
   Run `pkg-config --modversion doca-pcc-ztr-rttcc-algo`.
   A successful version string confirms the algorithm is
   shipped on this install. `pkg-config --cflags --libs
   doca-pcc-ztr-rttcc-algo` returns the compile and link
   flags the application uses; the agent surfaces the
   output verbatim rather than reconstructing flags from
   memory. If the call fails, the algorithm component
   was not selected in the DOCA install — route to
   [`doca-setup`](../../doca-setup/SKILL.md) to add it.
3. **Is the matched-version DPACC compiler installed?**
   The algorithm is DPA-side code compiled by DPACC; per
   the DOCA Compatibility Policy at
   <https://docs.nvidia.com/doca/sdk/doca-compatibility-policy/index.html>,
   the DPACC version must match the DOCA version. Inherit
   the verification ladder from
   [`doca-pcc TASKS.md ## install`](../doca-pcc/TASKS.md#configure)
   and
   [`doca-dpa TASKS.md ## install`](../doca-dpa/TASKS.md#configure).
4. **Are the host-side `doca-pcc` and the DPA-side
   prerequisite chain present?** Per the shipped README's
   build chain (DPACC → flexio-sources → doca-sdk-common →
   doca-sdk-pcc → doca-sdk-pcc-ztr-rttcc-algo →
   doca-sdk-argp → doca-samples), confirm
   `pkg-config --modversion doca-pcc`,
   `pkg-config --modversion doca-common`, and the presence
   of the shipped DOCA PCC application sample at
   `/opt/mellanox/doca/applications/pcc/`. If anything is
   missing, route to
   [`doca-setup`](../../doca-setup/SKILL.md) — do NOT
   piecewise-install components by hand from this skill.

The agent does not invent install commands beyond what the
shipped DOCA install procedure documents. If the user is
mid-install and asking for the right install incantation,
route to [`doca-setup`](../../doca-setup/SKILL.md).

## configure

Configuration of the shipped ZTR RTTCC algorithm is *light*
by design (the algorithm is the no-config baseline). The
agent walks four checks before any code change.

1. **Decide whether this is the right algorithm.** Walk the
   triage from
   [CAPABILITIES.md ## Capabilities and modes](CAPABILITIES.md#capabilities-and-modes).
   If the user's requirements (latency target, fairness
   policy, convergence behavior, specific signal source)
   diverge from RTT-based RoCE-v2 congestion control, the
   right answer may be a custom algorithm — route to
   [`doca-pcc`](../doca-pcc/SKILL.md) and the public PCC
   programming guide via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
2. **Pick the variant.** Per
   [CAPABILITIES.md ## Capabilities and modes](CAPABILITIES.md#capabilities-and-modes)
   the DPA-side source ships several variants (vanilla /
   path-migration / RX-rate / RX-rate + PM / multipath /
   multipath + credits / window-probeless). The choice is
   made at DPACC build time, not at runtime. The default
   is the vanilla variant unless the user has a measured
   reason otherwise. Document the choice explicitly so the
   build and test workflows downstream know what to
   verify.
3. **Decide which algo slot the algorithm occupies.** Per
   the README and per the host-side
   [`doca-pcc CAPABILITIES.md ## Capabilities and modes`](../doca-pcc/CAPABILITIES.md#capabilities-and-modes),
   the user picks an `algo_idx` for the algorithm. The
   shipped sample uses algo slot 0 for the rtt-template
   demo; if the user reuses slot 0 for the shipped
   algorithm, the sample's rtt-template dispatch in
   `doca_pcc_dev_user_algo()` must be removed (per the
   README's "Modifying" step 2).
4. **Decide the initial parameter posture.** Default is
   "ship the algorithm with no host-side parameter
   overrides; let the algorithm's compiled-in defaults
   take effect". Adjust this only after the
   `doca-pcc-counters` baseline confirms the algorithm is
   running but a documented behavior gap exists; see
   [`## use`](#use).

The agent does NOT do any host-OS-level configuration of
RDMA (route to
[`doca-rdma`](../doca-rdma/SKILL.md)), or any firmware-
level PCC configuration (route to
[`doca-setup`](../../doca-setup/SKILL.md) and the public
firmware-config guide via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)).

## build

Building means producing the DOCA PCC application that
embeds the algorithm. Per the README's "Building DOCA PCC
ZTR application" section, the canonical sequence is:

1. **Confirm the in-place edits in
   [`## modify`](#modify) are applied.** A `build` against
   the unmodified shipped sample produces an application
   that does NOT dispatch to ZTR RTTCC — the in-place edits
   in `device/rp/rtt_template/rp_rtt_template_dev_main.c`,
   in `build_device_code.sh`, and in
   `dependencies/meson.build` are what change that.
2. **Compile the DPA-side image via DPACC, linking the
   algorithm static library.** Per the README, the
   linkage is `-ldoca_pcc_ztr_rttcc_algo_dev` added to
   the DPACC compilation line in
   `/opt/mellanox/doca/applications/pcc/build_device_code.sh`.
   The agent does NOT rewrite the build script; the
   minimum-diff is the linker-flag append documented in
   the README.
3. **Build the host application via meson/ninja.** Per the
   README, the canonical invocation is `cd
   /opt/mellanox/doca/applications` then `meson
   <BUILD_DIR> -D=enable_all_applications=false
   -Denable_pcc=true && ninja -C <BUILD_DIR>`. The
   `enable_pcc=true` flag turns on the PCC application
   inside the DOCA applications meson project; the
   `enable_all_applications=false` flag scopes the build
   to PCC only, which is the fastest path for an iterate-
   on-PCC loop.
4. **Rebuild BOTH SIDES on every algorithm-affecting
   change.** Inherited two-side-program rule from
   [`doca-pcc CAPABILITIES.md ## Safety policy`](../doca-pcc/CAPABILITIES.md#safety-policy)
   and
   [`doca-dpa CAPABILITIES.md ## Safety policy`](../doca-dpa/CAPABILITIES.md#safety-policy):
   changing the variant, changing the
   `build_device_code.sh` linkage, or changing the
   algorithm dispatch in
   `doca_pcc_dev_user_algo()` is a DPA-side rebuild AND a
   host-side rebuild.

If the user is consuming the algorithm from a Bazel /
custom build rather than the shipped applications meson
project, the canonical authority is `pkg-config --cflags
--libs doca-pcc-ztr-rttcc-algo` for the host side plus the
same DPACC linkage line documented in the README. Do not
infer extra flags from memory; surface the `pkg-config`
output verbatim.

## modify

Modify = the in-place edits on the shipped DOCA PCC
application sample that wire ZTR RTTCC in. The README
prescribes the EXACT edit list; the agent walks them and
verifies each one before triggering the rebuild in
[`## build`](#build).

1. **`rp_rtt_template_dev_main.c` — header include.** Add
   `#include "doca_pcc_dev_ztr_rttcc_algo.h"` near the
   existing PCC device-side includes. This is the only
   header the user includes from this algorithm.
2. **`rp_rtt_template_dev_main.c` —
   `doca_pcc_dev_user_algo()`.** Add a dispatch case for
   the chosen algo slot that calls
   `doca_pcc_dev_ztr_rttcc_algo(algo_ctxt, event, attr,
   results)`. Per the README, if the user reuses algo slot
   0 (the shipped rtt-template demo's default), the
   existing rtt-template dispatch must be removed; if the
   user picks a different algo slot, leave the existing
   dispatch in place and add the ZTR RTTCC dispatch under
   the new slot.
3. **`rp_rtt_template_dev_main.c` —
   `doca_pcc_dev_user_init()`.** Add
   `doca_pcc_dev_ztr_rttcc_init(algo_idx)` for the chosen
   `algo_idx`. This is what registers the algorithm
   structure on the DPA at init time.
4. **`rp_rtt_template_dev_main.c` —
   `doca_pcc_dev_user_set_algo_params()`.** Add the
   set-params call:
   `doca_pcc_dev_set_ztr_rttcc_params(param_id_base,
   param_num, new_param_values, params)` per the README's
   modify step 4. The framework dispatches host-set
   parameter updates here. (Note: this is the call shape as
   used from *inside* the `doca_pcc_dev_user_set_algo_params()`
   callback, where `port_num` and `algo_slot` are already
   resolved in the surrounding scope. The full public-header
   signature carries those two as the first two arguments —
   `doca_pcc_dev_set_ztr_rttcc_params(port_num, algo_slot,
   param_id_base, param_num, new_param_values, params)` — and
   is what
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   and
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   document. Use whichever form the surrounding context
   exposes — both are the same call.)
5. **`build_device_code.sh` — DPACC linker flag.** Add
   `-ldoca_pcc_ztr_rttcc_algo_dev` to the `-device-libs`
   option of the DPACC compilation command, per the
   README. This is the linkage that pulls the algorithm
   static library into the DPA-side image.
6. **`dependencies/meson.build` — dependency entry.** Add
   `app_doca_depends += ['pcc_ztr_rttcc_algo']` so the
   meson build pulls in the algorithm's pkg-config
   module on the host side.
7. **Rebuild.** Walk [`## build`](#build) end-to-end. Do
   not skip the rebuild — the modifications take effect
   only after the rebuild completes successfully.

If the user is starting from a custom application rather
than the shipped sample, the conceptual checklist is the
same (include the header; dispatch from the user-algo
callback under a chosen slot; init from user-init; set
params from set-algo-params; link the device-side static
library; depend on the host-side pkg-config module) but
the exact file paths differ — the agent surfaces the
checklist conceptually and leaves the file selection to
the user.

## run

Running means launching the modified DOCA PCC application
so the host-side `doca-pcc` framework loads the embedded
DPA-side algorithm onto the BlueField DPA, attaches the
`doca_pcc` context to the chosen port, and starts the
algorithm. Most run-side mechanics are owned by
[`doca-pcc TASKS.md ## run`](../doca-pcc/TASKS.md#run); the
algorithm-specific overlay is:

1. **Confirm the build is the rebuilt one.** The
   canonical first-launch failure is launching a *prior*
   build of the application that does not embed the
   algorithm. Confirm timestamps on the host binary and
   the DPA-side image before launch.
2. **Confirm the host CLI flags select the device and
   the algo slot correctly.** Per the shipped DOCA PCC
   application help (`<application> --help`), the
   application accepts a device selector and PCC-specific
   flags. Surface those verbatim from `--help` rather
   than from memory — flag names drift between DOCA
   versions and the agent does not invent them.
3. **Confirm the algorithm loads cleanly.** The host-side
   framework reports the algorithm's load via the DOCA
   logger; with `DOCA_LOG_LEVEL=trace` on the application
   binary, the agent surfaces the load event (and the
   first CC event handed to the algorithm) to confirm
   the dispatch is wired.
4. **Confirm there is RoCE-v2 traffic on the port the
   `doca_pcc` is attached to.** Per
   [CAPABILITIES.md ## Safety policy](CAPABILITIES.md#safety-policy),
   the algorithm modulates *existing* traffic — without
   traffic, the algorithm is loaded but idle. Confirm
   with the user; if no traffic, route to
   [`doca-rdma`](../doca-rdma/SKILL.md) to bring up
   RoCE-v2 flows or to a fleet operator who can drive
   load.

The actual run command (`<sample-binary>
--device-pci-addr <pci> ...`) is what the shipped DOCA PCC
application help documents; the agent quotes the help
output rather than reconstructing it.

## test

Testing the algorithm is fundamentally an *observability
exercise*: the algorithm has a small surface (one public
`*_algo` entry point) and the question is whether it is
shaping flows on the wire. The agent walks two ladder
tests, in order.

1. **Loaded-and-running smoke.** With the application
   running per [`## run`](#run), confirm — via the host
   DOCA logger and via `doca-pcc-counters` — that the
   algorithm is receiving CC events and the per-event
   handle counter is incrementing. List counters first
   (per
   [`doca-pcc-counters TASKS.md ## use`](../../tools/doca-pcc-counters/TASKS.md#run)),
   snapshot once, run a brief load (RoCE-v2 traffic mix
   representative of the deployment), snapshot again,
   diff. If the per-event handle counter is incrementing
   AND the AI / decrement counters are incrementing, the
   algorithm is loaded AND running.
2. **Shaping-under-load smoke.** With sustained
   representative RoCE-v2 load on the port, observe the
   RTT statistics counters and the rate envelope
   counters (categories per
   [CAPABILITIES.md ## Observability](CAPABILITIES.md#observability)):
   under congestion, decrement counters should rise;
   under uncongested operation, the rate-envelope max
   counter should rise toward the configured bandwidth
   envelope. Diff snapshots before vs after the load
   period.

A test that passes (1) but stalls on (2) — handle counters
incrementing, but rate / RTT counters flat — is the
canonical signal that the algorithm has been wired into a
slot that does not see this port's CC events, OR that the
parameter posture is too conservative for the load; route
to [`## debug`](#debug) and to
[`## use`](#use) respectively.

For correctness beyond smoke (does the algorithm achieve
the user's specific fairness / latency target on the user's
specific workload), that is workload-level evaluation —
out of scope for this skill, and the answer when it fails
is usually "swap in a custom algorithm written against
[`doca-pcc`](../doca-pcc/SKILL.md)".

## debug

Debug is layered: most failures are NOT in the algorithm —
they are in the host-side framework precondition chain or
in the build. Walk the ladder; do not jump.

1. **`DOCA_ERROR_NOT_PERMITTED` / `DOCA_ERROR_DRIVER` /
   `DOCA_ERROR_INVALID_VALUE` from `doca_pcc` host-side
   calls at startup.** Algorithm wiring has not even
   begun yet — this is a host-side precondition failure.
   Route to
   [`doca-pcc TASKS.md ## debug`](../doca-pcc/TASKS.md#debug)
   first: firmware custom-PCC slot, DOCA + DPACC match,
   BlueField generation, `doca_dev` access. Do NOT
   modify the algorithm to chase a host-side error.
2. **DPACC link error or load-time
   `DOCA_ERROR_DRIVER`/`DOCA_ERROR_INVALID_VALUE` from the
   algorithm image.** Per
   [CAPABILITIES.md ## Version compatibility](CAPABILITIES.md#version-compatibility)
   and the README's build sequence, the two most
   common causes are (a) the DPA-side library was
   compiled at a different DOCA / DPACC version from
   the host application and (b) the
   `build_device_code.sh` `-device-libs` flag was not
   updated. Verify both `pkg-config --modversion
   doca-pcc-ztr-rttcc-algo` and the rebuilt
   `build_device_code.sh` content; rebuild both sides
   per [`## build`](#build).
3. **`DOCA_PCC_DEV_STATUS_FAIL` returned from
   `doca_pcc_dev_ztr_rttcc_init` at first launch.** Per
   the public header, the canonical cause is a duplicate
   or invalid `algo_idx`. Confirm the chosen `algo_idx`
   does not collide with another algorithm registered
   on the same `doca_pcc_app`; if it does, pick a
   different slot per [`## configure`](#configure).
4. **`DOCA_PCC_DEV_STATUS_FAIL` from the user-algo
   dispatch at runtime.** Per
   [CAPABILITIES.md ## Error taxonomy](CAPABILITIES.md#error-taxonomy),
   this typically means the algorithm was dispatched
   against a CC event from an algo slot it was NOT
   assigned to. Fix the dispatch in
   `doca_pcc_dev_user_algo()` so the ZTR RTTCC call
   runs only under the algo slot the algorithm was
   `_init`'d on.
5. **`DOCA_PCC_DEV_STATUS_FAIL` from
   `doca_pcc_dev_set_ztr_rttcc_params`.** Per the public
   header, when the set-params call fails NO parameters
   are changed. The failure means one or more parameter
   values are out of the documented range — fix the
   values, do not retry. Walk
   [`## use`](#use) for the canonical set-params
   workflow.
6. **Algorithm loaded but flows look unchanged.** This is
   the canonical "is the algorithm modulating traffic"
   question and is FIRST an observability problem, not
   an error. Walk the
   [`doca-pcc-counters TASKS.md ## debug`](../../tools/doca-pcc-counters/TASKS.md#debug)
   "loaded but unchanged" recipe to discriminate:
   - **No CC events landing on the algorithm.** The
     algorithm is in the wrong slot, OR the
     `doca_pcc` is not attached to the right port, OR
     there is no RoCE-v2 traffic on the port. Walk
     [`## run`](#run) step 4 and
     [`## configure`](#configure) step 3.
   - **CC events landing, but AI / decrement counters
     flat.** Parameters too conservative, OR the
     variant compiled in does not match the user's
     intent. Walk [`## use`](#use) for parameter
     adjustment first; if that does not move the
     needle, consider variant rebuild per
     [`## modify`](#modify).
   - **AI / decrement counters moving, but workload
     does not see the expected shaping.** This is a
     workload-vs-algorithm-fit question, not a wiring
     question — route to the public DOCA PCC
     programming guide via
     [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
     for the algorithm's documented behavior model,
     and consider whether a custom algorithm is
     warranted (route to
     [`doca-pcc`](../doca-pcc/SKILL.md)).
7. **DPA-side hang.** The algorithm's per-event call is
   stuck. Inherit the DPA-side debugging surface from
   [`doca-dpa TASKS.md ## debug`](../doca-dpa/TASKS.md#debug);
   the algorithm is DPA code and the same DPA-side
   developer tools apply.

For cross-cutting debug primitives
(`--sdk-log-level`, `DOCA_LOG_LEVEL=trace`, the trace
build flavor, the DOCA debug ladder, the structured-tools
contract for inspection commands) see
[`doca-debug`](../../doca-debug/SKILL.md) and
[`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md).

## use

Use = ongoing operation of the algorithm: tuning parameters
without rebuilding, deciding when to swap variants, and
deciding when to abandon the reference algorithm for a
custom one. Three workflows.

1. **Tuning a parameter from the host without rebuilding
   the DPA-side image.** The host-side flow is: the
   application's set-algo-params path (driven by
   `doca_pcc` per
   [`doca-pcc CAPABILITIES.md ## Capabilities and modes`](../doca-pcc/CAPABILITIES.md#capabilities-and-modes))
   dispatches into the DPA-side `doca_pcc_dev_user_set_algo_params()`
   which calls `doca_pcc_dev_set_ztr_rttcc_params(port_num,
   algo_slot, param_id_base, param_num, new_param_values,
   params)`. The agent's discipline:
   - Quote the parameter identifier verbatim from the
     installed DPA-side header set (under the algorithm's
     source tree). Do NOT invent parameter identifiers
     or quote them from agent memory.
   - Apply the parameter via the framework path; do not
     write to the DPA-side parameter array directly.
   - The set-params call enforces the documented range;
     if `DOCA_PCC_DEV_STATUS_FAIL` returns, no
     parameters are changed.
   - Observe via [`## test`](#test): the AI / decrement
     counter slope should shift visibly under the same
     load profile, OR the rate envelope counter should
     shift. If it does not, the parameter was applied
     but does not bind the algorithm under this load —
     pick a different parameter category or escalate
     to variant change.
2. **Swapping variants.** When the user's measurement
   says the chosen variant is wrong for the workload
   (e.g. the deployment is multipath but the variant in
   the image is vanilla), the agent walks: confirm
   the variant via the
   [CAPABILITIES.md ## Capabilities and modes](CAPABILITIES.md#capabilities-and-modes)
   variants table; pick the target variant; modify the
   `build_device_code.sh` linkage per
   [`## modify`](#modify); rebuild both sides per
   [`## build`](#build); re-run the smoke per
   [`## test`](#test). The variant is a build-time
   choice; it is not a runtime knob.
3. **Replacing with a custom algorithm.** The right
   answer when the workload's latency / fairness /
   convergence requirements diverge from what the
   reference can express with parameter tuning. Route to
   [`doca-pcc`](../doca-pcc/SKILL.md) for the
   framework's user-algo / user-init / user-set-algo-
   params callbacks the custom algorithm fills in, and
   to the public DOCA PCC programming guide via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
   for algorithm-design content. This skill's job is to
   make the *decision* explicit (keep the reference,
   tune the reference, or replace it) and to surface
   the rollback path per
   [CAPABILITIES.md ## Safety policy](CAPABILITIES.md#safety-policy).

## Deferred task verbs

This skill deliberately keeps a tight surface; the verbs
below are out of scope and the agent routes them rather
than attempting an answer.

- **`develop`, `extend`, `customize` (the algorithm
  itself).** The library is a *reference* implementation;
  customizing it means writing a new algorithm against
  the host-side `doca-pcc` framework. Route to
  [`doca-pcc`](../doca-pcc/SKILL.md) plus the public
  DOCA PCC programming guide via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
- **`benchmark`, `compare-algorithms`,
  `tune-for-workload`.** Workload-level evaluation. Route
  the user to their own measurement discipline plus the
  public DOCA PCC application guide at
  <https://docs.nvidia.com/doca/sdk/DOCA+PCC+Application+Guide.md>.
- **`inspect counters` (deep / read-only).** This is the
  read-only counter CLI side; route to
  [`doca-pcc-counters`](../../tools/doca-pcc-counters/SKILL.md).
- **`provision`, `configure firmware`,
  `mlxconfig`-side configuration.** Firmware-side
  configuration of PCC slots and PCC defaults. Route to
  [`doca-setup`](../../doca-setup/SKILL.md) and the public
  firmware-config guide via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
- **`upgrade DOCA`.** DOCA install / upgrade. Route to
  [`doca-setup`](../../doca-setup/SKILL.md). The shipped
  algorithm versions with DOCA itself, so DOCA upgrades
  include algorithm upgrades.
- **`tune RoCE-v2 traffic`, `bring up RDMA`,
  `configure DCQCN at the firmware level`.** RDMA and
  RoCE-v2 lifecycle. Route to
  [`doca-rdma`](../doca-rdma/SKILL.md) plus public docs.
- **`deploy to a fleet`, `multi-host rollout`,
  `Kubernetes integration`.** Fleet-level deployment is
  not algorithm-specific; the algorithm is part of an
  application binary, and the fleet rollout is owned by
  the user's deployment tooling. The
  [CAPABILITIES.md ## Safety policy](CAPABILITIES.md#safety-policy)
  replica-first rule applies regardless of the rollout
  tooling.
- **`prove correctness of the algorithm`.** The algorithm
  is the reference NVIDIA ships; the public DOCA PCC
  programming guide and the algorithm's own DPA-side
  header set are the authoritative behavior contract.
  Route algorithmic correctness questions to those
  public sources.
