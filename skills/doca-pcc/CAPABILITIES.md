# DOCA PCC capabilities, version overlay, errors, observability, safety

**Where to start:** Pick the H2 anchor that matches your question
(host-side `doca_pcc` context / loaded algorithm image /
attach-to-port semantics / triple-axis capability discovery /
env preconditions / errors) and read that section end-to-end.
The tables in each section are the load-bearing content; the
prose around them is interpretation.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the *how* of executing each pattern
(the verbs `configure / build / modify / run / test / debug`),
jump to [TASKS.md](TASKS.md). For the canonical DOCA
version-handling rules that this skill layers a PCC overlay on
top of, see [`doca-version`](../../doca-version/SKILL.md). For
the host-side DPA lifecycle this skill conceptually inherits
(custom PCC algorithms *are* DPA-side code), see
[`doca-dpa`](../doca-dpa/SKILL.md).

## Pattern overview

Every PCC question this skill teaches resolves into one of SIX
patterns. The patterns are CLASSES — they apply across every
DOCA PCC release and every host + BlueField + DPACC + firmware
combination.

| Pattern | When it applies (class shape) | Where the substance lives |
| --- | --- | --- |
| 1. Walk the two-side-program model first | Every custom PCC deployment has TWO translation units (host-side using `doca-pcc` + DPA-side algorithm compiled by `dpacc`); they are coupled by the algorithm entry points and by the parameter shape the host sets | [`## Capabilities and modes`](#capabilities-and-modes) two-side-program rule + [TASKS.md ## configure](TASKS.md#configure) step 2 |
| 2. Create the per-PCC-instance `doca_pcc` context | One `doca_pcc` Core context per BlueField port whose RDMA / RoCE traffic the custom algorithm is meant to control | [`## Capabilities and modes`](#capabilities-and-modes) per-instance-context rule + [TASKS.md ## configure](TASKS.md#configure) step 4 |
| 3. Load the PCC algorithm image and attach it to the port | `doca_pcc_app` is the loaded image of the user's DPACC-compiled DPA-side algorithm; loading it into the `doca_pcc` and starting the context attaches it to the port the `doca_dev` represents | [`## Capabilities and modes`](#capabilities-and-modes) app + attach tables + [TASKS.md ## configure](TASKS.md#configure) step 5 |
| 4. Honor triple-axis preconditions: BlueField generation, firmware custom-PCC slot, DOCA + DPACC version | An older BlueField may not expose the DPA at all; a newer one may expose the DPA but have the firmware-level custom-PCC slot disabled; a matched BlueField + firmware combo may still fail at load if DOCA + DPACC are skewed | [`## Safety policy`](#safety-policy) env-precondition matrix + [TASKS.md ## configure](TASKS.md#configure) step 1 |
| 5. Choose `doca-pcc` only when a CUSTOM algorithm is needed | The default factory PCC in ConnectX firmware does not need this library; the `pcc_counters` CLI does not either; the library is for *custom* algorithms only | [`## Capabilities and modes`](#capabilities-and-modes) path-selection rule + [`## Deferred topic boundaries`](#deferred-topic-boundaries) |
| 6. Diagnose a PCC error | Map `DOCA_ERROR_NOT_SUPPORTED` / `_NOT_PERMITTED` / `_INVALID_VALUE` / `_BAD_STATE` / `_DRIVER` to a root cause without leaving the PCC layer prematurely | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **The two-side-program model is non-negotiable.** Every custom
  PCC deployment is two programs: the host side (this skill,
  `doca-pcc`) and the DPA side (the user's congestion control
  algorithm compiled by `dpacc`). An agent that treats the
  deployment as one program — for example by proposing that
  the host code *"compute the rate-update directly"* — has the
  model wrong for every version of PCC. The host loads,
  parameterizes, and starts; the DPA-side algorithm executes
  per packet / per event; the two sides are coupled by the
  algorithm's entry points and the parameter shape.
- **Capability is a TRIPLE-axis question, not a single-axis
  one.** *"Is custom PCC available on this host"* requires
  ALL THREE of: a DOCA cap-query against the active
  `doca_devinfo` (the `doca_pcc_cap_*` family), the BlueField
  generation actually carrying a DPA processor exposed to the
  host, AND the BlueField firmware having the custom-PCC slot
  enabled. An agent that quotes only one or two axes will
  miss the *"the doc page says custom PCC exists but
  `doca_pcc_*` returns `DOCA_ERROR_NOT_PERMITTED` on my
  hardware"* cases.

## Capabilities and modes

The two orthogonal selection axes for any custom-PCC design
from the host side are *which BlueField port whose RDMA /
RoCE traffic the algorithm controls* (`doca_pcc` per `doca_dev`
that maps to a DPA-capable BlueField with the firmware
custom-PCC slot enabled) and *which PCC algorithm image* (the
`doca_pcc_app` produced by `dpacc` from the user's DPA-side
algorithm source) the host is going to load. Choose both
before writing any host-side load code, then drill into the
relevant capability-query.

**Two-side-program model — the host side and the DPA side.**

| Side | What runs there | Toolchain | What this skill covers |
| --- | --- | --- | --- |
| Host side | C / C++ (or any language that can FFI a C library) using `doca-pcc` to load / parameterize / start / stop the custom PCC algorithm and to observe what it reports back | Host system compiler + `pkg-config doca-pcc` | All of `## Capabilities and modes` / `## Error taxonomy` / `## Observability` / `## Safety policy` below |
| DPA side | The custom congestion-control algorithm body that runs on the BlueField DPA processor and affects RDMA / RoCE flows on the attached port; the user's source compiled by `dpacc` into the binary embedded in the host executable as a `doca_pcc_app` | `dpacc` (DPACC compiler) — same compiler as [`doca-dpa`](../doca-dpa/SKILL.md), and the same shape of compile step | This skill names the DPA side and routes via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) to the public DOCA PCC programming guide and the companion DPA / DPACC guides; it does not redefine the DPA-side API surface or design any specific algorithm body |

The agent's rule: when the user asks *"what should my
algorithm compute"*, that is a domain question (research /
workload tuning), NOT an API question — route via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
to the public DOCA PCC programming guide and to the user's
own congestion-control expertise. When the user asks *"how do
I load my algorithm from the host and attach it to a port"*,
that is this skill's scope. Two distinct questions; two
distinct surfaces.

**Path selection — custom PCC vs default firmware PCC vs
counter CLI.** Before any `doca-pcc` setup, the agent must
confirm `doca-pcc` is even the right artifact:

| User intent | Right artifact | Why this skill is / isn't it |
| --- | --- | --- |
| Default firmware PCC works; just want it on | None of this skill — firmware-side knobs only; route via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) | `doca-pcc` only loads *custom* algorithms; firmware-shipped algorithms run without it |
| Want to inspect PCC counters at runtime, no algorithm change | `pcc_counters` CLI — a separate diagnostic tool; route via [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools) | The counter tool is read-only inspection; `doca-pcc` is a control / load library. They share a name prefix but are different artifacts |
| Need a CUSTOM congestion control algorithm; researching a new algorithm; tuning to a specific workload | `doca-pcc` library (this skill) — load a DPACC-compiled custom algorithm onto a BlueField port | This is the only path `doca-pcc` is designed for; if none of the rows above match, stay here |

**The per-PCC-instance `doca_pcc` context — one per BlueField
port the host is driving.**

| Object | Lifetime | What it owns | Key calls |
| --- | --- | --- | --- |
| `doca_pcc` | Per BlueField port the host is driving a custom PCC algorithm on; created against a `doca_dev` that maps to a DPA-capable BlueField port with the firmware custom-PCC slot enabled | The DOCA-side bookkeeping for that PCC instance, the registration of the loaded PCC algorithm image, and the host-side observability for what the running algorithm reports back | `doca_pcc` create / configure / start / stop / destroy (DOCA Core lifecycle); `doca_pcc_cap_*` for what this device + firmware combo actually supports |

A host driving custom PCC on more than one BlueField port
needs one `doca_pcc` per port — there is no *"global PCC
context"*. The agent must ask which BlueField port (which
`doca_dev`, mapping to which physical port handling the user's
RDMA / RoCE traffic) the user intends to control before
recommending any `doca_pcc_*` call.

**The loaded PCC algorithm image (`doca_pcc_app`) — the
output of `dpacc` made visible to the host.**

| Object | Lifetime | What it represents | Key calls |
| --- | --- | --- | --- |
| `doca_pcc_app` | Loaded into a `doca_pcc` before that `doca_pcc` is started; replaced only by destroying the parent `doca_pcc` and creating a new one | The DPA-side congestion-control algorithm binary `dpacc` produced from the user's DPA-side source, embedded into the host executable at link time, and now addressable on the DPA processor that the BlueField port belongs to via the host-side DOCA Core lifecycle | Image-load helpers exposed by the host-side PCC API; the entry-point set in the loaded image is the set of algorithm-side callbacks the user marked in their DPA-side source that `dpacc` compiled |

The agent's rule: the host's view of *"which custom PCC
algorithm is running on this port"* is exactly the set of
entry points in the loaded `doca_pcc_app`. If the user is
*"trying to attach an algorithm whose entry points aren't in
the image"*, that is a build-side question — go back to
[`## build`](#capabilities-and-modes) and to the DPACC guide
via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md),
not a host-side load-call fix.

**Attach-to-port semantics — the algorithm binds to a
BlueField port carrying RDMA / RoCE.**

| Surface | What it does | Why the agent must surface it explicitly |
| --- | --- | --- |
| `doca_dev` selection at `doca_pcc` create | The `doca_dev` the `doca_pcc` is created against is the BlueField port whose RDMA / RoCE traffic the custom algorithm will affect | A custom PCC algorithm with no RDMA / RoCE traffic on its attached port has nothing to do. If the user has not stood up RDMA / RoCE on this port, route to [`doca-rdma`](../doca-rdma/SKILL.md) FIRST; PCC only modulates existing traffic |
| Starting the `doca_pcc` context | `doca_pcc_start()` is the moment the loaded algorithm becomes live on the port; before this point the algorithm is loaded but inert. (The `doca_pcc` is NOT a `doca_ctx` — it has its own `doca_pcc_create` / `_start` / `_stop` / `_destroy` lifecycle and no `doca_pcc_as_ctx`.) | Stopping the `doca_pcc` (`doca_pcc_stop`) reverts the port to whatever the firmware's default PCC behavior is (or no PCC, depending on the firmware). The agent must surface this transition explicitly — *"I started the program but my flows look unchanged"* is almost always a *"the start hasn't been called"* or *"there's no RDMA / RoCE traffic on this port to affect"* bug |

**Triple-axis capability discovery — the only rule.** Before
deploying any custom PCC algorithm, run ALL THREE of: a DOCA
cap-query against the active `doca_devinfo`, confirm the
BlueField generation actually exposes the DPA processor (the
algorithm runs on the DPA), AND confirm the BlueField firmware
has the custom-PCC slot enabled. Any axis missing the support
fails the deployment.

| Axis | What to call | Why the agent must ask |
| --- | --- | --- |
| DOCA side | The `doca_pcc_cap_*` family against the active `doca_devinfo` for the BlueField port the host is driving | PCC-side compatibility of custom-algorithm load with this device + this DOCA install is device-conditional; do not assume the capability is on every BlueField + DOCA combo |
| BlueField generation + DPA presence | `pkg-config --modversion doca-pcc` agrees with `doca_caps --version`; the user's BlueField is on a generation that carries a DPA processor (older generations may not). This is the same hardware axis as [`doca-dpa CAPABILITIES.md ## Capabilities and modes`](../doca-dpa/CAPABILITIES.md#capabilities-and-modes) | A BlueField without DPA hardware will fail the cap query no matter how recent the DOCA install is. Surface that distinction so the user does not chase a software upgrade for a hardware gap |
| Firmware-level custom-PCC slot | The BlueField firmware's custom-PCC slot must be enabled before any host-side load call. This is a firmware configuration knob set via the env-side firmware tools, not via `doca-pcc` itself — route to [`doca-setup`](../../doca-setup/SKILL.md) | A BlueField whose firmware does not enable the custom-PCC slot will reject load calls with `DOCA_ERROR_NOT_PERMITTED` even if DOCA and DPA capability queries are happy; this is a firmware-side fix, not a code change |

**Configuration shape.** *Mandatory* preconditions before any
`doca_pcc_start()`: the `doca_pcc`
context must be created (via `doca_pcc_create`) against a `doca_dev` that maps to a
DPA-capable BlueField port whose firmware has the custom-PCC
slot enabled; the PCC algorithm image (`doca_pcc_app`)
compiled by `dpacc` must be loaded into the `doca_pcc`; the
algorithm's host-side parameters (whatever knobs the
algorithm exposes) must be set; the `doca_pcc_cap_*` queries
must agree the deployment is supported on this combo.
*Optional* configurations (algorithm parameter retunes,
observability queue sizing) are program-side tunables that
ride on top of the same cap-query rule.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way
match rule, NGC container semantics, and the headers-win-over-docs
rule, see [`doca-version`](../../doca-version/SKILL.md). The body
lives there; this skill does not duplicate it.

**The PCC-specific overlay** is:

- **DOCA must match the DPACC compiler per the DOCA Compatibility Policy.** The DPACC compiler is the build-time component that turns the user's DPA-side PCC algorithm source into the binary embedded in the host executable as a `doca_pcc_app`; the host-side runtime that loads it is `doca-pcc`. Mismatched DOCA + DPACC versions fail at link time (missing symbols on either side) or at load time (`DOCA_ERROR_DRIVER` or `DOCA_ERROR_INVALID_VALUE` from the host-side load call) in ways that look like hardware bugs but are version-skew bugs. This is the same overlay [`doca-dpa CAPABILITIES.md ## Version compatibility`](../doca-dpa/CAPABILITIES.md#version-compatibility) documents for DPA itself, because the custom PCC algorithm *is* DPA-side code. The agent must surface BOTH `pkg-config --modversion doca-pcc` AND the installed `dpacc` version, cross-check them against the DOCA Compatibility Policy at <https://docs.nvidia.com/doca/sdk/doca-compatibility-policy/index.html>, and route any disagreement to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) before any PCC-layer diagnosis. Per the cross-cutting cap-query rule in [`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability), the `doca_pcc_cap_*` query against the active `doca_devinfo` is the runtime authority for *"is this custom PCC deployment supported on this hardware + this DOCA install + this firmware"*, and the four-way-match check (`doca-pcc.pc` plus `doca-common.pc` plus the matching DPACC) per [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility) catches the *DOCA upgraded but DPACC didn't* partial-install pattern before it surfaces as a load failure.

## Error taxonomy

PCC-specific overlays on the cross-library `DOCA_ERROR_*`
taxonomy. The cross-library taxonomy itself lives in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy);
the rows below are the *PCC surface* meaning that the agent
must disambiguate before falling back to the cross-library
response.

| Error | PCC context where it shows up | PCC-specific cause |
| --- | --- | --- |
| `DOCA_ERROR_NOT_SUPPORTED` | `doca_pcc` create / start; `doca_pcc_cap_*` family; first algorithm-load call | The BlueField in this host does not support custom PCC algorithms at all — typically because the BlueField generation is too old or the BlueField does not have a DPA processor exposed to the host (which custom PCC requires, since the algorithm runs on the DPA). Run the matching `doca_pcc_cap_*` against the active `doca_devinfo`; surface BOTH which DOCA version is installed AND which BlueField generation the host sees. Do not paper over with a retry. |
| `DOCA_ERROR_NOT_PERMITTED` | `doca_pcc` create / start; algorithm-load call | Either the standard `doca_dev` access is missing (the user / process cannot open the target `doca_dev` — same baseline as every other DOCA library), OR — and this is the PCC-specific case the agent MUST surface — the BlueField firmware does not have the custom-PCC slot enabled. The two look identical at the `doca-pcc` API surface and the fix is different: `doca_dev` access is a host-OS / group-membership fix per [`doca-setup ## Safety policy`](../../doca-setup/CAPABILITIES.md#safety-policy); the firmware-level custom-PCC slot is a firmware-side enable per [`doca-setup`](../../doca-setup/SKILL.md). The agent must check the firmware-side enable BEFORE concluding "this is a permission problem". |
| `DOCA_ERROR_INVALID_VALUE` | Algorithm-load call; algorithm-parameter set calls | Either the loaded algorithm image is incompatible with the target device (the image was built for a different BlueField generation, or against a different DOCA install), or a host-side algorithm parameter is out of the range the algorithm advertises. Re-check the build provenance of the `doca_pcc_app` against this host's DOCA + DPACC versions; re-read the DPA-side algorithm's parameter shape. Do not adjust the parameter value blindly without confirming the parameter range. |
| `DOCA_ERROR_BAD_STATE` | Any `doca_pcc_*` call before the `doca_pcc` is started or after it is stopped; teardown ordering between the loaded `doca_pcc_app` and the parent `doca_pcc` | Lifecycle violation. The most common case is calling host-side parameter or observability helpers before `doca_pcc_start()`, or destroying the `doca_pcc` while the loaded `doca_pcc_app` is still being referenced. Walk the universal Core lifecycle in [`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes); reverse the configure order on teardown. |
| `DOCA_ERROR_DRIVER` | `doca_pcc` create; algorithm-load call when DOCA + DPACC versions are skewed; first start when the firmware-side custom-PCC slot is in a transitional state | The PCC driver layer reported failure to DOCA. Most common cause is a DOCA + DPACC version mismatch per the DOCA Compatibility Policy; second most common is the algorithm image was built against a different DOCA install than the host runtime; third is a firmware-side custom-PCC slot that is enabled but in a transitional state (e.g. firmware was reconfigured but the BlueField has not been reset since). Route to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) layer 5 (driver) AND to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) for the version-skew side. |

The agent's rule: **never recommend a retry loop on
`DOCA_ERROR_*` without first identifying which of the rows
above is the cause**. None of the PCC errors above want a
retry — every row points at investigation (env / firmware /
version / lifecycle / two-side-program signature mismatch),
not at retry.

## Observability

PCC observability surface is **two-sided**: there is a
host-side observability surface (what the running algorithm
reports back to the host through the host-side `doca-pcc`
API, the DOCA logger, cap-query snapshots) AND an
infrastructure-side observability surface (the
`pcc_counters` diagnostic CLI documented in the public
DOCA Tools umbrella reachable via
[`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools)
plus the DPA-side developer tools inherited from
[`doca-dpa CAPABILITIES.md ## Observability`](../doca-dpa/CAPABILITIES.md#observability)).
The agent must reach for both, not just one — a custom PCC
algorithm that loads cleanly but produces no visible change
in RDMA / RoCE traffic is almost always visible on one of the
infrastructure-side tools.

Three primary signals the agent should reach for:

1. **Host-side reports surfaced by the running algorithm.**
   Whatever observability the host-side `doca-pcc` API
   exposes (algorithm-defined report stream, per-event
   callbacks, status counters) is the first place to look
   when the user reports *"the algorithm loaded but I'm not
   sure it's doing anything"*. Absence of a report when the
   algorithm should be emitting one is *always* a host-side
   missing-progress bug OR a DPA-side algorithm body that
   has no effect path; confirm by progressing the PE and by
   re-reading the algorithm's report-emission points.
2. **Capability snapshot at configure time.** The output of
   `doca_pcc_cap_*` against the active `doca_devinfo`
   together with the installed `pkg-config --modversion
   doca-pcc` and the installed `dpacc` version is the
   baseline of *"what the library + the hardware + the
   firmware + the DPACC compiler said was possible"* before
   any algorithm was loaded. Save it; if a runtime call
   later returns `DOCA_ERROR_NOT_SUPPORTED` or
   `DOCA_ERROR_NOT_PERMITTED` the diff against this baseline
   is the bug.
3. **Infrastructure-side counter tool + DPA-side developer
   tools (route via
   [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools)).**
   When the host-side surface shows the algorithm is loaded
   and reports look fine but the user's RDMA / RoCE traffic
   shows no change in congestion behavior, the
   `pcc_counters` CLI is the diagnostic the agent should
   route to for read-only inspection of PCC counters at the
   port. For DPA-side hangs (the algorithm body is running
   but stuck), the DPA-side developer tools named in
   [`doca-dpa CAPABILITIES.md ## Observability`](../doca-dpa/CAPABILITIES.md#observability)
   apply. The agent must NAME the existence of these tools
   and route the user there; the per-tool surface is out of
   scope for this skill.

For cross-cutting observability primitives (`--sdk-log-level`,
the `DOCA_LOG_LEVEL` env var, the trace build flavor) see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).
For the install-tree observability (logger names, package
layout, sample tree) defer to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

PCC's safety surface is **env-precondition-driven AND
two-side-program-driven AND firmware-configuration-driven**.
The three most common PCC first-app failures are (1) the
host's BlueField generation does not expose a DPA processor
to the host at all; (2) the BlueField firmware does not have
the custom-PCC slot enabled; and (3) the host-side load call
and the DPA-side algorithm image disagree because one side
was rebuilt while the other was not. The agent's job is to
verify all three before any host-side load call, not after
the first `DOCA_ERROR_DRIVER`.

The **env-precondition matrix** the agent must walk for any
new host-side custom PCC setup:

| Precondition | What must be true | How the agent verifies | Where to fix |
| --- | --- | --- | --- |
| BlueField with a DPA processor visible to the host | The host's `doca_dev` enumeration includes a BlueField whose generation carries a DPA processor and whose mode exposes that DPA to the host (since the custom PCC algorithm runs on the DPA) | `doca_pcc_cap_*` against the active `doca_devinfo`; cross-check with `doca_caps --list-devs`; confirm BlueField mode via the env-side BlueField checks. Same hardware axis [`doca-dpa CAPABILITIES.md ## Safety policy`](../doca-dpa/CAPABILITIES.md#safety-policy) documents | [`doca-setup`](../../doca-setup/SKILL.md) for the env-side BlueField mode; this is **not** a code fix in the host-side PCC program |
| BlueField firmware has the custom-PCC slot enabled | The BlueField firmware must be configured to permit a custom (non-factory) PCC algorithm to be loaded; this is a firmware-side knob, not a host-side library call | Via the env-side firmware configuration tools — the agent must NAME this precondition explicitly even though the exact firmware tool sits in [`doca-setup`](../../doca-setup/SKILL.md). A successful `doca_dev` open with no PCC features in `doca_pcc_cap_*` strongly suggests the slot is disabled | [`doca-setup`](../../doca-setup/SKILL.md) for the firmware-side enable; the BlueField typically needs a reset after the slot is flipped before the new state takes effect |
| DOCA install paired with a matching DPACC compiler | `pkg-config --modversion doca-pcc` and the installed `dpacc` are at versions the DOCA Compatibility Policy lists as compatible | `pkg-config --modversion doca-pcc`; check the installed `dpacc` version; cross-check against the [DOCA Compatibility Policy](https://docs.nvidia.com/doca/sdk/doca-compatibility-policy/index.html). Same overlay [`doca-dpa CAPABILITIES.md ## Safety policy`](../doca-dpa/CAPABILITIES.md#safety-policy) documents | [`doca-setup`](../../doca-setup/SKILL.md) for the install-side; route to [`doca-version`](../../doca-version/SKILL.md) for the four-way-match check |
| Standard DOCA `doca_dev` access | The user / process can open the target `doca_dev` for the BlueField port — same baseline DOCA access rule as every other DOCA library; typically requires sudo or membership in the host's standard mlnx-style group | The DOCA `doca_dev` enumeration succeeds for the target device; if it does not, that is an env-side problem | [`doca-setup`](../../doca-setup/SKILL.md) for the env-side; do **not** modify the program |
| Host-side parameter shape matches DPA-side algorithm | The host-side parameter set calls match the parameter shape (count, sizes, types) the DPA-side algorithm in the loaded `doca_pcc_app` actually exposes | Re-read the DPA-side algorithm source and compare with the host-side parameter calls; if the DPA-side source changed, rebuild the PCC algorithm image via `dpacc` AND rebuild the host executable that embeds it | Program-layer fix on the two sides together; do **not** patch only one side |
| Single small-algorithm smoke succeeded before complex algorithms | A trivial PCC algorithm (no-op rate-update, or a pass-through that emits a single report) loads and runs end-to-end on this exact host + this exact image + RDMA / RoCE traffic actually flowing on the attached port, before any sophisticated algorithm is attempted | Walk the smoke step in [TASKS.md ## test](TASKS.md#test) step 1; a smoke that fails identifies *env-side* or *firmware-side* or *two-side-program* gaps cheaply, before any algorithm design effort is wasted | Diagnose the smoke failure first; do NOT scale a broken smoke into a complex algorithm design |

**Do not partial-rebuild one side.** A host-side rebuild
against a new DOCA install with the PCC algorithm image still
built by an old `dpacc`, or a DPA-side rebuild without
rebuilding the host that embeds it, fails PCC in non-obvious
ways: the host load call may succeed and the algorithm may
silently misbehave on the wire or return `DOCA_ERROR_DRIVER`
at load. The fix is to rebuild both sides against the matched
DOCA + DPACC versions per the DOCA Compatibility Policy, not
to silence the error.

**Lifecycle ordering is PCC-aware.** The loaded `doca_pcc_app`
must be released BEFORE the parent `doca_pcc` is destroyed;
the `doca_pcc` itself follows the universal DOCA Core
teardown order on top of the `doca_dev`. Out-of-order teardown
surfaces as `DOCA_ERROR_BAD_STATE` on subsequent calls and may
also leave the BlueField port with a half-configured custom
PCC state that the firmware has to recover from on the next
start — the agent must surface this ordering explicitly.

**This skill does not define an algorithm.** `doca-pcc`
*loads* a congestion control algorithm the user supplies; it
does not implement one. When the user asks *"what algorithm
should I write"*, the agent must refuse to invent an
algorithm body and must route the user to the public DOCA PCC
programming guide via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
and to the user's own congestion-control domain expertise.

## Deferred topic boundaries

This skill scopes itself to the **host-side** `doca-pcc`
library for **custom** PCC algorithms. Adjacent topics the
agent will get asked but should route elsewhere:

- **DPA-side congestion-control algorithm design itself**
  (the algorithm body that runs on the DPA processor, the
  per-event entry-point set, the DPA-side memory model the
  algorithm uses, congestion-control theory) — outside this
  skill. Route to the public *DOCA PCC* programming guide and
  the *DPACC* compiler guide via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md);
  this skill assumes the user has the DPA-side algorithm and
  is asking *how to load and attach it from the host*.
- **`pcc_counters` diagnostic CLI** (the real artifact is the
  `pcc_counters.sh` script under `tools/pcc_counters/`) — a separate
  artifact with its own public page; route via
  [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools).
  It is read-only inspection; this skill is control + load.
  Conflating the tool with the library is the single most
  common PCC first-app design error.
- **Default factory PCC algorithm shipped in ConnectX
  firmware** — does not need this library. Configuration of
  the factory PCC is a firmware-side concern; route via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
- **Host-side DPA control library (`doca-dpa`)** — the
  generic *run a DPA kernel from the host* library. PCC
  conceptually inherits the DPA two-side-program shape from
  it, but `doca-dpa` is the right skill when the user wants
  a generic DPA workload that is NOT congestion control on
  RDMA / RoCE traffic. See [`doca-dpa`](../doca-dpa/SKILL.md).
- **DPA developer tools** (the DPA debugger, the DPA
  process-state inspector, the DPA statistics tool) — named
  in [`## Observability`](#observability) for routing, but
  the per-tool surface lives in the public *DPA Tools*
  umbrella via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
- **DPACC compiler internals** (flags, target options, how
  the host + DPA split-build is wired) — out of scope. Route
  to the public *DOCA DPACC Compiler* guide via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
- **Setting up the RDMA / RoCE traffic the custom PCC
  algorithm controls** — owned by
  [`doca-rdma`](../doca-rdma/SKILL.md). This skill *modulates*
  RDMA / RoCE traffic on the attached port; it does not
  stand up that traffic. A `doca-pcc` deployment with no
  RDMA / RoCE traffic on the attached port is a
  pre-flight gap, not a `doca-pcc` bug.
- **DOCA Core context and progress engine internals** —
  owned by
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
  This skill *uses* the Core lifecycle; it does not redefine
  it.
- **Cross-cutting `DOCA_ERROR_*` taxonomy** — owned by
  [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
  This skill adds the PCC overlay, not the taxonomy itself.
- **Cross-cutting debug ladder** (install / version / build /
  link / runtime / program / driver) — owned by
  [`doca-debug ## debug`](../../doca-debug/TASKS.md#debug).
  This skill's `## debug` redirects there for layer 1-4;
  layers 5-7 carry the PCC-specific overlay (including the
  firmware-side custom-PCC slot route and the two-side-program
  parameter-shape mismatch).
