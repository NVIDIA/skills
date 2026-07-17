# DOCA DPA capabilities, version overlay, errors, observability, safety

**Where to start:** Pick the H2 anchor that matches your question
(host-side `doca_dpa` context / loaded DPA image / DPA execution
context / kernel launch + completion / dual capability discovery
/ env preconditions / errors) and read that section end-to-end.
The tables in each section are the load-bearing content; the
prose around them is interpretation.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the *how* of executing each pattern
(the verbs `configure / build / modify / run / test / debug`),
jump to [TASKS.md](TASKS.md). For the canonical DOCA
version-handling rules that this skill layers a DPA overlay on
top of, see [`doca-version`](../../doca-version/SKILL.md).

## Pattern overview

Every DPA question this skill teaches resolves into one of SIX
patterns. The patterns are CLASSES — they apply across every
DOCA DPA release and every host + BlueField + DPACC combination.

| Pattern | When it applies (class shape) | Where the substance lives |
| --- | --- | --- |
| 1. Walk the two-side-program model first | Every DPA application has TWO translation units (host-side using `doca-dpa` + DPA-side compiled by `dpacc`); they are coupled by the function signature of the DPA kernel and by the launch-argument shape | [`## Capabilities and modes`](#capabilities-and-modes) two-side-program rule + [TASKS.md ## configure](TASKS.md#configure) step 2 |
| 2. Create the per-DPA-instance `doca_dpa` context | One `doca_dpa` Core context per BlueField DPA instance the host is driving | [`## Capabilities and modes`](#capabilities-and-modes) per-instance-context rule + [TASKS.md ## configure](TASKS.md#configure) step 3 |
| 3. Load the DPA application image and create execution contexts | `doca_dpa_app` is the loaded image of the user's DPACC-compiled DPA-side binary; `doca_dpa_thread` is a DPA execution context that runs kernels from it | [`## Capabilities and modes`](#capabilities-and-modes) app + thread tables + [TASKS.md ## configure](TASKS.md#configure) step 5 |
| 4. Launch a DPA kernel from the host and observe its completion | Use the `doca_dpa_kernel_launch_update_*` family to invoke a DPA kernel function with arguments; attach a `doca_dpa_completion` so the host knows when async DPA work is done | [`## Capabilities and modes`](#capabilities-and-modes) launch + completion section + [TASKS.md ## run](TASKS.md#run) |
| 5. Honor env preconditions: DPA-capable BlueField, DOCA matched to DPACC, DPA-side image agrees with host expectations | Mismatched DOCA + DPACC combos fail at link or launch in confusing ways; an older BlueField generation may simply not expose the DPA feature | [`## Safety policy`](#safety-policy) env-precondition matrix + [TASKS.md ## configure](TASKS.md#configure) step 1 |
| 6. Diagnose a DPA error | Map `DOCA_ERROR_NOT_SUPPORTED` / `_DRIVER` / `_AGAIN` / `_INVALID_VALUE` / `_BAD_STATE` to a root cause without leaving the DPA layer prematurely | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **The two-side-program model is non-negotiable.** Every DPA
  application is two programs: the host side (this skill,
  `doca-dpa`) and the DPA side (the user's source compiled by
  `dpacc`, possibly using the DPA device-side components — the
  comm archive `libdoca_dpa_dev_comm.a` and the verbs archive
  `libdoca_dpa_dev_verbs.a` (shipped within `doca-dpa`) — from
  inside the kernel). An agent that
  treats the application as one program — for example by
  proposing that the host code *"call the DPA kernel directly
  as a function"* — has the model wrong for every version of
  DPA. The host launches; the DPA executes; the two sides are
  coupled by the kernel function signature and the launch
  arguments.
- **Capability is a TWO-axis question.** *"Is this DPA feature
  available on this host"* requires BOTH a DOCA cap-query
  against the active `doca_devinfo` (the `doca_dpa_cap_*`
  family) AND the BlueField generation actually carrying a DPA
  processor exposed to the host. An agent that quotes only one
  axis will miss the *"the doc page says feature X exists but
  it returns `DOCA_ERROR_NOT_SUPPORTED` on my hardware"* cases.

## Capabilities and modes

The two orthogonal selection axes for any DPA design from the
host side are *which BlueField DPA instance* (`doca_dpa` per
`doca_dev` that maps to a DPA-capable BlueField) and *which
DPA application image* (the `doca_dpa_app` produced by `dpacc`
from the user's DPA-side source) the host is going to load
onto that DPA. Choose both before writing any host-side launch
code, then drill into the relevant capability-query.

**Two-side-program model — the host side and the DPA side.**

| Side | What runs there | Toolchain | What this skill covers |
| --- | --- | --- | --- |
| Host side | C / C++ (or any language that can FFI a C library) using `doca-dpa` to drive the DPA | Host system compiler + `pkg-config doca-dpa` | All of `## Capabilities and modes` / `## Error taxonomy` / `## Observability` / `## Safety policy` below |
| DPA side | The kernel function bodies that run on the DPA processor; the user's source compiled by `dpacc` into the binary embedded in the host executable as a `doca_dpa_app` | `dpacc` (DPACC compiler) plus, optionally inside the kernel, the DPA device-side archives `libdoca_dpa_dev_comm.a` and `libdoca_dpa_dev_verbs.a` (part of `doca-dpa`) | This skill names the DPA side and routes via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) to the public DOCA DPA / DPACC / DPA-Comms / DPA-Verbs guides; it does not redefine the DPA-side API surface |

The agent's rule: when the user asks *"how do I write the DPA
kernel"*, that is the DPA-side question — route to the public
DPA / DPACC guide via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
When the user asks *"how do I load my DPA kernel from the host
and launch it"*, that is this skill's scope. Two distinct
questions; two distinct surfaces.

**The per-DPA-instance `doca_dpa` context — one per DPA the
host drives.**

| Object | Lifetime | What it owns | Key calls |
| --- | --- | --- | --- |
| `doca_dpa` | Per BlueField DPA instance the host is driving; created against a `doca_dev` that maps to a DPA-capable BlueField | The DOCA-side bookkeeping for that DPA, the registration of the loaded DPA application image, the DPA execution contexts (`doca_dpa_thread`) created on top, and the completion objects attached for async observation | `doca_dpa` create / configure / start / stop / destroy (DOCA Core lifecycle); `doca_dpa_cap_*` for what this DPA actually supports |

A multi-BlueField host that wants to drive more than one DPA
needs one `doca_dpa` per BlueField — there is no *"global DPA
context"*. The agent must ask which BlueField (which
`doca_dev`) the user intends to drive before recommending any
`doca_dpa_*` call.

**The loaded DPA application image (`doca_dpa_app`) — the
output of `dpacc` made visible to the host.**

| Object | Lifetime | What it represents | Key calls |
| --- | --- | --- | --- |
| `doca_dpa_app` | Loaded into a `doca_dpa` before that `doca_dpa` is started | The DPA-side binary `dpacc` produced from the user's DPA-side source, embedded into the host executable at link time, and now made addressable on the DPA processor via the host-side DOCA Core lifecycle | Image-load helpers exposed by the host-side DPA API; the symbol surface in the host link line resolves DPA kernel function entry points named in the user's DPA-side source |

The agent's rule: the host's view of *"which DPA kernel functions
exist"* is exactly the set of entry points in the loaded
`doca_dpa_app`, which is exactly the set of function names the
user marked as DPA kernels in the DPA-side source that `dpacc`
compiled. If the user is *"trying to launch a kernel that's
not in the image"*, that is a build-side question — go back to
[`## build`](#capabilities-and-modes) and to the DPACC guide via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md),
not a host-side launch-call fix.

**The DPA execution context (`doca_dpa_thread`) — one DPA
processor thread per context.**

| Object | Lifetime | What it represents | Key calls |
| --- | --- | --- | --- |
| `doca_dpa_thread` | Created on the `doca_dpa` after the application image is loaded; lives until the `doca_dpa` is destroyed | A DPA-processor execution context that DPA kernels run on. Multiple threads on the same DPA give parallelism; one thread is fine for a single-stream control workload | DOCA Core lifecycle (`doca_ctx_start` on the parent `doca_dpa`); the thread is the addressable "where do I launch this kernel" handle for the host-side launch call |

The agent's anti-pattern alert: *"create one DPA thread per
kernel invocation"* defeats the entire reason to use the DPA
(thread-create overhead per launch). A persistent DPA thread
processing a stream of launches is the right shape; a
fresh-thread-per-launch is the wrong shape.

**Host-initiated kernel launch + completion model.**

| Surface | What it does | Why the agent must surface it explicitly |
| --- | --- | --- |
| `doca_dpa_kernel_launch_update_*` family | Host-side call that invokes a DPA kernel function (named in the loaded `doca_dpa_app`) with caller-supplied arguments on a chosen `doca_dpa_thread` | The launch is **asynchronous from the host's point of view**. The host call returns once the launch is *submitted*; the DPA kernel may still be running. Forgetting this is the most common DPA first-app bug |
| `doca_dpa_completion` | An attachable completion mechanism for observing when async DPA work has finished | Without a completion attached, the host has no portable way to know the kernel finished — it sees a `_submit`-like success and the program then races against the DPA. Attach the completion before the launch |

**Dual-axis capability discovery — the only rule.** Before
sizing any thread count, assuming a DPA feature is available,
or proposing a kernel-launch shape works on the user's
BlueField, run BOTH a DOCA cap-query AND confirm the BlueField
generation actually exposes the DPA processor. Either axis
missing the support fails the feature.

| Axis | What to call | Why the agent must ask |
| --- | --- | --- |
| DOCA side | The `doca_dpa_cap_*` family against the active `doca_devinfo` for the BlueField the host is driving | DPA-side compatibility of a specific feature with this BlueField + this DOCA install is device-conditional; do not assume the feature is on every BlueField + DOCA combo |
| BlueField generation | `pkg-config --modversion doca-dpa` agrees with `doca_caps --version`; the user's BlueField is on a generation that carries a DPA processor (older generations may not) | The DPA is hardware — a BlueField without DPA hardware will fail the cap query no matter how recent the DOCA install is. Surface that distinction so the user does not chase a software upgrade for a hardware gap |

**Configuration shape.** *Mandatory* preconditions before any
`doca_dpa_kernel_launch_update_*` call: the `doca_dpa` Core
context must be created against a `doca_dev` that maps to a
DPA-capable BlueField; the DPA application image
(`doca_dpa_app`) compiled by `dpacc` must be loaded into the
`doca_dpa`; at least one `doca_dpa_thread` must exist on top
of it; the `doca_dpa` Core context must be at the started
lifecycle stage; the host launch arguments must match the
shape that the DPA-side kernel function expects. *Optional*
configurations (thread count, completion attachment topology)
are program-side tunables that ride on top of the same
cap-query rule.

## comms

DPA Comms is the **DPA-SIDE** component — the archive
`libdoca_dpa_dev_comm.a` with device header
`doca_dpa_dev_comch_msgq.h`, shipped as part of `doca-dpa`
(there is NO separate `doca-dpa-comms` pkg-config module) — the
DPA kernel itself calls from INSIDE the DPA
processor to send / receive small messages between DPA threads or
to signal another DPA thread on the same `doca_dpa_app`. It is NOT
host-side; it does NOT replace `doca-comch` (host ↔ DPU PCIe
control-plane messaging) and does NOT replace `doca-rdma`
(host-to-remote-peer RDMA). The host-side parent surface this
section sits beneath is everything in
[`## Capabilities and modes`](#capabilities-and-modes) above; this
section adds only what DPA Comms changes on top.

**Routing rule the agent must surface first.** When the user asks
about messaging on or from the DPA, walk this disambiguation
BEFORE recommending any symbol:

| User wants to | Right library | Side |
| --- | --- | --- |
| Host program ↔ DPU process control messaging over PCIe | `doca-comch` | host |
| Host ↔ remote peer RDMA | `doca-rdma` | host |
| Inter-DPA-thread messaging or signaling on the same DPA | DPA Comms device-side component `libdoca_dpa_dev_comm.a` (this section) | **DPA** |
| RDMA from inside the DPA kernel to a remote peer | DPA Verbs device-side component `libdoca_dpa_dev_verbs.a` (see [`## verbs`](#verbs)) | **DPA** |

The agent's anti-pattern alert: adding the DPA device-side comm
archive (`libdoca_dpa_dev_comm.a`) to the host link line is a
strong signal the user has the side wrong. Its symbols are
linked into the DPA-side translation unit by `dpacc`, not into
the host executable.

**DPA-side primitive families — pick at least one per kernel.**
The exact symbol set is install-bound and lives in the DPA-side
headers DPACC compiles against (`doca_dpa_dev_comch_msgq.h`);
the agent must NOT quote symbol
names from memory and must route the user to the on-disk samples
at `/opt/mellanox/doca/samples/doca_dpa/` plus the public
*DOCA DPA Comms* guide via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
The family-level shape is stable enough to teach:

| Family | Right shape for | Wrong shape for |
| --- | --- | --- |
| DPA-side endpoint handle | Addressable unit a DPA kernel sends to / receives from; host creates it through the parent flow and passes it into the kernel via launch arguments | Treating the endpoint as host-owned — host sets it up; the actual send / receive happens DPA-side |
| Small-message send / receive | Inter-thread hand-off, control messages, latency-optimized coordination between DPA threads on the same `doca_dpa_app` | Bulk data movement (use the DPA Verbs device-side component for remote RDMA from inside the kernel) |
| Signal / event | Lightweight wake-up / wait between DPA threads in the same loaded `doca_dpa_app` | A substitute for `doca_dpa_completion` — the host's view of *"the kernel finished"* still lives in the parent's completion mechanism, not in a DPA-side signal |

**Capability discovery — no per-primitive cap family.** There is
NO `doca_dpa_comms_cap_*` host cap family. The only DPA host-side
capability queries are `doca_dpa_cap_is_supported` (is a DPA
exposed to the host on this `doca_devinfo` at all) and
`doca_dpa_cap_get_max_kernel_time_alive_supported`. Whether a
specific DPA-Comms primitive is available is a function of the
BlueField generation and the matched DOCA + DPACC install, read
from the device header `doca_dpa_dev_comch_msgq.h` and the
shipped sample — NOT from a runtime cap query. An agent that
proposes a *"DPA-side runtime cap check"* or quotes a
`doca_dpa_comms_cap_*` call has the model wrong. Because the
comm component is the archive `libdoca_dpa_dev_comm.a` shipped
within `doca-dpa` (no separate `doca-dpa-comms.pc`), the version
anchor is the single `pkg-config --modversion doca-dpa`, which
must agree with `doca_caps --version` and the installed `dpacc`
per the
[DOCA Compatibility Policy](https://docs.nvidia.com/doca/sdk/doca-compatibility-policy/index.html).
A `doca-dpa` / `dpacc` version skew is the canonical *"my
DPA-side comms call returns `DOCA_ERROR_DRIVER` on the host
completion"* root cause — surface BOTH versions in the report.

**DPA-Comms error overlay.** Every error surfaces back to the host
through the parent skill's `doca_dpa_completion`; the DPA kernel
itself does not surface errors to anyone but its host program.
Disambiguate the rows below BEFORE falling back to the parent
overlay in [`## Error taxonomy`](#error-taxonomy):

| Error (as observed on host completion) | DPA-Comms-specific cause |
| --- | --- |
| `DOCA_ERROR_AGAIN` | DPA-side comms queue is full. **The DPA kernel must yield** — return from the launch and let the host drain via `doca_pe_progress` per [`doca-programming-guide`](../../doca-programming-guide/SKILL.md), then re-submit on the next launch. A tight in-kernel retry pins the DPA processor and starves the host's drain |
| `DOCA_ERROR_NOT_SUPPORTED` | The primitive the kernel called is not on this BlueField generation OR not in this matched DOCA + DPACC install. Confirm the DPA itself is exposed via `doca_dpa_cap_is_supported`, then re-check `doca_dpa_dev_comch_msgq.h` and the shipped sample for the primitive; do NOT retry on the same device |
| `DOCA_ERROR_BAD_STATE` | DPA-side initialization-order violation INSIDE the kernel — distinct from the parent's host-side `_BAD_STATE`. Both exist; both surface on the host completion; the agent must tell them apart. Walk the kernel-side initialization order in the shipped sample at `/opt/mellanox/doca/samples/doca_dpa/` BEFORE adjusting kernel code |
| `DOCA_ERROR_INVALID_VALUE` | Bad endpoint handle (commonly an out-of-scope handle from a different `doca_dpa` instance), payload past the per-primitive size limit, or the DPA-side function signature drifted from the host's launch-argument shape. Per the parent's *do not partial-rebuild one side* rule, rebuild BOTH sides via `dpacc` + host build |

**Safety overlay — parent-skill prerequisite matrix.** This
section inherits the parent's env-precondition matrix in
[`## Safety policy`](#safety-policy) plus: the parent host-side
flow must already be green end-to-end (a trivial DPA kernel with
NO DPA-Comms calls launches and completes on this host + image
per [TASKS.md ## test](TASKS.md#test) step 1); the single
`doca-dpa` version agrees with the installed `dpacc` (there is
no separate `doca-dpa-comms.pc`); the BlueField generation +
matched install actually expose the comm primitives the kernel
will call; the
two-side-program signature on the DPA-Comms endpoint handles is
consistent across host launch call and DPA-side kernel signature.
Do NOT start writing DPA-side comms code on a broken parent flow
— fix that first. The DPA kernel does NOT own DPA-Comms teardown;
teardown follows the host-side `doca-dpa` lifecycle (destroy
threads → release the loaded app → destroy the `doca_dpa`).

For the configure / build / modify / run / test / debug shape
specific to DPA-Comms, see [TASKS.md ## comms](TASKS.md#comms).

## verbs

DPA Verbs is the **DPA-SIDE** raw-verbs RDMA surface — the
archive `libdoca_dpa_dev_verbs.a` with device header
`doca_dpa_dev_verbs.h`, shipped as part of `doca-dpa` (there is
NO separate `doca-dpa-verbs` pkg-config module) — the DPA kernel
itself calls from INSIDE the DPA
processor to post work requests (sends, RDMA reads, RDMA writes,
atomics) on QPs the host configured. It is the **latency-tuning
escape hatch** beneath the host-side parent surface in
[`## Capabilities and modes`](#capabilities-and-modes) above —
not a default. Most RDMA work belongs on host-side
[`doca-rdma`](../doca-rdma/SKILL.md); DPA Verbs exists when the
host round-trip is the measured bottleneck AND the DPA kernel
needs RDMA semantics from inside the kernel body.

**The 4-way RDMA matrix — the load-bearing first decision.**
DOCA splits RDMA across two axes: execution side (host CPU vs DPA
processor) and abstraction level (high-level tasks vs raw verbs):

| Library | Execution | Abstraction | When it's the right answer |
| --- | --- | --- | --- |
| [`doca-rdma`](../doca-rdma/SKILL.md) | Host CPU | High-level tasks | Default for the vast majority of RDMA work |
| [`doca-verbs`](../doca-verbs/SKILL.md) (the DOCA host-side raw-verbs escape hatch) | Host CPU | Raw verbs | `doca-rdma` does not expose the specific verb / opcode / WR flag the user needs, but the host is still the right execution side |
| [`## comms`](#comms) (`libdoca_dpa_dev_comm.a`) | DPA processor | Local DPA-side messaging | DPA kernels coordinating among themselves — **not RDMA at all** |
| **DPA Verbs** (`libdoca_dpa_dev_verbs.a`, this section) | DPA processor | Raw verbs | Host round-trip is the measured latency bottleneck AND the kernel needs RDMA from inside its body |

The agent's rule, before any code-level discussion: (1) confirm
DPA-resident compute is already on `doca-dpa` (the parent); (2)
confirm the host round-trip is the **measured** latency bottleneck
(profile / histogram / per-op cost — not an intuition); (3)
confirm the user needs RDMA semantics, not local DPA messaging
(that is [`## comms`](#comms)). If any check fails, the right
answer is to climb back up to host-side RDMA — recommending
DPA-side verbs for a host-bottleneck workload is a misroute.

**Host-configures-QP / DPA-uses-QP — the coupling rule.** Single
non-negotiable invariant for everything below.

| Side | What it does |
| --- | --- |
| Host | Creates and **configures** the RDMA QP(s) the DPA kernel will post against (transport, max message size, feature flags, state transitions); makes the QP handle(s) visible to the kernel via launch arguments or DPA-visible memory; runs the cap query; launches the kernel via the parent's [`## Capabilities and modes`](#capabilities-and-modes) launch surface; drains `doca_dpa_completion` |
| DPA | The kernel function body (DPACC-compiled) calls `doca_dpa_dev_verbs_*` primitives to post WRs on host-configured QP handles; optionally polls completions inline if the pattern requires the kernel to react to its own completions |

The agent's anti-pattern alert: *"the DPA kernel creates its own
QP"* has the model wrong — QPs are host-side. The DPA-side code
consumes handles the host setup made available; it does NOT own
QP create / configure / state transitions.

**DPA-side primitive surface.** WR-post (send, RDMA read, RDMA
write, atomic) on a host-configured QP handle, plus optional
completion-poll inside the kernel. Exact symbol names are
install-bound (they live in `doca_dpa_dev_verbs.h`); the agent
must NOT quote them from memory and must
route the user to `/opt/mellanox/doca/samples/doca_dpa/`
plus the public *DOCA DPA Verbs* guide via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

**Capability discovery — no per-verb cap family.** There is NO
`doca_dpa_verbs_cap_*` host cap family. The only DPA host-side
capability queries are `doca_dpa_cap_is_supported` and
`doca_dpa_cap_get_max_kernel_time_alive_supported`. Which
specific verb / opcode the DPA hardware exposes is determined by
the BlueField generation and the matched DOCA + DPACC install,
read from `doca_dpa_dev_verbs.h` and the shipped sample — the
DPA-side translation unit cannot cap-query from inside the
kernel, and there is no host-side per-verb query to substitute.
Because the verbs component is the archive
`libdoca_dpa_dev_verbs.a` shipped within `doca-dpa` (no separate
`doca-dpa-verbs.pc`), the version chain is the single
`pkg-config --modversion doca-dpa` agreeing with
`doca_caps --version` and the installed `dpacc` per the
[DOCA Compatibility Policy](https://docs.nvidia.com/doca/sdk/doca-compatibility-policy/index.html).

**DPA-Verbs error overlay.** Add to the parent overlay in
[`## Error taxonomy`](#error-taxonomy):

| Error | DPA-Verbs-specific cause |
| --- | --- |
| `DOCA_ERROR_NOT_SUPPORTED` | The BlueField generation (or this matched DOCA + DPACC install) does not expose the requested DPA-side verb / opcode. Confirm the DPA is exposed via `doca_dpa_cap_is_supported` and re-check `doca_dpa_dev_verbs.h` / the shipped sample for the verb; if the verb truly is not there, the answer is the hardware — climb back to host-side RDMA if a host-side alternative covers the case |
| `DOCA_ERROR_INVALID_VALUE` | Bad QP handle, payload past the host-configured QP's max message size, or a WR flag the QP does not support. Two-side-program signature bug; fix BOTH sides together per the parent's *do not partial-rebuild one side* rule |
| `DOCA_ERROR_IO_FAILED` | The WR submitted but the **completion reports an error**. The DPA-side post return value is NOT the answer; the CQE error field on whichever side reads the completion (host via `doca_dpa_completion` by default; in-kernel poll if the pattern requires it) IS. Direct the user to drain the relevant CQ and read the CQE error field verbatim |
| `DOCA_ERROR_DRIVER` | Most often DOCA + DPACC version skew, or the DPA-side image was built against a different install than the host runtime. Capture the versions (`doca-dpa.pc`, `dpacc`, `doca_caps --version` — there is no separate `doca-dpa-verbs.pc`) and cross-check against the Compatibility Policy |
| `DOCA_ERROR_BAD_STATE` | The host-configured QP is not in a state that accepts WRs, OR the parent `doca_dpa` was not started before the kernel that posts was launched. The DPA side cannot manufacture readiness the host has not yet established |

**Climb-back rule.** DPA-side verbs is a *targeted* latency-tuning
surface. Once the specific RDMA op that drove the drop into DPA-
side verbs is wired and the latency win is measured, the agent
should explicitly ask whether remaining RDMA work can move back
up to host-side [`doca-rdma`](../doca-rdma/SKILL.md). The
two-side-program maintenance cost (DPACC + DOCA version pinning,
cap-query density, host / DPA boundary debugging) is real; do not
pay it for RDMA the host can handle.

For the configure / build / modify / run / test / debug shape
specific to DPA-Verbs, see [TASKS.md ## verbs](TASKS.md#verbs).

## Version compatibility

For the canonical DOCA version-detection chain, the four-way
match rule, NGC container semantics, and the headers-win-over-docs
rule, see [`doca-version`](../../doca-version/SKILL.md). The body
lives there; this skill does not duplicate it.

**The DPA-specific overlay** is:

- **DOCA must match the DPACC compiler per the DOCA Compatibility Policy.** The DPACC compiler is the build-time component that turns DPA-side source into the binary embedded in the host executable as a `doca_dpa_app`; the host-side runtime that loads it is `doca-dpa`. Mismatched DOCA + DPACC versions fail at link time (missing symbols on either side) or at launch time (`DOCA_ERROR_DRIVER` from a `doca_dpa_kernel_launch_update_*` call) in ways that look like hardware bugs but are version-skew bugs. The agent must surface BOTH `pkg-config --modversion doca-dpa` AND the installed `dpacc` version, cross-check them against the DOCA Compatibility Policy at <https://docs.nvidia.com/doca/sdk/doca-compatibility-policy/index.html>, and route any disagreement to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) before any DPA-layer diagnosis. Per the cross-cutting cap-query rule in [`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability), the `doca_dpa_cap_*` query against the active `doca_devinfo` is the runtime authority for *"is this DPA feature on this hardware + this DOCA install"*, and the four-way-match check (`doca-dpa.pc` plus `doca-common.pc` plus the matching DPACC) per [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility) catches the *DOCA upgraded but DPACC didn't* partial-install pattern before it surfaces as a launch failure.

## Error taxonomy

DPA-specific overlays on the cross-library `DOCA_ERROR_*`
taxonomy. The cross-library taxonomy itself lives in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy);
the rows below are the *DPA surface* meaning that the agent
must disambiguate before falling back to the cross-library
response.

| Error | DPA context where it shows up | DPA-specific cause |
| --- | --- | --- |
| `DOCA_ERROR_NOT_SUPPORTED` | `doca_dpa` create / start; `doca_dpa_cap_*` family; first kernel launch | The BlueField in this host does not have a DPA processor exposed to the host (older generation, or the BlueField is in a mode that does not expose the DPA), or the specific DPA feature the kernel uses is not in this hardware generation. Run the matching `doca_dpa_cap_*` against the active `doca_devinfo`; surface BOTH which DOCA version is installed AND which BlueField generation the host sees. Do not paper over with a retry. |
| `DOCA_ERROR_INVALID_VALUE` | `doca_dpa_kernel_launch_update_*`; image-load helpers | The host-side launch arguments do not match the DPA-side kernel's signature (size, count, or type), or the launch argument buffer is oversized for the version-quoted limit. Re-read the DPA-side function signature in the user's DPA-side source; rebuild via `dpacc` if the signature changed and the host launch call was not updated. Do not adjust the launch call without confirming the DPA-side signature. |
| `DOCA_ERROR_BAD_STATE` | Any `doca_dpa_kernel_launch_update_*` call before the `doca_dpa` is started; teardown ordering between the `doca_dpa_thread`, the loaded `doca_dpa_app`, and the parent `doca_dpa` | Lifecycle violation. The most common case is launching a DPA kernel before `doca_ctx_start()` has been called on the `doca_dpa`, or destroying the `doca_dpa` while a `doca_dpa_thread` still references it. Walk the universal Core lifecycle in [`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes); reverse the configure order on teardown. |
| `DOCA_ERROR_AGAIN` | `doca_dpa_kernel_launch_update_*` when the DPA-side completion queue is full; `doca_dpa_completion` drain from the host | The DPA-side queue feeding the `doca_dpa_completion` is full. This is *not* a hardware error; the host must drain pending completions via the progress engine before re-submitting. Same as the cross-library *"would-block, retry after progress"* pattern. |
| `DOCA_ERROR_DRIVER` | `doca_dpa` create; `doca_dpa_kernel_launch_update_*`; image-load helpers when DOCA + DPACC versions are skewed | The DPA driver layer reported failure to DOCA. Most common cause is a DOCA + DPACC version mismatch per the DOCA Compatibility Policy; second most common is the DPA-side image was built against a different DOCA install than the host runtime. Route to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) layer 5 (driver) AND to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) for the version-skew side. |

The agent's rule: **never recommend a retry loop on
`DOCA_ERROR_*` without first identifying which of the rows
above is the cause**. `_AGAIN` is the only one that wants a
drain-then-retry; the others want investigation (env / version
/ lifecycle / two-side-program signature mismatch), not retry.

## Observability

DPA observability surface is **two-sided**: there is a host-side
observability surface (per-launch completions delivered through
`doca_dpa_completion`, the DOCA logger, cap-query snapshots)
AND a DPA-side observability surface (the DPA developer tools
documented in the public *DPA Tools* umbrella reachable via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
— the DPA debugger, the DPA process-state inspector, and the
DPA statistics tool). The agent must reach for both, not just
one — a hung DPA kernel that produces no host-side completion
is almost always visible on the DPA side via one of the DPA
Tools.

Three primary signals the agent should reach for:

1. **Host-side per-launch completions via `doca_dpa_completion`.**
   Every kernel launch the host submits produces a completion
   (success or failure) the host reads through the
   `doca_dpa_completion` attached to that launch's
   `doca_dpa_thread`. Absence of a completion for a submitted
   launch is *always* a host-side missing-progress bug OR a
   DPA-side kernel that has not yet exited (commonly because
   the kernel is in an infinite loop with no termination
   signal). Confirm by progressing the PE before reaching for
   DPA-side tooling.
2. **Capability snapshot at configure time.** The output of
   `doca_dpa_cap_*` against the active `doca_devinfo` together
   with the installed `pkg-config --modversion doca-dpa` and
   the installed `dpacc` version is the baseline of *"what the
   library + the hardware + the DPACC compiler said was
   possible"* before any kernel was launched. Save it; if a
   runtime call later returns `DOCA_ERROR_NOT_SUPPORTED` the
   diff against this baseline is the bug.
3. **DPA-side developer tools (route via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)).**
   When the host-side completion never arrives and the host
   side is healthy, the DPA-side kernel is running but stuck.
   The public *DPA Tools* umbrella names the DPA debugger
   (attach-and-step inside the DPA kernel), the DPA
   process-state inspector, and the DPA statistics tool. The
   agent must NAME the existence of these tools and route the
   user there; the per-tool surface is out of scope for this
   skill.

For cross-cutting observability primitives (`--sdk-log-level`,
the `DOCA_LOG_LEVEL` env var, the trace build flavor) see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).
For the install-tree observability (logger names, package
layout, sample tree) defer to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

DPA's safety surface is **env-precondition-driven AND
two-side-program-driven**. The two most common DPA first-app
failures are (1) the host's BlueField is not on a generation
that exposes the DPA processor to the host, and (2) the
host-side launch call and the DPA-side kernel signature
disagree because one side was rebuilt while the other was not.
The agent's job is to verify both before any
`doca_dpa_kernel_launch_update_*` call, not after the first
`DOCA_ERROR_DRIVER`.

The **env-precondition matrix** the agent must walk for any
new host-side DPA setup:

| Precondition | What must be true | How the agent verifies | Where to fix |
| --- | --- | --- | --- |
| BlueField with a DPA processor visible to the host | The host's `doca_dev` enumeration includes a BlueField whose generation carries a DPA processor and whose mode exposes that DPA to the host | `doca_dpa_cap_*` against the active `doca_devinfo`; cross-check with `doca_caps --list-devs`; confirm BlueField mode via the env-side BlueField checks | [`doca-setup`](../../doca-setup/SKILL.md) for the env-side BlueField mode; this is **not** a code fix in the host-side DPA program |
| DOCA install paired with a matching DPACC compiler | `pkg-config --modversion doca-dpa` and the installed `dpacc` are at versions the DOCA Compatibility Policy lists as compatible | `pkg-config --modversion doca-dpa`; check the installed `dpacc` version; cross-check against the [DOCA Compatibility Policy](https://docs.nvidia.com/doca/sdk/doca-compatibility-policy/index.html) | [`doca-setup`](../../doca-setup/SKILL.md) for the install-side; route to [`doca-version`](../../doca-version/SKILL.md) for the four-way-match check |
| Standard DOCA `doca_dev` access | The user / process can open the target `doca_dev` for the BlueField — same baseline DOCA access rule as every other DOCA library; on the BlueField with a DPA, the BlueField must be in the right mode for the DPA to be exposed (not all BlueField modes expose the DPA equally) | The DOCA `doca_dev` enumeration succeeds for the target device; if it does not, that is an env-side problem | [`doca-setup`](../../doca-setup/SKILL.md) for the env-side; do **not** modify the program |
| Host launch call signature matches DPA-side kernel signature | The host-side `doca_dpa_kernel_launch_update_*` call passes arguments whose count, sizes, and types match the function `dpacc` compiled into the loaded `doca_dpa_app` | Re-read the DPA-side source and compare with the host launch call; if the DPA-side source changed, rebuild the DPA-side image via `dpacc` AND rebuild the host executable that embeds it | Program-layer fix on the two sides together; do **not** patch only one side |
| Single-kernel-launch smoke succeeded before scaling | A trivial DPA kernel (no-op, or counter increment) launches and completes end-to-end on this exact host + this exact image before any larger workload is attempted | Walk the smoke step in [TASKS.md ## test](TASKS.md#test) step 1; a smoke that fails identifies *env-side* or *two-side-program* gaps cheaply | Diagnose the smoke failure first; do NOT scale a broken smoke into a high-throughput design |

**Do not partial-rebuild one side.** A host-side rebuild
against a new DOCA install with the DPA-side image still built
by an old `dpacc`, or a DPA-side rebuild without rebuilding the
host that embeds it, fails DPA in non-obvious ways: the host
launch call may succeed at submit and the DPA kernel may
diverge silently or return `DOCA_ERROR_DRIVER` at launch. The
fix is to rebuild both sides against the matched DOCA + DPACC
versions per the DOCA Compatibility Policy, not to silence the
error.

**Lifecycle ordering is DPA-aware.** The `doca_dpa_thread`
objects must be destroyed BEFORE the `doca_dpa_app` is
released; the `doca_dpa_app` release must precede destroying
the `doca_dpa`; the `doca_dpa` itself follows the universal
DOCA Core teardown order on top of the `doca_dev`. Out-of-order
teardown surfaces as `DOCA_ERROR_BAD_STATE` on subsequent
calls but also leaves the DPA processor in an undefined state
for the next process; the agent must surface this ordering
explicitly.

## Deferred topic boundaries

This skill scopes itself to the **host-side** DOCA DPA library.
Adjacent topics the agent will get asked but should route
elsewhere:

- **DPA-side kernel programming itself** (how to write the
  function body the DPA processor runs; the DPA-side memory
  model; DPA-side allocation; intrinsics) — outside this
  skill. Route to the public *DOCA DPA* programming guide and
  the *DPACC* compiler guide via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md);
  this skill assumes the user has the DPA-side kernel and is
  asking *how to load and launch it from the host*.
- **DPA device-side comm component `libdoca_dpa_dev_comm.a`**
  (communication primitives the DPA kernel itself calls, header
  `doca_dpa_dev_comch_msgq.h`) — a DPA-side archive shipped
  within `doca-dpa` (NOT a separate pkg-config module), with its
  own public guide. Route via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  to the public *DOCA DPA Comms* guide. Treating its DPA-side
  surface as host-side `doca-dpa` is the most common DPA
  library-selection error.
- **DPA device-side verbs component `libdoca_dpa_dev_verbs.a`**
  (ibverbs-like RDMA verbs the DPA kernel itself calls, header
  `doca_dpa_dev_verbs.h`) — a DPA-side archive shipped within
  `doca-dpa` (NOT a separate pkg-config module), with its own
  public guide. Route via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  to the public *DOCA DPA Verbs* guide. Do not redefine its
  surface here.
- **DPA developer tools** (the DPA debugger, the DPA
  process-state inspector, the DPA statistics tool) — named in
  [`## Observability`](#observability) for routing, but the
  per-tool surface lives in the public *DPA Tools* umbrella
  via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
- **DPACC compiler internals** (flags, target options, how the
  host + DPA split-build is wired) — out of scope. Route to
  the public *DOCA DPACC Compiler* guide via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
- **DOCA Core context and progress engine internals** — owned
  by
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
  This skill *uses* the Core lifecycle; it does not redefine
  it.
- **Cross-cutting `DOCA_ERROR_*` taxonomy** — owned by
  [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
  This skill adds the DPA overlay, not the taxonomy itself.
- **Cross-cutting debug ladder** (install / version / build /
  link / runtime / program / driver) — owned by
  [`doca-debug ## debug`](../../doca-debug/TASKS.md#debug).
  This skill's `## debug` redirects there for layer 1-4; layers
  5-7 carry the DPA-specific overlay (including the DPA-driver
  route and the two-side-program signature mismatch).
- **GPU-initiated networking from a CUDA kernel** — a separate
  *DOCA-into-a-non-CPU-target* path (CUDA on an NVIDIA GPU,
  not the DPA on a BlueField). Lives in
  [`doca-gpunetio`](../doca-gpunetio/SKILL.md). The two skills
  share the *DOCA-into-target-processor* shape (paired-library
  setup, env-precondition matrix, two-axis capability rule,
  smoke-before-scale, dedicated completion mechanism) but
  differ in the target processor, the kernel toolchain (DPACC
  vs CUDA toolkit), the allocation primitives, and the
  kernel-side library set; the agent should treat them as
  siblings, not synonyms.
