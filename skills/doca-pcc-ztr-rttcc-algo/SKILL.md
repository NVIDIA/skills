---
license: Apache-2.0
name: doca-pcc-ztr-rttcc-algo
description: >
  Use this skill when the user is doing hands-on deployment, tuning,
  or evaluation of the DOCA-shipped Zero-Touch RoCE RTT-based
  Congestion Control (ZTR RTTCC) reference algorithm on a BlueField-3
  DPA — wiring `doca_pcc_dev_ztr_rttcc_algo` into the shipped DOCA
  PCC sample, picking a variant (vanilla / PM / RX-rate / multipath /
  window-probeless) at DPACC build time, tuning host-set parameters,
  or diagnosing `DOCA_PCC_DEV_STATUS_FAIL` from the algorithm.
  Trigger even when the user does not say 'DOCA PCC' or 'ZTR RTTCC' —
  typical implicit phrasings: 'my RoCE-v2 flows aren't being
  throttled', 'PCC sample isn't dispatching to my algo', 'how do I
  pick the multipath PCC variant', 'set-params returns fail',
  'algorithm loaded but counters are flat', or 'do I need a custom
  CC algorithm on BF3'. Refuse and route elsewhere for writing a
  custom PCC algorithm from scratch, read-only PCC counter
  inspection, the host-side `doca-pcc` lifecycle, or firmware-only
  pre-Programmable PCC — those belong to other skills.
metadata:
  kind: library
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux (Ubuntu
  22.04/24.04 or RHEL/SLES) with a BlueField-3 DPU exposing the DPA
  processor, the firmware custom-PCC slot enabled, a matched-version
  DPACC compiler, and live RoCE-v2 traffic on the attached port.
  Reads `pkg-config doca-pcc-ztr-rttcc-algo` and inspects
  /opt/mellanox/doca/{lib,include,applications/pcc}.
---

# DOCA PCC ZTR RTTCC Algorithm

**Where to start:** This skill assumes DOCA is already
installed, the user's BlueField has a DPA processor that
the host can see through DOCA (a BlueField-3-generation
device per the README), the BlueField firmware has the
custom-PCC slot enabled, the DPACC compiler is installed at
a matched version per the DOCA Compatibility Policy, and the
user is doing **hands-on deployment of the DOCA-shipped ZTR
RTTCC reference algorithm** on a BlueField port that
already carries RoCE-v2 traffic — i.e. either deploying it
as the no-config-required baseline, tuning its documented
parameters, or evaluating it against a custom algorithm the
user intends to write. Open [`TASKS.md`](TASKS.md) if the
user wants to *do* something (install / configure / build /
modify / run / test / debug / use); open
[`CAPABILITIES.md`](CAPABILITIES.md) when the question is
*what does the algorithm express, what are its variants and
parameters, what does it ship vs not ship*. If the user has
not installed DOCA yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first; if the user
has not stood up the host-side `doca-pcc` framework yet,
route to [`doca-pcc`](../doca-pcc/SKILL.md) first (this
algorithm is a *library consumed by* the PCC framework, not
a standalone program); if the user only wants to *inspect*
PCC counters at runtime without changing the running
algorithm, route to
[`doca-pcc-counters`](../../tools/doca-pcc-counters/SKILL.md);
if the user wants to *write their own algorithm from
scratch*, that is the `doca-pcc` library plus the public
PCC programming guide — this skill is for the shipped
reference algorithm specifically.

## Example questions this skill answers well

The CLASSES of ZTR RTTCC questions this skill is built to
answer, each with one worked example. The agent should
treat the *class* as the load-bearing piece — the worked
example is a single instance.

- **"Is the ZTR RTTCC reference algorithm the right
  baseline for my deployment, or should I write a custom
  algorithm?"** — worked example: *"I have a BlueField-3
  carrying production RoCE-v2 traffic from a GPU cluster;
  is the shipped algorithm a fine default or do I need
  custom logic?"*. Answered by the decision rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  ("when to use the reference vs custom") + the env
  preconditions in
  [`TASKS.md ## install`](TASKS.md#install).
- **"How do I wire the shipped algorithm into the DOCA
  PCC application that's already running on my host?"** —
  worked example: *"`/opt/mellanox/doca/applications/pcc`
  is already building from sample sources; what do I
  change so the user algo callback dispatches to
  `doca_pcc_dev_ztr_rttcc_algo` under a chosen algo
  slot?"*. Answered by the integration sequence in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the in-place edits in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **"Which variant of the algorithm am I getting — vanilla
  RTT-CC, path-migration mode, RX-rate mode, multipath,
  multipath with credits, window-probeless?"** — worked
  example: *"the shipped library exposes one public
  symbol `doca_pcc_dev_ztr_rttcc_algo` but the device-
  side source ships several variants; how do I know
  which one I get and how do I pick another?"*.
  Answered by the variants table in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
- **"How do I confirm the algorithm is actually
  modulating my RDMA / RoCE traffic, and not just
  loading?"** — worked example: *"I followed the
  integration steps; the application starts; how do I
  know the algorithm is shaping flows under load?"*.
  Answered by the observability surface in
  [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability)
  + the counter-watch loop in
  [`TASKS.md ## test`](TASKS.md#test) which routes to
  [`doca-pcc-counters`](../../tools/doca-pcc-counters/SKILL.md).
- **"Which tunables does the algorithm expose, and how
  do I change them from the host without rebuilding the
  DPA-side image?"** — worked example: *"my workload is
  more latency-sensitive than the default profile assumes
  — which parameter knob do I adjust?"*. Answered by
  the parameter surface in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the `doca_pcc_dev_set_ztr_rttcc_params` workflow in
  [`TASKS.md ## use`](TASKS.md#use).
- **"What does this `DOCA_PCC_DEV_STATUS_FAIL` or
  `DOCA_ERROR_*` from a `doca_pcc_dev_ztr_rttcc_*` call
  mean and which layer caused it?"** — worked example:
  *"my init callback returns `DOCA_PCC_DEV_STATUS_FAIL`
  on first launch"*. Answered by the algorithm overlay
  on the host-side PCC taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + the layered ladder in
  [`TASKS.md ## debug`](TASKS.md#debug) that escalates
  through
  [`doca-pcc`](../doca-pcc/SKILL.md) and
  [`doca-debug`](../../doca-debug/SKILL.md).

## Audience

This skill serves **external developers operating a
BlueField-3-class DPU who want to deploy NVIDIA's shipped
reference PCC algorithm on RoCE-v2 traffic, OR who are
evaluating it against a custom algorithm they intend to
write**. The reference algorithm is *zero-touch* by design
— the no-config-required baseline — and the canonical use
case is dropping it onto a port and confirming it shapes
flows correctly under congestion. It is *not* for NVIDIA
developers contributing to the algorithm itself, nor for
users who want general PCC programming theory (route via
the public DOCA PCC programming guide), nor for users who
only want to *inspect* PCC counters (route to
[`doca-pcc-counters`](../../tools/doca-pcc-counters/SKILL.md)).

**Language scope.** The algorithm ships as a DPA-side
library (`pkg-config` module `doca-pcc-ztr-rttcc-algo`)
plus a public header `doca_pcc_dev_ztr_rttcc_algo.h` that
DPA-side translation units include. The shipped algorithm
binary is the static library
`libdoca_pcc_ztr_rttcc_algo_dev.a` per the README; the
device-side translation unit that consumes it is C and is
compiled by DPACC. The host-side that drives the PCC
context comes from [`doca-pcc`](../doca-pcc/SKILL.md);
this library does NOT add a host-side surface beyond the
host-side helpers (also shipped as
`libdoca_pcc_ztr_rttcc_algo.{a,so}` per the README) that
the `doca-pcc` framework links. Other-language host-side
wrappers around `doca-pcc` can drive this algorithm through
the same lifecycle described in `doca-pcc`; the DPA-side
integration always stays C-via-DPACC.

## When to load this skill

Load this skill when the user is doing hands-on deployment,
tuning, or evaluation of the DOCA-shipped ZTR RTTCC
reference algorithm on a BlueField port carrying RoCE-v2
traffic, in any host language plus the DPA-side translation
unit built by `dpacc`. Concretely:

- Deciding whether the shipped algorithm is the right
  baseline for the user's RoCE-v2 workload, or whether the
  user needs a custom algorithm.
- Wiring the algorithm into the DOCA PCC sample
  application by patching the user-algo / user-init /
  user-set-algo-params callbacks per the steps the
  shipped README documents.
- Picking which of the algorithm's documented variants
  (vanilla, path-migration, RX-rate, RX-rate + PM,
  multipath, multipath + credits, window-probeless)
  matches the user's intent — and surfacing that the
  *public* surface only ships one symbol
  (`doca_pcc_dev_ztr_rttcc_algo`); the variants live in
  the DPA-side source the user compiles against.
- Tuning the host-set parameters the algorithm exposes
  (per the parameter list in
  `doca_pcc_dev_ztr_rttcc_algo.h` and the shipped
  `doca_pcc_dev_set_ztr_rttcc_params`).
- Observing whether the running algorithm is actually
  modulating RoCE-v2 flows under load (route to
  [`doca-pcc-counters`](../../tools/doca-pcc-counters/SKILL.md)
  for the read-only inspection side).
- Deciding when to *replace* the shipped algorithm with a
  custom one because latency target, fairness policy, or
  convergence behavior requirements diverge from what the
  reference provides.

Do **not** load this skill for general DOCA orientation;
for the host-side `doca-pcc` lifecycle (route to
[`doca-pcc`](../doca-pcc/SKILL.md)); for writing a custom
algorithm from scratch (route to
[`doca-pcc`](../doca-pcc/SKILL.md) and the public PCC
programming guide via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md));
for read-only PCC counter inspection (route to
[`doca-pcc-counters`](../../tools/doca-pcc-counters/SKILL.md));
or for the default firmware-shipped PCC algorithms that
predate Programmable Congestion Control entirely (no
host-side code, no DPACC compile — that is a firmware-only
path routed via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)).

## What this skill provides

This is a **thin loader**. The body keeps only the
orientation needed to pick the right next file. The
substantive algorithm-specific material lives in two
companion files:

- `CAPABILITIES.md` — what the shipped ZTR RTTCC
  algorithm expresses on this version + this BlueField
  generation + this firmware: the public DPA-side API
  surface (`doca_pcc_dev_ztr_rttcc_init`,
  `doca_pcc_dev_ztr_rttcc_algo`,
  `doca_pcc_dev_set_ztr_rttcc_params`,
  `doca_pcc_dev_ztr_rttcc_get_param_num`,
  `doca_pcc_dev_ztr_rttcc_get_counter_num`,
  `doca_pcc_dev_ztr_rttcc_get_num_of_histograms`), the
  documented variants (vanilla / path-migration / RX-rate
  / multipath / window-probeless — pick one at
  DPA-side compile time), the relationship to the
  host-side `doca-pcc` framework (this is an algorithm
  body the framework loads), the relationship to the
  `doca-pcc-counters` tool (which is the canonical
  inspection surface), the algorithm's parameter and
  counter surface (RTT-based congestion signal,
  per-feature parameter blocks), the error taxonomy in
  `DOCA_PCC_DEV_STATUS_OK` / `_FAIL`, and the safety
  policy.
- `TASKS.md` — step-by-step workflows for the in-scope
  algorithm verbs: `install`, `configure`, `build`,
  `modify`, `run`, `test`, `debug`, `use`. Plus a
  `Deferred task verbs` block that points out-of-scope
  questions at the right next skill.

The skill assumes DOCA + the DPACC compiler + the
[`doca-pcc`](../doca-pcc/SKILL.md) host-side framework are
already installed; the BlueField is a generation that
exposes the DPA processor (the algorithm runs on the DPA);
the BlueField firmware has the custom-PCC slot enabled
(inherited from
[`doca-pcc CAPABILITIES.md ## Safety policy`](../doca-pcc/CAPABILITIES.md#safety-policy));
and the BlueField port the algorithm will modulate has
RoCE-v2 traffic actually flowing on it (the algorithm
modulates *existing* RDMA / RoCE traffic — without traffic,
there is nothing for it to do).

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **The algorithm body itself, in any language.** The
  shipped algorithm is the static library
  `libdoca_pcc_ztr_rttcc_algo_dev.a` (plus the host-side
  helpers) installed by the matching DOCA host package
  per the README. The agent's job is to route the user to
  the installed library and header
  (`doca_pcc_dev_ztr_rttcc_algo.h`) and to prescribe the
  in-place edits documented in the README on the
  shipped DOCA PCC application source, not to author the
  algorithm.
- **A standalone PCC application.** The DOCA PCC
  application that hosts this algorithm is the shipped
  C sample under
  `/opt/mellanox/doca/applications/pcc/` (per the
  README). The agent's job is to prescribe the
  minimum-diff modifications the README documents and to
  walk the user through the rebuild — not to author a
  parallel application.
- **A description of every internal variant's
  behavior.** The DPA-side source ships several
  variants (vanilla, path-migration, RX-rate, multipath,
  multipath + credits, window-probeless). The agent
  names the variant set, says which one is reachable via
  the public symbol on this install, and routes
  algorithm-design questions to the public PCC
  programming guide via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
  It does not redefine each variant's mathematical
  behavior.
- **A specific congestion-control algorithm tutorial.**
  Congestion-control theory, RoCE-v2 fairness analysis,
  workload-specific tuning — out of scope. Route to the
  user's own domain expertise and to the public DOCA PCC
  guide.

## Loading order

1. Read this `SKILL.md` first to confirm the user's
   question is in scope (deployment / tuning / evaluation
   of the shipped reference algorithm, not algorithm
   design from scratch and not read-only counter
   inspection).
2. **For the algorithm capability matrix, the public
   DPA-side API surface, the variant set, the parameter
   and counter surface, the host-side `doca-pcc`
   framework relationship, the
   `doca-pcc-counters` inspection-side relationship, the
   error taxonomy, the observability surface, and the
   safety policy, see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — install, configure,
   build, modify, run, test, debug, use — see
   [TASKS.md](TASKS.md).**

Both companion files cross-link to each other,
[`doca-pcc`](../doca-pcc/SKILL.md) for the host-side PCC
lifecycle that loads this algorithm,
[`doca-pcc-counters`](../../tools/doca-pcc-counters/SKILL.md)
for the read-only counter-inspection side of validating that
the algorithm is modulating traffic,
[`doca-dpa`](../doca-dpa/SKILL.md) for the DPA-side
two-side-program model and the DPACC compiler discipline,
[`doca-version`](../../doca-version/SKILL.md) for the
canonical DOCA version-handling rules (with the DPACC
overlay inherited from
[`doca-dpa`](../doca-dpa/SKILL.md) and
[`doca-pcc`](../doca-pcc/SKILL.md)), and
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
whenever the right answer is "look it up in the public DOCA
PCC programming guide or the on-disk install layout".

## Related skills

- [`doca-pcc`](../doca-pcc/SKILL.md) — the host-side PCC
  control library. This algorithm is loaded INTO a
  `doca_pcc` context that `doca-pcc` stands up; the
  host-side lifecycle (`doca_pcc` create / configure /
  start / stop / destroy, the algorithm image
  `doca_pcc_app`, the attach-to-port semantics) is owned
  by `doca-pcc`. This skill prescribes only the DPA-side
  algorithm integration on top.
- [`doca-pcc-counters`](../../tools/doca-pcc-counters/SKILL.md) —
  the read-only diagnostic CLI for PCC counters at the
  port. The canonical *"is the algorithm actually
  modulating traffic"* check goes through the counter
  tool; this skill names what counters the algorithm
  emits (CNP / NACK / AI / HAI / decrement / RTT-band
  counters per the public header) and routes the
  inspection workflow to the tool skill.
- [`doca-dpa`](../doca-dpa/SKILL.md) — the host-side
  DPA control library. The algorithm runs on the DPA,
  compiled by DPACC; the two-side-program rule and the
  DOCA-and-DPACC version-match overlay inherited from
  here apply.
- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
  the routing table for every public DOCA documentation
  source (the DOCA PCC programming guide at
  <https://docs.nvidia.com/doca/sdk/doca-pcc/index.html>;
  the DOCA PCC application guide; the DOCA Compatibility
  Policy) and the on-disk layout of an installed DOCA
  package.
- [`doca-setup`](../../doca-setup/SKILL.md) — env
  preparation, install verification, DPACC compiler
  install / verification, BlueField firmware
  configuration (including the custom-PCC slot enable),
  and the *I have no install yet* path with the public
  NGC DOCA container. This skill assumes its
  preconditions are satisfied AND that DPACC is installed
  at a version that matches DOCA AND that the firmware-
  level custom-PCC slot is enabled.
- [`doca-version`](../../doca-version/SKILL.md) —
  canonical DOCA version-handling rules. This skill's
  `## Version compatibility` cross-links the four-way
  match rule plus the DOCA-and-DPACC overlay inherited
  from [`doca-pcc`](../doca-pcc/SKILL.md).
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md) —
  the bundle's structured-tools precedence rule (detect /
  prefer / fall back / report). The Command appendix in
  [TASKS.md](TASKS.md) honors this contract.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md) —
  general DOCA programming patterns. This skill layers
  algorithm-specific overlays on top of the universal
  build, modify-a-shipped-sample, and Core lifecycle
  patterns.
- [`doca-debug`](../../doca-debug/SKILL.md) — cross-cutting
  debug ladder. Algorithm-specific debug (the algorithm
  loaded but counters do not move; the algorithm fails to
  initialize; the algorithm modulates traffic too
  aggressively / too gently for the workload) overlays on
  top of that ladder.
- [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) —
  cross-cutting hardware-safety meta-policy. Because the
  algorithm modulates production RoCE-v2 flows on a
  BlueField port, the meta-policy's pre-flight inventory,
  replica-first, and rollback rules apply via this
  skill's `## Safety policy` overlay.

