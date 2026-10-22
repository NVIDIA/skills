---
license: Apache-2.0
name: doca-pcc-ztr-rttcc-algo
description: >
  DOCA-shipped Zero-Touch RoCE RTT congestion control (ZTR RTTCC)
  reference algorithm on a BlueField-3 DPA: deploy, tune and evaluate
  doca_pcc_dev_ztr_rttcc_algo through the shipped DOCA PCC sample. Trigger
  on "my RoCE-v2 flows aren't being throttled", "PCC sample isn't
  dispatching to my algo", "set-params returns fail", "algorithm loaded
  but counters are flat", "how do I pick the multipath PCC variant", or
  "do I need a custom CC algorithm on BF3", even without naming ZTR RTTCC.
  Your own algorithm through the host library belongs to doca-pcc; the
  doca_spcx_cc command line to doca-spcx-cc. Covers picking a variant
  (vanilla / PM / RX-rate / multipath / window-probeless) at DPACC build
  time, tuning host-set parameters, and DOCA_PCC_DEV_STATUS_FAIL from the
  algorithm. Refuse and route elsewhere for writing a custom PCC algorithm
  from scratch, read-only PCC counter inspection, or firmware-only
  pre-Programmable PCC.
metadata:
  author: NVIDIA <doca@nvidia.com>
  kind: library
  doca_version: "3.5"
  tags:
    - doca
    - pcc
    - ztr-rttcc
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux (Ubuntu
  22.04/24.04 or RHEL/SLES) with a BlueField-3 DPU exposing the DPA
  processor, the firmware custom-PCC slot enabled, a matched-version
  DPACC compiler, and live RoCE-v2 traffic on the attached port. Reads
  `pkg-config doca-pcc-ztr-rttcc-algo` and inspects
  /opt/mellanox/doca/{lib,include,applications/pcc}. The validation
  steps require a live DOCA install.
---

# DOCA PCC ZTR RTTCC Algorithm

## Examples

- **"How do I pick the multipath PCC variant?"** — this is in
  scope. Pick the ZTR RTTCC variant at DPACC build time; do not
  invent a custom algorithm.
- **"Algorithm loaded but counters are flat."** — confirm live
  RoCE-v2 traffic and host-set params before rewriting.

## Instructions

Follow Loading order below. Open the tasks companion for configure /
tune / debug, and the capabilities companion for variants and the
FAIL status. Do not open every companion file.

**Where to start:** This skill assumes DOCA + DPACC are
installed, the BlueField is a BlueField-3-generation device
with its DPA visible to the host, the firmware custom-PCC
slot is enabled, and the user is doing **hands-on deployment
of the DOCA-shipped ZTR RTTCC reference algorithm** on a port
already carrying RoCE-v2 traffic — deploying it as the
no-config baseline, tuning its parameters, or evaluating it
against a custom algorithm. Open the tasks companion to *do*
something (install / configure / build / modify / run / test /
debug / use); open the capabilities companion for *what the
algorithm expresses, its variants and parameters, what it
ships vs not*. If DOCA is not installed, route to
`doca-setup/SKILL.md` first; if the
host-side `doca-pcc` framework is not up yet, route to
`doca-pcc/SKILL.md` first (this algorithm is a
*library consumed by* the PCC framework); for read-only
counter inspection, route to
`doca-pcc-counters/SKILL.md`;
for writing an algorithm from scratch, that is `doca-pcc`
plus the public PCC guide — this skill is for the shipped
reference algorithm specifically.

## Example questions this skill answers well

The CLASSES of ZTR RTTCC questions this skill answers, each
with one worked example. The *class* is the load-bearing
piece; the worked example is one instance.

- **"Is the ZTR RTTCC reference algorithm the right
  baseline for my deployment, or should I write a custom
  algorithm?"** — worked example: *"I have a BlueField-3
  carrying production RoCE-v2 traffic from a GPU cluster;
  is the shipped algorithm a fine default or do I need
  custom logic?"*. Answered by the decision rule in
  ## Capabilities and modes
  ("when to use the reference vs custom") + the env
  preconditions in
  ## install.
- **"How do I wire the shipped algorithm into the DOCA
  PCC application that's already running on my host?"** —
  worked example: *"`/opt/mellanox/doca/applications/pcc`
  is already building from sample sources; what do I
  change so the user algo callback dispatches to
  `doca_pcc_dev_ztr_rttcc_algo` under a chosen algo
  slot?"*. Answered by the integration sequence in
  ## Capabilities and modes
  + the in-place edits in
  ## modify.
- **"Which variant of the algorithm am I getting — vanilla
  RTT-CC, path-migration mode, RX-rate mode, multipath,
  multipath with credits, window-probeless?"** — worked
  example: *"the shipped library exposes one public
  symbol `doca_pcc_dev_ztr_rttcc_algo` but the device-
  side source ships several variants; how do I know
  which one I get and how do I pick another?"*.
  Answered by the variants table in
  ## Capabilities and modes.
- **"How do I confirm the algorithm is actually
  modulating my RDMA / RoCE traffic, and not just
  loading?"** — worked example: *"I followed the
  integration steps; the application starts; how do I
  know the algorithm is shaping flows under load?"*.
  Answered by the observability surface in
  ## Observability
  + the counter-watch loop in
  ## test which routes to
  `doca-pcc-counters/SKILL.md`.

> **"Flows aren't being throttled" / "counters are flat" is first an
> observability question, not a tuning one.** This is a debug
> question, so do not answer it by walking the launch checklist in
> ## run — that sequence is for bringing a
> new build up, not for a system already running. Go to the
> discrimination in ## debug step 6 and
> start there: are CC events reaching the algorithm at all? If
> none are, the cause is the wrong algo slot, a `doca_pcc` attached to
> the wrong port, or — the documented common trap — no RoCE-v2 traffic
> on that port to modulate, in which case the algorithm is loaded but
> idle and the run proves nothing. Only once events are landing and
> the AI / decrement counters are still flat do parameters or the
> compiled variant come into question. Jumping to parameter tuning or
> a variant rebuild first is the mistake
> ## Safety policy
> calls out by name; the discrimination recipe is
> ## debug step 6.
- **"Which tunables does the algorithm expose, and how do I
  change them from the host without rebuilding the DPA-side
  image?"** — worked example: *"my workload is more
  latency-sensitive than the default profile — which knob do
  I adjust?"*. Answered by the parameter surface in
  ## Capabilities and modes
  + the `doca_pcc_dev_set_ztr_rttcc_params` workflow in
  ## use.
- **"What does this `DOCA_PCC_DEV_STATUS_FAIL` or
  `DOCA_ERROR_*` from a `doca_pcc_dev_ztr_rttcc_*` call
  mean and which layer caused it?"** — worked example:
  *"my init callback returns `DOCA_PCC_DEV_STATUS_FAIL`
  on first launch"*. Answered by the algorithm overlay
  on the host-side PCC taxonomy in
  ## Error taxonomy
  + the layered ladder in
  ## debug that escalates
  through
  `doca-pcc/SKILL.md` and
  `doca-debug/SKILL.md`.

## Audience

This skill serves **external developers operating a
BlueField-3-class DPU who want to deploy NVIDIA's shipped
reference PCC algorithm on RoCE-v2 traffic, OR who are
evaluating it against a custom algorithm they intend to
write**. The algorithm is *zero-touch* by design — the
no-config-required baseline. It is *not* for NVIDIA
developers contributing to the algorithm itself, nor for
general PCC programming theory (route via the public DOCA
PCC guide), nor for users who only want to *inspect* PCC
counters (route to
`doca-pcc-counters/SKILL.md`).

**Language scope.** The algorithm ships as a DPA-side
library (`pkg-config` module `doca-pcc-ztr-rttcc-algo`)
plus the public header `doca_pcc_dev_ztr_rttcc_algo.h`.
The shipped binary is the static library
`libdoca_pcc_ztr_rttcc_algo_dev.a` per the README; the
device-side translation unit that consumes it is C
compiled by DPACC. The host-side that drives the PCC
context comes from `doca-pcc/SKILL.md`;
this library adds no host-side surface beyond the helpers
(`libdoca_pcc_ztr_rttcc_algo.{a,so}`) `doca-pcc` links.
Other-language host-side wrappers around `doca-pcc` drive
this algorithm through the same lifecycle; the DPA-side
integration always stays C-via-DPACC.

## When to load this skill

Load this skill when the user is doing hands-on deployment,
tuning, or evaluation of the DOCA-shipped ZTR RTTCC
reference algorithm on a BlueField port carrying RoCE-v2
traffic. Concretely:

- Deciding whether the shipped algorithm is the right
  baseline for the RoCE-v2 workload, or a custom algorithm
  is needed.
- Wiring the algorithm into the DOCA PCC sample application
  by patching the user-algo / user-init /
  user-set-algo-params callbacks the shipped README
  documents.
- Picking which documented variant (vanilla, path-migration,
  RX-rate, RX-rate + PM, multipath, multipath + credits,
  window-probeless) matches the user's intent — the *public*
  surface ships one symbol (`doca_pcc_dev_ztr_rttcc_algo`);
  the variants live in the DPA-side source the user compiles.
- Tuning the host-set parameters via
  `doca_pcc_dev_ztr_rttcc_algo.h` and
  `doca_pcc_dev_set_ztr_rttcc_params`.
- Observing whether the running algorithm modulates RoCE-v2
  flows under load (route to
  `doca-pcc-counters/SKILL.md`
  for read-only inspection).
- Deciding when to *replace* the reference algorithm with a
  custom one because latency, fairness, or convergence
  requirements diverge.

Do **not** load this skill for general DOCA orientation; for
the host-side `doca-pcc` lifecycle or for writing a custom
algorithm from scratch (route to
`doca-pcc/SKILL.md` plus the public PCC guide
via
`doca-public-knowledge-map/SKILL.md`);
for read-only PCC counter inspection (route to
`doca-pcc-counters/SKILL.md`);
or for the factory firmware PCC algorithms that predate
Programmable Congestion Control (firmware-only path, routed
via
`doca-public-knowledge-map/SKILL.md`).

## What this skill provides

This is a **thin loader**. The body keeps only the
orientation needed to pick the right next file. The
substantive algorithm-specific material lives in two
companion files:

- the capabilities companion — what the shipped ZTR RTTCC
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
- the tasks companion — step-by-step workflows for the in-scope
  algorithm verbs: `install`, `configure`, `build`,
  `modify`, `run`, `test`, `debug`, `use`. Plus a
  `Deferred task verbs` block that points out-of-scope
  questions at the right next skill.

The skill assumes DOCA + DPACC + the
`doca-pcc/SKILL.md` host-side framework are
installed; the BlueField exposes the DPA processor; the
firmware custom-PCC slot is enabled (inherited from
`doca-pcc ## Safety policy`); and the port the algorithm
modulates actually carries RoCE-v2 traffic — the algorithm
modulates *existing* traffic; without it there is nothing
to do.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **The algorithm body itself, in any language.** The
  shipped algorithm is the static library
  `libdoca_pcc_ztr_rttcc_algo_dev.a` (plus host-side
  helpers) installed by the matching DOCA host package.
  The agent routes the user to the installed library and
  header (`doca_pcc_dev_ztr_rttcc_algo.h`) and prescribes
  the README-documented in-place edits on the shipped DOCA
  PCC application source — it does not author the
  algorithm.
- **A standalone PCC application.** The DOCA PCC
  application hosting this algorithm is the shipped C
  sample under `/opt/mellanox/doca/applications/pcc/`.
  The agent prescribes the README's minimum-diff
  modifications and walks the rebuild — it does not
  author a parallel application.
- **A description of every internal variant's
  behavior.** The agent names the variant set, says which
  one is reachable via the public symbol on this install,
  and routes algorithm-design questions to the public PCC
  guide via
  `doca-public-knowledge-map/SKILL.md`;
  it does not redefine each variant's math.
- **A congestion-control tutorial.** CC theory, RoCE-v2
  fairness analysis, workload-specific tuning — out of
  scope. Route to the user's domain expertise and the
  public DOCA PCC guide.

## Loading order

1. Read this skill first to confirm the user's
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
   safety policy, see the capabilities companion.**
3. **For step-by-step workflows — install, configure,
   build, modify, run, test, debug, use — see
   the tasks companion.**

Both companion files cross-link to each other and to the skills
under **Related skills** below.

## Related skills

- `doca-pcc/SKILL.md` — the host-side PCC control
  library. This algorithm is loaded INTO a `doca_pcc` context that
  `doca-pcc` stands up, and owns the host-side lifecycle (create /
  configure / start / stop / destroy, the `doca_pcc_app` image, the
  attach-to-port semantics). This skill prescribes only the DPA-side
  algorithm integration on top.
- `doca-pcc-counters/SKILL.md` —
  the read-only CLI for port PCC counters, and the canonical *"is
  the algorithm actually modulating traffic"* check. This skill
  names the counters the algorithm emits (CNP / NACK / AI / HAI /
  decrement / RTT-band) and routes the inspection there.
- `doca-dpa/SKILL.md` — the host-side DPA control
  library. The algorithm runs on the DPA, compiled by DPACC; the
  two-side-program rule and the DOCA-and-DPACC version-match
  overlay apply.
- `doca-public-knowledge-map/SKILL.md`
  — routing to the public DOCA PCC programming guide
  (<https://docs.nvidia.com/doca/sdk/doca-pcc/index.html>), the PCC
  application guide, the Compatibility Policy, and the on-disk
  layout of an installed DOCA package.
- `doca-setup/SKILL.md` — env preparation,
  install and DPACC compiler verification, BlueField firmware
  configuration (including the custom-PCC slot enable), and the *no
  install yet* path via the public NGC DOCA container. This skill
  assumes those preconditions hold, DPACC matches DOCA, and the
  custom-PCC slot is enabled.
- `doca-version/SKILL.md` — canonical DOCA
  version handling. `## Version compatibility` cross-links the
  four-way match rule and the DOCA-and-DPACC overlay inherited from
  `doca-pcc/SKILL.md`.
- `doca-structured-tools-contract/SKILL.md`
  — the detect / prefer / fall back / report precedence rule the
  Command appendix in the tasks companion honors.
- `doca-programming-guide/SKILL.md`
  — general DOCA programming patterns; this skill layers
  algorithm-specific overlays on the universal build and
  modify-a-shipped-sample workflows.
- `doca-debug/SKILL.md` — the cross-cutting
  debug ladder. Algorithm-specific debug (loaded but counters
  flat; init fails; modulates too aggressively or too gently)
  overlays on top.
- `doca-hardware-safety/SKILL.md` —
  cross-cutting hardware-safety meta-policy. Because the
  algorithm modulates production RoCE-v2 flows, its pre-flight,
  replica-first, and rollback rules apply via this skill's
  `## Safety policy` overlay.
