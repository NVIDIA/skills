# DOCA SPCX Congestion-Control Tool — Tasks

**Where to start:** The verbs that carry real workflow
content are `## install` (host-side DOCA + DPA + firmware
slot prerequisites), `## configure` (the SPCX-vs-PCC
decision tree + role + algorithm + parameters + probe-
packet format), `## run` (the prepare → smoke →
contention-positive evaluation flow), `## test`
(iterative loop on the replica), `## debug` (layered
diagnosis), and `## use` (the *"safe to roll forward"*
decision with evidence + rollback + escalation). The two
routing-stub verbs (`build`, `modify`) are kept because
the agent's task-verb contract is uniform across the
bundle and each carries a meaningful pointer to where the
user's question actually belongs.

This file is loaded by [`SKILL.md`](SKILL.md) after
[`CAPABILITIES.md`](CAPABILITIES.md). It walks the agent
through the documented invocations of `doca_spcx_cc`, the
replica-first evaluation discipline, and the hand-off to
the production roll-forward gate that the bundle-wide
hardware-safety meta-policy gates.

## install

`doca_spcx_cc` is **shipped pre-built** as part of every
DOCA install that includes the SPCX optional component,
under `/opt/mellanox/doca/tools/`. The operator-side
install path:

1. **Confirm DOCA is installed with the SPCX component.**
   If the binary is missing, route to
   [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
   to install or repair the host-side DOCA package
   selection; confirm the version per
   [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure).
2. **Confirm the host-side
   [`doca-pcc`](../../libs/doca-pcc/SKILL.md) library is
   installed at the matching version.** The SPCX tool
   links this library; the two must come from the same
   DOCA release band per
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility).
3. **Confirm the DPACC compiler is installed and
   version-matched.** Per the
   [`doca-dpa`](../../libs/doca-dpa/SKILL.md) overlay,
   the DPA-side algorithm image is built by DPACC and
   the host-side DOCA must match DPACC.
4. **Confirm a BlueField with a DPA processor is
   physically present and visible to the host through
   DOCA.** Walk
   [`doca-dpa TASKS.md ## configure`](../../libs/doca-dpa/TASKS.md#configure)
   step 1 to enumerate the device.
5. **Confirm the firmware custom-PCC slot is enabled.**
   Per
   [`doca-pcc CAPABILITIES.md ## Capabilities and modes`](../../libs/doca-pcc/CAPABILITIES.md#capabilities-and-modes)
   triple-axis precondition: custom-PCC requires a
   firmware-level slot to be enabled via the
   `mlxconfig`-class knob the public guide names. If
   the slot is disabled, the SPCX session will fail
   with the firmware-precondition error layer. The
   slot flip is a hardware-touching change per the
   bundle-wide hardware-safety meta-policy in
   [`doca-hardware-safety CAPABILITIES.md ## Safety policy`](../../doca-hardware-safety/CAPABILITIES.md#safety-policy)
   — capture the pre-flight inventory, confirm OOB,
   and follow the cold-power-cycle commit rule.
6. **Confirm SPCX availability on the installed DOCA +
   BlueField generation.** SPCX is the newer of the
   two programmable-CC surfaces; the public DOCA SPCX
   guide names the supported combinations. The agent
   does NOT assume SPCX is available — the gate is
   `doca_spcx_cc --help` succeeding on the installed
   binary + the public guide naming the user's setup.
7. **Confirm a non-prod RDMA / RoCE fabric with
   controllable contention is reachable** for the
   evaluation flow in [`## test`](#test). Without
   contention there is no signal to evaluate against.
8. **Confirm the operator has the privileges the
   public DOCA SPCX guide requires** for binding the
   device, loading the algorithm image, and writing
   the captured artifacts.

If the binary is not at the standard path, the fix is to
install / repair the host-side DOCA package selection
that includes the SPCX component, not to patch the file
in place.

## configure

The tool's configuration is the invocation + the loaded
DPA-side algorithm image. Steps the agent walks the user
through, in order:

1. **Gate — pick the right programmable-CC surface.**
   Walk the SPCX-vs-PCC-vs-factory-firmware decision
   tree in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
   If the user wants a documented default algorithm,
   route to firmware configuration; if the user wants
   the established PCC surface, route to
   [`doca-pcc`](../../libs/doca-pcc/SKILL.md); only
   commit to SPCX when the install + algorithm +
   BlueField combination supports it AND the user
   accepts the newer-surface tradeoff.
2. **Confirm the live-link / contention precondition.**
   The fabric the tool will exercise must be live,
   carry RDMA / RoCE traffic, and have contention that
   matches the algorithm's design. Document the
   contention shape explicitly; an unstated contention
   pattern means the run is uninformative per
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   live-link rule.
3. **Source the DPA-side algorithm.** Either build the
   user-authored algorithm via DPACC (walk the public
   DOCA SPCX / DOCA PCC programming guides via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
   for the algorithm-side discipline) or stage the
   documented shipped reference per
   [`doca-pcc-ztr-rttcc-algo`](../../libs/doca-pcc-ztr-rttcc-algo/SKILL.md).
   Pin a SHA + version of the algorithm image; the
   evaluation evidence depends on this.
4. **Pick the role.** The role decision (RP — Reaction Point —
   vs NP — Notification Point) is a deployment-shape
   choice the user makes per-endpoint. The shipped
   reference sample binary in DOCA 3.3 (`doca_spcx_cc`)
   hard-codes `cfg.role = PCC_ROLE_RP` in `pcc.c` — it does
   NOT register a `--role` CLI flag in
   `pcc_core.c:register_pcc_params`. To run the NP role with the
   shipped reference sample, the user either rebuilds the
   sample with `cfg.role = PCC_ROLE_NP` or uses the public
   DOCA SPCX-CC guide's documented method for the user's
   installed version (`--help` first to confirm the actual
   registered flag set on that DOCA release; do NOT quote a
   `--role` flag the binary does not expose). The two endpoints
   of the deployment must still agree on the role assignment;
   quote the assignment back to the user so they can challenge it.
5. **Pick the probe-packet format.** `--probe-packet-
   format` (e.g. CCMAD per the public default — re-
   confirm the exact token against `--help`). Both
   ends of the deployment must agree on the format.
6. **Stage the parameters.** Each algorithm exposes a
   parameter set; the parameters live in the algorithm
   documentation, not in this skill. Pin the parameter
   set in writing; the parameter set is part of the
   evaluation evidence tuple per
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability).
7. **Plan thread count + core list.** Per the shipped
   sample (the host-side `doca-pcc` defaults), the
   session uses a thread count + a per-thread core
   list. The agent does not invent the defaults; the
   shipped sample + `--help` on the installed binary
   are the authoritative source.
8. **Stage the capture surface.** Decide where the
   per-port / per-flow tracer output goes, and pair
   it with a planned read of
   [`doca-pcc-counters`](../doca-pcc-counters/SKILL.md)
   snapshots before / during / after the evaluation.

For the canonical DOCA universal lifecycle on the
host-side PCC library this tool drives, see
[`doca-pcc TASKS.md ## configure`](../../libs/doca-pcc/TASKS.md#configure)
and
[`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure).
This skill is concerned with the *operator-side*
configuration of the SPCX harness.

## build

`doca_spcx_cc` is **shipped pre-built** as part of every
DOCA install that includes the SPCX optional component
(`/opt/mellanox/doca/tools/doca_spcx_cc`). There is no
source tree the external user is expected to compile for
the tool itself.

The user-built half is the **DPA-side algorithm image**:
the DPA-side translation unit the user wrote (or the
shipped reference algorithm code), compiled by DPACC into
an image the tool loads via the
[`doca-pcc`](../../libs/doca-pcc/SKILL.md) library.

Routing for nearby "build" questions:

- *"The binary isn't there — do I need to build it?"* → no.
  Route to
  [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
  to install or repair the host-side DOCA package
  selection that includes the SPCX component.
- *"I want to build my own SPCX algorithm."* → that is
  the DPA-side build. Route to
  [`doca-pcc TASKS.md ## build`](../../libs/doca-pcc/TASKS.md#build)
  for the cross-side build pattern + the public DOCA
  SPCX / DPACC guides via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  for the SPCX-specific algorithm authoring detail.
- *"I want to build the shipped reference RTT-based
  algorithm."* → route to
  [`doca-pcc-ztr-rttcc-algo`](../../libs/doca-pcc-ztr-rttcc-algo/SKILL.md)
  for the algorithm-side discipline; the tool's harness
  is the same.
- *"I want to extend the tool with a new mode."* → out
  of scope here; this skill is for external operators
  consuming the shipped tool, not for contributors
  extending it.

The `## What this skill deliberately does not ship` block
in [`SKILL.md`](SKILL.md) explicitly forbids adding a
build recipe for the SPCX tool; revisit that policy
before changing this section.

## modify

**Do not modify the shipped `doca_spcx_cc` binary.** It
is an NVIDIA-shipped CLI; there is no documented public
way to change its behaviour, output format, or
parameter surface, and none should be invented.

What the agent *does* modify, every time, is:

1. The **tool invocation** — the registered argp flags (`--device`,
   `--threads`, `--wait-time`, `--probe-packet-format`),
   the algorithm-image path (`--app` or whatever the
   public guide names on this DOCA release), and any
   capture-surface flags the binary registers. The role
   selection (RP vs NP) is hard-coded in the shipped
   sample's `pcc.c` (`cfg.role = PCC_ROLE_RP`) on DOCA 3.3
   and is NOT a runtime knob; switching role means
   rebuilding the sample or following the public guide
   for the user's installed version.
2. The **DPA-side algorithm image** — the user-authored
   algorithm body (rebuilt via DPACC) or the shipped
   reference algorithm's parameter set
   ([`doca-pcc-ztr-rttcc-algo`](../../libs/doca-pcc-ztr-rttcc-algo/SKILL.md)).

That is the configuration loop in [`## configure`](#configure)
above and the iteration loop in [`## test`](#test) below;
treat *modify the invocation + algorithm, not the
binary* as the operating mode.

Routing for nearby "modify" questions:

- *"My algorithm is mis-behaving on the live link — can
  I tweak it in place?"* → no; tweak the source,
  rebuild via DPACC, re-load. The runtime is
  immutable.
- *"I want a different observability surface than the
  tool exposes."* → re-examine the documented
  observability surface in
  [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability);
  pair with
  [`doca-pcc-counters`](../doca-pcc-counters/SKILL.md)
  and
  [`doca-rdma`](../../libs/doca-rdma/SKILL.md) for
  complementary surfaces. Patching the tool's output
  is out of scope.

## run

The prepare → smoke → contention-positive evaluation
flow — every SPCX session goes through it, no
exceptions. The full invocation surface lives in the
public DOCA SPCX page; this section names the *shape*
of the flow, not verbatim command lines (per
[`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
*"do not invent flags"*).

> **Do-not-invent guard (flags vs subcommands).** The
> `doca_spcx_cc` binary takes **flags only — it has no
> subcommand surface**. Real downstream agents have
> hallucinated a `load` / `start` / `observe` / `stop`
> subcommand family for this binary; it does not exist.
> The flag inventory the bundle enumerates verbatim is
> `--device`, `--threads`, `--wait-time`,
> `--probe-packet-format`, plus algorithm-image and
> capture-surface flags (see
> [`## Command appendix`](#command-appendix)). The role
> (RP vs NP) is hard-coded in the shipped sample's
> `pcc.c` (`cfg.role = PCC_ROLE_RP`) on this DOCA
> release and is **not** registered as a CLI flag — do
> not quote `--role` against `--help` output that does
> not list it. The host-side status values reported by
> the bundle (`Active`, `Standby`, `Deactivated`,
> `Error` per
> [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes))
> are status indicators, not subcommands.

1. **Confirm prerequisites.** Per [`## install`](#install)
   and [`## configure`](#configure): binary present,
   library + DPACC + firmware-slot + SPCX availability
   all aligned, algorithm image pinned, live-link +
   contention fabric reachable.
2. **Capture the pre-flight inventory.** Per the
   bundle-wide hardware-safety meta-policy
   ([`doca-hardware-safety CAPABILITIES.md ## Capabilities and modes`](../../doca-hardware-safety/CAPABILITIES.md#capabilities-and-modes)):
   PCIe topology, link state, firmware level, BFB
   level, current CC algorithm in effect (factory or
   prior programmable), host-side env snapshot. This
   is the baseline the rollback compares against.
3. **Confirm OOB management path is reachable.** Per
   the meta-policy
   [`doca-hardware-safety CAPABILITIES.md ## Safety policy`](../../doca-hardware-safety/CAPABILITIES.md#safety-policy)
   OOB rule, if the cutover touches a BlueField the
   operator manages over the same RDMA link the SPCX
   algorithm controls.
4. **Smoke run — load the algorithm, briefly observe
   the host-side status.** Run `doca_spcx_cc` with
   the chosen role + probe-packet format + algorithm
   image + thread / core layout for a short window.
   Confirm the host-side status reaches `Active` per
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability)
   and the tool emits the documented start banner.
   If the session does not reach `Active`, stop and
   walk [`## debug`](#debug) layer by layer.
5. **Inject the planned contention pattern.** The
   evaluation only produces signal when the fabric is
   contended in a way that matches the algorithm's
   design. Confirm via
   [`doca-pcc-counters`](../doca-pcc-counters/SKILL.md)
   snapshots and
   [`doca-rdma`](../../libs/doca-rdma/SKILL.md) flow
   observation that the contention is real.
6. **Capture per-port / per-flow runtime output.**
   The tool's tracer-style output is the primary
   evidence surface; pair it with `doca-pcc-counters`
   snapshots before / during / after, and with
   `doca-rdma`-side metrics (throughput, latency,
   completion ordering).
7. **Stop the algorithm cleanly.** Send SIGINT /
   SIGTERM per the shipped sample's signal handling
   (the agent re-confirms against `--help`); confirm
   the host-side status transitions to `Deactivated`
   cleanly; capture the final tracer output.
8. **Confirm post-stop state on the link.** The
   fabric should return to the algorithm-in-effect-
   before state (the captured pre-flight baseline).
   If it does not, the rollback path is the
   immediate next step.

When recording the run for downstream consumers (the
*evaluation evidence* pattern), write down: the DOCA
version, the DPACC compiler version, the BlueField
identity + firmware version + custom-PCC slot state,
the algorithm name + SHA + version + parameter set,
the role assignment per endpoint, the probe-packet
format, the fabric topology + contention shape, the
capture window, the tracer output, the
`doca-pcc-counters` snapshots, and the `doca-rdma`
metrics. The downstream [`## test`](#test),
[`## debug`](#debug), and [`## use`](#use) workflows
depend on those fields.

## test

`doca_spcx_cc` is **a high-stakes evaluation tool**, so
its `## test` verb is about *iteratively building
evidence* — confirming the algorithm's behaviour is
what the user expects under controlled contention, on a
replica that matches production's hardware class — not
unit-testing the tool itself.

**`## test` is an iterative loop, not a one-shot
pass.** A single contention-positive run is the
beginning of the evidence pile, not the end. Each
iteration tightens one axis (parameter set, contention
shape, algorithm version, observability completeness,
fabric-topology match) and loops back to
[`## run`](#run).

The eval-loop overlay (rows apply to every SPCX
session):

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| Host-side status never reaches `Active` | Install / device-binding / firmware-slot / DPA-image / algorithm-precondition layer of the error taxonomy | Walk [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy) layers 1–5 in order; do not jump ahead. |
| Status reaches `Active` but tracer / counters show no effect on the link | Live-link / contention layer — the link is idle, the contention is below the algorithm's signal floor, or the contention pattern is wrong | Inject the planned contention; confirm via `doca-pcc-counters` that the link is in fact congested; re-run. Do NOT conclude the algorithm is broken from an idle-link run. |
| Algorithm running on a congested link but the captured behaviour is wrong (oscillation, rate collapse, persistent under-utilisation) | Runtime layer of the error taxonomy | Do NOT roll forward; stop the algorithm, capture the runtime evidence, walk [`doca-pcc TASKS.md ## debug`](../../libs/doca-pcc/TASKS.md#debug); fix in the algorithm source, rebuild via DPACC, re-evaluate. |
| Behaviour looks right on this replica but the replica's hardware class does not match production | Per the bundle-wide hardware-safety replica-first rule | Re-run on a replica that matches production on BlueField generation, firmware level, host kernel, and fabric topology class; do NOT roll forward against a non-matching replica. |
| Two consecutive runs on the same setup produce different captured behaviour | Non-determinism in the contention generator, the algorithm itself, or the observability surface | Pin the contention generator; pin the parameter set; pin the algorithm SHA; re-run; if still divergent, the algorithm's non-determinism is the answer the user came for. |
| Algorithm passes replica testing on every captured axis | Evidence is ready for the production-rollout gate per [`## use`](#use) | Move to [`## use`](#use); do NOT skip the bounded-blast-radius and rollback-rehearsed steps. |

The agent's rule: every change to the algorithm or its
parameters re-opens the loop. Re-running with a
tweaked parameter set and quoting the *previous*
captured behaviour without re-checking under the new
parameter set is exactly the failure mode this loop
replaces.

Loop termination: stop iterating once the captured
behaviour is consistent across two consecutive runs on
a production-matching replica AND the rollback path
has been rehearsed end-to-end on the replica AND the
bounded-blast-radius rollout plan is in writing.

This skill does NOT ship a "test fixture" or pre-
recorded expected output. CC behaviour is fabric-,
topology-, contention-, and algorithm-specific; pinning
one would mislead operators on a different setup.

## debug

When `doca_spcx_cc` fails to load, fails to reach
`Active`, or behaves wrongly on a congested link, walk
the layered error taxonomy in
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
in order. The shape of the diagnosis:

1. **Install.** Confirm the binary exists, the SPCX
   optional component is installed, and the host-side
   [`doca-pcc`](../../libs/doca-pcc/SKILL.md) library
   is present. Route to
   [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
   if not.
2. **Device-binding.** Confirm the BlueField is
   visible to DOCA per
   [`doca-pcc TASKS.md ## configure`](../../libs/doca-pcc/TASKS.md#configure)
   step 1.
3. **Firmware custom-PCC slot.** Confirm the slot is
   enabled via the `mlxconfig`-class knob the public
   guide names. The slot flip is a hardware-touching
   change per
   [`doca-hardware-safety CAPABILITIES.md ## Safety policy`](../../doca-hardware-safety/CAPABILITIES.md#safety-policy).
4. **DPA-image.** Confirm the DPA-side algorithm
   image was built by a DPACC version matched to the
   host-side DOCA per
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility);
   re-build if not.
5. **Algorithm-precondition.** Confirm the parameter
   set is within the algorithm's documented ranges;
   confirm the algorithm's required probe-packet
   format is exposed on this install; confirm the
   far-side endpoint exists and has the matching
   role assignment.
6. **Live-link / contention.** Confirm contention
   actually exists on the fabric and that its shape
   matches the algorithm's design. This is the most
   common *"my algorithm did nothing"* root cause.
7. **Runtime.** If the algorithm is running on a
   congested link but the captured behaviour is
   wrong, capture the evidence and walk
   [`doca-pcc TASKS.md ## debug`](../../libs/doca-pcc/TASKS.md#debug);
   the rollback path is the immediate operational
   response — do NOT keep iterating on a production-
   exposed deployment.
8. **Version.** Walk
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   end-to-end; the tool ↔ library ↔ DPACC ↔ firmware
   quadruple is the common version mismatch failure.
9. **Cross-cutting.** Cause is below DOCA — driver,
   firmware, BlueField mode, host kernel. Hand off
   to [`doca-debug ## debug`](../../doca-debug/SKILL.md)
   and
   [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug).

In every case: **quote what the tool reported.** Do
not paraphrase the tracer output, do not summarize the
per-port / per-flow stream into a single number, and
do not skip the live-link / contention layer.

## use

The captured evaluation evidence is consumed by the
**roll-forward vs roll-back decision** — the load-
bearing safety gate the bundle's hardware-safety meta-
policy gates. The agent's hand-off:

1. **Confirm every replica-first criterion is met**
   per
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy):
   evaluation on a replica that matches production
   on hardware class, contention shape matched to
   the algorithm's design, two consecutive runs
   produced consistent behaviour, rollback rehearsed
   end-to-end on the replica.
2. **Confirm the production rollout plan is bounded.**
   First production cutover is one BlueField pair, not
   the fleet. Then a small bounded set. Then
   progressively larger bounded sets, with the
   observability gate proven at each step.
3. **Confirm the rollback to factory PCC is
   documented.** The rollback names the exact action
   (revert the loaded algorithm, fall back to the
   factory PCC behaviour in firmware) and the gate
   that triggers it (observability surface goes
   silent, queue depth exceeds threshold, host-side
   `doca-pcc` status transitions to `Error`, fabric-
   level link instability).
4. **Confirm OOB management path is reachable** for
   every production endpoint the algorithm is loaded
   onto.
5. **Confirm maintenance window discipline** per
   [`doca-hardware-safety CAPABILITIES.md ## Safety policy`](../../doca-hardware-safety/CAPABILITIES.md#safety-policy)
   — cutovers run inside an explicit, time-boxed
   window with operations notified.
6. **Refuse the cutover when any gate fails.** The
   agent surfaces the failed gate as blocking and
   does NOT find a creative workaround. Escalate per
   [`doca-hardware-safety CAPABILITIES.md ## Safety policy`](../../doca-hardware-safety/CAPABILITIES.md#safety-policy)
   no-rollback rule.

The agent's rule: the tool evaluates; this verb gates
the cutover; the
[`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md)
meta-policy is the final authority. A *"safe to roll
forward"* answer requires evidence + bounded plan +
documented rollback + OOB + maintenance window;
anything less is refused.

## Deferred task verbs

The verbs below are not `doca_spcx_cc` work and should
be routed out before the agent does any of them under
this skill's name.

- **DPA-side algorithm authoring detail** →
  [`doca-pcc`](../../libs/doca-pcc/SKILL.md) +
  [`doca-dpa`](../../libs/doca-dpa/SKILL.md) + the
  public DOCA SPCX programming guide via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
  This skill is the operator-side harness; the
  algorithm-side discipline is owned upstream.
- **Factory PCC configuration** → firmware-level
  knobs, no host-side library or tool needed. Route
  via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
- **PCC counter inspection without loading an
  algorithm** →
  [`doca-pcc-counters`](../doca-pcc-counters/SKILL.md).
  The counter tool is read-only and side-effect-free;
  SPCX loads an algorithm.
- **Raw DPA cycle profiling** → different surface,
  different tool. Route via
  [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools);
  this tool's tracer output is at the SPCX-runtime
  layer, not the DPA-instruction layer.
- **RDMA library programming questions** →
  [`doca-rdma`](../../libs/doca-rdma/SKILL.md).
- **General DOCA install / repair** →
  [`doca-setup ## install`](../../doca-setup/TASKS.md#configure).

## Command appendix

`doca_spcx_cc`-specific invocation classes the verbs
above reach for. Every row is a CLASS — the agent must
not invent flags, role tokens, probe-packet format
tokens, or metric names beyond `--help` on the
installed binary and the public DOCA SPCX page.

**Infra-aware preamble (every row below).** Per the
bundle's detect → prefer → fall back → report contract
documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent should:

1. Probe for the matching structured helper FIRST
   (`doca-env --json` for version + devices + DPA
   availability + custom-PCC slot state in one shot;
   `doca-capability-snapshot` for per-device
   capability flags).
2. If the probe succeeds, the structured tool's
   output is the authoritative answer.
3. If the probe fails, fall back to the manual
   command in the row.
4. The schemas the structured tools emit are defined
   in
   [`doca-structured-tools-contract ## Schemas`](../../doca-structured-tools-contract/SKILL.md#schemas).

| Purpose (class) | Invocation (shape) | Owning step | Reads as healthy when … |
| --- | --- | --- | --- |
| Discover the documented flag surface | `doca_spcx_cc --help` + the public DOCA SPCX page | [`## configure`](#configure); [`## debug`](#debug) layer 1 | Prints the documented inventory the shipped binary actually registers via argp: `--device`, `--threads`, `--wait-time`, `--probe-packet-format`, plus algorithm-image / capture-surface flags. The role (RP vs NP) is hard-coded in the shipped sample's `pcc.c` (`cfg.role = PCC_ROLE_RP`) and is NOT registered as a CLI flag on this DOCA release; do not quote `--role` against `--help` output that does not list it. |
| Smoke-load the algorithm in RP role | `doca_spcx_cc --device <mlx5_*> --probe-packet-format <token> --app <algorithm-image> ... ` (role defaults to RP per the shipped sample's hard-coded `cfg.role = PCC_ROLE_RP`; for NP, see `## configure` step 4 — rebuild or follow the public guide for that DOCA release) | [`## run`](#run) step 4 | Host-side status reaches `Active`; the documented start banner prints; the per-port tracer surface begins emitting. |
| Drive the contention-positive evaluation | Smoke-load above + the planned contention pattern injected on the fabric | [`## run`](#run) steps 5–6; [`## test`](#test) iteration | The tracer surface shows the algorithm reacting to the contention; `doca-pcc-counters` snapshots show non-trivial PCC counter activity; `doca-rdma` metrics show the modulation the algorithm is designed for. |
| Stop the algorithm cleanly | SIGINT / SIGTERM to the running session per the shipped sample's signal handling | [`## run`](#run) step 7 | Host-side status transitions to `Deactivated`; the post-stop fabric state matches the captured pre-flight baseline. |
| Capture the evidence tuple | Pair the tool's tracer output with `doca-pcc-counters` snapshots, `doca-rdma` metrics, and the metadata tuple per [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability) | [`## test`](#test) baseline-capture rule; [`## use`](#use) gate | The full (DOCA + DPACC + BlueField + firmware + algorithm SHA + parameter set + role + probe-packet format + fabric topology + contention shape + window) tuple is attached to the captured artifact. |
| Roll back to the factory PCC | The documented rollback path the operator rehearsed on the replica | [`## use`](#use) | The previous CC algorithm (factory or prior programmable) is restored; the captured post-rollback state matches the pre-flight baseline. |

Three cross-cutting rules for this appendix:

- **Never invent a flag, role token, probe-packet
  format token, algorithm parameter name, or metric
  name.** `--help` on the installed binary and the
  public DOCA SPCX page are the joint contract.
- **Smoke before contention, contention before
  conclusion.** Every row above presumes the smoke
  reached `Active` first AND the contention
  precondition was met. Drawing a conclusion from
  either step in isolation is the canonical SPCX
  evaluation failure.
- **Cross-link instead of duplicate.** Cross-cutting
  commands (`pkg-config --modversion doca-pcc`,
  `doca_caps --list-devs`, `mlxconfig -d <bdf> q` for
  the custom-PCC slot state) live in
  [`doca-debug ## debug`](../../doca-debug/SKILL.md) and
  [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug);
  this appendix names only `doca_spcx_cc`-specific
  invocation classes.

## Cross-cutting

A few rules that apply across every verb in this file,
restated here so they are visible at the point of
action and not buried in [`SKILL.md`](SKILL.md):

- The **public DOCA SPCX page** plus the installed
  `--help` are the joint source of truth. When they
  disagree, the *installed* `--help` wins for the
  user's actual run.
- **A CC algorithm has no signal under no contention.**
  The agent refuses to declare an algorithm safe
  based on an idle-link or contention-absent run.
- **Replica-first is mandatory.** No production
  cutover without a contention-positive evaluation on
  a hardware-matched replica and a rehearsed
  rollback.
- **Factory-PCC rollback is the always-available
  escape hatch.** Every deployment plan names it
  explicitly; a plan without it is refused per the
  bundle-wide hardware-safety meta-policy.
- **Quote the (DOCA + DPACC + BlueField + firmware +
  algorithm SHA + parameter set + role + probe-packet
  format + fabric topology + contention shape +
  window) tuple.** Evidence without the tuple is
  unreplicable.
- This skill **assumes a healthy DOCA install** with
  the SPCX component, a paired
  [`doca-pcc`](../../libs/doca-pcc/SKILL.md) library
  + DPACC, a BlueField with a visible DPA and the
  firmware custom-PCC slot enabled, and a
  contention-positive replica fabric. If any of
  those is in doubt, route to
  [`doca-setup`](../../doca-setup/SKILL.md),
  [`doca-pcc`](../../libs/doca-pcc/SKILL.md), or
  [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md)
  before running anything else here.
