# DOCA upgrade workflows

**Where to start:** The verbs run `configure → build → modify → run
→ test → debug`. The load-bearing flow is `configure` (detect +
report the gap), then `run` (the confirmation-gated guided upgrade),
then `test` (prove it landed), with `debug` for a move that went
wrong. `build` and `modify` are routing stubs — the version-pin
change that selects a target release is owned by other skills.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the upgrade-mode taxonomy, the never-auto
rule, the sunset concern, the error taxonomy, observability, and the
safety overlay, see [CAPABILITIES.md](CAPABILITIES.md). The
version-detection chain is owned by
[`doca-version`](../doca-version/SKILL.md); the hardware / firmware /
reboot discipline is owned by
[`doca-hardware-safety`](../doca-hardware-safety/SKILL.md).

Each verb below describes the **shape of the workflow**, not a
copy-paste recipe. The agent's job is to walk the user through the
steps in order, verifying preconditions before recommending the
next call — and, above all, to STOP and ask before any upgrade
command runs.

## configure

Goal: detect what DOCA is installed, discover what newer release is
available, run the apt-source consistency precheck, and report the
gap — all WITHOUT issuing any upgrade.

Steps the agent should walk the user through:

1. **Detect the installed version.** Route to
   [`doca-version ## configure`](../doca-version/TASKS.md#configure)
   and capture the four-source chain plus the four-way match status.
   This skill consumes that result; it does NOT restate the
   detection chain. Record the version and host kind for the session.
2. **Discover the available target release.** Look up what release
   is current via the DOCA Release Notes, reached through
   [`doca-public-knowledge-map ## Public documentation entry points`](../doca-public-knowledge-map/SKILL.md#public-documentation-entry-points).
   Never quote a "latest" version from memory; the target is the one
   the user names or the one the release notes confirm is current.
3. **Run the apt-source consistency precheck.** For any host that
   will receive an `apt`-shaped upgrade, route to
   [`doca-version ## apt-source consistency`](../doca-version/TASKS.md#apt-source-consistency)
   to confirm the configured source channel matches the intended
   target before any move is contemplated.
4. **Check for sunset / deprecation status.** Per
   [CAPABILITIES.md ## Capabilities and modes](CAPABILITIES.md#capabilities-and-modes),
   surface whether an installed component appears to be on a
   deprecation track and route the user to the release notes via
   [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md)
   to confirm, rather than recommending continued investment.
5. **Report the gap.** State the installed release, the target
   release, the upgrade mode (per the mode table), and what moving
   costs. Then STOP — the actual move is gated on explicit
   confirmation in [`## run`](#run).

If detection fails (a `pkg-config` not-found or a four-way mismatch),
route through [`doca-version ## debug`](../doca-version/TASKS.md#debug)
before reporting any gap.

## build

> **Anchor exists for lint compliance — routing stub.** An upgrade
> does not produce a build artifact; there is no compile-time output
> of moving a host from one DOCA release to the next. The build
> pattern a DOCA consumer rebuilds against AFTER an upgrade is owned
> by [`doca-programming-guide ## build`](../doca-programming-guide/TASKS.md#build),
> and the build-time-vs-runtime version match is owned by
> [`doca-version ## build`](../doca-version/TASKS.md#build).
>
> This skill's only build-side concern is to SURFACE that an upgrade
> that moves the runtime release may require the user to rebuild
> their application against the new headers — the rebuild itself is
> routed to the two skills above, and the post-upgrade build/runtime
> match is verified in [`## test`](#test). The agent does not invent
> build flags or a rebuild recipe here.

## modify

> **Anchor exists for lint compliance — routing stub.** The concrete
> "modify" an upgrade entails is changing the version pin that
> selects the target release — the apt-source channel pin, or a
> build-manifest minimum, or a container tag. Those edits are owned
> by other skills and this skill does NOT redefine them:
>
> - The apt-source channel pin and the local-repo / network-URL
>   shapes are owned by
>   [`doca-version ## apt-source consistency`](../doca-version/TASKS.md#apt-source-consistency)
>   and the install path in
>   [`doca-setup ## configure`](../doca-setup/TASKS.md#configure).
> - A build-manifest `pkg-config` minimum or a container-tag pin is
>   owned by [`doca-version ## modify`](../doca-version/TASKS.md#modify).
>
> This skill's role is to name *which* pin a chosen upgrade target
> requires changing and to route the edit to the owning skill; the
> agent does not invent a channel URL, a version literal, or a tag
> string here.

## run

Goal: walk the guided upgrade — but ONLY after the gap has been
reported in [`## configure`](#configure) AND the user has explicitly
confirmed. This verb is where the never-auto rule is enforced.

Steps the agent should walk the user through (in order):

1. **Confirm before acting.** Restate the `installed → target` gap
   and the upgrade mode, then obtain explicit user confirmation per
   [CAPABILITIES.md ## Safety policy](CAPABILITIES.md#safety-policy).
   No upgrade command runs until the user says yes; *"should I
   upgrade?"* is answered with the gap and a request to confirm.
2. **Confirm a rollback path first.** Before the move, confirm the
   prior version anchors are captured and a rollback exists for the
   chosen mode (reinstall the prior release, reflash the prior BFB,
   redeploy the prior container tag) per the rollback-first rule.
3. **Apply the mode-appropriate move.** For a host apt upgrade,
   route the package-set convergence to
   [`doca-setup ## configure`](../doca-setup/TASKS.md#configure)
   after the apt source is reconciled. The agent does not invent the
   package list; it uses the documented install procedure.
4. **Delegate every hardware / firmware / reboot step.** A BFB
   reflash, a BlueField mode flip, an `mlxconfig` write, or any cold
   power cycle the target release requires is routed to
   [`doca-hardware-safety ## modify`](../doca-hardware-safety/TASKS.md#modify)
   and the pre-flight discipline in
   [`doca-hardware-safety ## configure`](../doca-hardware-safety/TASKS.md#configure).
   This skill names *when* such a step is part of the upgrade; that
   skill names *how* it is applied safely. The agent never redefines
   the reflash / mode-flip / power-cycle steps here.
5. **Gate the workload behind verification.** Do not declare the
   upgrade done until [`## test`](#test) passes; route the
   post-change workload gate through
   [`doca-hardware-safety ## run`](../doca-hardware-safety/TASKS.md#run)
   when the move touched hardware state.

## test

Goal: prove the upgrade landed — the install converged on the target
release and is internally consistent — before any workload depends
on it.

This is **a loop, not a one-shot pass.** Each iteration re-reads the
version state and compares against the target; the loop terminates
when the four-way match holds on the target release.

Iteration shape:

1. **Post-upgrade four-way match.** Route to
   [`doca-version ## test`](../doca-version/TASKS.md#test) and
   confirm `pkg-config --modversion doca-common`,
   `cat /opt/mellanox/doca/applications/VERSION`, `doca_caps
   --version`, and (on BlueField) the BFB version all agree on the
   *target* release. A partial result is a failed upgrade.
2. **Compare against the captured baseline.** Per
   [CAPABILITIES.md ## Observability](CAPABILITIES.md#observability),
   the pre-upgrade state was captured in [`## configure`](#configure);
   the post-upgrade state must show the target release, not the old
   one, on every source.
3. **Confirm host/BFB are in step.** On BlueField hosts, confirm the
   BFB moved with the host packages; a skew is the host/BFB row in
   [CAPABILITIES.md ## Error taxonomy](CAPABILITIES.md#error-taxonomy)
   and routes back to [`## debug`](#debug).
4. **Confirm the consumer still builds / runs.** If the user has an
   application, confirm it builds and runs against the new release
   per [`doca-version ## test`](../doca-version/TASKS.md#test); a
   build/runtime skew after the move is a failed upgrade.
5. **Loop back if any source still reads the old release.** Re-run
   from step 1 after any corrective action until the four-way match
   holds on the target.

## debug

Goal: when an upgrade fails or lands partially, diagnose it before
any other layer is considered. The upgrade-level failure modes are
enumerated in
[CAPABILITIES.md ## Error taxonomy](CAPABILITIES.md#error-taxonomy);
this verb is the recovery ladder.

**The recovery ladder.**

1. **Read all four sources.** Route to
   [`doca-version ## debug`](../doca-version/TASKS.md#debug) and
   capture the post-upgrade four-source state. Identify which sources
   moved to the target and which did not.
2. **Classify the failure.** Map the captured state to a row in
   [CAPABILITIES.md ## Error taxonomy](CAPABILITIES.md#error-taxonomy):
   - **Apt-source drift** → reconcile the source first per
     [`doca-version ## apt-source consistency`](../doca-version/TASKS.md#apt-source-consistency),
     then re-converge.
   - **Aborted transaction / `dpkg` interrupted** → recover the
     package database via the documented `dpkg`/`apt` repair path
     before retrying the move. The agent quotes the documented repair
     step, not an invented one.
   - **Partial upgrade** → converge the package set on one release
     via the install path in
     [`doca-setup ## debug`](../doca-setup/TASKS.md#debug).
   - **Host/BFB skew** → the BFB reflash is the missing move; route
     it to [`doca-hardware-safety ## debug`](../doca-hardware-safety/TASKS.md#debug).
3. **Re-verify against the target.** After the corrective action,
   re-run [`## test`](#test) step 1; the four-way match must hold on
   the target release before any other layer is reconsidered.
4. **Hand off a residual software-layer symptom.** Once the version
   state is known-good and a symptom remains, route to
   [`doca-debug ## debug`](../doca-debug/TASKS.md#debug) with the
   captured upgrade state as evidence.

When the failure is a hardware-state failure that has no documented
recovery (a one-way firmware roll, a bricked BFB), the agent applies
the refuse-and-escalate rule owned by
[`doca-hardware-safety ## debug`](../doca-hardware-safety/TASKS.md#debug)
— it does not guess at a recovery.

## Deferred task verbs

The following verbs are out of scope for this skill but are commonly
asked in the same conversations. Route them as follows so the agent
does not invent guidance:

- **install / first-time setup.** Owned by
  [`doca-setup ## recognize`](../doca-setup/TASKS.md#recognize) and
  [`doca-setup ## no-install`](../doca-setup/TASKS.md#no-install).
  This skill assumes something is already installed; an upgrade is
  *moving from one installed release to another*.
- **version detection in isolation.** Owned by
  [`doca-version ## configure`](../doca-version/TASKS.md#configure).
  This skill consumes the detected version; it does not own the
  detection chain.
- **the hardware / firmware / reboot step itself.** Owned by
  [`doca-hardware-safety ## modify`](../doca-hardware-safety/TASKS.md#modify).
  This skill names when a reflash / mode-flip / cold-power-cycle is
  part of an upgrade; that skill owns how it is applied safely.
