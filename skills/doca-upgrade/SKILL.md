---
license: Apache-2.0
name: doca-upgrade
description: >
  Use this skill when the user is contemplating a DOCA upgrade or
  downgrade — moving a host to a newer DOCA release, refreshing the
  BlueField BFB, bumping the NGC DOCA container tag, or rolling back.
  The discipline is detect → report → ASK → only-then guided upgrade:
  detect what is installed, discover what newer release exists, report
  the gap, then STOP and ask for explicit confirmation — never upgrade
  automatically. Trigger even without the word "upgrade": "is there a
  newer DOCA", "should I move to the next release", "I want the latest
  features", "my component is being deprecated, what now", or "roll me
  back". Route elsewhere for version detection (doca-version),
  first-time install (doca-setup), any hardware/firmware/reboot step
  (doca-hardware-safety), and public-docs / sunset routing
  (doca-public-knowledge-map).
metadata:
  kind: library
compatibility: >
  No DOCA install required to read this skill (it is a cross-cutting
  overlay loaded against any DOCA artifact skill); the detection,
  gap-report, and guided-upgrade steps within DO require a live DOCA
  install at /opt/mellanox/doca, and the hardware / firmware / reboot
  steps require a BlueField DPU or ConnectX NIC plus out-of-band
  console reachability as gated by doca-hardware-safety.
---

# DOCA upgrade

**Where to start:** This skill is the bundle's single source of
truth for the discipline that wraps a DOCA *upgrade or downgrade*.
Open [`TASKS.md`](TASKS.md) when the user wants to *do* something
with an upgrade (detect the gap, ask for confirmation, walk the
guided upgrade, verify it, or diagnose a failed one); open
[`CAPABILITIES.md`](CAPABILITIES.md) when the question is *what does
an upgrade even cover* (the host apt-upgrade vs BFB reflash vs
container-tag bump vs downgrade/rollback modes, the never-auto rule,
the confirmation gate, the sunset-awareness concern, the error
taxonomy, and the upgrade-specific safety overlay).

**The single most important rule this skill exists to enforce:**
**detect → report → ASK → only-then guided upgrade. Never upgrade
automatically.** The skill detects the installed DOCA env and what
newer release is available, reports the gap clearly, then STOPS and
asks the user for explicit confirmation. Only on an explicit "yes"
does it walk a guided upgrade — and every hardware / firmware /
reboot step is delegated to
[`doca-hardware-safety`](../doca-hardware-safety/SKILL.md), never
redefined here. Version detection itself is owned by
[`doca-version`](../doca-version/SKILL.md); this skill routes there
and does not restate the detection chain.

## Example questions this skill answers well

The CLASSES of upgrade questions this skill is built to answer, each
with one worked example. The agent should treat the *class* as the
load-bearing piece — the worked example is a single instance.

- **"Is there a newer DOCA than what I have, and how far behind am
  I?"** — worked example: *"`pkg-config --modversion doca-common`
  says 3.1.0; what's current and what does moving cost me?"*.
  Answered by the detect-and-report-the-gap workflow in
  [`TASKS.md ## configure`](TASKS.md#configure), which routes the
  detection to [`doca-version`](../doca-version/SKILL.md) and the
  *what is current* lookup to
  [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md).
- **"Should I just run `apt upgrade` to move to the next DOCA
  release?"** — worked example: *"I want the newest Flow features;
  can I upgrade in place right now?"*. Answered by the never-auto
  rule and the confirmation gate in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  plus the ask-before-acting step in
  [`TASKS.md ## run`](TASKS.md#run) — the agent reports the gap and
  STOPS for explicit confirmation before any upgrade command runs.
- **"My upgrade left the host on a different release than the
  BlueField — what happened?"** — worked example: *"after the
  upgrade `pkg-config` says 3.3.0 but `bfver` still says 3.1.0"*.
  Answered by the host/BFB-skew row in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  and the failed-upgrade diagnosis in
  [`TASKS.md ## debug`](TASKS.md#debug), which re-routes the
  detection to [`doca-version ## test`](../doca-version/TASKS.md#test).
- **"The upgrade aborted half-way and now `dpkg` is wedged — how do
  I recover?"** — worked example: *"`apt upgrade` was interrupted;
  `dpkg` reports an interrupted transaction"*. Answered by the
  partial-upgrade and aborted-transaction rows in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  plus the recovery ladder in [`TASKS.md ## debug`](TASKS.md#debug).
- **"The upgrade needs me to reflash the BFB / flip BlueField mode /
  reboot — is that safe?"** — worked example: *"the target release
  requires a new BFB; can I reflash now?"*. Answered by routing
  every hardware / firmware / reboot step to
  [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md) via
  [`TASKS.md ## run`](TASKS.md#run); this skill never redefines the
  reflash, mode-flip, or cold-power-cycle discipline.
- **"My installed component is being sunset — should I keep building
  on it or move?"** — worked example: *"I rely on DOCA App Shield /
  the DOCA Flow Inspector service; are they on a
  deprecation track?"*. Answered by the sunset / deprecation
  awareness concern in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
  which routes the user to the public release notes via
  [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md)
  to confirm lifecycle status rather than recommending continued
  investment in a component that may be sunsetting.

## When to load this skill

Load this skill whenever an upgrade or downgrade of DOCA is the
load-bearing concern — the user is asking whether a newer release
exists, whether to move to it, how to move safely, or how to recover
a move that went wrong. The decision must be made **before** the
agent composes its first sentence; the activation checklist below is
mirrored here so the activation rule is at hand whenever this skill
is consulted. The skill's contract is unconditional: it reports the
gap and then STOPS for explicit user confirmation; it never issues an
upgrade command on the user's behalf without that confirmation.

## Agent activation checklist — load this skill at the START of the answer when any cell below is true

| Trigger class | Concrete prompt-side signals (any one fires the overlay) |
| --- | --- |
| Upgrade / downgrade intent | *"is there a newer DOCA"*, *"should I move to the next release"*, *"upgrade me to the latest"*, *"roll me back to the previous DOCA"*, *"downgrade to 3.1 for a bug repro"* |
| Accidental / drifted upgrade | *"`apt upgrade` pulled a newer DOCA point release and broke my build"*, *"my host and BlueField are on different releases now"*, *"the source channel is `latest` and I didn't mean to move" * |
| Failed / partial upgrade | *"the upgrade aborted half-way"*, *"`dpkg` is wedged after an interrupted transaction"*, *"upgrade finished but the four-way match no longer holds"* |
| Container tag bump | the user wants to move an NGC DOCA container deployment from one tag to a newer one, or asks whether bumping the tag is an upgrade |
| Sunset / deprecation awareness | the user asks whether the component they depend on is deprecated, being sunset, or worth continued investment |

When any cell above fires, the agent MUST:

1. Route version detection to
   [`doca-version ## configure`](../doca-version/TASKS.md#configure)
   — do NOT restate the four-source detection chain here; this skill
   consumes the detected version and the four-way match status.
2. **Report the gap and STOP for explicit confirmation.** Present
   the installed release, the available target release, and what
   moving costs (per
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility)),
   then ask the user to confirm. Do NOT proceed to any upgrade
   command until the user explicitly says yes.
3. Route every hardware / firmware / reboot step to
   [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md). This
   skill names *when* such a step is part of the upgrade; that skill
   names *how* it is applied safely.

Do **not** load this skill for first-time install (use
[`doca-setup`](../doca-setup/SKILL.md)), for the version-detection
chain in isolation (use [`doca-version`](../doca-version/SKILL.md)),
or for general orientation (use
[`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md)).

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive upgrade material
lives in two companion files:

- `CAPABILITIES.md` — the upgrade surface: the upgrade-mode taxonomy
  (host apt upgrade vs BFB reflash vs container-tag bump vs
  downgrade/rollback), the never-auto rule and the confirmation
  gate, the sunset / deprecation-awareness concern, the
  version-compatibility overlay that redirects to `doca-version`,
  the error taxonomy (partial upgrade, apt-source drift, host/BFB
  skew, aborted transaction / `dpkg` interrupted), the observability
  surface (which commands prove the upgrade did or did not happen),
  and the upgrade-specific safety overlay on top of
  `doca-hardware-safety`.
- `TASKS.md` — step-by-step workflows for the in-scope verbs:
  `configure` (detect + discover target + apt-source precheck),
  `build` and `modify` (routing stubs — the version-pin change is
  owned elsewhere), `run` (the confirmation-gated guided upgrade),
  `test` (post-upgrade four-way verification), `debug` (diagnose a
  failed / partial upgrade), plus a `Deferred task verbs` block.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is in
   scope (an upgrade / downgrade is being contemplated or recovered).
2. **For the upgrade-mode taxonomy, the never-auto rule, the
   sunset-awareness concern, the error taxonomy, observability, and
   the safety overlay, see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — detect-and-report, the
   confirmation-gated guided upgrade, post-upgrade verification, and
   failed-upgrade diagnosis — see [TASKS.md](TASKS.md).**

## Related skills

- [`doca-version`](../doca-version/SKILL.md) — owns the four-source
  detection chain and the four-way match rule. This skill's detect
  step routes there and does not redefine version detection; the
  gap report consumes the detected version.
- [`doca-setup`](../doca-setup/SKILL.md) — owns first-time install,
  install verification, and the apt-source path. This skill's
  apt-source consistency precheck cross-links there before any
  `apt`-shaped upgrade is contemplated.
- [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md) — owns
  every hardware / firmware / reboot step (mlxconfig, BFB reflash,
  BlueField mode flip, cold power cycle). This skill's `## Safety
  policy` overlays that meta-policy and adds only the
  upgrade-specific surface (confirmation gate, rollback-first,
  maintenance window).
- [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md) —
  the routing table to public DOCA docs, including the release notes
  and the Compatibility Policy. This skill routes *what is the
  current release* and *is this component sunsetting* there; it does
  not duplicate the routing.
- [`doca-debug`](../doca-debug/SKILL.md) — the cross-cutting debug
  ladder. A failed upgrade whose residual symptom lives at a
  software layer hands off there once the upgrade state is known.
- [`doca-programming-guide`](../doca-programming-guide/SKILL.md) —
  the build pattern a consumer rebuilds against after an upgrade.
  This skill surfaces *that* a rebuild may be needed; the build
  mechanics live there.
- [`doca-structured-tools-contract`](../doca-structured-tools-contract/SKILL.md) —
  the JSON schemas the agent prefers when present; the upgrade
  detect step reuses the same `doca-env` one-shot the version skill
  uses, when the host has it.
