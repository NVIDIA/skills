---
license: Apache-2.0
name: doca-upgrade
description: >
  Changing DOCA versions safely: upgrading or downgrading the host DOCA
  packages, refreshing the BlueField BFB, bumping the NGC DOCA container
  tag, or rolling back, with the discipline detect -> report -> ASK ->
  only then a guided upgrade. Trigger on "is there a newer DOCA", "I want the newest DOCA
  features, can I just apt upgrade in place", "should I move to the
  next release", "the upgrade stopped midway and dpkg is in a broken
  state", "my component is being deprecated, what now", or "roll me back",
  even without the word upgrade. Detect what is installed, find the newer
  release, report the gap, then STOP and ask for explicit confirmation;
  never upgrade automatically. Do not use it for a runtime error on an
  unchanged install (doca-debug). Version detection alone belongs to
  doca-version, first-time install to doca-setup, any hardware / firmware
  / reboot step to doca-hardware-safety, and public-docs or sunset routing
  to doca-public-knowledge-map.
metadata:
  author: NVIDIA <doca-feedback@nvidia.com>
  kind: library
  doca_version: "3.5"
  tags:
    - doca
    - upgrade
    - version
compatibility: >
  No DOCA install required to read this skill (it is a cross-cutting
  overlay loaded against any DOCA artifact skill); the detection,
  gap-report, and guided-upgrade steps within DO require a live DOCA
  install at /opt/mellanox/doca, and the hardware / firmware / reboot
  steps require a BlueField DPU or ConnectX NIC plus out-of-band
  console reachability as gated by doca-hardware-safety.
---

# DOCA upgrade

## Examples

- **"Should I move to the next DOCA release?"** — detect what is
  installed, report the gap, then STOP and ask. Never auto-upgrade.
- **"apt upgrade drifted my DOCA packages."** — this is recovery,
  not a silent repair. Detect, report, ask, then guide.

## Quick answers (answer from this file)

- **Not changing versions** (a runtime error on the current, pinned
  install)? This skill does not apply: name `doca-debug` and stop.

These two answers are complete here; open `TASKS.md` only for a guided
upgrade after the user confirms. Run only the read-only detection
commands yourself. Package commands (`sudo dpkg`, `apt`, `apt-get`) need
root and change the system: give them to the user to run and stop; in an
agent shell `sudo` waits for a password and the session stalls.

- **"Can I just `apt upgrade` to the next DOCA?"** Never upgrade
  automatically. Detect the installed release with the `doca-version`
  four-source chain (`pkg-config --modversion doca-common`,
  `cat /opt/mellanox/doca/applications/VERSION`, `doca_caps --version`, and
  the BFB version on BlueField), report the gap to the target release, then
  STOP and ask for explicit confirmation. BFB, firmware or reboot steps follow
  `doca-hardware-safety`.
- **"The upgrade aborted half-way and dpkg reports an interrupted
  transaction."** Recover the package database first:
  after the user explicitly confirms (both commands change system package
  state), the user runs `sudo dpkg --configure -a`, then
  `sudo apt-get -f install` (Debian/Ubuntu). Do not re-run the upgrade as the first action; re-detect
  the release with `doca-version` before deciding the next step.

## Instructions

Follow Loading order below. Open the tasks companion for detect →
report → ASK → guided upgrade. Open the capabilities companion for
the upgrade-mode taxonomy and refuse-auto-upgrade rule. Do not open
every companion file.

**Where to start:** This skill is the bundle's single source of
truth for the discipline that wraps a DOCA *upgrade or downgrade*.
Open the tasks companion when the user wants to *do* something
with an upgrade (detect the gap, ask for confirmation, walk the
guided upgrade, verify it, or diagnose a failed one); open
the capabilities companion when the question is *what does
an upgrade even cover* (the host apt-upgrade vs BFB reflash vs
container-tag bump vs downgrade/rollback modes, the never-auto rule,
the confirmation gate, the sunset-awareness concern, the error
taxonomy, and the upgrade-specific safety overlay).

**How to use this skill.** Read this file, then search the companion file
for the `## <name>` heading the question needs and read that whole section.
Answer completely from it, quoting only its symbols and commands. Do not
browse docs; use the entry points in `doca-public-knowledge-map`. Give
environment checks as commands for the user's DOCA host. For out-of-scope
requests, name the skill that owns them and stop.

**For a planned move, the single most important rule this skill
exists to enforce is detect → report → ASK → only-then guided
upgrade. Never upgrade automatically.** The skill detects the
installed DOCA env and what newer release is available, reports the
gap clearly, then STOPS and asks the user for explicit confirmation.
Only on an explicit "yes" does it walk a guided upgrade. A failed or
partial move instead enters the recovery ladder without presenting a
new target move. In both paths every hardware / firmware / reboot
step is delegated to
`doca-hardware-safety/SKILL.md`, never
redefined here. Version detection itself is owned by
`doca-version/SKILL.md`; this skill routes there
and does not restate the detection chain.

## Example questions this skill answers well

The CLASSES of upgrade questions this skill is built to answer, each
with one worked example. The agent should treat the *class* as the
load-bearing piece — the worked example is a single instance.

- **"Is there a newer DOCA than what I have, and how far behind am
  I?"** — worked example: *"`pkg-config --modversion doca-common`
  says `<installed X.Y.Z>`; what's current and what does moving cost me?"*.
  Answered by the detect-and-report-the-gap workflow in
  ## configure, which routes the
  detection to `doca-version/SKILL.md` and the
  *what is current* lookup to
  `doca-public-knowledge-map/SKILL.md`.
- **"Should I just run `apt upgrade` to move to the next DOCA
  release?"** — worked example: *"I want the newest Flow features;
  can I upgrade in place right now?"*. Answered by the never-auto
  rule and the confirmation gate in
  ## Safety policy
  plus the ask-before-acting step in
  ## run — the agent reports the gap and
  STOPS for explicit confirmation before any upgrade command runs.
- **"My upgrade left the host on a different release than the
  BlueField — what happened?"** — worked example: *"after the
  upgrade `pkg-config` reports the new release but `bfver` still reports the old one"*.
  Answered by the host/BFB-skew row in
  ## Error taxonomy
  and the failed-upgrade diagnosis in
  ## debug, which re-routes the
  detection to `doca-version ## test`.
- **"The upgrade aborted half-way and now `dpkg` is wedged — how do
  I recover?"** — worked example: *"`apt upgrade` was interrupted;
  `dpkg` reports an interrupted transaction"*. Answered by the
  partial-upgrade and aborted-transaction rows in
  ## Error taxonomy
  plus the recovery ladder in ## debug.
- **"The upgrade needs me to reflash the BFB / flip BlueField mode /
  reboot — is that safe?"** — worked example: *"the target release
  requires a new BFB; can I reflash now?"*. Answered by routing
  every hardware / firmware / reboot step to
  `doca-hardware-safety/SKILL.md` via
  ## run; this skill never redefines the
  reflash, mode-flip, or cold-power-cycle discipline.
- **"My installed component is being sunset — should I keep building
  on it or move?"** — worked example: *"I rely on DOCA App Shield /
  the DOCA Flow Inspector service; are they on a
  deprecation track?"*. Answered by the sunset / deprecation
  awareness concern in
  ## Capabilities and modes,
  which routes the user to the public release notes via
  `doca-public-knowledge-map/SKILL.md`
  to confirm lifecycle status rather than recommending continued
  investment in a component that may be sunsetting.

## When to load this skill

Load this skill whenever an upgrade or downgrade of DOCA is the
load-bearing concern — the user is asking whether a newer release
exists, whether to move to it, how to move safely, or how to recover
a move that went wrong. The decision must be made **before** the
agent composes its first sentence; the activation checklist below is
mirrored here so the activation rule is at hand whenever this skill
is consulted. For a planned upgrade or downgrade, it reports the gap
and then STOPS for explicit user confirmation. For a failed or partial
upgrade, it enters ## debug first to
capture and stabilize the current state; it never uses another
upgrade command as the first recovery action, retries the move, or
issues a new upgrade command without explicit confirmation.

## Agent activation checklist — load this skill at the START of the answer when any cell below is true

| Trigger class | Concrete prompt-side signals (any one fires the overlay) |
| --- | --- |
| Upgrade / downgrade intent | *"is there a newer DOCA"*, *"should I move to the next release"*, *"upgrade me to the latest"*, *"roll me back to the previous DOCA"*, *"downgrade to 3.1 for a bug repro"* |
| Accidental / drifted upgrade | *"`apt upgrade` pulled a newer DOCA point release and broke my build"*, *"my host and BlueField are on different releases now"*, *"the source channel is `latest` and I didn't mean to move" * |
| Failed / partial upgrade | *"the upgrade aborted half-way"*, *"`dpkg` is wedged after an interrupted transaction"*, *"upgrade finished but the four-way match no longer holds"* |
| Container tag bump | the user wants to move an NGC DOCA container deployment from one tag to a newer one, or asks whether bumping the tag is an upgrade |
| Sunset / deprecation awareness | the user asks whether the component they depend on is deprecated, being sunset, or worth continued investment |

First classify the request. A **planned move** (including a requested
rollback) follows detect → report → ASK. A **failed, partial, or
accidentally drifted move** is already a diagnosis/recovery request:
load this skill, capture the current four-source state, and enter
TASKS.md `## debug`; do not ask for permission to perform an upgrade
that has already failed, and do not issue another upgrade command as
the first recovery action.

For a planned move, the agent MUST:

1. Route planned-upgrade version detection to
   `doca-version ## configure`.
   For failed or partial upgrades, start with
   `doca-version ## debug` and
   ## debug. Do NOT restate the
   four-source detection chain here; this skill consumes the detected
   state and four-way match status.
2. **Report the gap and STOP for explicit confirmation.** Present
   the installed release, the available target release, and what
   moving costs (per
   ## Version compatibility),
   then ask the user to confirm. Do NOT proceed to any upgrade
   command until the user explicitly says yes.
3. Route every hardware / firmware / reboot step to
   `doca-hardware-safety/SKILL.md`. This
   skill names *when* such a step is part of the upgrade; that skill
   names *how* it is applied safely.

For a failed/partial move, the agent MUST instead classify the
captured state against
CAPABILITIES.md `## Error taxonomy`,
delegate the single documented corrective action, and re-enter the
recovery ladder. Any unsupported `installed → target` jump fails
closed: do not improvise intermediate releases or commands; require
an upgrade path explicitly documented by the Compatibility Policy or
release notes.

Do **not** load this skill for first-time install (use
`doca-setup/SKILL.md`), for the version-detection
chain in isolation (use `doca-version/SKILL.md`),
or for general orientation (use
`doca-public-knowledge-map/SKILL.md`).

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive upgrade material
lives in two companion files:

- the capabilities companion — the upgrade surface: the upgrade-mode taxonomy
  (host apt upgrade vs BFB reflash vs container-tag bump vs
  downgrade/rollback), the never-auto rule and the confirmation
  gate, the sunset / deprecation-awareness concern, the
  version-compatibility overlay that redirects to `doca-version`,
  the error taxonomy (partial upgrade, apt-source drift, host/BFB
  skew, aborted transaction / `dpkg` interrupted), the observability
  surface (which commands prove the upgrade did or did not happen),
  and the upgrade-specific safety overlay on top of
  `doca-hardware-safety`.
- the tasks companion — step-by-step workflows for the in-scope verbs:
  `configure` (detect + discover target + apt-source precheck),
  `build` and `modify` (routing stubs — the version-pin change is
  owned elsewhere), `run` (the confirmation-gated guided upgrade),
  `test` (post-upgrade four-way verification), `debug` (diagnose a
  failed / partial upgrade), plus a `Deferred task verbs` block.

## Loading order

1. Read this skill first to confirm the user's question is in
   scope (an upgrade / downgrade is being contemplated or recovered).
2. **For the upgrade-mode taxonomy, the never-auto rule, the
   sunset-awareness concern, the error taxonomy, observability, and
   the safety overlay, see the capabilities companion.**
3. **For step-by-step workflows — detect-and-report, the
   confirmation-gated guided upgrade, post-upgrade verification, and
   failed-upgrade diagnosis — see the tasks companion.**

## Related skills

- `doca-version/SKILL.md` — owns the four-source
  detection chain and the four-way match rule. This skill's detect
  step routes there and does not redefine version detection; the
  gap report consumes the detected version.
- `doca-setup/SKILL.md` — owns first-time install,
  install verification, and the apt-source path. This skill's
  apt-source consistency precheck cross-links there before any
  `apt`-shaped upgrade is contemplated.
- `doca-hardware-safety/SKILL.md` — owns
  every hardware / firmware / reboot step (mlxconfig, BFB reflash,
  BlueField mode flip, cold power cycle). This skill's `## Safety
  policy` overlays that meta-policy and adds only the
  upgrade-specific surface (confirmation gate, rollback-first,
  maintenance window).
- `doca-public-knowledge-map/SKILL.md` —
  the routing table to public DOCA docs, including the release notes
  and the Compatibility Policy. This skill routes *what is the
  current release* and *is this component sunsetting* there; it does
  not duplicate the routing.
- `doca-debug/SKILL.md` — the cross-cutting debug
  ladder. A failed upgrade whose residual symptom lives at a
  software layer hands off there once the upgrade state is known.
- `doca-programming-guide/SKILL.md` —
  the build pattern a consumer rebuilds against after an upgrade.
  This skill surfaces *that* a rebuild may be needed; the build
  mechanics live there.
- `doca-structured-tools-contract/SKILL.md` —
  the JSON schemas the agent prefers when present; the upgrade
  detect step reuses the same `doca-env` one-shot the version skill
  uses, when the host has it.
