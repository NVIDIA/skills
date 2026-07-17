# DOCA upgrade — capabilities, version compatibility, errors, observability, safety

**Where to start:** Pick the H2 anchor that matches your question
(what an upgrade covers / how the target release is chosen / which
failure modes to recognize / how to prove the upgrade happened /
the upgrade-specific safety overlay) and read that section
end-to-end. The tables in each section are the load-bearing content;
the prose is interpretation.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the *how* of executing each workflow, jump
to [TASKS.md](TASKS.md). The version-detection chain this skill
consumes is owned by
[`doca-version`](../doca-version/SKILL.md); the hardware / firmware /
reboot discipline this skill delegates to is owned by
[`doca-hardware-safety`](../doca-hardware-safety/SKILL.md).

## Pattern overview

Every upgrade question this skill teaches resolves into the same
shape: **detect what is installed → report the gap to the target →
ASK for explicit confirmation → only then walk a guided upgrade.**
The patterns are CLASSES — they apply across every DOCA release,
every upgrade mode, and every host kind.

| Upgrade pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Detect + discover the gap | Read the installed version (via `doca-version`) and the available target release (via the public release notes) and state the delta | [`## Capabilities and modes`](#capabilities-and-modes) + [TASKS.md ## configure](TASKS.md#configure) |
| 2. Report + confirm | Present installed, target, and cost; STOP; obtain explicit user confirmation before any command runs | [`## Safety policy`](#safety-policy) confirmation gate + [TASKS.md ## run](TASKS.md#run) |
| 3. Walk the guided upgrade | Execute the mode-appropriate steps; delegate every hardware / firmware / reboot step to `doca-hardware-safety` | [`## Capabilities and modes`](#capabilities-and-modes) mode table + [TASKS.md ## run](TASKS.md#run) |
| 4. Verify the upgrade landed | Re-run the four-way match (via `doca-version`) and prove the new version is what is loaded | [`## Observability`](#observability) + [TASKS.md ## test](TASKS.md#test) |
| 5. Diagnose a failed / partial upgrade | Map the symptom (partial upgrade, source drift, host/BFB skew, aborted transaction) to a recovery action | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **Never upgrade automatically.** The agent detects and reports;
  the user decides. No upgrade command runs until the user has seen
  the gap report and explicitly confirmed (see
  [`## Safety policy`](#safety-policy)).
- **Surface sunset / deprecation status before recommending more
  investment.** When the gap report touches a component that may be
  on a deprecation track, route the user to the public release notes
  to confirm lifecycle status rather than recommending continued
  investment.

## Capabilities and modes

An "upgrade" is not one operation. The agent identifies which
**upgrade mode** the user is in before reporting the gap, because
the mode determines the steps, the rollback shape, and which
hardware-touching steps (if any) are delegated to
[`doca-hardware-safety`](../doca-hardware-safety/SKILL.md).

| Upgrade mode | What moves | What it touches | Who owns the touching step |
| --- | --- | --- | --- |
| **Host apt upgrade** | The host-side DOCA packages (`doca-common`, the per-tool packages, `doca-host`) move to a newer release | Userspace packages; may pull a new kernel module / DKMS / OFED step | The apt-source path is owned by [`doca-setup ## configure`](../doca-setup/TASKS.md#configure); the precheck is [`doca-version ## apt-source consistency`](../doca-version/TASKS.md#apt-source-consistency) |
| **BFB reflash** | The BlueField OS image and DPU-side DOCA install move to a newer BFB | The BlueField OS image, the DPU PCIe surface, every hosted service container; a link-breaking, reboot-class change | Delegated entirely to [`doca-hardware-safety ## modify`](../doca-hardware-safety/TASKS.md#modify); never redefined here |
| **Container image tag bump** | An NGC DOCA container deployment moves from one tag to a newer one | The container image only; the host packages and BFB are independent anchors | The tag-selection rule is owned by [`doca-setup`](../doca-setup/SKILL.md); the version inside the new tag is verified via [`doca-version ## test`](../doca-version/TASKS.md#test) |
| **Downgrade / rollback** | The host packages, BFB, or container tag move *back* to a prior release (e.g. for a bug repro or a regression rollback) | The same surfaces as the forward modes, in reverse | Same delegation as the forward mode; the rollback discipline overlays [`doca-hardware-safety ## debug`](../doca-hardware-safety/TASKS.md#debug) |

The agent's rule: **identify the mode, then report the gap for that
mode.** A host apt upgrade and a BFB reflash are different changes
with different blast radii even when they move the same release
number; the agent does not conflate them.

### Sunset / deprecation awareness

When the agent detects what is installed, part of the gap report is
checking whether an installed component is on a deprecation / sunset
track. The team has flagged DOCA App Shield and the DOCA Flow
Inspector service — both now policy-excluded from this public
bundle (see [AGENTS.md `## Non-goals`](../../AGENTS.md#non-goals-questions-the-agent-should-recognize-and-refuse-politely)
item 7) — as worked examples to watch on this front. The discipline is:

- Treat a possible sunset as a finding to **surface**, not a fact to
  assert from memory. The authoritative statement of a component's
  lifecycle status is the public release notes, reached via
  [`doca-public-knowledge-map ## Public documentation entry points`](../doca-public-knowledge-map/SKILL.md#public-documentation-entry-points)
  (DOCA Release Notes) — the agent routes the user there to confirm.
- When a component does appear to be sunsetting, the agent does NOT
  recommend continued investment in it as part of an upgrade. It
  surfaces the sunset status, points at the public migration / docs
  path via
  [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md),
  and lets the user decide whether to upgrade-in-place or migrate.
- This is class-shaped: DOCA App Shield and the Flow Inspector
  service are the worked examples, but the discipline applies to any
  component whose release notes mark it deprecated.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way match
rule, NGC container semantics, and the headers-win-over-docs rule,
see [`doca-version`](../doca-version/SKILL.md). The body lives there;
this skill does not duplicate it.

**The upgrade overlay** is:

- **The gap is `installed → target`.** The installed version comes
  from the four-source chain in
  [`doca-version ## Capabilities and modes`](../doca-version/CAPABILITIES.md#capabilities-and-modes);
  the target release is the one the user wants, confirmed against the
  public release notes — never a version string from agent memory.
- **The target must be a supported pairing.** Before recommending a
  move, the agent confirms the `installed → target` jump is inside
  the window NVIDIA documents in the
  [DOCA Compatibility Policy](https://docs.nvidia.com/doca/sdk/doca-compatibility-policy/index.html)
  (cited once via
  [`doca-version ## Version compatibility`](../doca-version/CAPABILITIES.md#version-compatibility)).
- **Host, BFB, firmware, and container tag are independent anchors.**
  An upgrade that moves one anchor must bring the others into a
  supported pairing; a move that leaves them skewed is a failed
  upgrade, caught by the post-upgrade four-way match in
  [`## Observability`](#observability).
- **Never quote a version literal from memory.** The version-handling
  anti-hallucination rules in
  [`doca-version ## Safety policy`](../doca-version/CAPABILITIES.md#safety-policy)
  apply here without modification.

## Error taxonomy

The upgrade-level failure modes the agent should recognize and
disambiguate before continuing. For the cross-library `DOCA_ERROR_*`
taxonomy itself, see
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../doca-programming-guide/CAPABILITIES.md#error-taxonomy);
the rows below are the *upgrade-level* causes that bubble up as
those errors.

| Symptom | Most-likely upgrade cause | First action |
| --- | --- | --- |
| Some DOCA packages moved to the target release; others did not | **Partial upgrade** — the package set converged unevenly (a held package, a missing per-tool package, or an interrupted run) | Re-detect the four sources per [`doca-version ## debug`](../doca-version/TASKS.md#debug); converge the package set on one release before any other diagnosis |
| `apt-cache policy doca-common` shows a Candidate from a different release than what is installed | **Apt-source drift** — the configured source channel disagrees with the installed release | Reconcile the apt source first per [`doca-version ## apt-source consistency`](../doca-version/TASKS.md#apt-source-consistency); do not upgrade until the source matches the intended target |
| After the upgrade, `pkg-config`/`doca_caps` report the new release but `bfver` still reports the old BFB | **Host/BFB version skew post-upgrade** — the host moved but the BFB did not | The BFB reflash is the missing step; route it to [`doca-hardware-safety ## modify`](../doca-hardware-safety/TASKS.md#modify), and re-verify per [`doca-version ## test`](../doca-version/TASKS.md#test) |
| `dpkg` reports an interrupted / half-configured transaction; further `apt` invocations refuse to proceed | **Aborted transaction / `dpkg` interrupted** — the upgrade run was interrupted mid-flight | Recover the package database per the documented `dpkg`/`apt` repair path before retrying; see [TASKS.md ## debug](TASKS.md#debug) for the recovery ladder |
| Upgrade completed but the program now returns `DOCA_ERROR_NOT_SUPPORTED` / builds against a symbol the runtime lacks | **Build/runtime skew after upgrade** — the runtime `*.so` and the headers/`pkg-config` are now from different releases | Same partial-install diagnosis owned by [`doca-version ## debug`](../doca-version/TASKS.md#debug); reconverge, do not patch around in code |
| Downgrade left the host on the older release but a co-tenant workload now fails | **Rollback blast radius** — the downgrade affected shared host state | Treat the rollback with the same discipline as the forward move; route any hardware-touching rollback step to [`doca-hardware-safety ## debug`](../doca-hardware-safety/TASKS.md#debug) |

## Observability

The upgrade observability surface is the set of commands that prove
whether the upgrade *did or did not* happen. There is no DOCA
"upgrade counter"; the visibility comes from comparing the version
state before and after against the sources
[`doca-version`](../doca-version/SKILL.md) owns. The agent reads the
same chain before the upgrade (to establish the baseline) and after
(to prove the move).

| Phase | Signal | What it tells the agent |
| --- | --- | --- |
| Pre-upgrade | The four-source detection chain + four-way match status, captured per [`doca-version ## configure`](../doca-version/TASKS.md#configure) | The baseline the post-upgrade state is compared against; without it the agent cannot prove the move happened |
| Pre-upgrade | `apt-cache policy <pkg>` / the configured apt source channel | Whether the host *will* resolve to the intended target before any command runs; the apt-source precheck owns this |
| Post-upgrade | The same four-source chain re-read, plus the four-way match per [`doca-version ## test`](../doca-version/TASKS.md#test) | Whether the install converged on the target release and is internally consistent; a partial result is a failed upgrade |
| Post-upgrade | The BFB version (`bfver`) on BlueField hosts | Whether the BFB moved in step with the host packages, or is now skewed |
| Post-upgrade | The structured one-shot `doca-env --json` `version.consistent` field when present, per [`doca-structured-tools-contract ## Schemas`](../doca-structured-tools-contract/SKILL.md#schemas) | The four-way match pre-computed in one read; prefer it when the host has the helper |

The agent prefers the structured one-shot when present and falls
back to the manual chain `doca-version` documents otherwise. The
load-bearing rule: **prove the upgrade with the same sources before
and after; a claim that the upgrade "worked" without a post-upgrade
four-way match is not evidence.**

## Safety policy

This skill's safety surface is the **confirmation gate** on top of
the hardware-safety meta-policy. For the cross-cutting discipline
that wraps any hardware / firmware / reboot step — pre-flight
inventory, out-of-band access, maintenance window, replica-first
validation, rollback discipline, refuse-and-escalate — see
[`doca-hardware-safety ## Safety policy`](../doca-hardware-safety/CAPABILITIES.md#safety-policy).
The body lives there; this skill does not redefine it.

**The upgrade-specific overlay** is:

- **Confirmation gate (never auto-upgrade).** The agent reports the
  `installed → target` gap and STOPS. No upgrade command runs until
  the user has seen the report and explicitly confirmed. *"Should I
  upgrade?"* is answered with the gap and a request to confirm, never
  with an upgrade command issued on the user's behalf.
- **Rollback-first.** Before the guided upgrade begins, the agent
  confirms a rollback path exists for the chosen mode (reinstall the
  prior release, reflash the prior BFB, redeploy the prior container
  tag) and that the prior version anchors are captured — the same
  documented-and-rehearsed rollback rule the hardware-safety
  meta-policy requires for any hardware-touching step.
- **Maintenance window.** Any upgrade that touches a host carrying
  real workload — and every BFB reflash or reboot-class step — runs
  inside an explicit, time-boxed maintenance window with stakeholders
  notified, per the hardware-safety maintenance-window rule.
