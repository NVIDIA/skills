---
license: Apache-2.0
name: doca-dms
description: >
  Operate NVIDIA DOCA Management Service on a BlueField, Arm/x86 host,
  or Kubernetes pod: from DOCA 3.5 the local `dms-cli` (one command per
  YANG path, no server), and the `dmsd` + `dmspe` service: choose deployment and
  authentication, configure `-allowed_users` and `dmsgroup`, use gNMI
  Get/Set/Subscribe, run supported gNOI workflows, and debug
  frontend/backend failures. Trigger even without "DMS" for "manage a
  remote BlueField over gRPC", "gNOI reboot from orchestrator", or
  fleet-management requests. SAFETY: reboot, OS install, factory-reset,
  and managed-file deletion are destructive and require target-bound
  explicit confirmation; never invoke them speculatively. Route
  installation and library/API build questions elsewhere, and route
  turnkey aggregation to the externally-productized DOCA Telemetry
  Service.
metadata:
  author: NVIDIA <doca@nvidia.com>
  kind: service
  doca_version: "3.5"
  tags:
    - doca
    - dms
    - gnmi
compatibility: >
  DOCA service shipped with the DOCA install at /opt/mellanox/doca on
  the management endpoint (x86 host (non-DPU), BlueField Arm, or
  Kubernetes pod) on Linux (Ubuntu 22.04/24.04 or RHEL/SLES); `dmsd` +
  `dmspe` run there, and DMS is also pulled as an NGC container image.
  Verify the public DMS guide version matches the installed DOCA release
  before quoting flags or YANG paths. The validation steps require a
  live DOCA install.
---

# DOCA Management Service (DMS)

## Examples

- **"Read or change a BlueField setting locally."** — use
  `dms-cli` (DOCA 3.5+); explore the path with `--tree` first.
- **"Manage a remote BlueField over gRPC."** — this is in
  scope. Use `dmsd` + `dmspe` with the chosen auth; do not
  drop to ad-hoc SSH for the management plane.
- **"gNOI reboot from orchestrator."** — treat reboot / OS
  install / factory-reset as destructive: require target-bound
  explicit confirmation first.

## Instructions

Follow Loading order below. Open the tasks companion for configure /
gNMI / debug. Open the capabilities companion for auth and gNOI
scope. Do not open every companion file.

> **⚠️ Destructive operations.** The gNOI `reboot`, `OS install`,
> `factory-reset`, and managed-file deletion operations are
> **irreversible** and **service-impacting**
> — they can take a production BlueField or ConnectX offline or wipe its
> configuration. Before issuing any of them the agent MUST: (1) verify the
> target device identity, and (2) obtain explicit confirmation bound to
> that target and action. In an interactive session this is an explicit
> user reply naming/accepting both; in unattended execution it must be an
> approved-system authorization artifact bound to both. Otherwise stop
> with `confirmation_required`. Never invoke them speculatively or as
> a side effect of another task. See the public DMS guide's safety
> guidance for these operations.

**Where to start:** This skill is for *operating* DMS, not for
*linking against* a library. If the user wants to *deploy* or *run*
the daemon, open the tasks companion and start at
`## configure`. If the question is *what shape
of service is DMS and what protocols does it speak*, start at
the capabilities companion. If DOCA is not installed on the
management endpoint yet, route to
`doca-setup/SKILL.md` first.

**How to use this skill.** Read this file, then search the companion file
for the `## <name>` heading the question needs and read that whole section.
Answer completely from it, quoting only its symbols and commands. Do not
browse docs; use the entry points in `doca-public-knowledge-map`. Give
environment checks as commands for the user's DOCA host. For out-of-scope
requests, name the skill that owns them and stop.

**DOCA 3.5 and later: start with `dms-cli`.** The DMS guide now centres on
`/opt/mellanox/doca/services/dms/dms-cli`, run directly on the host or
BlueField Arm with no server or daemon. One syntax per YANG path:
`dms-cli -t pci/<BDF> /nvidia/mode/operating-mode` reads a value, the same
path plus a value sets it, and action paths run operations. Explore before
writing: `--tree /nvidia`, `--describe <path>` and `list-xpaths <path>`;
script with `--json`; validate with `--dry-run`. VFs and SFs take
`-t pci/<BDF>,pf<N>vf<M>` or `pf<N>sf<M>`. Observed on DOCA 3.5: as a normal
user it fails with `backend fork/exec .../dmspe: permission denied` — run it
with `sudo` or add the user to `dmsgroup`; some example paths do not exist on
every device (for example congestion-control slots are under
`/nvidia/cc/algo/slot` there), so confirm every path with `--tree` /
`list-xpaths` before using it. Firmware or BFB installs
(`/nvidia/provisioning/install`), resets and mode changes are
hardware-state changes: follow `doca-hardware-safety`. The `dmsd` + `dmspe`
gRPC (gNMI / gNOI) service below remains for remote and fleet management.

## Example questions this skill answers well

> **Never invent the image tag.** It comes from the NGC catalog's DOCA
> org or the DMS page on `docs.nvidia.com`, not from memory — a
> plausible tag reads as authoritative and mismatches the installed
> DOCA. Cite where it came from, or send the user to look it up.

The CLASSES of DMS questions this skill is built to answer, each with
one worked example. The class is the load-bearing piece; the worked
example is one instance.

- **"Where should DMS run for my target topology?"** — worked example:
  *"I have a host without a DPU; can I still manage a remote
  ConnectX?"*. Answered by the deployment-shape decision in
  the capabilities companion
  + the tasks companion.
- **"Which authentication mode should I pick for my security
  posture?"** — worked example: *"a multi-tenant production env vs a
  single-tenant lab"*. Answered by the auth-mode trade-off table in
  the capabilities companion
  + the auth-wiring step in
  the tasks companion.
- **"How do I issue a gNMI `Get` / `Set` against a modeled path?"** —
  worked example: *"set `/interfaces/interface/config/mtu` on a remote
  interface"*. Answered by the gNMI/gNOI surface in
  the capabilities companion
  + the request-shape workflow in
  the tasks companion.
- **"How do I run a gNOI operation (reboot, OS install, file
  transfer)?"** — worked example: *"trigger a clean reboot of the
  target via gNOI"*. Answered by the gNOI catalog in
  the capabilities companion
  + the run workflow in the tasks companion.
- **"My DMS request returned an error — was it the frontend or the
  backend?"** — worked example: *"`mlxconfig` failed under DMS but
  works on the shell"*. Answered by the frontend-vs-backend split in
  the capabilities companion
  + the layered ladder in
  the tasks companion.
- **"Where do I read DMS logs and how do I rotate them?"** — worked
  example: *"persistent log directory + journald + log-rotation
  policy"*. Answered by the logging surface in
  the capabilities companion
  + the log-rotation step in
  the tasks companion.

## Audience

This skill serves **external operators and platform teams who deploy and
operate DMS** to manage NVIDIA® BlueField® networking platforms or
NVIDIA® ConnectX® SmartNICs from a centralized control plane. Concretely:
people running `dmsd`, integrating gNMI/gNOI clients against it, choosing
an authentication mode, or wiring DMS into a Kubernetes deployment.

It is **not** for NVIDIA developers contributing to DMS itself, and it
is **not** a programming guide for *building applications on top of*
DOCA libraries (that is `doca-programming-guide` plus the matching
library skill under `libs/`). DMS is a **service**, not a library: the
user invokes it as a daemon and talks to it over gRPC; they do not link
against `libdms.so` to write their own program.

**Status note.** Per the public DMS guide, DMS is currently in **beta**,
with General Availability scoped to SPC-X use cases. The skill reflects
the public guide's posture: prescribe the documented launch / auth /
deployment paths, follow the documented security best practices, and
defer roadmap and GA-scope questions to the live public guide rather
than guessing.

## When to load this skill

Load this skill when the user is doing **hands-on DMS operation work**
against a BlueField or ConnectX target where DOCA is already installed
on the management endpoint (host, DPU, or pod). Concretely:

- Deciding *where* DMS should run (host non-DPU / BlueField Arm /
  Kubernetes pod) for a given target topology.
- Bringing up the `dmsd` daemon — choosing SystemD vs manual launch,
  selecting an authentication mode, wiring `-allowed_users` (the gRPC
  client authorization boundary) and, if needed, `dmsgroup` (the
  `dmspe` backend-helper Unix group).
- Issuing `gNMI` `Get` / `Set` requests against modeled paths
  (e.g. `/interfaces/interface/config/mtu`).
- Issuing `gNOI` operations: OS install, reboot, file transfer,
  factory-reset, `mlxconfig`, containerz.
- Choosing an authentication mode (localhost / PAM / credentials / mTLS)
  and understanding the security trade-offs the public guide calls out.
- Reading or rotating DMS logs, configuring config persistency, or
  recovering from a crashed daemon.
- Debugging a DMS request that returned an error — separating
  "frontend rejected before reaching backend" from "backend executed and
  the underlying tool (e.g. `mlxconfig`) failed".

Do **not** load this skill for general DOCA orientation, install of
DOCA itself, or library-API questions. For those, route via
`doca-public-knowledge-map/SKILL.md`,
`doca-setup/SKILL.md`, or the matching
`libs/<library>` skill.

## What this skill provides

This is a **thin loader**. Substantive material lives in two companion
files:

- the capabilities companion — DMS architecture (frontend `dmsd` / privileged
  backend `dmspe`), management protocols (gNMI, gNOI), the YANG-based
  unified configuration dictionary, deployment shapes, authentication
  modes with their security trade-offs, the configuration-persistency
  model, the logging surface, and DMS's documented security posture.
- the tasks companion — step-by-step workflows for the in-scope DMS verbs:
  `configure`, `build`, `modify`, `run`, `test`, `debug`, plus a
  `Deferred task verbs` block routing out-of-scope questions.

The skill assumes a host where DOCA is already installed and the
operator has root / `sudo` access where the public guide says it is
required. It does not cover installing DOCA — that path goes through
`doca-setup/SKILL.md`.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a templates or sample-config
bundle. To keep the boundary clean, it deliberately does not contain —
and pull requests should not add:

- **Pre-baked DMS configuration files** (YANG instance documents,
  full-stack example configs, ready-to-run `dmsd` flag bundles)
  intended to be copy-pasted into production. Configs are deployment-
  specific and the safe answer for an external operator is to derive
  them from the public guide against their own target. The agent's
  job is to prescribe the *procedure* and quote the documented flags
  and paths, not to ship a config the user might run unmodified.
- **Pre-written gNMI / gNOI client programs in any language.** The
  client surface is standard gNMI / gNOI (publicly documented); the
  skill describes which paths and operations DMS supports, not how to
  build a gRPC client in language X.
- **TLS material, credentials, or PAM stanzas.** These are
  user-environment artifacts; the skill points at the documented
  configuration knobs and the documented security best practices and
  stops there.
- **A `samples/`, `templates/`, or `reference/` subtree** of any
  kind. A mock or incomplete artifact in this skill's tree, even one
  labeled "reference", is misleading: operators will read it as
  production-ready.

## Loading order

1. Read this skill first to confirm the user's question is in scope.
2. **For the DMS architecture, deployment shapes, auth modes,
   protocol/path inventory, persistency, logging, and security
   posture, see the capabilities companion.**
3. **For step-by-step workflows — configure, build, modify, run, test,
   debug — see the tasks companion.**
4. **Apply the destructive-operation gate in every phase.** Before any
   gNOI OS install, reboot, factory reset, or managed-file deletion —
   including a test or debug action — verify the exact target and obtain explicit
   confirmation for that specific operation using the interactive or
   approved-system mechanism defined in the warning above. No earlier
   confirmation or workflow phase carries authorization forward.

## Related skills

- `doca-public-knowledge-map/SKILL.md`
  — the routing table to the public DMS guide and the rest of the
  public DOCA documentation set.
- `doca-setup/SKILL.md` — env preparation and
  install verification on the host where `dmsd` will run, including
  the *I have no install yet* path via the public NGC DOCA container.
  This skill assumes its preconditions are satisfied at the management
  endpoint.
- `doca-programming-guide/SKILL.md` —
  general DOCA patterns. DMS is service-shaped not library-shaped, so
  the build / modify / first-app pattern there does not apply directly,
  but the cross-library `DOCA_ERROR_*` taxonomy and the
  layered-debug order remain useful when DMS reports errors that
  originated in a DOCA library it called.
