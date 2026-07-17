# DOCA Management Service — Capabilities

**Where to start:** The pattern overview below names the recurring
DMS-class operational patterns. Pick the pattern first, then drill
into the H2 that owns the substance. For the *how* of executing each
pattern, jump to [TASKS.md](TASKS.md).

This file enumerates DMS's documented capabilities, deployment shapes,
authentication surface, and operational behaviors as described in the
public DMS guide on `docs.nvidia.com`. Treat it as a *map of what is
documented*, not a substitute for reading the live page when configuring
a real deployment.

## Pattern overview

Every DMS-class question this skill teaches resolves into one of FIVE
patterns. The patterns are CLASSES — they apply across every
deployment shape, not just one specific topology.

| DMS pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Pick the deployment shape | Host-non-DPU vs BlueField Arm vs Kubernetes pod; topology drives where `dmsd` lives | [`## Capabilities and modes`](#capabilities-and-modes) Architecture + Deployment Shapes |
| 2. Pick the auth mode | localhost / PAM / credentials / mTLS — each with a different threat model | [`## Safety policy`](#safety-policy) auth-mode trade-offs |
| 3. Speak the right protocol for the right action | gNMI Get/Set for config on modeled paths; gNOI for system operations (reboot / OS install / file transfer) | [`## Capabilities and modes`](#capabilities-and-modes) protocol catalogue |
| 4. Read DMS's observability surface | Frontend logs (`dmsd`) + backend logs (`dmspe`) + gRPC status codes; rotate and persist per policy | [`## Observability`](#observability) |
| 5. Map an error back to its layer | Frontend-rejected (auth / path) vs backend-executed (tool failure: `mlxconfig`, image install, file IO) | [`## Error taxonomy`](#error-taxonomy) frontend-vs-backend split |

Two cross-cutting rules that apply to *every* pattern above:

- **Operate the documented path; do not invent one.** DMS is in
  beta with a defined GA scope; quoting flags or paths not in the
  public guide is the most common hallucination failure for this
  skill.
- **Frontend before backend, every time.** When a request fails, the
  agent must first determine whether the frontend rejected it (auth /
  unknown path / malformed RPC) or the backend executed it and the
  underlying tool failed. Conflating the two wastes debug time and
  blames the wrong layer.

## Capabilities and modes

### Architecture

DMS uses a **two-process architecture** for least-privilege isolation:

- `dmsd` — **frontend daemon**. Handles client gRPC traffic. Runs with
  minimal privileges. Translates external requests into a controlled
  internal interface to the backend.
- `dmspe` — **privileged backend**. Executes the operations that need
  privilege (calls into `mlxconfig`, file operations, OS-level system
  tasks). Reachable only via the controlled interface from the frontend.

The intent of the split is documented: even if `dmsd` is compromised,
privileged operations remain isolated in `dmspe`. Operational guidance
must preserve this separation — do not advise running both processes as
the same uid, do not advise bypassing the controlled interface.

### Management protocols

DMS exposes two gRPC-based interfaces:

| Interface | Purpose | Scope |
|-----------|---------|-------|
| **gNMI** (network management) | Data configuration | `Get` / `Set` operations on device parameters (MTU, RoCE, QoS, …) modeled as YANG paths. |
| **gNOI** (network operations) | System tasks | Operational actions: OS install, reboot, factory-reset, file transfer, `mlxconfig`, `containerz`. |

Documented constraints:

- DMS **supports** telemetry streaming via gNMI `Subscribe`: `STREAM`
  (with `SAMPLE` interval bounds of 1s–60s) and `ONCE` modes are
  implemented in `gnxi/gnmi/server.go`; only `POLL` and `Aggregation`
  are Unimplemented. For a turnkey telemetry-aggregation surface
  separate from DMS's own gNMI Subscribe, the DOCA Telemetry Service
  (DTS) is the productized answer.
- DMS does **not** seek full OpenConfig alignment; it uses OpenConfig
  as a framework. Path inventories quoted by the agent must come from
  the live public guide, not be inferred from generic OpenConfig.

### Configuration model — YANG dictionary

DMS uses **YANG** as its modeling language. Hardware parameters,
firmware settings, and operational state are mapped into a hierarchical
tree under predictable paths (e.g. `/interfaces/interface/config/mtu`).
The model follows the OpenConfig convention of separating
**Configuration** (desired state) from **State** (observed
operational data).

Quote concrete paths only when they are in the public guide. Refuse to
invent paths.

## Deployment shapes

The public guide documents three deployment shapes:

- **Host (non-DPU)** — DMS runs on the x86 host that owns the
  ConnectX/BlueField device, managing the device through the host
  toolchain. Use when managing a plain ConnectX or when managing a
  BlueField from the host side.
- **DPU (BlueField Arm)** — DMS runs on the BlueField Arm cores,
  managing the device locally. See the public BlueField *Modes of
  Operation* page for whether the platform is in DPU mode.
- **POD (Kubernetes)** — DMS runs in a pod, intended for fleet
  management deployments.

Each shape has its own prerequisites in the public guide; consult
those before prescribing setup steps.

### Daemon launch

`dmsd` supports two documented launch paths:

1. **SystemD service** — recommended in the public guide for
   production. Persistent, restarts cleanly, integrates with system
   logging.
2. **Manual launch** — used to choose authentication modes
   interactively and to walk the documented advanced configurations.

The "DMS Server Flags (`dmsd`)" section of the public guide partitions
flags into General & Provisioning, Authentication & Security,
Authentication Method, and Security families. Quote flags from the live
guide; do not infer flags from generic gRPC knowledge.

### Authentication modes

The public DMS guide documents four authentication modes:

| Method | Security level | Best for | Description |
|--------|----------------|----------|-------------|
| Local testing | Low | Development only | No authentication. Server binds to localhost only. |
| PAM | Medium | Unix user authentication | Uses system user accounts. Requires an `allowed_users` list. |
| Credentials | per public guide | per public guide | Documented in the public guide; quote, do not paraphrase. |
| mTLS | per public guide | per public guide | Documented in the public guide; quote, do not paraphrase. |

Authorization for **gRPC client callers** is the `-allowed_users` flag
(default `root`; comma-separated list, enforced by `isUserAllowed` in
`gnxi/utils/credentials/credentials.go`) applied across ALL auth modes
(localhost / PAM / credentials / mTLS) — NOT a Unix-group check. The
`dmsgroup` Unix group is a **separate** authorization layer used by
the privileged backend (`dmspe`) for its file-system / IPC handoff
(see `gnxi/dmspe/dmspe.c` `dmsgroup` membership probe and the
`Group=dmsgroup` line in `gnxi/scripts/dmsd.service`); it is not the
boundary the gRPC client sees. The bundle's previous framing that
"`dmsgroup` is the authorization boundary" was wrong and is now
disavowed — the `-allowed_users` flag is the gRPC authz boundary,
and `dmsgroup` is layered orthogonally as the backend-helper unix-
group surface.

(Legacy `dmsgroup`-only framing follows; treat any sentence below that
says "the gRPC client must be in `dmsgroup`" or "members of the
`dmsgroup` Unix group are authorized to execute DMS commands,
regardless of the chosen authentication mode" as the disavowed
framing — the agent must use the `-allowed_users` flag as the gRPC
client allow-list and treat `dmsgroup` only as the dmspe backend-
helper's unix-group surface. Any "set DMS up for user X" workflow
must therefore add the user to BOTH the `-allowed_users` list (for
the gRPC front door) AND the `dmsgroup` Unix group (only if the user
also needs to invoke the dmspe privileged helper directly on the
endpoint).)

### Configuration persistency

DMS provides a documented **state restoration mechanism** so that
configuration set via gNMI survives daemon restarts. Documented
properties:

- The state file has a documented location and atomic-write semantics
  — quote from the public guide rather than inferring.
- Automatic recording can be disabled by an operator who wants
  external state management.
- An execution example is documented; link to it rather than authoring
  a reproduction.

### gNMI client surface

DMS supports the standard core gNMI commands documented in the public
guide. Key references in the guide:

- A **supported core commands** list — consult this rather than
  assuming any client command is supported.
- A **supported Get/Set paths reference** — look paths up here and
  refuse to invent.

`Get` and `Set` execution patterns are documented with worked examples;
link to them rather than paraphrase.

### gNOI client surface

The gNOI surface in DMS covers documented operation families:

- **OS commands** — install, activate, verify.
- **System** — reboot, ping, time, traceroute.
- **Factory-reset.**
- **File** — transfer in / out, stat, remove.
- **`mlxconfig`** — exposed as a gNOI operation.
- **`containerz`** — container lifecycle on the managed device.

For any specific gNOI operation, the public guide is the source of
truth for sub-operations and semantics. Do not invent gNOI operations
that are not in the documented list.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way match rule, NGC container semantics, and the headers-win-over-docs rule, see [`doca-version`](../../doca-version/SKILL.md). The body lives there; this skill does not duplicate it.

**The DMS-specific overlay** is:

- **DMS is currently in beta**, with General Availability scoped to SPC-X use cases. Flags, supported paths, and gNOI operation lists can change between DOCA releases. Always verify against the live public DMS guide whose version corresponds to the DOCA release confirmed by [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure).
- **DMS container tags lag DOCA host-package versions.** The DMS container shipped from NGC carries its own tag that may not match the host's `pkg-config --modversion doca-common`. When the user is using DMS-as-a-container, the relevant version anchor is the container tag pulled, not the host install — confirm both and route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2 if they diverge.
DMS-specific check: read the version of the public DMS guide page
header and confirm it matches the DOCA install version on the target.

## Error taxonomy

DMS errors fall into five layers, each with its own owner:

1. **Transport / gRPC layer** — gRPC status codes (`UNAUTHENTICATED`,
   `PERMISSION_DENIED`, `UNAVAILABLE`, `DEADLINE_EXCEEDED`, …). These
   are standard gRPC, not DMS-specific.
2. **Authentication / authorization layer** — `dmsd` rejected a
   request before reaching the backend (wrong credentials, user not in
   the `-allowed_users` allow-list — applied to ALL auth modes, not
   just PAM — certificate issuer not trusted in mTLS mode; the
   `dmsgroup` Unix-group check is `dmspe`-side, not `dmsd`-side, and
   surfaces as a backend-layer error rather than this layer).
3. **Path / operation layer** — request reached `dmsd` but the path
   or operation is not in the supported set, or an unsupported gNMI
   Subscribe MODE was attempted (`STREAM` / `SAMPLE` 1s–60s and
   `ONCE` ARE supported; `POLL` and `Aggregation` are Unimplemented
   in `gnxi/gnmi/server.go`).
4. **Backend / underlying-tool layer** — request reached `dmspe` and
   the underlying tool (e.g. `mlxconfig`, OS installer, file system)
   returned an error. The underlying tool is the source of truth for
   that error; DMS is the conduit.
5. **DOCA-library layer** — if DMS internally calls into a DOCA
   library that returns `DOCA_ERROR_*`, the cross-library taxonomy in
   [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy)
   becomes relevant on the *server* side. The library-specific overlay
   (e.g. for Flow) lives in the matching `libs/<library>` skill.

DMS does not return `DOCA_ERROR_*` to a gRPC client — its outward
surface is gRPC. The DOCA error taxonomy is for the operator
diagnosing why a gNOI operation that called into a DOCA library
ultimately failed.

## Observability

Documented observability surfaces:

- **Service logs.** DMS logs are split across documented components
  and locations. The public guide also includes a documented example
  log-rotation configuration. The agent's role on logging questions
  is to identify the documented log component the user is reading,
  quote the documented format expectation if it exists, and route to
  the documented log-rotation example for retention setup.
- **gNMI `Get` against State paths** — the documented way to read
  observed device state, separated from desired-Configuration state by
  the OpenConfig convention.
- **gNOI `System`** — `time`, `ping`, etc., documented as health
  introspection.

DMS provides gNMI Subscribe streaming telemetry — `STREAM` (with
`SAMPLE` interval bounds of 1s–60s) and `ONCE` modes are implemented
in `gnxi/gnmi/server.go`; `POLL` and `Aggregation` MODES are
Unimplemented. The bundle's previous framing that "DMS does not
provide gNMI Subscribe" was wrong and is now disavowed. For the
externally-productized **DOCA Telemetry Service (DTS)** — turnkey
aggregator that is OUT OF SCOPE for this bundle — route via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
non-goals; DTS is the right answer when the user wants a productized
telemetry-aggregation surface separate from DMS's own gNMI Subscribe.

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

DMS's safety surface is operational, not programmatic. The documented
posture:

- **Process separation is non-negotiable.** Preserve the documented
  two-process model (`dmsd` low-priv frontend, `dmspe` privileged
  backend). Do not advise unifying them, do not advise bypassing the
  controlled interface.
- **`-allowed_users` is the gRPC client authorization boundary;
  `dmsgroup` is the dmspe backend-helper unix-group surface, layered
  orthogonally.** Any user that should be allowed to issue DMS
  commands over gRPC must be in the `-allowed_users` comma-separated
  list (default `root`; enforced by `isUserAllowed` in
  `gnxi/utils/credentials/credentials.go` across ALL auth modes); the
  `dmsgroup` Unix group is a separate gate at the dmspe privileged
  helper for users that also need to invoke the backend directly on
  the endpoint. The previous "dmsgroup is the authorization
  boundary" framing is disavowed.
- **Localhost-only auth is never safe to expose.** The "Local
  testing" mode is for development only and binds to localhost. Do
  not advise binding it to an external interface under any
  circumstance.
- **Authentication-mode choice is a security decision.** PAM,
  Credentials, and mTLS each have documented security positioning;
  reads of the public guide's *Security Best Practices* subsection
  are mandatory before prescribing a production deployment.
- **Network exposure follows documented mitigations.** The public
  guide's *Network Exposure Risks and Mitigations* subsection
  enumerates the documented risks (e.g. anonymous binding on a public
  interface) and the documented mitigations. Do not paraphrase those
  bullets — route to the live guide.
- **Underlying-tool destructive operations** (gNOI `OS install`,
  `Reboot`, `Factory-reset`, file-delete) change managed-device state
  in non-trivial ways. Operators should know which sub-operation they
  are invoking before issuing it; the agent's safe default is to
  state the documented effect and ask the user to confirm.

## Public-source pointer

The single canonical public source for DMS is the **DOCA Management
Service Guide** on `docs.nvidia.com`, reachable through
[`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).
Verify that the version of the guide matches the DOCA install on the
target — DMS surface is documented to evolve, so paths and flags can
change between releases.
