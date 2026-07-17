# DOCA Management Service — Tasks

**Where to start:** The order is `configure → build → modify → run →
test → debug`. The `## test` verb is an iterative loop, not a
one-shot pass — see the eval-loop overlay in `## test` below. For DMS,
`build` and `modify` are about *daemon configuration* (SystemD unit,
flags, YANG-instance fragments), not about compiling source.

These verbs cover the in-scope DMS operational workflows for an
external operator deploying and using DMS. Every step assumes the
operator has consulted the live public DMS guide on `docs.nvidia.com`
and is using it as the authoritative reference; this file prescribes
the *order* and *what to look up where*, not a copy-paste runbook.

## configure

Preparing the management endpoint and choosing a deployment shape.

1. **Confirm the env is healthy first.** This skill expects DOCA to be
   installed at the management endpoint (host, BlueField Arm, or pod
   image base). If that has not been verified, run
   [`doca-setup ## test`](../../doca-setup/TASKS.md#test) first. If
   the user has no install yet, route to
   [`doca-setup ## no-install`](../../doca-setup/TASKS.md#no-install)
   for the public NGC DOCA container path.
2. **Identify the deployment shape.** Read the public DMS guide's
   "Service Deployment" section and map the user's situation to one of
   the documented shapes (Host non-DPU / BlueField Arm / Kubernetes
   pod). Each shape has its own "Prerequisites" subsection — read it
   before any further step.
3. **Identify the management protocol scope.** Confirm whether the
   user needs gNMI (data configuration), gNOI (system operations), or
   both. gNMI Subscribe streaming IS supported (`STREAM` with `SAMPLE`
   1s–60s and `ONCE`; only `POLL` / `Aggregation` are Unimplemented per
   `gnxi/gnmi/server.go`). For a turnkey, productized
   telemetry-aggregation surface separate from DMS's own gNMI
   Subscribe, route to the DOCA Telemetry Service (DTS) instead.
4. **Plan the authentication mode.** Choose one of the four
   documented modes (localhost-only / PAM / credentials / mTLS) and
   read the *Security Best Practices* subsection before committing.
   Localhost-only is for development only and must not be exposed.
5. **Plan user authorization.** Identify the human/service users who
   will issue DMS commands over gRPC and ensure they are in the
   `-allowed_users` list (default `root`; enforced by `isUserAllowed`
   in `gnxi/utils/credentials/credentials.go`). The `dmsgroup` Unix
   group is a separate, `dmspe`-only gate — add a user to it only if
   they also need to invoke the `dmspe` privileged helper directly on
   the endpoint.

## build

DMS is a service, not a library. There is no DMS *application*
artifact for the operator to build — the daemon (`dmsd`/`dmspe`) ships
with the DOCA install, and clients are standard gNMI/gNOI tooling that
the operator already has.

If the user is asking how to build a **gNMI/gNOI client** in their own
language, the right routing is:

- **C / C++** — generic gRPC + gNMI/gNOI client patterns. Outside DMS
  scope. The DMS-specific contribution is the path inventory and the
  operation list documented in `CAPABILITIES.md ## gNMI client surface`
  and `## gNOI client surface`.
- **Other languages (Go / Python / Rust / …)** — same answer. Use
  the standard gNMI/gNOI client libraries for that language; DMS does
  not author wrappers and does not maintain language-specific clients.

If the user is instead asking about building **a DOCA application** in
the general sense (linking against `libdoca-*`), that is *not* a DMS
question — route them to
[`doca-programming-guide ## build`](../../doca-programming-guide/TASKS.md#build)
and the matching `libs/<library>` skill.

## modify

DMS does not have a "modify a sample" workflow analogous to DOCA
libraries; there is no DMS sample program a user starts from. The
DMS analog of "modify" is **adapt the documented launch / auth /
deployment recipe to the user's environment**:

1. **Start from the documented recipe.** Identify the public guide's
   recipe that matches the user's deployment shape and authentication
   mode. Quote it; do not author a new one.
2. **Diff against the user's environment.** Note the specific
   substitutions the user must make: hostnames, certificate paths,
   `allowed_users` lists for PAM mode, port assignments,
   network-exposure decisions.
3. **Apply minimum-change.** Change only what the user's environment
   forces. Every additional deviation from the documented recipe
   widens the surface for an unintended exposure.
4. **Re-validate against the Security Best Practices subsection.**
   Each substitution is a chance to accidentally weaken the documented
   posture (e.g. binding the localhost-only mode to `0.0.0.0`).

## run

Bringing up `dmsd` and exercising it.

1. **Decide SystemD vs manual launch.** SystemD is the documented
   recommended path for production. Manual launch is appropriate for
   bring-up, debugging, and walking the documented advanced
   configurations.
2. **Apply the documented launch flags.** Flag inventory lives under
   "DMS Server Flags (`dmsd`)" in the public guide, partitioned into
   General & Provisioning / Authentication & Security / Authentication
   Method / Security flag families. Quote flags from the live guide;
   do not infer flags from generic gRPC knowledge.
3. **Verify the daemon is reachable.** Use a documented gNMI `Get`
   against a known-supported State path as a connectivity probe.
   "Documented" matters: a `Get` against an unsupported path returns
   an error from `dmsd` regardless of whether the daemon is healthy.
4. **Verify authorization works as expected.** Confirm a user in the
   `-allowed_users` list can issue gRPC commands and a user outside it
   cannot. If the test fails, the failure is at the authorization
   layer, not the protocol layer — see `## debug` step 2.
5. **Issue the user's first real `Set` against a documented path.**
   Then `Get` the corresponding State path to confirm the change
   landed. The `Set` / `Get` pattern across Configuration vs State
   branches is the documented validation idiom.

For gNOI operations: read the operation's documented sub-operation
list first, then issue. Operations like `OS install`, `Reboot`, and
`Factory-reset` change the managed device state in non-trivial ways —
the operator should know which sub-operation they are invoking and
why.

> **⚠️ Destructive and irreversible — confirm before issuing.** `OS
> install`, `Reboot`, and `Factory-reset` are service-impacting and
> cannot be undone: they can take a production BlueField or ConnectX
> offline or wipe its configuration. Before issuing any of them the
> agent MUST (1) verify the target device identity and (2) get explicit
> user confirmation for that specific action. Never issue them
> speculatively or as an implicit side effect of another task.

## test

DMS has no "compile and unit-test" workflow — testing is operational.

**`## test` is an iterative loop, not a one-shot pass.** Every
configuration mutation (auth mode, listener, `-allowed_users`
membership, persistency setting) re-opens the smoke sweep. Skipping the re-run
after a mutation is the failure mode this loop replaces.

The eval-loop overlay (rows apply to every DMS deployment, not just one
topology):

| Step | Why this is a loop, not a step | Where the substance lives |
| --- | --- | --- |
| 1 → 4 → 1 | Capability-snapshot drift (step 4) often reveals an as-deployed gap that needs a configuration change; loop back to step 1 | [`## test`](#test) step 4 |
| 2 → ## debug | When the auth-mode smoke does NOT reject what it should, the deployment is unsafe — escalate to `## debug` immediately, do not run later steps | [`## debug`](#debug) |
| 3 → ## configure → 3 | When persistency does not survive restart, the persistency configuration is wrong — loop back to `## configure` and re-run step 3 | [`## configure`](#configure) |
| 1..4 → ## run | Each loop iteration ends with a documented smoke; if all four pass, hand off to live `## run` traffic | [`## run`](#run) |

The agent's rule: every mutation re-opens the sweep. A
configuration change followed by "it probably still works" is exactly
the failure mode the iterative loop is here to prevent.

1. **Smoke-test the daemon.** After launch (`## run` step 3), confirm
   the daemon answers a documented gNMI `Get` and returns expected
   shape.
2. **Smoke-test the auth mode.** Confirm the chosen authentication
   mode rejects unauthenticated/unauthorized requests as documented.
   For mTLS, this means an explicit "request without cert is
   rejected" check. For PAM, this means an "user not in `allowed_users`
   is rejected" check.
3. **Smoke-test persistency.** If the deployment relies on
   configuration persistency, set a benign value, restart `dmsd`,
   and verify the value survives. The documented state restoration
   mechanism is what is being exercised; if it does not behave as
   documented, that is a bug to file (against DMS, not DOCA Flow or
   any library).
4. **Capability snapshot.** Save the *as-deployed* answer to: which
   gNMI paths your environment supports, which gNOI operations your
   environment supports, which auth mode is active, who is in the
   `-allowed_users` list. This snapshot is the artifact that lets
   future debug sessions skip rediscovery.

## debug

Layered diagnosis. Walk the layers in this order; do not skip down
without clearing the layer above.

1. **Transport layer.** Is the gRPC channel even reaching `dmsd`?
   Symptoms: `UNAVAILABLE`, `DEADLINE_EXCEEDED`, TLS handshake
   failures. Causes: wrong host/port, firewall, daemon not running,
   TLS material misconfigured. Resolution: confirm `dmsd` is running
   (SystemD status or process list), confirm the listener address
   matches the client's target, confirm TLS material if mTLS mode is
   in use.
2. **Authentication / authorization layer.** Symptoms:
   `UNAUTHENTICATED`, `PERMISSION_DENIED`, "user not authorized".
   Causes: wrong credentials, user not in the `-allowed_users` list
   (applies to ALL auth modes, enforced by `isUserAllowed`),
   certificate issuer not trusted by `dmsd`. Resolution: walk the
   documented authentication-mode troubleshooting in the public guide
   for the specific mode in use.
3. **Path / operation layer.** Symptoms: `INVALID_ARGUMENT`, "path
   not found", "operation not supported". Causes: the path is not in
   the DMS-supported set, the operation is not in the documented
   gNOI list, or an unsupported gNMI Subscribe MODE was attempted
   (`POLL` / `Aggregation` are Unimplemented; `STREAM` / `SAMPLE`
   1s–60s and `ONCE` ARE supported).
   Resolution: re-read the supported-paths reference and the gNOI
   operation list in `CAPABILITIES.md`. If the path or operation is
   genuinely not in the public set, the answer is "not supported" —
   not "invent a workaround".
4. **Backend / underlying-tool layer.** Symptoms: the operation
   reached `dmspe` and the underlying tool (e.g. `mlxconfig`, OS
   installer, file system) returned an error. Resolution: extract
   the underlying-tool error from the response and consult that
   tool's own documentation. DMS is the conduit, not the source of
   truth for the underlying tool's failures.
5. **State persistency layer.** Symptoms: a previously-set value
   does not survive a daemon restart. Resolution: confirm automatic
   recording is enabled (it can be disabled per the docs), confirm
   the configuration-persistency file is writable and not being
   overwritten by an external process, then consult the documented
   state restoration mechanism.
6. **Library-level errors.** If DMS is acting as a thin wrapper over
   a DOCA library call and that library returned `DOCA_ERROR_*`, the
   relevant cross-library taxonomy lives in
   [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
   The library-specific overlay (e.g. for Flow) lives in the
   matching `libs/<library>` skill.

## Command appendix

DMS-specific commands the verbs above reach for, grouped by purpose
so the agent picks the right family without searching prose. Every
row is a class — the agent must not invent flags beyond what the row
names; flag discovery is `--help` on the installed binary or the
SystemD unit file, not prose recall.

| Purpose | Command (class shape) | Owning step | Reads as healthy when … |
| --- | --- | --- | --- |
| Daemon lifecycle (SystemD) | `systemctl status dmsd` / `start` / `stop` / `restart` | [`## run`](#run) | `active (running)` with no recent restart loops. |
| Daemon launch (manual) | `dmsd --help` first, then the documented flag set | [`## run`](#run) | Daemon binds the documented listener and emits the expected startup banner. |
| Daemon logs (frontend) | `journalctl -u dmsd --since "5 min ago"` or the documented log file | [`## debug`](#debug) layer Transport/Auth | Lines present for the request window; no auth-rejection storm. |
| Daemon logs (backend) | The documented `dmspe` log destination | [`## debug`](#debug) layer Backend | Backend execution lines present; tool stderr captured. |
| Sanity gNMI Get | A gNMI `Get` on a documented path (e.g. `/system/state/hostname`) | [`## test`](#test) step 1 | Returns the expected typed value. |
| Sanity gNMI Set | A gNMI `Set` on a benign path the user controls | [`## test`](#test) step 1 | Returns success; subsequent `Get` reflects the new value. |
| gNOI sanity (read-only) | A gNOI `System.Time` (or equivalent read-only op the guide lists) | [`## test`](#test) step 1 | Returns the expected time / status. |
| Auth-mode negative test | A request *without* credentials | [`## test`](#test) step 2 | Frontend rejects with the documented `PERMISSION_DENIED` / `UNAUTHENTICATED`. |
| Persistency check | Set a value, restart `dmsd`, re-`Get` | [`## test`](#test) step 3 | The previously set value survives the restart. |
| Capability snapshot | A documented gNMI `Get` enumerating supported paths / ops | [`## test`](#test) step 4 | Output matches the deployment-shape capability matrix. |
| Cross-cutting health | `ss -tlnp \| grep dmsd` (port listener), `ps -ef \| grep dmsd` | [`## debug`](#debug) layer Transport | Daemon listens on the documented port; one `dmsd` process. |

Three cross-cutting rules for this appendix:

- **Never invent a DMS flag.** The public guide is the contract;
  `dmsd --help` against the installed binary is the only secondary
  source. Prose-derived flags are the most common hallucination
  failure for this skill.
- **Frontend logs before backend logs.** When triaging, read
  `journalctl -u dmsd` (or the documented frontend log) first; only
  drop to `dmspe` once the frontend confirms the request reached the
  backend.
- **Cross-link instead of duplicate.** Cross-cutting commands (the
  read-only triple, `dmesg`, `mlxconfig -d <pcie> q`) live in
  [`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
  this appendix names only the DMS-specific ones.

## Deferred task verbs

- **Installing DOCA on the management endpoint** — out of scope here.
  Route to [`doca-setup ## configure`](../../doca-setup/TASKS.md#configure)
  for env preparation and
  [`doca-setup ## test`](../../doca-setup/TASKS.md#test) for install
  health verification, or
  [`doca-setup ## no-install`](../../doca-setup/TASKS.md#no-install)
  for the public NGC DOCA container Path 0 if there is no DOCA install
  yet.
- **Building a custom DOCA application** — not a DMS question. Route
  to [`doca-programming-guide ## build`](../../doca-programming-guide/TASKS.md#build)
  for the canonical build pattern, plus the matching `libs/<library>`
  skill for the API surface.
- **Turnkey telemetry aggregation** — out of scope for DMS. DMS's own
  gNMI Subscribe DOES stream (`STREAM` / `SAMPLE` 1s–60s and `ONCE`;
  only `POLL` / `Aggregation` are Unimplemented); for a productized
  telemetry-aggregation surface, route to the DOCA Telemetry Service
  (DTS), discoverable through
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
- **Library-internal API questions** (Flow pipe construction, RDMA
  queue setup, …) — outside DMS. Route to the matching
  `libs/<library>` skill.

## Cross-cutting

- The public DMS guide is the single source of truth. Any flag,
  path, operation, or auth-mode detail the agent quotes must come
  from there, not from generic gNMI / gNOI / gRPC knowledge.
- DMS is currently **beta** with GA scoped to SPC-X use cases. Treat
  any "what is the long-term roadmap" question as out of scope and
  defer to the live public guide.
- Localhost-only auth is **never** safe to expose externally.
- All operational guidance must preserve the documented two-process
  separation (`dmsd` low-priv frontend, `dmspe` privileged backend).
- For URL routing to the DMS guide and other public DOCA documentation,
  see [doca-public-knowledge-map ## DOCA services](../../doca-public-knowledge-map/SKILL.md#doca-services).
