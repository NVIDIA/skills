# DOCA Flow gRPC Server — Tasks

**Where to start:** The verbs that carry real workflow content
are `## configure`, `## run`, `## test`, and `## debug`. The
other verbs (`## install`, `## build`, `## modify`, `## use`)
carry routing stubs or a tightly-scoped agent-side workflow,
because `doca_flow_grpc` is a build artifact
(`install: false` in `tools/flow_grpc_server/meson.build`,
gated by `flag_enable_grpc_support`) built from the DOCA
source tree against a fixed `.proto` contract, not a binary
the external user patches.

This file is loaded by [`SKILL.md`](SKILL.md) after
[`CAPABILITIES.md`](CAPABILITIES.md). It walks the agent through
the task verbs every artifact in this bundle exposes.

## install

`doca_flow_grpc` is **NOT installed by a DOCA package**
(`install: false` in `tools/flow_grpc_server/meson.build`). It
is a build artifact produced only when DOCA is built from
source with `flag_enable_grpc_support` (and
`flag_enable_grpc_flow_library`) enabled. "Install" for this
tool therefore means "build it from the source tree" — see
[`## build`](#build).

Routing for nearby "install" questions:

- *"The binary isn't there — do I need to install something?"*
  → it is not shipped pre-built; you must build it from the
  DOCA source tree with gRPC support enabled (see
  [`## build`](#build)). For the DOCA SDK / source prerequisites,
  route to
  [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
  or to [`doca-setup ## no-install`](../../doca-setup/TASKS.md#no-install).
- *"I need to install gRPC tooling for my client language."*
  → not a DOCA-side install question. The
  [gRPC quickstart](https://grpc.io/docs/) index on
  `grpc.io` is the authoritative source for the
  language-specific `protoc` + gRPC plugin install.

## configure

`configure` for `doca_flow_grpc` is *"decide whether
gRPC is the right surface AND pick the auth / TLS posture AND
pick the network segment AND locate the `.proto` files BEFORE
starting the server"*. Skipping any of those is the canonical
failure mode.

Steps the agent should walk the user through, in order:

1. **Confirm DOCA is installed and the Flow library is
   present.** Run
   [`doca-setup ## test`](../../doca-setup/TASKS.md#test);
   confirm `pkg-config --modversion doca-flow` resolves. If
   the user has no install yet, route to
   [`doca-setup ## no-install`](../../doca-setup/TASKS.md#no-install)
   before any gRPC discussion.
2. **Decide remote-vs-direct.** Per
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   remote-vs-direct table; if the controlling process can
   link `libdoca_flow.so` directly, that is the simpler,
   smaller-attack-surface answer. The gRPC server is the
   right answer only when the deployment topology requires
   a network boundary or a non-C++ client language without
   bindings.
3. **Locate the `.proto` files on the user's install.** The
   shipped DOCA install contains the `.proto` contract under
   the tool's source tree (per the shipped `server/`
   subtree); the exact path is install-specific. The agent
   should ask the operator to confirm the path rather than
   invent one. The `.proto` file is the authoritative
   contract — see
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   `.proto`-as-contract.
4. **Pick the auth posture.** mTLS-with-client-cert,
   token-based, or plaintext-on-trusted-segment. Per the
   [gRPC auth concepts](https://grpc.io/docs/guides/auth/)
   page on `grpc.io`, gRPC has documented standard
   mechanisms; the agent routes the operator there for the
   design discussion. Plaintext-without-auth is acceptable
   only on a genuinely trusted segment.
5. **Pick the TLS posture.** TLS-encrypted, mTLS, or
   plaintext. The operator must supply the CA / cert / key
   material; the agent never invents a path to a CA or a
   key.
6. **Pick the network segment.** Loopback, a control-plane-
   only subnet, or a broader segment. The agent must
   surface that binding on `0.0.0.0` without an auth +
   firewall posture is the canonical exposure failure.
7. **Decide whether the packet-buffering or DPA-side
   companion is required.** Per the shipped
   `packet_buffering/` and `dpa_device/` source subtrees,
   these are optional components for specific configurations.
   Confirm against the operator's intended Flow workload
   shape per
   [`doca-flow CAPABILITIES.md`](../../libs/doca-flow/CAPABILITIES.md);
   when DPA-offload is involved, cross-link
   [`doca-flow-dpa-perf`](../doca-flow-dpa-perf/SKILL.md)
   for the underlying DPA performance surface.
8. **Sanity check before any invocation.** Confirm with the
   user: what is the client language? Where will the client
   run? Which network segment is the endpoint on? Which auth
   posture? Where are the `.proto` files? If any answer is
   unclear, stop and ask.

Do not invent CLI flag strings, default endpoint shapes, or
`.proto` field names. The public DOCA Flow gRPC Server guide
and the installed binary's `--help` plus the shipped `.proto`
files are the joint source of truth.

## build

`doca_flow_grpc` is **built from the DOCA source tree**, not
shipped pre-built. Its meson target is
`executable('doca_flow_grpc', ..., install: false)` in
`tools/flow_grpc_server/meson.build`, so it is produced in the
build directory but never staged into an install prefix. The
target is gated: `meson` skips it entirely unless
`flag_enable_grpc_support` AND `flag_enable_grpc_flow_library`
are set (the meson `subdir_done()` guards bail out otherwise).

Routing for nearby "build" questions:

- *"The binary isn't there — do I need to build it?"* → yes,
  if you built DOCA without gRPC support. Re-configure the DOCA
  build with `flag_enable_grpc_support` (and
  `flag_enable_grpc_flow_library`) enabled, rebuild, and look
  for `doca_flow_grpc` in the build directory (it is not
  installed). For the DOCA source / build prerequisites, route
  to [`## install`](#install).
- *"I want to generate client stubs for my language."* → that
  is gRPC tooling, not a DOCA build. Route to the
  [gRPC language support](https://grpc.io/docs/languages/)
  index on `grpc.io` for the language-specific `protoc` +
  gRPC plugin invocation; the input is the shipped `.proto`
  files on the user's install.
- *"I want to write my own gRPC server in front of doca-flow."*
  → out of scope here; this skill is for external operators
  consuming the shipped server, not contributors building
  their own.

## modify

**Do not modify the shipped `doca_flow_grpc` binary or
the shipped `.proto` files.** They are the contract; modifying
them in place breaks every client generated against the
unmodified contract and breaks the version-overlay rule per
[`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility).

What the agent *does* modify, every time, is the **server
deployment**: the configuration file (when one exists per the
public guide), the TLS / auth posture, the bind address, the
optional companion components. That is the configuration loop
in [`## configure`](#configure) above.

Routing for nearby "modify" questions:

- *"Can I patch the binary to add a custom RPC?"* → out of
  scope; this skill is for consumers of the shipped server,
  not contributors to it.
- *"Can I subset the `.proto` to expose only some RPCs?"* →
  no, not by editing the shipped `.proto`. Subset by
  *access control* on the server side (auth + per-method
  authorization) instead; the documented surface is the
  documented surface.
- *"I need an RPC the contract doesn't expose."* → out of
  scope here; that is a contract-evolution request that
  belongs upstream of DOCA, not in an external-consumer
  skill.

## run

The start → bind → smoke flow. The full invocation surface
lives in the public DOCA Flow gRPC Server guide; this section
names the *shape* of the flow.

1. **Confirm preconditions.** Per
   [`## configure`](#configure) steps 1-7.
2. **Confirm the surrounding doca-flow application is
   healthy.** Run the Flow application's own smoke per
   [`doca-flow TASKS.md ## test`](../../libs/doca-flow/TASKS.md#test);
   the Flow port is up, at least one pipe is created and
   validated, counters are wired. The gRPC server has no
   value if the Flow side is not yet useful.
3. **Start the server with the chosen auth / TLS / bind
   configuration.** Quote the operator's chosen TLS / auth
   posture back to them before invocation so they can
   challenge it. Capture the server's start / bind log
   lines verbatim — the public DOCA Flow gRPC Server guide
   documents the log shape; do not invent it.
4. **Confirm the server bound the configured endpoint.** On
   success the server's log should show the bind line; on
   failure see the binding-failed layer in
   [`## debug`](#debug). The agent must NOT assume bound;
   it must require evidence.
5. **STOP here.** Do NOT point any client at the server
   until the smoke-before-bulk loop in [`## test`](#test)
   has passed.

When recording the run for downstream consumers, write down:
the DOCA version, the host the server runs on, the auth /
TLS / bind configuration, the configured optional companion
components, and the server's start / bind log block. The
downstream [`## test`](#test) and [`## debug`](#debug)
workflows depend on those fields.

## test

The gRPC server's `## test` is **the canonical
smoke-before-bulk loop** for the deployment. *"Test"* in this
skill means *"prove one client can dial, handshake, and issue
one read-only RPC end-to-end before the endpoint is exposed to
additional clients or any state-changing RPC"*, not
*"unit-test the server"*.

**`## test` is an iterative loop, not a one-shot pass.** Every
mutation — a TLS / auth posture change, a bind-address change,
a `.proto` regeneration, a Flow-application redeploy, a
driver / firmware change — re-opens the smoke.

The smoke-before-bulk shape:

1. **Start the server in a known-good Flow setup.** Per
   [`## run`](#run) steps 1-4.
2. **Generate client stubs in the client language.** Use
   `protoc` + the language-specific gRPC plugin per the
   [gRPC language support](https://grpc.io/docs/languages/)
   index on `grpc.io`, with the shipped `.proto` files on
   the user's install as input. Quote the `protoc` command
   the user actually ran; do not paraphrase it.
3. **Dial the endpoint with the chosen auth / TLS posture.**
   The client confirms the TLS handshake and any auth check
   passes; failures here belong in [`## debug`](#debug)
   layers 2-3.
4. **Issue ONE read-only RPC.** The RPC should be a
   listing / status RPC the `.proto` contract documents;
   the agent must NOT invent the method name from memory.
   The response confirms the server is programming the
   right doca-flow application.
5. **Cross-check the response against the live Flow
   application's own view.** Per
   [`doca-flow CAPABILITIES.md ## Observability`](../../libs/doca-flow/CAPABILITIES.md#observability),
   the Flow library exposes pipe state programmatically;
   if the gRPC response disagrees with the application's
   own view, that is a finding — walk
   [`## debug`](#debug) layer 6 (version) first.
6. **Only after steps 1-5 read clean** may the agent
   recommend exposing the server to additional clients or
   to state-changing RPCs.

Eval-loop overlay (rows apply to every gRPC server
deployment, not just one):

| Step | Why this is a loop, not a step | Where the substance lives |
| --- | --- | --- |
| 1 → ## debug | Server did not bind; walk the binding-failed layer, then re-run step 1 | [`## debug`](#debug) layer 2 |
| 3 → ## debug | TLS handshake or auth fails; walk the TLS / auth layer, then re-run step 1 with the corrected client posture | [`## debug`](#debug) layer 3 |
| 4 → ## debug | RPC returns a non-OK gRPC status code; map the code to the documented contract before retrying | [`## debug`](#debug) layer 4 |
| 5 → ## debug | gRPC response disagrees with the Flow application's own view; walk the version layer before any wider exposure | [`## debug`](#debug) layer 6 |
| 1 → auth / TLS / bind change → 1 | After widening / narrowing the access surface, re-run the smoke; the prior smoke is stale | [`## configure`](#configure) steps 4-6 + [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy) |
| 1 → Flow-side pipe change → 1 | After the Flow application creates or destroys pipes, re-run the smoke to confirm the gRPC server's view reflects the change | [`doca-flow TASKS.md ## modify`](../../libs/doca-flow/TASKS.md#modify) |

The agent's rule: every state-changing action on the server
configuration, the `.proto` contract, or the Flow application
re-opens the smoke. Saving a stale smoke is exactly the
failure mode this loop is here to prevent.

This skill does **not** ship a "test fixture" or pre-recorded
expected output. The expected RPC response is install-,
version-, and application-state-specific; pinning one would
mislead operators on a different platform / version.

## debug

When the user reports a stuck client connection, a TLS / auth
failure, an RPC returning a non-OK status code, or a response
that disagrees with the Flow application, walk the
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
layers in order:

1. **Server-not-started.** Confirm DOCA is installed, the
   Flow library is present, and the binary is actually
   running; check the server's own logs.
2. **Server-binding-failed.** Confirm the configured
   address / port / TLS material; the server's own error
   log is ground truth.
3. **TLS-or-auth-rejected.** Quote the standard gRPC
   status codes (`UNAUTHENTICATED`, `PERMISSION_DENIED`);
   route the client-side auth fix through the
   [gRPC auth concepts](https://grpc.io/docs/guides/auth/)
   page on `grpc.io`.
4. **RPC-call-error.** Match the gRPC status code
   (`INVALID_ARGUMENT`, `NOT_FOUND`, `FAILED_PRECONDITION`,
   etc.) to the documented RPC contract in the `.proto`
   files; the
   [gRPC status codes](https://grpc.io/docs/guides/status-codes/)
   reference on `grpc.io` is the canonical interpretation
   guide.
5. **Flow-precondition-failed.** Confirm the underlying
   Flow application is in a state to accept the RPC per
   [`doca-flow TASKS.md ## modify`](../../libs/doca-flow/TASKS.md#modify);
   re-running the RPC against an unready Flow application
   is the wrong move.
6. **Version.** Walk
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   end-to-end; common gRPC-server-specific symptom is a
   client generated from a different DOCA release's
   `.proto` than the server binary.
7. **Cross-cutting.** Hand off to
   [`doca-debug ## debug`](../../doca-debug/SKILL.md) and
   [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug)
   for the env-side layers (driver, firmware, BlueField
   mode, network reachability).

In every case: **quote the standard gRPC status code and the
server's log line verbatim.** Paraphrasing the status code is
the canonical lost-fidelity failure for this skill.

## use

`## use` is the agent-side workflow for *consuming* a
captured `doca_flow_grpc` session as evidence.

1. **Read the captured server logs and client status codes
   together.** A server-side bind line without a
   corresponding client-side connect line (or vice versa)
   is half the picture.
2. **Read the gRPC status code, not the user's prose
   summary of it.** The status code is the contract; the
   prose is the user's interpretation.
3. **Cross-check the gRPC response against the Flow
   application's own counter / inspector view.** Disagreement
   is signal — it routes to [`## debug`](#debug) layer 6.
4. **Route to [`doca-flow TASKS.md ## modify`](../../libs/doca-flow/TASKS.md#modify)**
   only when the agent confirms the next step is a
   Flow-program change, not a gRPC-server-configuration
   change.

## Deferred task verbs

The verbs below are not `doca_flow_grpc` work and
should be routed out before the agent does any of them under
this skill's name.

- **install DOCA** ⇒
  [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
  and [`## no-install`](../../doca-setup/TASKS.md#no-install).
- **write a doca-flow application** ⇒
  [`doca-flow`](../../libs/doca-flow/SKILL.md), layered on
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
  The gRPC server is a remote control plane on top of the
  Flow library; it is not a template for creating Flow
  applications.
- **library-internal pipe / counter / inspector deep dive**
  ⇒ [`doca-flow`](../../libs/doca-flow/SKILL.md). The gRPC
  server transports the same data the Flow library exposes
  programmatically; the deeper per-pipe semantics belong to
  the library.
- **generic gRPC tooling** (`protoc` install, language
  bindings, auth design) ⇒ [`grpc.io`](https://grpc.io/)
  directly. This skill does not duplicate the gRPC ecosystem
  documentation.
- **streaming telemetry / live metrics export of Flow
  KPIs** ⇒ not a feature of this tool. The DOCA Telemetry
  Service (DTS) is the documented telemetry surface;
  routing belongs in
  [`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).

## Command appendix

`doca_flow_grpc`-specific invocation classes the
verbs above reach for. Every row is a CLASS — the agent
must not invent RPC method names, message field names, or
endpoint paths beyond the shipped `.proto` files and the
public DOCA Flow gRPC Server guide.

**Infra-aware preamble (every row below).** Per the bundle's
detect → prefer → fall back → report contract documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent should:

1. Probe for the matching structured helper FIRST
   (`doca-env --json`; `doca-capability-snapshot`;
   `version-matrix.json`).
2. If the probe succeeds, the structured tool's output is the
   authoritative answer.
3. If the probe fails, fall back to the manual command in
   the row.
4. The schemas the structured tools emit are defined in
   [`doca-structured-tools-contract ## Schemas`](../../doca-structured-tools-contract/SKILL.md#schemas);
   the version-handling semantics are owned by
   [`doca-version`](../../doca-version/SKILL.md).

| Purpose (class) | Invocation (shape) | Owning step | Reads as healthy when … |
| --- | --- | --- | --- |
| Discover the documented CLI surface | `doca_flow_grpc --help` plus the public DOCA Flow gRPC Server guide | [`## configure`](#configure) step 8 + [`## debug`](#debug) layer 1 | Prints the documented flag inventory; the agent uses this as the only source of truth for flag names. |
| Confirm DOCA Flow library version | `pkg-config --modversion doca-flow` on the side the gRPC server runs | [`## configure`](#configure) step 1 + [`## debug`](#debug) layer 6 | Matches `doca_caps --version` and the version the client-side `.proto` was generated from; disagreement = partial install (route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2). |
| Locate the shipped `.proto` files | On the monorepo source tree the Flow gRPC `.proto` files live under `doca/libs/doca_flow/grpc/` (`common.proto`, `doca_flow.proto`, `packet_buffering/packet_buffering.proto`) — NOT under `doca/tools/flow_grpc_server/`. On a binary install they are shipped via the `doca-flow` include / share path; the agent runs `pkg-config doca-flow --variable=prefix` then `find <prefix> -name '*.proto'` to pin the actual install path on the user's host instead of memorizing one. | [`## configure`](#configure) step 3 | The `.proto` files exist on the user's install at the path the `find` confirms; the agent quotes the confirmed path, not memory. |
| Generate client stubs in the chosen language | `protoc` + the language-specific gRPC plugin per [grpc.io](https://grpc.io/docs/languages/), with the shipped `.proto` files as input | [`## test`](#test) step 2 | The generated stubs compile in the client language; the client can construct request / response messages matching the contract. |
| Confirm a client can dial the server end-to-end | The chosen client language's gRPC dial / channel / stub invocation against the configured endpoint with the configured auth / TLS posture | [`## run`](#run) step 4 + [`## test`](#test) step 3 | The TLS handshake succeeds; the auth check passes; the channel is ready to issue RPCs. |
| Issue ONE read-only RPC to confirm the server is wired to the live Flow application | The client invocation for a listing / status RPC documented in the shipped `.proto` files (the specific method name comes from the `.proto`, NOT from agent memory) | [`## test`](#test) step 4 | Exit 0 / `OK` status; the response reflects the live Flow application's state. |
| Save a session snapshot for debug | Capture (a) the server's start / bind log, (b) the client's full command line + stub generation command + dial + RPC trace, (c) the gRPC status code(s) verbatim, (d) the four-tuple of (DOCA version, host, auth / TLS / bind config, Flow application state) | [`## test`](#test) save step + [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug) | The saved bundle is consumed by the cross-cutting debug ladder. |

Three cross-cutting rules for this appendix:

- **Never invent an RPC method name, message field name,
  endpoint path, or auth-mechanism identifier.** The
  shipped `.proto` files on the user's install plus the
  public DOCA Flow gRPC Server guide plus the
  `grpc.io` documentation are the joint contract;
  prose-derived names are the most common hallucination
  failure for this skill.
- **State-changing RPCs re-open the smoke.** They are not
  retryable in place; after any state-changing RPC, the
  agent re-runs the read-only smoke per [`## test`](#test)
  before issuing anything else.
- **Cross-link instead of duplicate.** Cross-cutting
  commands (`pkg-config --modversion`, `dmesg`, network
  tooling) live in
  [`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md);
  the Flow application build / port / pipe commands live in
  [`doca-flow TASKS.md ## Command appendix`](../../libs/doca-flow/TASKS.md);
  gRPC ecosystem commands (`protoc`, auth) live at
  `grpc.io`; this appendix names only Flow-gRPC-server-
  specific invocations on top.

## Cross-cutting

A few rules that apply across every verb in this file:

- The **public DOCA Flow gRPC Server guide** + the installed
  binary's `--help` + the shipped `.proto` files are the
  joint source of truth on the DOCA side; the
  [`grpc.io`](https://grpc.io/) documentation is the joint
  source of truth on the gRPC ecosystem side.
- The **`.proto` file is the contract.** Quote it; do not
  paraphrase RPC names from prose memory.
- The **endpoint is an admin attack surface.** Bind on a
  trusted segment, gate with auth + TLS, smoke before bulk.
- **Quote the gRPC status code, not the user's summary of
  it.** The status code is the contract; the summary is the
  user's interpretation.
- This skill **assumes a healthy DOCA install** (or the
  public NGC DOCA container) and a working
  [`doca-flow`](../../libs/doca-flow/SKILL.md) application
  to program against. If either is in doubt, route to
  [`doca-setup`](../../doca-setup/SKILL.md) and
  [`doca-flow`](../../libs/doca-flow/SKILL.md) before
  running anything else here.
