---
license: Apache-2.0
name: doca-flow-grpc-server
description: >
  PLAINTEXT-ONLY: the shipped `doca_flow_grpc` server uses
  `grpc::InsecureServerCredentials()` with NO TLS / mTLS / token-auth
  knob on the binary — transport security must come from external
  infrastructure (e.g. an mTLS proxy / sidecar) on a trusted segment.
  Use this skill when bringing up, configuring, hardening, or
  debugging `doca_flow_grpc` — the DOCA-shipped gRPC remote-control
  surface in front of `doca-flow` that lets non-C++ clients (Python,
  Go, Rust, Java) program Flow pipes and entries over RPC instead of
  linking `libdoca_flow.so` directly. Trigger even when the user
  doesn't say 'doca-flow-grpc-server' or 'gRPC' — e.g. 'program Flow
  rules from Python on another host', 'remotely configure pipes on the
  BlueField', 'client times out connecting to the Flow server', 'where
  is the .proto for Flow', 'UNAUTHENTICATED / FAILED_PRECONDITION on a
  Flow RPC'. Route elsewhere for the underlying doca-flow API, generic
  gRPC tooling (protoc, language bindings), or DOCA install / BFB
  bring-up.
metadata:
  kind: tool
compatibility: >
  Requires DOCA on Linux (Ubuntu 22.04/24.04 or RHEL/SLES) with a
  BlueField DPU or ConnectX NIC. The `doca_flow_grpc` binary is a build
  artifact (install: false in tools/flow_grpc_server/meson.build, gated by
  flag_enable_grpc_support + flag_enable_grpc_flow_library) — NOT
  installed under a default DOCA path; build it from the DOCA source tree
  with gRPC enabled. Its `.proto` lives under libs/doca_flow/grpc/.
  Confirm Flow via `pkg-config doca-flow`.

---

# DOCA Flow gRPC Server (`doca_flow_grpc`)

> **CRITICAL transport-security correction (Run-12 + R13).** The
> shipped `doca_flow_grpc` / `doca_flow_grpc_client`
> binaries hard-code the gRPC plaintext credentials surface:
> the **server** uses **`grpc::InsecureServerCredentials()`** (the
> C++ gRPC server-side API in `tools/flow_grpc_server/server/`);
> the **C++ client** uses
> **`grpc::InsecureChannelCredentials()`** (the C++ gRPC
> client-side API; the client lives in
> `libs/doca_flow/grpc/client/`, compiled into the
> `doca_flow` library, NOT under `tools/flow_grpc_client/`);
> the **Python
> client** uses `grpc.aio.insecure_channel(...)`. Do NOT cite the
> server-side string as `grpc::InsecureChannelCredentials()` —
> that is the **client-side** API name and a Grep-against-source
> verification will fail. There
> is **no TLS, no mTLS, and no token-auth** knob on the shipped
> control plane today. Any prose below (or in `CAPABILITIES.md`
> / `TASKS.md`) that frames "mTLS / token auth / TLS posture"
> as a configurable knob on **this** server is the bundle's
> previous aspirational framing and is wrong against the shipped
> source. Treat the server as **plaintext-on-a-trusted-segment
> only**: it MUST be bound on a control-plane-only network
> segment behind an external firewall / VPN / hardened bastion
> that itself enforces TLS + identity. Any "TLS / mTLS / token-
> auth" discussion below is about the operator's external
> hardening layer, NOT a knob on this binary. Routing for an
> in-binary TLS / auth design discussion must say so explicitly
> and route the user to a generic gRPC framework
> ([gRPC auth concepts](https://grpc.io/docs/guides/auth/)) for
> a future-state design, not to a shipped-today knob.

**Where to start:** This is a tool skill for standing up and
operating `doca_flow_grpc`, the DOCA-shipped gRPC remote-
control surface for `doca-flow`. Open [`TASKS.md`](TASKS.md) and
start at [`## configure`](TASKS.md#configure) to decide whether a
remote control plane is the right answer at all (vs talking to
`libdoca_flow.so` directly), then [`## run`](TASKS.md#run) for
the start → bind → one-client-smoke sequence, then
[`## test`](TASKS.md#test) for the smoke-before-bulk loop that
gates any RPC that mutates Flow / dataplane state. Open
[`CAPABILITIES.md`](CAPABILITIES.md) when the question is *what
the gRPC contract surface looks like* (the `.proto` files shipped
under the tool's source tree on the user's install), *how the
auth / TLS posture decision is made*, *which language bindings
the gRPC ecosystem covers*, or *how to interpret the server's
own logs alongside the live Flow application's logs*. If DOCA is
not installed, route to
[`doca-setup`](../../doca-setup/SKILL.md) first; if the user has
not stood up `doca-flow` yet, route to
[`doca-flow`](../../libs/doca-flow/SKILL.md) FIRST — the gRPC
server is a remote control plane on top of the Flow library, not
a replacement for it.

## Example questions this skill answers well

The CLASSES of `doca_flow_grpc` questions this skill is
built to answer, each with one worked example. The class is the
load-bearing piece; the worked example is one instance.

- **"Do I actually need a remote control plane for my Flow
  pipeline, or should my client just link `libdoca_flow.so`
  directly?"** — worked example: *"my client is a Python
  service on a different host; can it program Flow rules
  remotely?"*. Answered by the *when-to-use-gRPC* decision in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the routing into
  [`doca-flow`](../../libs/doca-flow/SKILL.md) when a direct
  library link is the better answer.
- **"Where is the gRPC contract surface actually defined on my
  install?"** — worked example: *"I want to generate a Python
  client; where do I get the `.proto` file?"*. Answered by the
  *the-`.proto`-file-is-the-source-of-truth* rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the language-bindings discussion of standard gRPC tooling
  (`protoc` + the language-specific gRPC plugin per the
  [official gRPC docs](https://grpc.io/docs/) on `grpc.io`).
- **"How do I harden the gRPC endpoint so it isn't an open
  door into my dataplane?"** — worked example: *"the server is
  bound on `0.0.0.0`; what should I do before exposing it?"*.
  Answered by the *admin attack surface* posture in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + the auth / TLS / network-segment decision in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"How do I smoke ONE client end-to-end before opening the
  server to the fleet?"** — worked example: *"my Python client
  can dial the endpoint; what is the first RPC I run to prove
  it talks to the live Flow application?"*. Answered by the
  smoke-before-bulk loop in
  [`TASKS.md ## test`](TASKS.md#test) +
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  smoke-before-bulk rule.
- **"My client cannot reach the server — is the server down,
  the wrong endpoint, a TLS / auth mismatch, or a version
  mismatch?"** — worked example: *"the client times out
  connecting"*. Answered by the layered error taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + the layered ladder in
  [`TASKS.md ## debug`](TASKS.md#debug).
- **"Is my non-C++ client (Python / Go / Rust) actually the
  right shape for the gRPC contract, or is there a cleaner
  path?"** — worked example: *"I want a Rust client; what
  does the `.proto`-generated API look like?"*. Answered by the
  language-bindings discussion in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the routing through standard gRPC tooling.

## Audience

This skill serves **external operators, control-plane developers,
and AI agents who need to program a running DOCA Flow pipeline
from a non-C++ process across a network boundary** instead of
linking `libdoca_flow.so` directly into the controlling process.
Concretely:

- A control-plane engineer writing a Python / Go / Rust client
  that programs Flow rules on a BlueField from outside the
  BlueField's address space.
- A platform operator running a Flow-using service on
  BlueField who wants to expose a remote-control surface to a
  centralized control plane.
- An AI agent driving the *"can I program these Flow rules
  from this client / this network position"* triage step
  before recommending a code change to the surrounding
  doca-flow application.

It is **not** for users debugging the gRPC server's source code,
**not** a substitute for the live public DOCA Flow gRPC Server
guide on `docs.nvidia.com`, and **not** the place to learn the
`doca-flow` API — that audience belongs in
[`doca-flow`](../../libs/doca-flow/SKILL.md).

`doca_flow_grpc` is a **single CLI binary built from the DOCA
source tree** (`executable('doca_flow_grpc', ..., install: false)`
in `tools/flow_grpc_server/meson.build`, gated by
`flag_enable_grpc_support`), plus its companion `.proto`
contract files under `libs/doca_flow/grpc/`; per the tool's
source tree (`server/`, `dpa_device/`, `packet_buffering/`) the
tool can also be paired with a packet-buffering / DPA-side
helper on configurations that need them. The skill uses the
same `kind: tool` three-file shape (`SKILL.md` + `CAPABILITIES.md` + `TASKS.md`) the rest of the bundle's tool slot uses — front matter at the top of this file already says `kind: tool`. (Prior bundle revisions said "library three-file shape" here; that wording was internally inconsistent with the front matter and is corrected.)

## Language scope

This skill governs deployment, configuration, hardening, and
client-side bring-up across the languages standard gRPC tooling
covers — Python, Go, C++, Rust, Java, Node.js, C#, Kotlin, Ruby,
PHP, Dart — via the language-specific gRPC plugin generated
from the shipped `.proto` files (see the
[gRPC language support](https://grpc.io/docs/languages/) index
on `grpc.io`). The server itself is C++ + DOCA; the client
languages are open, gated only by the standard `protoc` plugin
set. For the `doca-flow` API the server programs, see
[`doca-flow`](../../libs/doca-flow/SKILL.md) — that surface is
C-language.

## When to load this skill

Load this skill when the user is — or the agent needs to — bring
up `doca_flow_grpc` against a running `doca-flow`
application (or its preconditions) and connect a non-C++ client
to it. Concretely:

- Deciding whether a remote gRPC control plane is the right
  surface (vs a direct `libdoca_flow.so` link in the client
  process).
- Locating the `.proto` files on the user's install so a
  language-binding client can generate the appropriate
  stubs.
- Deciding the deployment's transport-security posture and
  network segment. NOTE: the shipped server is plaintext-only
  (`grpc::InsecureServerCredentials()`); TLS / mTLS / token-auth
  are NOT binary configuration knobs — they are external
  infrastructure concerns (e.g. an mTLS proxy / sidecar) and the
  plaintext endpoint must stay on a trusted, isolated segment.
- Standing up the server alongside a known-good Flow setup
  and smoke-testing one client end-to-end before exposing
  the endpoint to the fleet.
- Diagnosing a connect / version / RPC failure through the
  layered taxonomy.

Do **not** load this skill for general DOCA orientation,
`doca-flow` API work, DOCA install, or general gRPC tooling
(use the [grpc.io](https://grpc.io/) docs directly for those).

## What this skill provides

This is a **thin loader**. Substantive material lives in two
companion files:

- `CAPABILITIES.md` — what `doca_flow_grpc` exposes:
  the gRPC remote-control surface in front of `doca-flow`,
  the `.proto`-files-as-authoritative-contract rule (the
  shipped `.proto` files under the tool's source on the
  user's install are the source of truth), the *when-to-use-
  gRPC vs direct-library-link* decision, the language-
  bindings story (any language standard gRPC tooling covers),
  the auth / TLS / network-segment decision axis, the
  packet-buffering / DPA-side option per the shipped
  `packet_buffering/` and `dpa_device/` subtrees, the
  version overlay (server rides the `doca-flow` library
  version it links against), the layered error taxonomy
  (server-not-started / server-binding-failed / TLS-or-auth-
  rejected / RPC-call-error / Flow-precondition-failed /
  version / cross-cutting), the observability surface (the
  server's own logs + the live Flow application's logs +
  the RPC client's status codes), and the safety policy
  that treats the endpoint as an admin attack surface.
- `TASKS.md` — step-by-step workflows for the in-scope task
  verbs: `install` (route to setup; binary is built from
  source with gRPC support enabled),
  `configure` (decide remote-vs-direct, pick auth / TLS,
  pick the network segment), `build` (route to install),
  `modify` (refuse — modify the deployment, not the binary),
  `run` (start → bind → smoke), `test` (the
  smoke-before-bulk loop with the client-side stub
  generation step), `debug` (the layered diagnosis ladder),
  `use` (the agent-side workflow for consuming a captured
  gRPC server session), plus a `Deferred task verbs` block
  and a `Command appendix`.

The skill assumes a host where DOCA is already installed (or
the NGC DOCA container is running) with the Flow library
present, a working `doca-flow` application to program against,
and the operator's awareness that exposing a gRPC control plane
is a high-stakes posture.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or scripts
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Verbatim RPC method names, message field inventories, or
  default endpoint paths.** The `.proto` files shipped under
  the tool's source tree on the user's install are the
  authoritative contract; copying them here pins the skill
  to one release and silently rots when the contract
  evolves.
- **Pre-baked client code in any language.** The
  language-specific gRPC plugin + the shipped `.proto` files
  are the contract; client code generated from them on the
  user's installed version is the right answer, not a stub
  pinned to a snapshot.
- **A pre-baked auth / TLS posture (which CA, which token
  source, which mTLS configuration).** That posture is a
  deployment-environment decision — route it to the
  operator's security review and the safety policy.
- **Wrappers, parsers, or scripts** that proxy the gRPC
  endpoint into another protocol. The endpoint is the
  endpoint; if a user wants HTTP/JSON instead, that is a
  separate concern outside this skill's scope.
- **A `samples/`, `bindings/`, or `reference/` subtree.**
  Even one labeled *"reference"* is misleading: operators
  will read it as buildable.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question
   is in scope (the user actually wants a remote gRPC control
   plane on top of `doca-flow`, not a direct library link or
   a different DOCA library).
2. **For what the server exposes, the `.proto`-as-contract
   rule, the language-bindings story, the auth / TLS /
   network-segment decision, version availability, the
   layered error surface, observability, and safety posture,
   see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For the documented start sequence and the
   smoke-before-bulk workflow — `install`, `configure`,
   `build`, `modify`, `run`, `test`, `debug`, `use` — see
   [TASKS.md](TASKS.md).**

## Related skills

- [`doca-flow`](../../libs/doca-flow/SKILL.md) — the **base
  library** the server's gRPC contract is a thin remote-
  control wrapper over. Pipe / entry / rule semantics, the
  validate-before-commit rule, the Flow counter / inspector
  surface all live there.
- [`doca-flow-tune`](../doca-flow-tune/SKILL.md) — the Flow
  tuning tool. When a Flow-program change is recommended,
  the change can be applied through the surrounding
  application or — when the control plane is remote —
  through this gRPC server's RPC surface.
- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  — routing to the public DOCA Flow gRPC Server page on
  `docs.nvidia.com` and the rest of the public DOCA
  documentation set.
- [`doca-version`](../../doca-version/SKILL.md) — canonical
  version-handling rules. The
  [`## Version compatibility`](CAPABILITIES.md#version-compatibility)
  section in this skill is a thin overlay on top.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder. gRPC server failures route into the ladder at
  the runtime layer.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, and the NGC DOCA container path.
- [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) —
  the cross-cutting hardware-safety meta-policy this skill's
  `## Safety policy` overlays. Any state-changing RPC is a
  potential dataplane-affecting change and must respect the
  meta-policy.
